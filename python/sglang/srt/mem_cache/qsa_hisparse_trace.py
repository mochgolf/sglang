"""Opt-in, bounded P4 B49 activation evidence; no numerical changes."""

import json
from pathlib import Path

import torch


class QSAHiSparseTrace:
    RID = "p4-diagnostic-B"
    PROMPT = 261118
    STEPS = 48

    def __init__(self, adapter, directory):
        if not adapter.strict or adapter.mode != "p2-offload":
            raise ValueError("P4 trace requires strict P2 offload")
        self.adapter = adapter
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        model = adapter.runner.model
        body = model.model
        if (len(body.layers) != 48 or adapter.layer_ids != list(range(3, 48, 4))
                or body.hc_count != 4 or not hasattr(body, "hyper_connection_mixer")):
            raise ValueError("P4 trace requires the 48-layer Qwen TP2 geometry")
        dtype = adapter.runner.model_config.dtype
        if dtype != torch.bfloat16:
            raise ValueError("P4 trace requires the frozen bfloat16 model")
        hidden = adapter.runner.model_config.hidden_size
        self.buffers, self.kinds, self.layouts, self.scales = {}, {}, {}, {}
        self.meta = self.identity = None
        self.next_step = 0

        def add(name, shape, field_dtype=dtype, kind="row"):
            self.buffers[name] = torch.empty(shape, dtype=field_dtype, device=adapter.device)
            self.kinds[name] = kind

        add("input_ids", (2,), torch.int64)
        add("positions", (2,), torch.int64)
        self.mrope = model.is_mrope_enabled
        if self.mrope:
            add("mrope_positions", (2, 3), torch.int64)
        add("embedding", (2, hidden))
        add("final_mix", (2, hidden))
        add("logits", (2, model.config.vocab_size), torch.float32)
        for index, layer in enumerate(body.layers):
            add(f"layer.{index}.hidden", (2, body.hc_count * hidden))
            add(f"projection.{index}" if index in adapter.layer_ids else f"linear.{index}", (2, hidden))
        self.base_fields = tuple(self.buffers)
        for index in adapter.layer_ids:
            for name in ("q", "output"):
                add(f"qsa.{index}.{name}", (2, 12, 256))
            add(f"qsa.{index}.indices", (2, 2051), torch.int32)
            add(f"qsa.{index}.valid_counts", (2,), torch.int32)
            for name in ("cu_q", "cu_k"):
                add(f"qsa.{index}.{name}", (3,), torch.int32, "cu")
            for name in ("k", "v"):
                add(f"qsa.{index}.{name}", (4102, 1, 256), dtype, "packed")
        self.field_indices = {name: index for index, name in enumerate(self.buffers)}
        self.epoch = torch.zeros(1, dtype=torch.int64, device=adapter.device)
        self.stamps = torch.full((len(self.buffers),), -1, dtype=torch.int64, device=adapter.device)
        adapter.workspace.extend((*self.buffers.values(), self.epoch, self.stamps))
        # Runners call the outer model.forward directly, bypassing its hooks.
        self.handles = [body.register_forward_pre_hook(self._inputs, with_kwargs=True),
                        model.logits_processor.register_forward_hook(self._logits),
                        body.embed_tokens.register_forward_hook(self._output_hook("embedding")),
                        body.register_forward_hook(self._output_hook("final_mix"))]
        for index, layer in enumerate(body.layers):
            self.handles.append(layer.register_forward_hook(self._layer_hook(index)))
            child = layer.o_proj if index in adapter.layer_ids else layer.linear_attn
            name = f"projection.{index}" if index in adapter.layer_ids else f"linear.{index}"
            self.handles.append(child.register_forward_hook(self._output_hook(name)))
        inventory = {name: {"shape": list(t.shape), "dtype": str(t.dtype),
                            "ptr": t.data_ptr(), "bytes": t.numel() * t.element_size()}
                     for name, t in {**self.buffers, "epoch": self.epoch, "stamps": self.stamps}.items()}
        self._write_json(f"inventory-rank-{adapter.rank}.json", inventory)

    def _write_json(self, name, value):
        path = self.directory / name
        temporary = path.with_suffix(".tmp")
        if path.exists() or temporary.exists():
            raise RuntimeError("P4 trace output already exists")
        temporary.write_text(json.dumps(value, indent=2) + "\n")
        temporary.replace(path)

    def begin(self, batch):
        if self.meta is not None:
            raise RuntimeError("P4 trace previous forward was not completed")
        states = self.adapter.batch_requests
        target = [s for s in states if s.lease.rid == self.RID]
        if not target:
            return
        if len(states) != 1:
            raise RuntimeError("P4 trace does not support a live B2 batch")
        state = target[0]
        lease = state.lease
        identity = (lease.req_pool_idx, lease.generation, lease.rid, lease.slot)
        if self.identity is not None and identity != self.identity:
            raise RuntimeError("P4 trace lease identity changed")
        self.identity = identity
        decode = batch.forward_mode.is_decode()
        if not decode and state.seq_len != self.PROMPT:
            return
        step = state.decode_steps if decode else 0
        if step != self.next_step or not 0 <= step <= self.STEPS:
            raise RuntimeError("P4 trace missing, duplicate or out-of-order step")
        if state.seq_len != self.PROMPT + step:
            raise RuntimeError("P4 trace sequence length differs")
        self.meta = dict(rid=lease.rid, req_pool_idx=lease.req_pool_idx,
                         generation=lease.generation, slot=lease.slot,
                         rank=self.adapter.rank, seq_len=state.seq_len, step=step,
                         phase="decode" if decode else "prefill",
                         forward_id=self.adapter.forward_id)
        self.epoch.fill_(self.adapter.forward_id)

    def _active(self):
        return self.adapter.graph_capture_size is not None or self.meta is not None

    def _stage(self, name, tensor):
        if not self._active():
            return
        target = self.buffers[name]
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != target.dtype:
            raise RuntimeError(f"P4 trace dtype/type differs: {name}")
        count = self.adapter.graph_capture_size
        key = ("graph", count) if count is not None else ("eager", self.meta["phase"])
        self.layouts.setdefault(key, {})[name] = {
            "shape": list(tensor.shape), "stride": list(tensor.stride()), "dtype": str(tensor.dtype)}
        if count is None and self.meta["phase"] == "prefill":
            tensor = tensor[-1:]
        if tensor.shape[1:] != target.shape[1:] or tensor.shape[0] > target.shape[0]:
            raise RuntimeError(f"P4 trace shape exceeds fixed staging: {name}")
        rows = count if count is not None else 1
        kind = self.kinds[name]
        if (kind == "row" and tensor.shape[0] != rows
                or kind == "cu" and tensor.shape[0] != rows + 1
                or kind == "packed" and tensor.shape[0] < rows * 2051):
            raise RuntimeError(f"P4 trace row count differs: {name}")
        target[:tensor.shape[0]].copy_(tensor)
        index = self.field_indices[name]
        self.stamps[index:index + 1].copy_(self.epoch)

    def _inputs(self, module, args, kwargs):
        if not self._active():
            return
        value = lambda name, index: kwargs[name] if name in kwargs else args[index]
        batch = value("forward_batch", 2)
        ids = value("input_ids", 0)
        self._stage("input_ids", batch.input_ids if ids is None else ids)
        self._stage("positions", batch.positions)
        if self.mrope:
            self._stage("mrope_positions", value("positions", 1).T)

    def _logits(self, module, args, output):
        if self._active():
            self._stage("logits", output.next_token_logits)

    def _output_hook(self, name):
        def hook(module, args, output):
            if self._active():
                self._stage(name, output[0] if isinstance(output, tuple) else output)
        return hook

    def _layer_hook(self, index):
        def hook(module, args, output):
            if self._active():
                hidden, residual = output
                if residual is not None:
                    raise RuntimeError("P4 Qwen4 layer residual must be None")
                self._stage(f"layer.{index}.hidden", hidden)
        return hook

    def attention(self, layer, q, k, v, indices, output, k_scale, v_scale,
                  valid_counts, cu_q, cu_k):
        if not self._active():
            return
        lid = layer.layer_id
        if not all(isinstance(value, (int, float)) for value in (k_scale, v_scale, layer.scaling)):
            raise RuntimeError("P4 trace descales must be host scalars")
        self.scales[lid] = (k_scale, v_scale, layer.scaling)
        for name, tensor in (("q", q), ("k", k), ("v", v), ("indices", indices),
                             ("output", output), ("valid_counts", valid_counts),
                             ("cu_q", cu_q), ("cu_k", cu_k)):
            self._stage(f"qsa.{lid}.{name}", tensor)

    def finish(self, *, graph):
        if self.meta is None:
            return
        if self.adapter.graph_capture_size is not None:
            raise RuntimeError("P4 trace cannot serialize during capture")
        meta = self.meta
        if graph != (self.adapter.graph_enabled and meta["phase"] == "decode"):
            raise RuntimeError("P4 trace completion path differs")
        state = self.adapter._request(meta["req_pool_idx"], meta["rid"])
        lease = state.lease
        identity = (lease.req_pool_idx, lease.generation, lease.rid, lease.slot)
        if (identity != self.identity or len(self.adapter.batch_requests) != 1
                or self.adapter.batch_requests[0] is not state
                or state.seq_len != meta["seq_len"] or state.decode_steps != meta["step"]
                or self.adapter.forward_id != meta["forward_id"]
                or graph and self.adapter.graph_batch != ((lease, state.seq_len),)):
            raise RuntimeError("P4 trace completion identity/step changed")
        self.adapter.slots.require(lease, "decode" if meta["phase"] == "decode" else "prefill")
        fields = self.base_fields if meta["phase"] == "prefill" else tuple(self.buffers)
        stamps = self.stamps.cpu().tolist()
        if any(stamps[self.field_indices[name]] != meta["forward_id"] for name in fields):
            raise RuntimeError("P4 trace has stale or missing device fields")
        key = ("graph", 1) if graph else ("eager", meta["phase"])
        layouts = self.layouts[key]
        if any(name not in layouts for name in fields):
            raise RuntimeError("P4 trace layout is incomplete")
        tensors = {}
        for name in fields:
            kind = self.kinds[name]
            size = 2 if kind == "cu" else 2051 if kind == "packed" else 1
            tensors[name] = self.buffers[name][:size].detach().to("cpu", copy=True)
        if meta["phase"] == "decode":
            for lid in self.adapter.layer_ids:
                prefix = f"qsa.{lid}."
                valid = int(tensors[prefix + "cu_k"][-1])
                if not 0 < valid <= 2051 or tensors[prefix + "valid_counts"].tolist() != [valid]:
                    raise RuntimeError("P4 trace packed valid prefix differs")
                for name in ("k", "v"):
                    tensors[prefix + name] = tensors[prefix + name][:valid].clone()
        path = self.directory / f"step-{meta['step']:03d}-rank-{self.adapter.rank}.pt"
        temporary = path.with_suffix(".tmp")
        if path.exists() or temporary.exists():
            raise RuntimeError("P4 trace refuses to overwrite a sample")
        torch.save({"schema": "qsa-p4-trace-v1", **meta, "graph": graph,
                    "layouts": {name: layouts[name] for name in fields},
                    "null_fields": [f"layer.{i}.residual" for i in range(48)],
                    "scales": dict(self.scales), "tensors": tensors}, temporary)
        temporary.replace(path)
        self.next_step += 1
        self.meta = None
