"""Opt-in B8/B1 CUDA-graph activation observer; no numerical changes."""

from pathlib import Path

import torch


class QSAHiSparseTrace:
    TARGETS = {"b8-shape-A0": "b8", "b8-shape-B1": "b1"}
    PROMPT = 261120
    STEPS = (8, 9)
    MAX_BATCH = 8

    def __init__(self, adapter, directory):
        if (not adapter.strict or adapter.mode != "p2-offload"
                or not adapter.graph_enabled or adapter.max_requests != self.MAX_BATCH):
            raise ValueError("B8 shape trace requires strict P2 offload and full B8 graphs")
        self.adapter = adapter
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        model = adapter.runner.model
        body = model.model
        if (len(body.layers) != 48 or adapter.layer_ids != list(range(3, 48, 4))
                or body.hc_count != 4 or not hasattr(body, "hyper_connection_mixer")):
            raise ValueError("B8 shape trace requires the 48-layer Qwen TP2 geometry")
        dtype = adapter.runner.model_config.dtype
        if dtype != torch.bfloat16:
            raise ValueError("B8 shape trace requires the frozen bfloat16 model")

        hidden = adapter.runner.model_config.hidden_size
        hc_hidden = body.hc_count * hidden
        experts = body.config.num_experts
        vocab = model.config.vocab_size
        self.buffers, self.layouts = {}, {}
        self.meta = None
        self.identities, self.seen, self.scales = {}, set(), {}

        def add(name, shape, field_dtype=dtype):
            self.buffers[name] = torch.empty(shape, dtype=field_dtype, device=adapter.device)

        add("input_ids", (self.MAX_BATCH,), torch.int64)
        add("positions", (self.MAX_BATCH,), torch.int64)
        self.mrope = model.is_mrope_enabled
        if self.mrope:
            add("mrope_positions", (self.MAX_BATCH, 3), torch.int64)
        add("embedding", (self.MAX_BATCH, hidden))
        add("final_mix", (self.MAX_BATCH, hidden))
        add("final_hc", (self.MAX_BATCH, hc_hidden))
        add("logits", (self.MAX_BATCH, vocab), torch.float32)
        for index, layer in enumerate(body.layers):
            add(f"layer.{index}.input", (self.MAX_BATCH, hidden if index == 0 else hc_hidden))
            add(f"attention.{index}.input", (self.MAX_BATCH, 12 * 256)
                if index in adapter.layer_ids else (self.MAX_BATCH, hidden))
            add(f"attention.{index}.output", (self.MAX_BATCH, hidden))
            add(f"mlp.{index}.input", (self.MAX_BATCH, hidden))
            add(f"router.{index}", (self.MAX_BATCH, experts))
            add(f"experts.{index}.output", (self.MAX_BATCH, hidden))
            shared = getattr(layer.mlp, "shared_expert", None)
            if shared is not None:
                add(f"shared_expert.{index}.output", (self.MAX_BATCH, hidden))
            add(f"mlp.{index}.output", (self.MAX_BATCH, hidden))
            add(f"layer.{index}.output", (self.MAX_BATCH, hc_hidden))
        for index in adapter.layer_ids:
            add(f"qsa.{index}.q", (self.MAX_BATCH, 12, 256))
            add(f"qsa.{index}.indices", (self.MAX_BATCH, 2051), torch.int32)
            add(f"qsa.{index}.output", (self.MAX_BATCH, 12, 256))
            add(f"qsa.{index}.valid_counts", (self.MAX_BATCH,), torch.int32)

        self.field_indices = {name: index for index, name in enumerate(self.buffers)}
        self.epoch = torch.zeros(1, dtype=torch.int64, device=adapter.device)
        self.stamps = torch.full(
            (len(self.buffers),), -1, dtype=torch.int64, device=adapter.device
        )
        adapter.workspace.extend((*self.buffers.values(), self.epoch, self.stamps))

        self.handles = [
            body.register_forward_pre_hook(self._inputs, with_kwargs=True),
            body.register_forward_hook(self._body_output),
            body.embed_tokens.register_forward_hook(self._output_hook("embedding")),
            model.logits_processor.register_forward_hook(self._logits),
        ]
        for index, layer in enumerate(body.layers):
            self.handles += [
                layer.register_forward_pre_hook(
                    self._input_hook(f"layer.{index}.input", "hidden_states"),
                    with_kwargs=True,
                ),
                layer.register_forward_hook(self._layer_output_hook(index)),
            ]
            attention = layer.o_proj if index in adapter.layer_ids else layer.linear_attn
            self.handles += [
                attention.register_forward_pre_hook(
                    self._input_hook(f"attention.{index}.input"), with_kwargs=True
                ),
                attention.register_forward_hook(
                    self._output_hook(f"attention.{index}.output")
                ),
                layer.mlp.register_forward_pre_hook(
                    self._input_hook(f"mlp.{index}.input"), with_kwargs=True
                ),
                layer.mlp.gate.register_forward_hook(self._output_hook(f"router.{index}")),
                layer.mlp.experts.register_forward_hook(
                    self._output_hook(f"experts.{index}.output")
                ),
                layer.mlp.register_forward_hook(self._output_hook(f"mlp.{index}.output")),
            ]
            shared = getattr(layer.mlp, "shared_expert", None)
            if shared is not None:
                self.handles.append(
                    shared.register_forward_hook(
                        self._output_hook(f"shared_expert.{index}.output")
                    )
                )

    @staticmethod
    def _tensor(value):
        while isinstance(value, (tuple, list)):
            value = value[0]
        return value

    def _active(self):
        return self.adapter.graph_capture_size is not None or self.meta is not None

    def _stage(self, name, value):
        if not self._active():
            return
        tensor = self._tensor(value)
        target = self.buffers[name]
        if (not isinstance(tensor, torch.Tensor) or tensor.dtype != target.dtype
                or tensor.ndim != target.ndim or tensor.shape[1:] != target.shape[1:]):
            raise RuntimeError(f"B8 shape trace layout differs: {name}")
        count = self.adapter.graph_capture_size
        rows = count if count is not None else self.meta["batch_size"]
        if tensor.shape[0] != rows or rows > self.MAX_BATCH:
            raise RuntimeError(f"B8 shape trace row count differs: {name}")
        key = ("graph", rows)
        self.layouts.setdefault(key, {})[name] = {
            "shape": list(tensor.shape),
            "stride": list(tensor.stride()),
            "dtype": str(tensor.dtype),
        }
        target[:rows].copy_(tensor)
        self.stamps[self.field_indices[name]:self.field_indices[name] + 1].copy_(self.epoch)

    def _inputs(self, _module, args, kwargs):
        if not self._active():
            return
        value = lambda name, index: kwargs[name] if name in kwargs else args[index]
        batch = value("forward_batch", 2)
        ids = value("input_ids", 0)
        self._stage("input_ids", batch.input_ids if ids is None else ids)
        self._stage("positions", batch.positions)
        if self.mrope:
            self._stage("mrope_positions", value("positions", 1).T)

    def _input_hook(self, name, keyword=None):
        def hook(_module, args, kwargs):
            if self._active():
                self._stage(name, kwargs[keyword] if keyword in kwargs else args[0])
        return hook

    def _output_hook(self, name):
        def hook(_module, _args, output):
            if self._active():
                self._stage(name, output)
        return hook

    def _body_output(self, module, _args, output):
        if self._active():
            self._stage("final_mix", output)
            self._stage("final_hc", module.last_hc_hidden_states)

    def _layer_output_hook(self, index):
        def hook(_module, _args, output):
            if self._active():
                hidden, residual = output
                if residual is not None:
                    raise RuntimeError("B8 Qwen4 layer residual must be None")
                self._stage(f"layer.{index}.output", hidden)
        return hook

    def _logits(self, _module, _args, output):
        if self._active():
            self._stage("logits", output.next_token_logits)

    def attention(self, layer, q, _k, _v, indices, output, k_scale, v_scale,
                  valid_counts, _cu_q, _cu_k):
        if not self._active():
            return
        lid = layer.layer_id
        if not all(isinstance(value, (int, float))
                   for value in (k_scale, v_scale, layer.scaling)):
            raise RuntimeError("B8 shape trace descales must be host scalars")
        self.scales[lid] = (k_scale, v_scale, layer.scaling)
        self._stage(f"qsa.{lid}.q", q)
        self._stage(f"qsa.{lid}.indices", indices)
        self._stage(f"qsa.{lid}.output", output)
        self._stage(f"qsa.{lid}.valid_counts", valid_counts)

    def begin(self, batch):
        if self.meta is not None:
            raise RuntimeError("B8 shape trace previous forward was not completed")
        if not batch.forward_mode.is_decode():
            return
        matches = [(row, state, self.TARGETS[state.lease.rid])
                   for row, state in enumerate(self.adapter.batch_requests)
                   if state.lease.rid in self.TARGETS]
        if not matches:
            return
        if len(matches) != 1:
            raise RuntimeError("B8 shape trace target count differs")
        row, state, label = matches[0]
        step = state.decode_steps
        if step not in self.STEPS:
            return
        if (label, step) in self.seen:
            raise RuntimeError("B8 shape trace duplicate step")
        if step == self.STEPS[1] and (label, self.STEPS[0]) not in self.seen:
            raise RuntimeError("B8 shape trace missing prior step")
        lease = state.lease
        identity = (lease.req_pool_idx, lease.generation, lease.rid, lease.slot)
        if label in self.identities and identity != self.identities[label]:
            raise RuntimeError("B8 shape trace lease identity changed")
        self.identities[label] = identity
        if state.seq_len != self.PROMPT + step:
            raise RuntimeError("B8 shape trace sequence length differs")
        self.meta = {
            "label": label,
            "rid": lease.rid,
            "req_pool_idx": lease.req_pool_idx,
            "generation": lease.generation,
            "slot": lease.slot,
            "rank": self.adapter.rank,
            "row": row,
            "batch_size": len(self.adapter.batch_requests),
            "seq_len": state.seq_len,
            "step": step,
            "forward_id": self.adapter.forward_id,
        }
        self.epoch.fill_(self.adapter.forward_id)

    def finish(self, *, graph):
        if self.meta is None:
            return
        if self.adapter.graph_capture_size is not None or not graph:
            raise RuntimeError("B8 shape trace requires graph replay completion")
        meta = self.meta
        row, count = meta["row"], meta["batch_size"]
        state = self.adapter._request(meta["req_pool_idx"], meta["rid"])
        lease = state.lease
        identity = (lease.req_pool_idx, lease.generation, lease.rid, lease.slot)
        graph_batch = self.adapter.graph_batch
        if (identity != self.identities[meta["label"]]
                or len(self.adapter.batch_requests) != count
                or self.adapter.batch_requests[row] is not state
                or state.seq_len != meta["seq_len"]
                or state.decode_steps != meta["step"]
                or self.adapter.forward_id != meta["forward_id"]
                or graph_batch is None or len(graph_batch) != count
                or graph_batch[row] != (lease, state.seq_len)):
            raise RuntimeError("B8 shape trace completion identity/step changed")
        self.adapter.slots.require(lease, "decode")
        stamps = self.stamps.cpu().tolist()
        if any(stamps[index] != meta["forward_id"] for index in range(len(self.buffers))):
            raise RuntimeError("B8 shape trace has stale or missing device fields")
        layouts = self.layouts.get(("graph", count), {})
        if set(layouts) != set(self.buffers):
            raise RuntimeError("B8 shape trace graph layout is incomplete")

        tensors = {
            name: value[row:row + 1].detach().to("cpu", copy=True)
            for name, value in self.buffers.items()
        }
        path = self.directory / (
            f"{meta['label']}-step-{meta['step']:03d}-rank-{self.adapter.rank}.pt"
        )
        temporary = path.with_suffix(".tmp")
        if path.exists() or temporary.exists():
            raise RuntimeError("B8 shape trace refuses to overwrite a sample")
        torch.save(
            {
                "schema": "qsa-b8-shape-trace-v1",
                **meta,
                "graph": True,
                "layouts": layouts,
                "scales": dict(self.scales),
                "tensors": tensors,
            },
            temporary,
        )
        temporary.replace(path)
        self.seen.add((meta["label"], meta["step"]))
        self.meta = None
