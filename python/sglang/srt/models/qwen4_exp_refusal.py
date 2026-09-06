"""Small runtime hooks for Qwen4-Exp refusal-direction experiments.

The hook is deliberately opt-in.  It records only eager prefill rows whose
absolute position is the end of the current full prompt, so a decode tail or
an incomplete chunked-prefill row cannot become a direction sample.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch


def _rank_suffix(rank_info: dict[str, Any]) -> str:
    return "-".join(
        f"{key}{int(rank_info.get(key, 0))}"
        for key in ("tp_rank", "pp_rank")
    )


_CAPTURE_FORWARD_MODES = frozenset({"EXTEND", "SPLIT_PREFILL", "DLLM_EXTEND"})


def _prefill_last_prompt_row_records(
    hidden_states: torch.Tensor,
    forward_batch,
    *,
    expected_prompt_token_count: int,
    sample_id: str,
) -> list[tuple[int, int]]:
    """Return physical rows that are the final token of a full prompt.

    ``extend_seq_lens`` describes only the current chunk.  The expected prompt
    length and request id are supplied by the caller after tokenization; this
    hook never infers prompt length from ``orig_seq_lens``.  Mixed/decode
    forwards are excluded entirely because their one-token rows are generated
    continuation states.
    """

    mode = getattr(forward_batch, "forward_mode", None)
    mode_name = getattr(mode, "name", None)
    if mode_name not in _CAPTURE_FORWARD_MODES:
        return []

    lengths = getattr(forward_batch, "extend_seq_lens_cpu", None)
    prefixes = getattr(forward_batch, "extend_prefix_lens_cpu", None)
    if lengths is None or prefixes is None:
        # A GPU-only batch is not safe for this diagnostic: converting a GPU
        # tensor here would silently make row selection depend on runtime sync.
        return []
    lengths = [int(x) for x in lengths]
    prefixes = [int(x) for x in prefixes]
    if len(lengths) != len(prefixes) or any(x < 0 for x in lengths + prefixes):
        return []

    expected_prompt_token_count = int(expected_prompt_token_count)
    if expected_prompt_token_count <= 0 or not isinstance(sample_id, str) or not sample_id:
        return []

    rids = list(getattr(forward_batch, "rids", None) or [])
    # Capture is deliberately a single-request operation.  Even when the
    # target rid could be located in a batched request, accepting a mixed batch
    # would make the "first full prompt" contract depend on scheduler layout.
    if len(rids) != 1:
        return []
    matching_requests = [index for index, rid in enumerate(rids) if str(rid) == sample_id]
    if matching_requests != [0]:
        return []
    request_index = matching_requests[0]
    if request_index >= len(lengths):
        return []

    # DP/attention sharding can make the local physical row count differ from
    # the sum of request lengths.  Refuse an ambiguous mapping instead of
    # treating the final local row as the final prompt token.  A non-None
    # ``num_padding`` identifies a graph replay, which is deliberately excluded
    # from this hook even when its real rows happen to be laid out contiguously.
    expected_rows = sum(lengths)
    real_rows = int(hidden_states.shape[0])
    if getattr(forward_batch, "num_padding", None) not in (None, 0):
        return []
    if expected_rows != real_rows:
        return []

    # The selected request must cover exactly the immutable prompt extent.  A
    # decode-to-extend row has prefix=L, length=1 and is therefore rejected for
    # an expected prompt length L; a re-prefill containing generated tokens is
    # rejected for the same reason.  No ``orig_seq_lens`` field is consulted.
    request_prefix = prefixes[request_index]
    request_length = lengths[request_index]
    if request_prefix + request_length != expected_prompt_token_count:
        return []

    # Use absolute positions from the actual ForwardBatch.  The selected row
    # must belong to the explicitly named request and equal expected_len - 1;
    # a chunk-end row is never accepted merely because it is last in a chunk.
    positions = getattr(forward_batch, "positions", None)
    if positions is None:
        return []
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().reshape(-1).tolist()
    positions = [int(value) for value in positions]
    if len(positions) != real_rows:
        return []
    start = sum(lengths[:request_index])
    end = start + lengths[request_index]
    request_positions = positions[start:end]
    if (
        not request_positions
        or max(request_positions) >= expected_prompt_token_count
        or min(request_positions) < request_prefix
    ):
        return []
    rows = [
        (row, request_index)
        for row in range(start, end)
        if 0 <= row < real_rows and positions[row] == expected_prompt_token_count - 1
    ]
    return rows if len(rows) == 1 else []


def _prefill_last_prompt_rows(
    hidden_states: torch.Tensor,
    forward_batch,
    *,
    expected_prompt_token_count: int,
    sample_id: str,
) -> list[int]:
    """Return rows matching an explicitly tokenized prompt end."""

    return [
        row
        for row, _request_index in _prefill_last_prompt_row_records(
            hidden_states,
            forward_batch,
            expected_prompt_token_count=expected_prompt_token_count,
            sample_id=sample_id,
        )
    ]


def _model_layers(model: torch.nn.Module) -> list[torch.nn.Module]:
    seen: set[int] = set()
    for candidate in (
        model,
        getattr(model, "model", None),
        getattr(model, "language_model", None),
    ):
        layers = getattr(candidate, "layers", None)
        if layers is None:
            continue
        result = []
        for layer in layers:
            if id(layer) not in seen and hasattr(layer, "layer_id"):
                seen.add(id(layer))
                result.append(layer)
        if result:
            return result
    return []


def _parameter_names(model: torch.nn.Module) -> dict[int, str]:
    return {
        id(param): name
        for name, param in model.named_parameters(remove_duplicate=False)
    }


def _param(module: Any) -> Optional[torch.Tensor]:
    weight = getattr(module, "weight", None)
    return weight if isinstance(weight, torch.Tensor) else None


def _writer_records(model: torch.nn.Module, rank_info: dict[str, Any]) -> list[dict[str, Any]]:
    names = _parameter_names(model)
    direct_names = dict(model.named_parameters())
    records: list[dict[str, Any]] = []
    for layer in _model_layers(model):
        layer_id = int(layer.layer_id)
        candidates = [
            ("attention_o_proj", getattr(layer, "o_proj", None)),
            (
                "gdn_out_proj",
                getattr(getattr(layer, "linear_attn", None), "out_proj", None),
            ),
            (
                "shared_expert_down",
                getattr(
                    getattr(getattr(layer, "mlp", None), "shared_expert", None),
                    "down_proj",
                    None,
                ),
            ),
        ]
        for kind, module in candidates:
            weight = _param(module)
            if weight is None:
                continue
            name = names.get(id(weight))
            if name is None:
                continue
            direct_weight = direct_names.get(name)
            if direct_weight is None or id(direct_weight) != id(weight):
                raise ValueError(
                    f"runtime writer {name} is not an exact direct-loader parameter "
                    "on the loaded model wrapper"
                )
            module_tp_size_attr = getattr(module, "tp_size", None)
            module_tp_rank_attr = getattr(module, "tp_rank", None)
            module_tp_size = int(
                rank_info.get("tp_size", 1)
                if module_tp_size_attr is None
                else module_tp_size_attr
            )
            module_tp_rank = int(
                rank_info.get("tp_rank", 0)
                if module_tp_rank_attr is None
                else module_tp_rank_attr
            )
            if module_tp_size_attr is not None and "tp_size" in rank_info:
                if module_tp_size != int(rank_info["tp_size"]):
                    raise ValueError(
                        f"runtime writer {name} TP size disagrees with worker: "
                        f"module={module_tp_size}, worker={rank_info['tp_size']}"
                    )
            if module_tp_rank_attr is not None and "tp_rank" in rank_info:
                if module_tp_rank != int(rank_info["tp_rank"]):
                    raise ValueError(
                        f"runtime writer {name} TP rank disagrees with worker: "
                        f"module={module_tp_rank}, worker={rank_info['tp_rank']}"
                    )
            if module_tp_size < 1 or not 0 <= module_tp_rank < module_tp_size:
                raise ValueError(
                    f"invalid runtime TP mapping for {name}: "
                    f"rank={module_tp_rank}, size={module_tp_size}"
                )
            if weight.ndim != 2:
                raise ValueError(
                    f"refusal writer {name} must be a 2-D projection, "
                    f"got shape={tuple(weight.shape)}"
                )
            # RowParallelLinear marks its sharded input dimension on the
            # parameter itself.  Preserve that runtime fact instead of
            # inferring a shard from an HF name or a hard-coded axis.
            parameter_input_axis = getattr(weight, "input_dim", None)
            if parameter_input_axis is not None:
                parameter_input_axis = int(parameter_input_axis)
                if parameter_input_axis not in (0, 1):
                    raise ValueError(
                        f"unsupported TP axis for refusal writer {name}: "
                        f"{parameter_input_axis}"
                    )
            tp_axis = parameter_input_axis if module_tp_size > 1 else None
            logical_shape = list(weight.shape)
            if tp_axis is not None:
                global_input_size = getattr(module, "input_size", None)
                if global_input_size is None:
                    raise ValueError(
                        f"runtime writer {name} has TP axis {tp_axis} but no input_size"
                    )
                logical_shape[tp_axis] = int(global_input_size)
            elif getattr(module, "input_size", None) is not None:
                # A replicated projection still exposes its logical input
                # size, which makes a rank snapshot self-describing.
                input_size = int(getattr(module, "input_size"))
                if input_size != int(weight.shape[1]):
                    raise ValueError(
                        f"replicated runtime writer {name} shape does not match "
                        f"module input_size: {tuple(weight.shape)} vs {input_size}"
                    )
            local_slice_size = int(weight.shape[tp_axis]) if tp_axis is not None else 0
            source_info = getattr(model, "_qwen4_exp_runtime_checkpoint_sources", {}).get(
                name, {}
            )
            records.append(
                {
                    "layer_id": layer_id,
                    "kind": kind,
                    "site": f"layer.{layer_id}.{kind}",
                    "direction_site": (
                        f"layer.{layer_id}.mlp_hc"
                        if kind == "shared_expert_down"
                        else f"layer.{layer_id}.attn_hc"
                    ),
                    "runtime_name": name,
                    "shape": list(weight.shape),
                    "logical_shape": logical_shape,
                    "dtype": str(weight.dtype).replace("torch.", ""),
                    "device": str(weight.device),
                    "data_ptr": int(weight.data_ptr()),
                    "output_axis": 0,
                    "tp_axis": tp_axis,
                    "tp_rank": module_tp_rank,
                    "tp_size": module_tp_size,
                    "pp_rank": int(rank_info.get("pp_rank", 0)),
                    "pp_size": int(rank_info.get("pp_size", 1)),
                    "slice_start": module_tp_rank * local_slice_size,
                    "slice_size": local_slice_size,
                    "module_class": type(module).__name__,
                    "parameter_input_axis": parameter_input_axis,
                    "direct_name_verified": True,
                    "checkpoint_name": source_info.get("checkpoint_name"),
                    "checkpoint_shape": source_info.get("checkpoint_shape"),
                    "checkpoint_dtype": source_info.get("checkpoint_dtype"),
                }
            )
    return records


def _validate_writer_set(
    model: torch.nn.Module,
    writers: list[dict[str, Any]],
    rank_info: Optional[dict[str, Any]] = None,
    *,
    expected_hc_site_count: Optional[int] = None,
) -> None:
    """Fail closed when a layer is fused or lacks a supported BF16 writer."""

    layer_list = _model_layers(model)
    layers = {int(layer.layer_id) for layer in layer_list}
    if len(layers) != len(layer_list):
        raise ValueError("loaded Qwen4-Exp layers have duplicate layer_id values")
    if expected_hc_site_count is not None and len(layer_list) * 2 != int(
        expected_hc_site_count
    ):
        raise ValueError(
            "loaded layer count does not match the expected two HC sites per layer: "
            f"layers={len(layer_list)}, expected_hc_site_count={expected_hc_site_count}"
        )
    by_layer: dict[int, set[str]] = {}
    for writer in writers:
        kind = str(writer["kind"])
        if kind not in {"attention_o_proj", "gdn_out_proj", "shared_expert_down"}:
            raise ValueError(f"unsupported refusal writer kind: {kind}")
        layer_id = int(writer["layer_id"])
        if layer_id not in layers:
            raise ValueError(f"writer refers to unknown layer {layer_id}: {writer}")
        layer_kinds = by_layer.setdefault(layer_id, set())
        if kind in layer_kinds:
            raise ValueError(f"duplicate refusal writer for layer {layer_id}: {kind}")
        layer_kinds.add(kind)
        if str(writer["dtype"]) != "bfloat16":
            raise ValueError(
                "refusal runtime requires separate BF16 writer storage, got "
                f"{writer['runtime_name']} dtype={writer['dtype']}"
            )
        if int(writer.get("output_axis", 0)) != 0:
            raise ValueError(
                f"refusal writer {writer['runtime_name']} does not expose output_axis=0"
            )
        shape = tuple(int(x) for x in writer["shape"])
        logical_shape = tuple(int(x) for x in writer["logical_shape"])
        if len(shape) != 2 or len(logical_shape) != 2:
            raise ValueError(f"refusal writer must be a 2-D matrix: {writer}")
        tp_size = int(writer.get("tp_size", 1))
        tp_rank = int(writer.get("tp_rank", 0))
        tp_axis = writer.get("tp_axis")
        if tp_axis is not None:
            tp_axis = int(tp_axis)
            if tp_axis not in (0, 1):
                raise ValueError(f"unsupported refusal writer TP axis: {writer}")
            if shape[tp_axis] * tp_size != logical_shape[tp_axis]:
                raise ValueError(
                    f"writer {writer['runtime_name']} local/logical TP shape mismatch: "
                    f"shape={shape}, logical_shape={logical_shape}, tp_size={tp_size}"
                )
            if int(writer.get("slice_size", shape[tp_axis])) != shape[tp_axis]:
                raise ValueError(f"writer slice_size mismatch: {writer}")
            if int(writer.get("slice_start", -1)) != tp_rank * shape[tp_axis]:
                raise ValueError(f"writer slice_start is not contiguous: {writer}")
        elif tp_size != 1:
            raise ValueError(
                f"sharded refusal writer {writer['runtime_name']} has no TP axis"
            )
        if rank_info is not None:
            for key in ("tp_rank", "tp_size", "pp_rank", "pp_size"):
                if key in rank_info and int(writer.get(key, -1)) != int(rank_info[key]):
                    raise ValueError(
                        f"writer {writer['runtime_name']} {key} does not match worker: "
                        f"writer={writer.get(key)}, worker={rank_info[key]}"
                    )
    missing = {}
    for layer_id in sorted(layers):
        kinds = by_layer.get(layer_id, set())
        output_kinds = kinds & {"attention_o_proj", "gdn_out_proj"}
        absent = {"shared_expert_down"} - kinds
        if len(output_kinds) != 1:
            absent.update({"exactly_one_attention_or_gdn_output"})
        if absent:
            missing[layer_id] = sorted(absent)
    if missing:
        raise ValueError(
            "refusal runtime target writer discovery is incomplete; refusing "
            f"to update fused/unknown storage: {missing}"
        )


def _state_rows(
    hidden_states: torch.Tensor,
    forward_batch,
    *,
    expected_prompt_token_count: int,
    sample_id: str,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    row_records = _prefill_last_prompt_row_records(
        hidden_states,
        forward_batch,
        expected_prompt_token_count=expected_prompt_token_count,
        sample_id=sample_id,
    )
    if not row_records:
        return hidden_states.new_empty((0, 4, 2560), dtype=torch.float32), []
    if hidden_states.ndim != 2 or hidden_states.shape[1] != 4 * 2560:
        raise ValueError(
            "Qwen4-Exp refusal capture requires raw trunk shape [tokens, 4*2560], "
            f"got {tuple(hidden_states.shape)}"
        )
    selected = hidden_states.detach().index_select(
        0,
        torch.tensor(
            [row for row, _request_index in row_records],
            device=hidden_states.device,
            dtype=torch.long,
        ),
    )
    selected = selected.reshape(-1, 4, 2560).to(dtype=torch.float32, device="cpu")
    rids = getattr(forward_batch, "rids", None) or []
    lengths = [int(x) for x in getattr(forward_batch, "extend_seq_lens_cpu", [])]
    prefixes = [int(x) for x in getattr(forward_batch, "extend_prefix_lens_cpu", [])]
    positions = getattr(forward_batch, "positions", None)
    if isinstance(positions, torch.Tensor):
        positions = positions.detach().cpu().reshape(-1).tolist()
    positions = [int(value) for value in (positions or [])]
    metadata = [
        {
            "row": int(row),
            "request_index": int(request_index),
            "request_id": rids[request_index] if request_index < len(rids) else None,
            "forward_mode": str(getattr(forward_batch, "forward_mode", "")),
            "absolute_position": positions[row] if row < len(positions) else None,
            "expected_prompt_token_count": int(expected_prompt_token_count),
            "extend_seq_len": lengths[request_index]
            if request_index < len(lengths)
            else None,
            "extend_prefix_len": prefixes[request_index]
            if request_index < len(prefixes)
            else None,
            "prompt_extent_matches": (
                request_index < len(lengths)
                and request_index < len(prefixes)
                and lengths[request_index] + prefixes[request_index]
                == int(expected_prompt_token_count)
            ),
        }
        for row, request_index in row_records
    ]
    return selected, metadata


@dataclass
class RefusalCaptureSession:
    output_dir: Path
    session_id: str
    rank_info: dict[str, Any]
    writer_records: list[dict[str, Any]]
    expected_prompt_token_count: int
    sample_id: str
    expected_hc_site_count: int
    expected_sites: tuple[str, ...]
    states: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    state_metadata: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    writer_outputs: dict[str, list[torch.Tensor]] = field(default_factory=dict)
    writer_output_metadata: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    active: bool = True
    failure: Optional[str] = None

    @classmethod
    def start(
        cls, model: torch.nn.Module, body: dict[str, Any], rank_info: dict[str, Any]
    ) -> "RefusalCaptureSession":
        output_dir_raw = body.get("output_dir")
        if not isinstance(output_dir_raw, str) or not output_dir_raw:
            raise ValueError("capture_start requires a non-empty output_dir")
        output_dir = Path(output_dir_raw).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        session_id = str(body.get("session_id") or "capture")
        expected_prompt_token_count = body.get("expected_prompt_token_count")
        sample_id = body.get("sample_id")
        if (
            isinstance(expected_prompt_token_count, bool)
            or not isinstance(expected_prompt_token_count, int)
            or expected_prompt_token_count <= 0
        ):
            raise ValueError(
                "capture_start requires positive integer expected_prompt_token_count"
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("capture_start requires non-empty sample_id")
        expected_hc_site_count = body.get("expected_hc_site_count", 96)
        if (
            isinstance(expected_hc_site_count, bool)
            or not isinstance(expected_hc_site_count, int)
            or expected_hc_site_count <= 0
        ):
            raise ValueError("expected_hc_site_count must be a positive integer")
        if expected_hc_site_count == 96 and (
            int(rank_info.get("tp_size", 1)) != 2
            or int(rank_info.get("pp_size", 1)) != 1
        ):
            raise ValueError(
                "the refusal resident experiment requires TP2/PP1 for its 96 HC sites: "
                f"rank={rank_info}"
            )
        layers = _model_layers(model)
        expected_sites = tuple(
            site
            for layer in layers
            for site in (
                f"layer.{int(layer.layer_id)}.attn_hc",
                f"layer.{int(layer.layer_id)}.mlp_hc",
            )
        )
        if len(expected_sites) != expected_hc_site_count:
            raise ValueError(
                "capture_start HC site count does not match the loaded model: "
                f"expected_hc_site_count={expected_hc_site_count}, "
                f"loaded={len(expected_sites)}"
            )
        writers = _writer_records(model, rank_info)
        _validate_writer_set(
            model,
            writers,
            rank_info,
            expected_hc_site_count=expected_hc_site_count,
        )
        session = cls(
            output_dir=output_dir,
            session_id=session_id,
            rank_info=dict(rank_info),
            writer_records=writers,
            expected_prompt_token_count=expected_prompt_token_count,
            sample_id=sample_id,
            expected_hc_site_count=expected_hc_site_count,
            expected_sites=expected_sites,
        )
        metadata = {
            "schema_version": 1,
            "session_id": session_id,
            "rank": session.rank_info,
            "sample_id": sample_id,
            "expected_prompt_token_count": expected_prompt_token_count,
            "expected_hc_site_count": expected_hc_site_count,
            "capture_requires_eager_prefill": True,
            "capture_requires_single_request": True,
            "capture_requires_prompt_cache_miss": True,
            "capture_selector": "absolute_position_equals_expected_prompt_length_minus_one",
            "sites": [
                {
                    "site": site,
                    "layer_id": int(site.split(".")[1]),
                    "kind": site.rsplit(".", 1)[-1],
                }
                for site in expected_sites
            ],
            "writers": writers,
        }
        (output_dir / f"{session_id}-{_rank_suffix(rank_info)}.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )
        return session

    def capture_state(self, site: str, hidden_states: torch.Tensor, forward_batch) -> None:
        if not self.active:
            return
        values, metadata = _state_rows(
            hidden_states,
            forward_batch,
            expected_prompt_token_count=self.expected_prompt_token_count,
            sample_id=self.sample_id,
        )
        if values.shape[0] == 0:
            return
        self.states.setdefault(site, []).append(values)
        self.state_metadata.setdefault(site, []).extend(metadata)

    def capture_writer_output(
        self,
        writer_name: str,
        output: torch.Tensor,
        forward_batch,
    ) -> None:
        if not self.active:
            return
        row_records = _prefill_last_prompt_row_records(
            output,
            forward_batch,
            expected_prompt_token_count=self.expected_prompt_token_count,
            sample_id=self.sample_id,
        )
        if not row_records:
            return
        if output.ndim != 2 or output.shape[1] != 2560:
            raise ValueError(
                "Qwen4-Exp writer capture requires [tokens, 2560], "
                f"got {tuple(output.shape)} for {writer_name}"
            )
        values = output.detach().index_select(
            0,
            torch.tensor(
                [row for row, _request_index in row_records],
                device=output.device,
                dtype=torch.long,
            ),
        )
        self.writer_outputs.setdefault(writer_name, []).append(
            values.to(dtype=torch.float32, device="cpu")
        )
        rids = getattr(forward_batch, "rids", None) or []
        lengths = [
            int(x) for x in getattr(forward_batch, "extend_seq_lens_cpu", [])
        ]
        prefixes = [
            int(x) for x in getattr(forward_batch, "extend_prefix_lens_cpu", [])
        ]
        positions = getattr(forward_batch, "positions", None)
        if isinstance(positions, torch.Tensor):
            positions = positions.detach().cpu().reshape(-1).tolist()
        positions = [int(value) for value in (positions or [])]
        self.writer_output_metadata.setdefault(writer_name, []).extend(
            {
                "row": int(row),
                "request_index": int(request_index),
                "request_id": (
                    rids
                )[request_index]
                if request_index < len(rids)
                else None,
                "forward_mode": str(getattr(forward_batch, "forward_mode", "")),
                "absolute_position": (
                    positions[row] if row < len(positions) else None
                ),
                "expected_prompt_token_count": int(
                    self.expected_prompt_token_count
                ),
                "extend_seq_len": (
                    lengths[request_index]
                    if request_index < len(lengths)
                    else None
                ),
                "extend_prefix_len": (
                    prefixes[request_index]
                    if request_index < len(prefixes)
                    else None
                ),
                "prompt_extent_matches": (
                    request_index < len(lengths)
                    and request_index < len(prefixes)
                    and lengths[request_index] + prefixes[request_index]
                    == int(self.expected_prompt_token_count)
                ),
            }
            for row, request_index in row_records
        )

    def stop(self) -> dict[str, Any]:
        if not self.active:
            return self.status()
        expected_sites = {
            str(site["site"])
            for site in self._expected_sites()
        }
        observed_sites = set(self.states)
        missing_sites = sorted(expected_sites - observed_sites)
        unexpected_sites = sorted(observed_sites - expected_sites)
        expected_writer_outputs = {
            str(record["site"])
            for record in self.writer_records
            if record["kind"] in {"attention_o_proj", "gdn_out_proj"}
        }
        missing_writer_outputs = sorted(
            expected_writer_outputs - set(self.writer_outputs)
        )
        site_sample_counts = {
            site: sum(chunk.shape[0] for chunk in chunks)
            for site, chunks in self.states.items()
        }
        wrong_site_sample_counts = {
            site: count
            for site, count in site_sample_counts.items()
            if site in expected_sites and count != 1
        }
        writer_output_sample_counts = {
            site: sum(chunk.shape[0] for chunk in chunks)
            for site, chunks in self.writer_outputs.items()
        }
        wrong_writer_output_sample_counts = {
            site: count
            for site, count in writer_output_sample_counts.items()
            if site in expected_writer_outputs and count != 1
        }
        unexpected_writer_outputs = sorted(
            set(self.writer_outputs) - expected_writer_outputs
        )
        malformed_sites = sorted(
            site
            for site in expected_sites & observed_sites
            if not self.states.get(site)
            or len(self.state_metadata.get(site, []))
            != sum(chunk.shape[0] for chunk in self.states[site])
        )
        if (
            missing_sites
            or unexpected_sites
            or malformed_sites
            or not observed_sites
            or missing_writer_outputs
            or wrong_site_sample_counts
            or wrong_writer_output_sample_counts
            or unexpected_writer_outputs
        ):
            self.failure = (
                "capture_stop rejected incomplete HC capture: "
                + (
                    f"missing_sites={missing_sites[:8]}"
                    + ("..." if len(missing_sites) > 8 else "")
                    if missing_sites
                    else "no HC site has a sample"
                )
                + (
                    f"; unexpected_sites={unexpected_sites[:8]}"
                    + ("..." if len(unexpected_sites) > 8 else "")
                    if unexpected_sites
                    else ""
                )
                + (
                    f"; missing_writer_outputs={missing_writer_outputs[:8]}"
                    + ("..." if len(missing_writer_outputs) > 8 else "")
                    if missing_writer_outputs
                    else ""
                )
                + (
                    f"; malformed_sites={malformed_sites[:8]}"
                    + ("..." if len(malformed_sites) > 8 else "")
                    if malformed_sites
                    else ""
                )
                + (
                    f"; wrong_site_sample_counts={wrong_site_sample_counts}"
                    if wrong_site_sample_counts
                    else ""
                )
                + (
                    f"; wrong_writer_output_sample_counts={wrong_writer_output_sample_counts}"
                    if wrong_writer_output_sample_counts
                    else ""
                )
                + (
                    f"; unexpected_writer_outputs={unexpected_writer_outputs[:8]}"
                    + ("..." if len(unexpected_writer_outputs) > 8 else "")
                    if unexpected_writer_outputs
                    else ""
                )
            )
            self.active = False
            failure_path = self.output_dir / (
                f"{self.session_id}-{_rank_suffix(self.rank_info)}.failure.json"
            )
            try:
                failure_path.write_text(
                    json.dumps(
                        self.status(
                            site_sample_counts=site_sample_counts,
                            writer_output_sample_counts=writer_output_sample_counts,
                        ),
                        ensure_ascii=False,
                        indent=2,
                    )
                    + "\n"
                )
            except OSError:
                # Preserve the original capture failure for the scheduler/RPC
                # caller even if the diagnostic directory is unavailable.
                pass
            raise RuntimeError(self.failure)
        self.active = False
        state_values = {
            site: torch.cat(chunks, dim=0) for site, chunks in self.states.items()
        }
        writer_values = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in self.writer_outputs.items()
        }
        payload = {
            "schema_version": 1,
            "session_id": self.session_id,
            "rank": self.rank_info,
            "sample_id": self.sample_id,
            "expected_prompt_token_count": self.expected_prompt_token_count,
            "expected_hc_site_count": self.expected_hc_site_count,
            "expected_sites": list(self.expected_sites),
            "states": state_values,
            "state_metadata": self.state_metadata,
            "writer_outputs": writer_values,
            "writer_output_metadata": self.writer_output_metadata,
            "writers": self.writer_records,
        }
        path = self.output_dir / f"{self.session_id}-{_rank_suffix(self.rank_info)}.pt"
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, path)
        return self.status(
            path=str(path),
            site_sample_counts=site_sample_counts,
            writer_output_sample_counts=writer_output_sample_counts,
        )

    def _expected_sites(self) -> list[dict[str, Any]]:
        # ``writer_records`` is intentionally not used to infer HC coverage:
        # a fused/quantized writer can be absent while the two HC insertion
        # points still exist.  The model layer list is supplied in start().
        return [
            {
                "site": site,
                "layer_id": int(site.split(".")[1]),
                "kind": site.rsplit(".", 1)[-1],
            }
            for site in self.expected_sites
        ]

    def status(self, **extra: Any) -> dict[str, Any]:
        expected_writer_outputs = {
            str(record["site"])
            for record in self.writer_records
            if record["kind"] in {"attention_o_proj", "gdn_out_proj"}
        }
        return {
            "success": self.failure is None,
            "active": self.active,
            "session_id": self.session_id,
            "rank": self.rank_info,
            "site_count": len(self.states),
            "sample_count": sum(x.shape[0] for chunks in self.states.values() for x in chunks),
            "expected_hc_site_count": self.expected_hc_site_count,
            "missing_hc_sites": sorted(set(self.expected_sites) - set(self.states)),
            "unexpected_hc_sites": sorted(set(self.states) - set(self.expected_sites)),
            "writer_output_site_count": len(self.writer_outputs),
            "missing_writer_outputs": sorted(
                expected_writer_outputs - set(self.writer_outputs)
            ),
            "failure": self.failure,
            **extra,
        }


def _qsa_indexers(model: torch.nn.Module) -> list[tuple[int, Any]]:
    """Return the loaded compressed-QSA indexers by their runtime layer IDs."""

    result: list[tuple[int, Any]] = []
    for layer in _model_layers(model):
        indexer = getattr(layer, "indexer", None)
        if indexer is None or not hasattr(indexer, "_qsa_r1_graph_capture"):
            continue
        result.append((int(layer.layer_id), indexer))
    return result


def _cpu_vector(value: Any, name: str) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().reshape(-1).tolist()
    if not isinstance(value, (list, tuple)):
        raise RuntimeError(f"r1 observer could not read {name} as a vector")
    result = [int(item) for item in value]
    return result


@dataclass
class QsaR1Observer:
    """Drain c1 QSA raw block selections after normal CUDA-graph replays.

    The indexer owns the captured graph tensor reference.  This object only
    reads that storage after replay while active and writes the agreed raw
    JSONL schema; it never adds an operation to the graph.
    """

    output_dir: Path
    session_id: str
    rank_info: dict[str, Any]
    indexers: tuple[tuple[int, Any], ...]
    expected_steps: Optional[int] = None
    active: bool = True
    failure: Optional[str] = None
    row_count: int = 0
    first_position: Optional[int] = None
    last_position: Optional[int] = None
    request_id: Optional[str] = None
    _raw_path: Path = field(init=False)
    _metadata_path: Path = field(init=False)
    _status_path: Path = field(init=False)
    _stream: Any = field(init=False, repr=False)

    @classmethod
    def start(
        cls, model: torch.nn.Module, body: dict[str, Any], rank_info: dict[str, Any]
    ) -> "QsaR1Observer":
        output_dir_raw = body.get("output_dir")
        if not isinstance(output_dir_raw, str) or not output_dir_raw:
            raise ValueError("r1_start requires a non-empty output_dir")
        output_dir = Path(output_dir_raw).expanduser()
        output_dir.mkdir(parents=True, exist_ok=True)
        session_id = str(body.get("session_id") or "r1")
        if not session_id or session_id in {".", ".."} or "/" in session_id:
            raise ValueError("r1_start session_id must be a simple path component")
        expected_steps = body.get("expected_steps")
        if expected_steps is not None and (
            isinstance(expected_steps, bool)
            or not isinstance(expected_steps, int)
            or expected_steps <= 0
        ):
            raise ValueError("r1_start expected_steps must be a positive integer")
        indexers = tuple(_qsa_indexers(model))
        if not indexers:
            raise RuntimeError("r1_start found no compressed QSA indexers")
        expected_layer_count = body.get("expected_qsa_layer_count")
        if expected_layer_count is not None and (
            isinstance(expected_layer_count, bool)
            or not isinstance(expected_layer_count, int)
            or expected_layer_count <= 0
        ):
            raise ValueError(
                "r1_start expected_qsa_layer_count must be a positive integer"
            )
        if expected_layer_count is not None and len(indexers) != expected_layer_count:
            raise RuntimeError(
                "r1_start QSA layer count mismatch: "
                f"loaded={len(indexers)}, expected={expected_layer_count}"
            )

        buffers: list[dict[str, Any]] = []
        buffer_ranges: list[tuple[int, int, int]] = []
        for layer_id, indexer in indexers:
            capture = getattr(indexer, "_qsa_r1_graph_capture", None)
            if not isinstance(capture, dict):
                raise RuntimeError(
                    f"r1_start has no captured c1 raw block tensor for layer {layer_id}"
                )
            required_metadata = {
                "tensor",
                "data_ptr",
                "shape",
                "dtype",
                "layer_id",
                "forward_mode",
                "cuda_graph",
                "block_topk",
                "compress_ratio",
                "producer",
            }
            missing_metadata = sorted(required_metadata - set(capture))
            if missing_metadata:
                raise RuntimeError(
                    f"r1_start missing c1 producer metadata for layer {layer_id}: "
                    f"{missing_metadata}"
                )
            tensor = capture.get("tensor")
            shape = tuple(int(value) for value in capture.get("shape", ()))
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_cuda
                or tensor.dtype != torch.int32
                or not tensor.is_contiguous()
                or shape != (1, 512)
                or tuple(tensor.shape) != shape
                or int(tensor.data_ptr()) != int(capture.get("data_ptr", -1))
                or int(capture["layer_id"]) != layer_id
                or capture["dtype"] != "int32"
                or capture["forward_mode"] != "decode"
                or capture["cuda_graph"] is not True
                or int(capture["block_topk"]) != 512
                or int(capture["compress_ratio"]) <= 0
                or capture["producer"] != "QSAIndexer.select_decode_tokens.fast_topk"
            ):
                raise RuntimeError(
                    "r1_start c1 raw block tensor metadata is invalid for "
                    f"layer {layer_id}: {capture}"
                )
            data_ptr = int(capture["data_ptr"])
            buffer_ranges.append(
                (data_ptr, data_ptr + 512 * tensor.element_size(), layer_id)
            )
            buffers.append(
                {
                    "layer_id": layer_id,
                    "producer": capture.get("producer"),
                    "data_ptr": data_ptr,
                    "shape": list(shape),
                    "dtype": capture.get("dtype"),
                    "block_topk": int(capture.get("block_topk", -1)),
                    "compress_ratio": int(capture.get("compress_ratio", -1)),
                    "forward_mode": capture.get("forward_mode"),
                    "cuda_graph": capture.get("cuda_graph"),
                }
            )
        if len({start for start, _end, _layer_id in buffer_ranges}) != len(buffer_ranges):
            raise RuntimeError("r1_start found overlapping c1 raw block buffers")
        for index, (start, end, layer_id) in enumerate(buffer_ranges):
            for other_start, other_end, other_layer_id in buffer_ranges[index + 1 :]:
                if start < other_end and other_start < end:
                    raise RuntimeError(
                        "r1_start found overlapping c1 raw block buffers: "
                        f"layers={layer_id},{other_layer_id}"
                    )

        observer = cls(
            output_dir=output_dir,
            session_id=session_id,
            rank_info=dict(rank_info),
            indexers=indexers,
            expected_steps=expected_steps,
        )
        observer._raw_path = output_dir / (
            f"{session_id}-{_rank_suffix(rank_info)}.jsonl"
        )
        observer._metadata_path = output_dir / (
            f"{session_id}-{_rank_suffix(rank_info)}.json"
        )
        observer._status_path = output_dir / (
            f"{session_id}-{_rank_suffix(rank_info)}.status.json"
        )
        if observer._raw_path.exists():
            raise FileExistsError(f"r1_start refuses existing raw file: {observer._raw_path}")
        observer._stream = observer._raw_path.open("x", encoding="utf-8")
        metadata = {
            "schema_version": "qsa-r1-v1",
            "session_id": session_id,
            "rank": observer.rank_info,
            "expected_steps": expected_steps,
            "layer_ids": [layer_id for layer_id, _indexer in indexers],
            "raw_path": str(observer._raw_path),
            "buffers": buffers,
            "observer": "post_graph_replay_cpu_drain",
            "expanded_indices_used": False,
        }
        observer._metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return observer

    def _fail(self, message: str) -> None:
        self.failure = message
        self.active = False
        raise RuntimeError(message)

    def drain(self, forward_batch: Any, graph_runner: Any) -> None:
        if not self.active:
            return
        if self.failure is not None:
            return
        mode = getattr(getattr(forward_batch, "forward_mode", None), "name", None)
        if mode != "DECODE":
            self._fail(f"r1 observer rejected non-DECODE forward mode: {mode!r}")
        if getattr(forward_batch, "spec_info", None) is not None:
            self._fail("r1 observer rejected speculative decode")
        if getattr(forward_batch, "_original_forward_mode", None) is not None:
            self._fail("r1 observer rejected a substituted forward mode")
        if int(getattr(forward_batch, "batch_size", -1)) != 1:
            self._fail(
                "r1 observer requires raw batch size 1, got "
                f"{getattr(forward_batch, 'batch_size', None)!r}"
            )
        if getattr(forward_batch, "num_padding", None) not in (None, 0):
            self._fail("r1 observer rejected padded forward batch")
        if int(getattr(graph_runner, "bs", -1)) != 1:
            self._fail(
                "r1 observer requires c1 graph replay, got "
                f"graph_bs={getattr(graph_runner, 'bs', None)!r}"
            )
        rids = list(getattr(forward_batch, "rids", None) or [])
        if len(rids) != 1 or not isinstance(rids[0], str) or not rids[0]:
            self._fail(f"r1 observer requires one runtime request id, got {rids!r}")
        rid = rids[0]
        try:
            positions = _cpu_vector(getattr(forward_batch, "positions", None), "positions")
            seq_lens_value = getattr(forward_batch, "seq_lens_cpu", None)
            if seq_lens_value is None:
                seq_lens_value = getattr(forward_batch, "seq_lens", None)
            seq_lens = _cpu_vector(seq_lens_value, "seq_lens")
        except (TypeError, ValueError, RuntimeError) as exc:
            self._fail(str(exc))
        if len(positions) != 1 or len(seq_lens) != 1:
            self._fail(
                f"r1 observer requires one position/sequence length, got {positions!r}/{seq_lens!r}"
            )
        position = int(positions[0])
        sequence_length = int(seq_lens[0])
        if position < 0 or sequence_length <= 0:
            self._fail(
                f"r1 observer received invalid position/sequence length: {position}/{sequence_length}"
            )
        if self.request_id is None:
            self.request_id = rid
            self.first_position = position
        elif rid != self.request_id:
            self._fail(
                "r1 observer request changed during one session: "
                f"{self.request_id!r} -> {rid!r}"
            )
        if self.last_position is not None and position != self.last_position + 1:
            self._fail(
                "r1 observer position is not consecutive: "
                f"previous={self.last_position}, current={position}"
            )

        rows: list[dict[str, Any]] = []
        for layer_id, indexer in self.indexers:
            capture = getattr(indexer, "_qsa_r1_graph_capture", None)
            if not isinstance(capture, dict):
                self._fail(f"r1 observer lost c1 producer metadata for layer {layer_id}")
            tensor = capture.get("tensor")
            expected_ptr = int(capture.get("data_ptr", -1))
            expected_shape = tuple(int(value) for value in capture.get("shape", ()))
            if (
                not isinstance(tensor, torch.Tensor)
                or not tensor.is_cuda
                or tensor.dtype != torch.int32
                or tuple(tensor.shape) != expected_shape
                or expected_shape != (1, 512)
                or int(tensor.data_ptr()) != expected_ptr
            ):
                self._fail(
                    "r1 observer producer tensor changed for "
                    f"layer {layer_id}: expected_ptr={expected_ptr}, "
                    f"actual={tensor!r}"
                )
            # This is the sole observer D2H operation and is outside graph
            # replay.  Do not replace it with a device-side copy or clone.
            values = [int(value) for value in tensor.detach().cpu().reshape(-1).tolist()]
            if len(values) != 512:
                self._fail(
                    f"r1 observer raw block shape changed for layer {layer_id}: {len(values)}"
                )
            ratio = int(capture.get("compress_ratio", 0))
            if ratio <= 0:
                self._fail(f"r1 observer has invalid compress_ratio for layer {layer_id}")
            compressed_length = sequence_length // ratio
            if compressed_length < 0:
                self._fail(
                    f"r1 observer computed invalid compressed_length={compressed_length}"
                )
            rows.append(
                {
                    "request_id": rid,
                    "rank": int(self.rank_info.get("tp_rank", 0)),
                    "layer_id": int(layer_id),
                    "position": position,
                    "compressed_length": compressed_length,
                    "block_indices": values,
                    "forward_mode": "decode",
                    "cuda_graph": True,
                }
            )
        for row in rows:
            self._stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._stream.flush()
        self.row_count += len(rows)
        self.last_position = position
        if (
            self.expected_steps is not None
            and self.row_count == self.expected_steps * len(self.indexers)
        ):
            # The scheduler may execute one ahead target step before the HTTP
            # response is assembled.  The requested generation budget ends at
            # this accepted decode position; leave the graph-owned references
            # intact but disable further D2H drains until r1_stop closes us.
            self.active = False

    def status(self, **extra: Any) -> dict[str, Any]:
        return {
            "success": self.failure is None,
            "active": self.active,
            "session_id": self.session_id,
            "rank": self.rank_info,
            "layer_count": len(self.indexers),
            "layer_ids": [layer_id for layer_id, _indexer in self.indexers],
            "row_count": self.row_count,
            "request_id": self.request_id,
            "first_position": self.first_position,
            "last_position": self.last_position,
            "expected_steps": self.expected_steps,
            "raw_path": str(self._raw_path),
            "metadata_path": str(self._metadata_path),
            "failure": self.failure,
            **extra,
        }

    def stop(self) -> dict[str, Any]:
        if self.active:
            self.active = False
        if getattr(self, "_stream", None) is not None and not self._stream.closed:
            self._stream.flush()
            self._stream.close()
        expected_rows = (
            None
            if self.expected_steps is None
            else self.expected_steps * len(self.indexers)
        )
        if expected_rows is not None and self.row_count != expected_rows:
            self.failure = (
                "r1_stop expected one raw row per layer per step: "
                f"rows={self.row_count}, expected={expected_rows}"
            )
        if self.row_count == 0 and self.failure is None:
            self.failure = "r1_stop rejected empty raw capture"
        result = self.status()
        try:
            self._status_path.write_text(
                json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError:
            pass
        if self.failure is not None:
            raise RuntimeError(self.failure)
        return result


@dataclass
class RefusalRuntimeState:
    capture: Optional[RefusalCaptureSession] = None
    r1: Optional[QsaR1Observer] = None


_RUNTIME_STATES: dict[int, RefusalRuntimeState] = {}


def _runtime_state(model: torch.nn.Module) -> RefusalRuntimeState:
    return _RUNTIME_STATES.setdefault(id(model), RefusalRuntimeState())


def _snapshot(
    model: torch.nn.Module, body: dict[str, Any], rank_info: dict[str, Any]
) -> dict[str, Any]:
    output_dir_raw = body.get("output_dir")
    if not isinstance(output_dir_raw, str) or not output_dir_raw:
        raise ValueError("snapshot requires a non-empty output_dir")
    output_dir = Path(output_dir_raw).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    writers = _writer_records(model, rank_info)
    expected_hc_site_count = body.get("expected_hc_site_count", 96)
    if (
        isinstance(expected_hc_site_count, bool)
        or not isinstance(expected_hc_site_count, int)
        or expected_hc_site_count <= 0
    ):
        raise ValueError("snapshot expected_hc_site_count must be a positive integer")
    if expected_hc_site_count == 96 and (
        int(rank_info.get("tp_size", 1)) != 2
        or int(rank_info.get("pp_size", 1)) != 1
    ):
        raise ValueError(
            "the refusal resident experiment requires TP2/PP1 for its 96 HC sites: "
            f"rank={rank_info}"
        )
    _validate_writer_set(
        model,
        writers,
        rank_info,
        expected_hc_site_count=expected_hc_site_count,
    )
    requested = body.get("kinds")
    if requested is not None:
        if not isinstance(requested, list) or not all(isinstance(x, str) for x in requested):
            raise ValueError("snapshot kinds must be a list of strings")
        unknown = set(requested) - {
            "attention_o_proj",
            "gdn_out_proj",
            "shared_expert_down",
        }
        if unknown:
            raise ValueError(f"snapshot requested unknown writer kinds: {sorted(unknown)}")
        writers = [x for x in writers if x["kind"] in requested]
    if body.get("require_bf16", True):
        bad = [x for x in writers if x["dtype"] != "bfloat16"]
        if bad:
            raise ValueError(
                "refusal runtime only updates BF16 writers; non-BF16 targets: "
                + ", ".join(x["runtime_name"] for x in bad)
            )
    if body.get("require_shared_expert_down", True) and sum(
        x["kind"] == "shared_expert_down" for x in writers
    ) != len(_model_layers(model)):
        raise ValueError(
            "runtime does not expose one separate BF16 shared_expert.down_proj "
            "for every loaded layer; refusing to silently target fused or INT4 "
            "expert storage"
        )
    names = {x["runtime_name"] for x in writers}
    params = {
        name: param.detach().to(device="cpu").clone()
        for name, param in model.named_parameters(remove_duplicate=False)
        if name in names
    }
    if set(params) != names:
        raise ValueError(
            "writer metadata does not cover the direct-loader parameter names: "
            f"missing={sorted(names - set(params))}"
        )
    label = str(body.get("label") or "snapshot")
    path = output_dir / f"{label}-{_rank_suffix(rank_info)}.pt"
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "schema_version": 1,
            "rank": rank_info,
            "weights": params,
            "writers": writers,
        },
        tmp,
    )
    os.replace(tmp, path)
    return {
        "success": True,
        "path": str(path),
        "rank": rank_info,
        "writers": writers,
    }


def refusal_control(
    model: torch.nn.Module,
    method: str,
    body: Optional[dict[str, Any]] = None,
    *,
    rank_info: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    body = body or {}
    rank_info = rank_info or {"tp_rank": 0, "tp_size": 1, "pp_rank": 0, "pp_size": 1}
    state = _runtime_state(model)
    if method == "capture_start":
        if state.capture is not None and state.capture.active:
            raise RuntimeError("refusal capture is already active")
        state.capture = RefusalCaptureSession.start(model, body, rank_info)
        return state.capture.status(writers=state.capture.writer_records)
    if method == "capture_status":
        return (
            state.capture.status() if state.capture is not None else {"success": True, "active": False}
        )
    if method == "capture_stop":
        if state.capture is None:
            raise RuntimeError("capture_stop rejected: no active capture session")
        return state.capture.stop()
    if method == "snapshot":
        return _snapshot(model, body, rank_info)
    if method == "capture_clear":
        state.capture = None
        return {"success": True, "active": False}
    if method == "r1_start":
        if state.r1 is not None and state.r1.active:
            raise RuntimeError("r1 observer is already active")
        state.r1 = QsaR1Observer.start(model, body, rank_info)
        # Keep the stopped observer object attached as well: QSA indexers retain
        # their graph-owned c1 tensors for the process lifetime, while the
        # model runner uses this object only as the active CPU-drain switch.
        model._qwen4_exp_r1_observer = state.r1
        return state.r1.status()
    if method == "r1_status":
        return (
            state.r1.status()
            if state.r1 is not None
            else {"success": True, "active": False, "row_count": 0}
        )
    if method == "r1_stop":
        if state.r1 is None:
            raise RuntimeError("r1_stop rejected: no observer session")
        return state.r1.stop()
    raise ValueError(f"unknown refusal runtime method: {method!r}")


def capture_hc_state(layer_id: int, site_kind: str, hidden_states, forward_batch) -> None:
    """Called from Qwen4-Exp's two HC mix sites; no-op when disabled."""

    # The model object is not passed through the layer helper.  The layer stores
    # the current owner lazily when the control endpoint starts a session.
    for state in _RUNTIME_STATES.values():
        if state.capture is not None and state.capture.active:
            state.capture.capture_state(
                f"layer.{int(layer_id)}.{site_kind}_hc", hidden_states, forward_batch
            )


def capture_writer_output(writer_name: str, output, forward_batch) -> None:
    for state in _RUNTIME_STATES.values():
        if state.capture is not None and state.capture.active:
            state.capture.capture_writer_output(writer_name, output, forward_batch)
