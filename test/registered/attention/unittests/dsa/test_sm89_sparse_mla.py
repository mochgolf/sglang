import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

_SM_SCALE = 1.0 / (512 + 64) ** 0.5


def _splitk_host_partial_state(logits, values, split_tokens):
    partial_lse = []
    partial_o = []
    for start in range(0, logits.numel(), split_tokens):
        split_logits = logits[start : start + split_tokens]
        split_values = values[start : start + split_tokens]
        valid = torch.isfinite(split_logits)
        if not valid.any():
            partial_lse.append(torch.tensor(float("-inf"), dtype=torch.float32))
            partial_o.append(torch.zeros(values.shape[1], dtype=torch.float32))
            continue
        lse = torch.logsumexp(split_logits[valid].float(), dim=0)
        probabilities = (
            torch.exp(split_logits[valid].float() - lse).to(torch.bfloat16).float()
        )
        partial_lse.append(lse)
        partial_o.append(probabilities @ split_values[valid].to(torch.bfloat16).float())
    return torch.stack(partial_lse), torch.stack(partial_o)


def _splitk_host_combine(partial_lse, partial_o):
    valid = torch.isfinite(partial_lse)
    if not valid.any():
        return torch.zeros(partial_o.shape[-1], dtype=torch.float32)
    global_lse = torch.logsumexp(partial_lse[valid], dim=0)
    weights = torch.zeros_like(partial_lse)
    weights[valid] = torch.exp(partial_lse[valid] - global_lse)
    return weights @ partial_o


def _cuda_kernel_source_body(source, kernel_name):
    start = source.index(kernel_name)
    body_start = source.index("{", start)
    depth = 0
    for offset in range(body_start, len(source)):
        if source[offset] == "{":
            depth += 1
        elif source[offset] == "}":
            depth -= 1
            if depth == 0:
                return source[body_start + 1 : offset]
    raise AssertionError(f"unterminated CUDA kernel {kernel_name}")


def _require_sm89():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if torch.cuda.get_device_capability() != (8, 9):
        pytest.skip("requires an SM89 CUDA device")


def _make_packed_case(*, rows, topk, seed=7):
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    pool_size = max(topk + 32, 96)
    q_nope = torch.randn(
        rows, 32, 512, device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    q_rope = torch.randn(
        rows, 32, 64, device=device, dtype=torch.float32, generator=generator
    ).to(torch.bfloat16)
    packed_bytes = torch.empty(pool_size, 1, 656, device=device, dtype=torch.uint8)
    kv_cache = packed_bytes.view(torch.float8_e4m3fn)
    kv_cache[:, :, :512].copy_(
        (
            torch.randn(
                pool_size,
                1,
                512,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * 0.125
        ).to(torch.float8_e4m3fn)
    )
    packed_bytes[:, :, 512:528].view(torch.float32).copy_(
        (
            0.03125
            + torch.rand(
                pool_size, 1, 4, device=device, dtype=torch.float32, generator=generator
            )
            * 0.125
        )
    )
    packed_bytes[:, :, 528:].view(torch.bfloat16).copy_(
        (
            torch.randn(
                pool_size,
                1,
                64,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            * 0.125
        ).to(torch.bfloat16)
    )
    page_table = torch.arange(topk, device=device, dtype=torch.int32).remainder(
        pool_size
    )
    page_table = page_table.unsqueeze(0).expand(rows, -1).clone()
    page_table[:, 3] = -1
    return {
        "q_nope": q_nope,
        "q_rope": q_rope,
        "kv_cache": kv_cache,
        "page_table": page_table,
        "cache_seqlens": torch.full((rows,), topk, device=device, dtype=torch.int32),
    }


def _packed_reference(case):
    packed = case["kv_cache"].view(torch.uint8)
    outputs = []
    for row, row_len in enumerate(case["cache_seqlens"].tolist()):
        token_ids = case["page_table"][row, :row_len]
        valid = (token_ids >= 0) & (token_ids < packed.shape[0])
        if not valid.any():
            outputs.append(torch.zeros_like(case["q_nope"][row]))
            continue
        selected = packed[token_ids[valid].long(), 0]
        scales = selected[:, 512:528].contiguous().view(torch.float32)
        nope = (
            selected[:, :512].contiguous().view(torch.float8_e4m3fn).float()
            * scales.repeat_interleave(128, dim=1)
        ).to(torch.bfloat16)
        rope = selected[:, 528:].contiguous().view(torch.bfloat16)
        q = torch.cat((case["q_nope"][row], case["q_rope"][row]), dim=-1)
        k = torch.cat((nope, rope), dim=-1)
        logits = q.float() @ k.float().transpose(0, 1) * _SM_SCALE
        probabilities = torch.softmax(logits, dim=-1).to(torch.bfloat16).float()
        outputs.append((probabilities @ nope.float()).to(torch.bfloat16))
    return torch.stack(outputs)


def _assert_close(actual, expected):
    diff = (actual.float() - expected.float()).abs()
    assert torch.isfinite(actual).all()
    assert diff.max().item() <= 5e-2
    assert diff.mean().item() <= 5e-3
    assert (
        torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        >= 0.995
    )


def test_splitk_host_ownership_and_softmax_oracle():
    for split_tokens in (32, 64):
        owned = [
            split * split_tokens + 16 * warp + token
            for split in range(2048 // split_tokens)
            for warp in range(split_tokens // 16)
            for token in range(16)
        ]
        assert sorted(owned) == list(range(2048))

    logits = torch.tensor(
        [float("-inf")] * 4 + [-1.5, 0.25, float("-inf"), 1.0] + [float("-inf")] * 4
    )
    values = torch.arange(84, dtype=torch.float32).reshape(12, 7) / 17
    partial_lse, partial_o = _splitk_host_partial_state(logits, values, 4)
    merged = _splitk_host_combine(partial_lse, partial_o)
    valid = torch.isfinite(logits)
    expected = (
        torch.softmax(logits[valid], dim=0).to(torch.bfloat16).float()
        @ values[valid].to(torch.bfloat16).float()
    )
    torch.testing.assert_close(merged, expected, atol=0, rtol=0)
    assert torch.equal(
        _splitk_host_combine(torch.full((2,), float("-inf")), torch.zeros(2, 7)),
        torch.zeros(7),
    )


def test_decode_selector_and_workspace_schema(monkeypatch):
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        select_sm89_sparse_mla_decode_kernel,
        splitk_workspace_schema,
    )

    monkeypatch.delenv("SGLANG_GLM_DSA_SM89_DECODE_KERNEL", raising=False)
    assert select_sm89_sparse_mla_decode_kernel() == "three_stage"
    assert select_sm89_sparse_mla_decode_kernel(" splitK32 ") == "splitk32"
    assert select_sm89_sparse_mla_decode_kernel("splitk64") == "splitk64"
    with pytest.raises(ValueError, match="three_stage.*splitk32.*splitk64"):
        select_sm89_sparse_mla_decode_kernel("invalid")
    assert splitk_workspace_schema(1, 32, 2048, 32) == {
        "splits": 64,
        "partial_o_bytes": 4_194_304,
        "partial_lse_bytes": 8_192,
        "workspace_bytes": 4_202_496,
    }


def test_splitk_partial_wrapper_uses_preallocated_workspace(monkeypatch):
    from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

    partial_o, partial_lse = object(), object()
    allocations = []

    def fake_empty(shape, *, device, dtype):
        allocations.append((shape, device, dtype))
        return partial_o if len(allocations) == 1 else partial_lse

    q = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(
        sm89_sparse_mla_cuda,
        "_validate_sm89_sparse_mla_decode_cuda_splitk_workspace",
        MagicMock(),
    )
    assert sm89_sparse_mla_cuda.allocate_sm89_sparse_mla_decode_cuda_splitk_workspace(
        q, 32
    ) == (partial_o, partial_lse)
    assert allocations == [
        ((64, 1, 32, 512), torch.device("cuda:0"), torch.float32),
        ((64, 1, 32), torch.device("cuda:0"), torch.float32),
    ]


def test_splitk_cuda_source_contract():
    from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

    source_path = (
        Path(sm89_sparse_mla_cuda.__file__).resolve().parent
        / "csrc"
        / "sm89_sparse_mla_cuda.cu"
    )
    source = source_path.read_text(encoding="utf-8")
    validation = inspect.getsource(
        sm89_sparse_mla_cuda._validate_sm89_sparse_mla_decode_cuda_splitk_workspace
    )
    partial_entry = _cuda_kernel_source_body(
        source, "sm89_sparse_mla_decode_cuda_splitk_partial"
    )
    assert "launch_sm89_sparse_mla_splitk_partial(" in partial_entry
    assert "torch::empty" not in partial_entry
    assert "sm89_sparse_mla_splitk_combine_kernel" in source
    assert 'm.def("sm89_sparse_mla_decode_cuda_splitk_partial"' in source
    for marker in (
        "partial_o.dtype != torch.float32",
        "partial_lse.dtype != torch.float32",
        "partial_o.shape != (splits, 1, 32, 512)",
        "partial_o.data_ptr() % 32",
    ):
        assert marker in validation


@pytest.mark.parametrize("split_tokens", (32, 64))
def test_sm89_splitk_packed_fp8_matches_reference(split_tokens):
    _require_sm89()
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        sm89_sparse_mla_decode_cuda_splitk,
    )

    case = _make_packed_case(rows=1, topk=2048)
    assert case["kv_cache"].shape[1:] == (1, 656)
    expected = _packed_reference(case)
    actual = sm89_sparse_mla_decode_cuda_splitk(
        **case,
        sm_scale=_SM_SCALE,
        logit_cap=0.0,
        v_head_dim=512,
        split_tokens=split_tokens,
    )
    torch.cuda.synchronize()
    _assert_close(actual, expected)


def test_sm89_triton_prefill_packed_fp8_masks_invalid_pages():
    _require_sm89()
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
        sm89_sparse_mla_prefill_triton,
    )

    case = _make_packed_case(rows=2, topk=64)
    case["cache_seqlens"][1] = 0
    expected = _packed_reference(case)
    actual = sm89_sparse_mla_prefill_triton(
        **case, sm_scale=_SM_SCALE, logit_cap=0.0, v_head_dim=512
    )
    torch.cuda.synchronize()
    _assert_close(actual, expected)
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))


def test_sm89_cuda_tensorcore_prefill_packed_fp8_matches_reference():
    _require_sm89()
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        sm89_sparse_mla_prefill_cuda,
    )

    case = _make_packed_case(rows=2, topk=64, seed=23)
    assert case["kv_cache"].shape[1:] == (1, 656)
    expected = _packed_reference(case)
    actual = sm89_sparse_mla_prefill_cuda(
        **case,
        sm_scale=_SM_SCALE,
        logit_cap=0.0,
        v_head_dim=512,
        cuda_impl="tensorcore",
    )
    torch.cuda.synchronize()
    _assert_close(actual, expected)


@pytest.mark.parametrize("split_tokens", (32, 64))
def test_sm89_splitk_cuda_graph_replays_updated_stable_inputs(split_tokens):
    _require_sm89()
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        sm89_sparse_mla_decode_cuda_splitk,
    )

    static = _make_packed_case(rows=1, topk=2048, seed=29)
    pointers = {name: tensor.data_ptr() for name, tensor in static.items()}
    sm89_sparse_mla_decode_cuda_splitk(
        **static,
        sm_scale=_SM_SCALE,
        logit_cap=0.0,
        v_head_dim=512,
        split_tokens=split_tokens,
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        output = sm89_sparse_mla_decode_cuda_splitk(
            **static,
            sm_scale=_SM_SCALE,
            logit_cap=0.0,
            v_head_dim=512,
            split_tokens=split_tokens,
        )

    for seed, sequence_length, page_shift in (
        (31, 1537, 17),
        (37, 1901, 31),
    ):
        updated = _make_packed_case(rows=1, topk=2048, seed=seed)
        updated["cache_seqlens"].fill_(sequence_length)
        valid_pages = updated["page_table"] >= 0
        updated["page_table"][valid_pages] = (
            updated["page_table"][valid_pages] + page_shift
        ).remainder(updated["kv_cache"].shape[0])
        for name, tensor in updated.items():
            static[name].copy_(tensor)
        assert {name: tensor.data_ptr() for name, tensor in static.items()} == pointers
        expected = _packed_reference(static)
        graph.replay()
        torch.cuda.synchronize()
        _assert_close(output, expected)
