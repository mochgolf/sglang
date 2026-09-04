import sys
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend_module
from sglang.srt.layers.attention.qsa.kernel import qsa_sparse_attention
from sglang.srt.layers.attention.qsa.sparse_attn import (
    qwen_sparse_kv_extraction_compact_triton,
    sparse_gqa_fwd_interface_triton,
    sparse_gqa_fwd_interface_triton_ck,
)
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _quantize_fp8(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    return (tensor / scale).to(torch.float8_e4m3fn)


def _torch_reference(q, k, v, slots, softmax_scale, k_scale=1.0, v_scale=1.0):
    """Independent sparse GQA oracle over already-quantized cache values."""
    repeats = q.shape[1] // k.shape[1]
    rows = []
    for row, row_slots in zip(q, slots):
        selected = row_slots[row_slots >= 0].long()
        keys = k[selected].float().repeat_interleave(repeats, dim=1) * k_scale
        values = v[selected].float().repeat_interleave(repeats, dim=1) * v_scale
        scores = torch.bmm(row.float().unsqueeze(1), keys.permute(1, 2, 0)).squeeze(1)
        probabilities = torch.softmax(scores * softmax_scale, dim=-1)
        rows.append(
            torch.bmm(probabilities.unsqueeze(1), values.permute(1, 0, 2)).squeeze(1)
        )
    return torch.stack(rows).to(q.dtype)


@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.25, 0.5)])
def test_qsa_sparse_attention_reference_applies_fp8_kv_descales(k_scale, v_scale):
    torch.manual_seed(23)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    k = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), k_scale)
    v = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), v_scale)
    slots = torch.tensor([[0, 2, 4, 6], [1, 3, 5, -1]], dtype=torch.int32)
    softmax_scale = 16**-0.5

    actual = qsa_sparse_attention(
        q, k, v, slots, softmax_scale, k_scale=k_scale, v_scale=v_scale
    )
    expected = _torch_reference(
        q, k, v, slots, softmax_scale, k_scale=k_scale, v_scale=v_scale
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_qsa_sparse_attention_bf16_control_path():
    torch.manual_seed(24)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    k = torch.randn(7, 2, 16, dtype=torch.bfloat16)
    v = torch.randn(7, 2, 16, dtype=torch.bfloat16)
    slots = torch.tensor([[0, 2, 4, 6], [1, 3, 5, -1]], dtype=torch.int32)
    softmax_scale = 16**-0.5

    actual = qsa_sparse_attention(q, k, v, slots, softmax_scale)
    expected = _torch_reference(q, k, v, slots, softmax_scale)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.25, 0.5)])
def test_qsa_fp8_cached_prefill_and_decode_use_layer_descales(k_scale, v_scale):
    torch.manual_seed(25)
    q = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    k = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), k_scale)
    v = _quantize_fp8(torch.randn(7, 2, 16, dtype=torch.bfloat16), v_scale)
    topk = torch.tensor([[0, 2, 4, 6], [1, 3, 5, -1]], dtype=torch.int32)
    metadata = SimpleNamespace(
        token_to_batch_idx=torch.tensor([0, 1]),
        sequence_lengths=torch.tensor([7, 7], dtype=torch.int32),
        token_slot_table=torch.arange(7).repeat(2, 1),
    )
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _layer_id: k,
        get_value_buffer=lambda _layer_id: v,
    )
    backend.forward_metadata = metadata
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=4,
        head_dim=16,
        scaling=16**-0.5,
        k_scale_float=k_scale,
        v_scale_float=v_scale,
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    expected = _torch_reference(
        q, k, v, topk, layer.scaling, k_scale=k_scale, v_scale=v_scale
    ).reshape(2, -1)

    prefill = backend.forward_extend(
        q, k, v, layer, forward_batch, save_kv_cache=False, topk_indices=topk
    )
    decode = backend._forward_paged_attention(q, layer, forward_batch, topk)
    torch.testing.assert_close(prefill, expected, rtol=0, atol=0)
    torch.testing.assert_close(decode, expected, rtol=0, atol=0)


@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.25, 0.5)])
def test_qsa_fp8_prefill_matches_independent_reference(k_scale, v_scale):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(27)
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device="cuda")
    k = _quantize_fp8(
        torch.randn(3, 1, 128, dtype=torch.bfloat16, device="cuda"), k_scale
    )
    v = _quantize_fp8(
        torch.randn(3, 1, 128, dtype=torch.bfloat16, device="cuda"), v_scale
    )
    indices = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [0, 1, 2]], dtype=torch.int32, device="cuda"
    )
    cu_seqlens = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    softmax_scale = 128**-0.5

    actual = sparse_gqa_fwd_interface_triton(
        q, k, v, 3, indices, cu_seqlens, softmax_scale, k_scale, v_scale
    )
    expected = _torch_reference(q, k, v, indices, softmax_scale, k_scale, v_scale)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


@pytest.mark.parametrize(
    "dtype,scales", [(torch.bfloat16, (1.0, 1.0)), (torch.float8_e4m3fn, (0.25, 0.5))]
)
def test_qsa_chunked_prefill_matches_independent_reference(dtype, scales):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(29)
    k_scale, v_scale = scales
    q = torch.randn(3, 4, 128, dtype=torch.bfloat16, device="cuda")
    source_k = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="cuda")
    source_v = torch.randn(6, 1, 128, dtype=torch.bfloat16, device="cuda")
    k = source_k if dtype == torch.bfloat16 else _quantize_fp8(source_k, k_scale)
    v = source_v if dtype == torch.bfloat16 else _quantize_fp8(source_v, v_scale)
    indices = torch.tensor(
        [[0, 1, 2, 3], [0, 2, 3, 4], [1, 3, 4, 5]],
        dtype=torch.int32,
        device="cuda",
    )
    cu_q = torch.tensor([0, 3], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 6], dtype=torch.int32, device="cuda")
    kv_lens = torch.tensor([6], dtype=torch.int32, device="cuda")
    softmax_scale = 128**-0.5
    scale_kwargs = (
        {"k_scale": k_scale, "v_scale": v_scale} if dtype == torch.float8_e4m3fn else {}
    )

    actual = sparse_gqa_fwd_interface_triton_ck(
        q, k, v, indices, cu_q, cu_k, kv_lens, softmax_scale, **scale_kwargs
    )
    expected = _torch_reference(q, k, v, indices, softmax_scale, **scale_kwargs)
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def test_qsa_fp8_compact_gather_dequantizes_for_flash_attention():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    torch.manual_seed(31)
    k_scale, v_scale = 0.25, 0.5
    k = _quantize_fp8(
        torch.randn(16, 1, 16, dtype=torch.bfloat16, device="cuda"), k_scale
    )
    v = _quantize_fp8(
        torch.randn(16, 1, 16, dtype=torch.bfloat16, device="cuda"), v_scale
    )
    req_to_token = torch.tensor(
        [[3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12]], dtype=torch.int32, device="cuda"
    )
    req_indices = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    indices = torch.tensor(
        [[0, 3, 5, -1], [1, 4, -1, -1]], dtype=torch.int32, device="cuda"
    )
    seq_lens = torch.tensor([6, 5], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 3, 5], dtype=torch.int32, device="cuda")
    out_k = torch.empty(8, 1, 16, dtype=torch.bfloat16, device="cuda")
    out_v = torch.empty_like(out_k)

    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k,
        out_v,
        2,
        4,
        k_scale,
        v_scale,
    )
    selected = torch.tensor([3, 9, 13, 4, 10], device="cuda")
    torch.testing.assert_close(
        out_k[:5].float(), k[selected].float() * k_scale, rtol=0, atol=2e-2
    )
    torch.testing.assert_close(
        out_v[:5].float(), v[selected].float() * v_scale, rtol=0, atol=2e-2
    )

    # TRTLLM applies descales in BMM1/BMM2 and therefore keeps packed KV in FP8.
    out_k_fp8 = torch.empty(8, 1, 16, dtype=torch.float8_e4m3fn, device="cuda")
    out_v_fp8 = torch.empty_like(out_k_fp8)
    qwen_sparse_kv_extraction_compact_triton(
        k,
        v,
        req_to_token,
        req_indices,
        indices,
        seq_lens,
        cu_k,
        out_k_fp8,
        out_v_fp8,
        2,
        4,
        k_scale,
        v_scale,
    )
    torch.testing.assert_close(
        out_k_fp8[:5].float(), k[selected].float(), rtol=0, atol=0
    )
    torch.testing.assert_close(
        out_v_fp8[:5].float(), v[selected].float(), rtol=0, atol=0
    )


@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.25, 0.5)])
def test_qsa_fp8_cache_write_preserves_live_kv(k_scale, v_scale):
    class Pool:
        dtype = torch.float8_e4m3fn

        def set_kv_buffer(
            self, layer, loc, cache_k, cache_v, write_k_scale=None, write_v_scale=None
        ):
            self.args = (cache_k, cache_v, write_k_scale, write_v_scale)
            if write_k_scale is not None:
                cache_k.div_(write_k_scale)
                cache_v.div_(write_v_scale)

    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = Pool()
    layer = SimpleNamespace(layer_id=0, k_scale_float=k_scale, v_scale_float=v_scale)
    k = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    v = torch.randn(3, 1, 16, dtype=torch.bfloat16)
    expected_k, expected_v = k.clone(), v.clone()

    backend._store_kv(layer, torch.arange(3, dtype=torch.int32), k, v)

    written_k, written_v, written_k_scale, written_v_scale = (
        backend.token_to_kv_pool.args
    )
    torch.testing.assert_close(k, expected_k)
    torch.testing.assert_close(v, expected_v)
    if (k_scale, v_scale) == (1.0, 1.0):
        assert written_k is k and written_v is v
        assert written_k_scale is None and written_v_scale is None
    else:
        assert written_k is not k and written_v is not v
        assert (written_k_scale, written_v_scale) == (k_scale, v_scale)


def test_qsa_trtllm_decode_receives_fp8_kv_descales(monkeypatch):
    seen = {}

    monkeypatch.setattr(
        qsa_backend_module,
        "qwen_sparse_valid_counts_triton",
        lambda *args: args[2].fill_(args[4]),
    )
    monkeypatch.setattr(
        qsa_backend_module,
        "qwen_sparse_kv_extraction_compact_triton",
        lambda *args, **kwargs: None,
    )

    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.req_to_token_pool = SimpleNamespace(
        req_to_token=torch.arange(8, dtype=torch.int32).reshape(1, 8)
    )
    backend._fa2_scratch = {}
    backend._trtllm_sparse_tables = {}
    backend._trtllm_workspace = torch.empty(1, dtype=torch.uint8)
    backend._cuda_graph_max_tokens = 0
    layer = SimpleNamespace(scaling=0.125, k_scale_float=0.25, v_scale_float=0.5)
    q = torch.randn(1, 4, 16, dtype=torch.bfloat16)
    k = torch.empty(8, 1, 16, dtype=torch.float8_e4m3fn)
    v = torch.empty_like(k)
    metadata = SimpleNamespace(
        sequence_lengths=torch.tensor([8], dtype=torch.int32),
        row_req_pool_indices=None,
        is_cuda_graph=False,
    )

    output = backend._forward_trtllm_sparse(
        q,
        k,
        v,
        layer,
        SimpleNamespace(req_pool_indices=torch.tensor([0])),
        metadata,
        torch.tensor([[0, 1, 2, 3]], dtype=torch.int32),
        lambda **kwargs: seen.update(kwargs) or torch.zeros_like(kwargs["query"]),
    )

    assert output.shape == (1, 64)
    assert seen["kv_cache"][0].dtype == torch.float8_e4m3fn
    assert seen["bmm1_scale"] == 0.125 * 0.25
    assert seen["bmm2_scale"] == 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
