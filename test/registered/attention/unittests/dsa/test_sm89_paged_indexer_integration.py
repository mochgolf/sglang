from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.layers.attention.dsa import sm89_paged_indexer
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)

from .test_sm89_paged_indexer import (
    HEAD_DIM,
    NUM_HEADS,
    PAGE_SIZE,
    _make_q,
    _make_raw_cache,
    _make_weights,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_indexer_glm_sm89_paged_branch_uses_raw_cache_without_deepgemm():
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = object.__new__(dsa_indexer.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.SM89
    indexer.index_topk = 8
    indexer.sm_count = 128
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0

    q_fp8 = _make_q(2)[:, 0]
    weights = _make_weights(2).unsqueeze(-1)
    raw_cache = _make_raw_cache(2)
    seq_lens = torch.tensor([93], device="cuda", dtype=torch.int32)
    page_table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    logits = torch.empty(1, 2 * PAGE_SIZE, device="cuda", dtype=torch.float32)
    real_topk = torch.arange(8, device="cuda", dtype=torch.int64).unsqueeze(0)
    expected_topk = torch.cat((real_topk, torch.full_like(real_topk, -1)), dim=0)

    metadata = MagicMock()
    metadata.paged_mqa_schedule_metadata = None
    metadata.get_page_table_64.return_value = page_table
    metadata.get_seqlens_int32.return_value = seq_lens
    metadata.get_dsa_extend_len_cpu.return_value = [1]
    metadata.topk_transform.return_value = real_topk
    token_to_kv_pool = SimpleNamespace(
        page_size=PAGE_SIZE,
        get_index_k_with_scale_buffer=MagicMock(return_value=raw_cache),
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    def fake_sm89_logits(
        actual_q,
        actual_cache,
        actual_weights,
        actual_seq_lens,
        actual_page_table,
        actual_max_seq_len,
    ):
        assert actual_q.shape == (1, 1, NUM_HEADS, HEAD_DIM)
        assert actual_q.is_contiguous()
        assert actual_cache is raw_cache
        assert actual_cache.ndim == 2
        assert actual_weights.shape == (1, NUM_HEADS)
        assert actual_weights.is_contiguous()
        assert actual_seq_lens is seq_lens
        assert actual_seq_lens.device.type == "cuda"
        assert actual_seq_lens.dtype == torch.int32
        assert actual_page_table is page_table
        assert actual_page_table.device.type == "cuda"
        assert actual_page_table.dtype == torch.int32
        assert actual_max_seq_len == 2 * PAGE_SIZE
        return logits

    with (
        patch.object(
            dsa_indexer,
            "get_token_to_kv_pool",
            return_value=token_to_kv_pool,
        ),
        patch.object(
            sm89_paged_indexer,
            "sm89_paged_fp8_index_logits",
            side_effect=fake_sm89_logits,
        ) as mock_sm89_logits,
        patch.object(
            dsa_indexer.deep_gemm,
            "get_paged_mqa_logits_metadata",
            side_effect=AssertionError("DeepGEMM schedule must not be constructed"),
        ) as mock_schedule,
    ):
        actual_topk = indexer._get_topk_paged(
            forward_batch,
            layer_id=7,
            q_fp8=q_fp8,
            weights=weights,
            metadata=metadata,
        )

    assert torch.equal(actual_topk, expected_topk)
    mock_sm89_logits.assert_called_once()
    mock_schedule.assert_not_called()
    metadata.get_dsa_extend_len_cpu.assert_called_once_with()
    metadata.topk_transform.assert_called_once_with(logits, 8)


def test_sm89_prefill_uses_portable_indexer_instead_of_deepgemm_ragged():
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = MagicMock()
    indexer.index_topk = 2048
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    expected = torch.arange(8, device="cuda", dtype=torch.int32).reshape(1, 8)
    indexer.forward_indexer.return_value = expected
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, attn_cp_metadata=None
    )
    q_fp8 = _make_q(1)[:, 0]
    weights = _make_weights(1).unsqueeze(-1)
    metadata = MagicMock()

    with (
        patch.object(
            dsa_indexer,
            "_broadcast_indexer_topk_from_rank0",
            side_effect=lambda value: value,
        ),
        patch.object(
            dsa_indexer,
            "maybe_capture_indexer_topk",
            side_effect=lambda _layer_id, value: value,
        ),
    ):
        actual = dsa_indexer._select_sm89_topk(
            indexer,
            q_fp8,
            weights,
            forward_batch,
            metadata,
            layer_id=7,
        )

    assert actual is expected
    indexer.forward_indexer.assert_called_once_with(
        q_fp8.contiguous(),
        weights,
        forward_batch,
        topk=2048,
        layer_id=7,
    )
    indexer._get_topk_paged.assert_not_called()
    indexer._get_topk_ragged.assert_not_called()


def test_forward_indexer_masks_future_positions_for_each_extend_request():
    from sglang.srt.layers.attention.dsa import dsa_indexer, tilelang_kernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = object.__new__(dsa_indexer.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_topk = 4
    q_lens = [2, 3]
    seq_lens = [5, 7]
    q_fp8 = torch.empty(sum(q_lens), NUM_HEADS, HEAD_DIM, device="cuda")
    weights = torch.empty(sum(q_lens), NUM_HEADS, 1, device="cuda")
    forward_batch = SimpleNamespace(
        batch_size=len(q_lens),
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.tensor(seq_lens, device="cuda", dtype=torch.int32),
        extend_seq_lens_cpu=torch.tensor(q_lens, dtype=torch.int32),
        req_pool_indices=torch.tensor([0, 1], device="cuda", dtype=torch.int32),
    )
    token_to_kv_pool = SimpleNamespace(
        page_size=PAGE_SIZE,
        get_index_k_continuous=lambda _layer_id, seq_len, _block_table: torch.empty(
            seq_len, HEAD_DIM, device="cuda", dtype=torch.uint8
        ),
        get_index_k_scale_continuous=lambda _layer_id, seq_len, _block_table: (
            torch.empty(seq_len, 1, device="cuda", dtype=torch.float32)
        ),
    )
    req_to_token_pool = SimpleNamespace(
        req_to_token=torch.zeros(2, PAGE_SIZE, device="cuda", dtype=torch.int32)
    )

    def future_favoring_fp8_index(actual_q, _weights, actual_k, _scale):
        q_len = actual_q.shape[1]
        seq_len = actual_k.shape[1]
        return (
            torch.arange(seq_len, device="cuda", dtype=torch.float32)
            .expand(1, q_len, -1)
            .clone()
        )

    with (
        patch.object(
            dsa_indexer, "get_token_to_kv_pool", return_value=token_to_kv_pool
        ),
        patch.object(
            dsa_indexer, "get_req_to_token_pool", return_value=req_to_token_pool
        ),
        patch.object(
            tilelang_kernel, "fp8_index", side_effect=future_favoring_fp8_index
        ),
    ):
        topk_indices = indexer.forward_indexer(
            q_fp8, weights, forward_batch, topk=indexer.index_topk, layer_id=7
        )

    q_start = 0
    for seq_len, q_len in zip(seq_lens, q_lens):
        prefix_len = seq_len - q_len
        for query_offset in range(q_len):
            causal_len = prefix_len + query_offset + 1
            consumed = topk_indices[q_start + query_offset, : min(causal_len, 4)]
            assert torch.all(consumed < causal_len)
        q_start += q_len


def test_sm89_idle_empty_batch_returns_all_invalid_without_paged_kernel():
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = MagicMock()
    indexer.index_topk = 8
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    indexer._get_topk_paged.return_value = torch.zeros(
        2, 8, device="cuda", dtype=torch.int32
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.IDLE,
        seq_lens=torch.empty(0, device="cuda", dtype=torch.int32),
        attn_cp_metadata=None,
    )

    with (
        patch.object(
            dsa_indexer,
            "_broadcast_indexer_topk_from_rank0",
            side_effect=lambda value: value,
        ),
        patch.object(
            dsa_indexer,
            "maybe_capture_indexer_topk",
            side_effect=lambda _layer_id, value: value,
        ),
    ):
        actual = dsa_indexer._select_sm89_topk(
            indexer,
            _make_q(2)[:, 0],
            _make_weights(2).unsqueeze(-1),
            forward_batch,
            MagicMock(),
            layer_id=7,
        )

    assert torch.equal(actual, torch.full((2, 8), -1, device="cuda", dtype=torch.int32))
    indexer._get_topk_paged.assert_not_called()
    indexer.forward_indexer.assert_not_called()


@pytest.mark.parametrize("forward_mode", ["TARGET_VERIFY", "DRAFT_EXTEND_V2"])
def test_sm89_rejects_speculative_paged_modes(forward_mode):
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = MagicMock()
    indexer.index_topk = 8
    forward_batch = SimpleNamespace(
        forward_mode=getattr(ForwardMode, forward_mode),
        seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
        attn_cp_metadata=None,
    )

    with pytest.raises(RuntimeError, match="SM89 DSA indexer.*speculative"):
        dsa_indexer._select_sm89_topk(
            indexer,
            _make_q(1)[:, 0],
            _make_weights(1).unsqueeze(-1),
            forward_batch,
            MagicMock(),
            layer_id=7,
        )

    indexer._get_topk_paged.assert_not_called()
    indexer.forward_indexer.assert_not_called()


@pytest.mark.parametrize(
    ("attribute", "value"), [("num_init_tokens", 1), ("num_local_tokens", 1)]
)
def test_sm89_rejects_nonzero_init_or_local_tokens(attribute, value):
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = MagicMock()
    indexer.index_topk = 8
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    setattr(indexer, attribute, value)
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.DECODE,
        seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
        attn_cp_metadata=None,
    )

    with pytest.raises(RuntimeError, match="num_init_tokens.*num_local_tokens.*zero"):
        dsa_indexer._select_sm89_topk(
            indexer,
            _make_q(1)[:, 0],
            _make_weights(1).unsqueeze(-1),
            forward_batch,
            MagicMock(),
            layer_id=7,
        )

    indexer._get_topk_paged.assert_not_called()
    indexer.forward_indexer.assert_not_called()


def test_forward_cuda_rejects_sm89_nonzero_tokens_before_short_prefill_fast_path():
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = object.__new__(dsa_indexer.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.alt_stream = None
    indexer.dsa_enable_prefill_cp = False
    indexer.index_topk = 2048
    indexer.num_init_tokens = 1
    indexer.num_local_tokens = 0
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.SM89
    indexer._forward_cuda_k_only = MagicMock(
        side_effect=AssertionError("SM89 validation must run before the short path")
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=None,
        seq_lens_cpu=torch.ones(1, dtype=torch.int32),
        seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
    )
    attn_backend = MagicMock()
    attn_backend.get_indexer_metadata.return_value = MagicMock()

    with (
        patch.object(
            dsa_indexer,
            "_is_in_piecewise_or_breakable_cuda_graph",
            return_value=False,
        ),
        patch.object(dsa_indexer, "get_attn_backend", return_value=attn_backend),
    ):
        with pytest.raises(
            RuntimeError, match="num_init_tokens.*num_local_tokens.*zero"
        ):
            indexer.forward_cuda(
                torch.empty(1, 1, device="cuda"),
                torch.empty(1, 1, device="cuda"),
                torch.zeros(1, device="cuda", dtype=torch.int64),
                forward_batch,
                layer_id=7,
            )

    indexer._forward_cuda_k_only.assert_not_called()


def test_sm89_rejects_prefill_context_parallel():
    from sglang.srt.layers.attention.dsa import dsa_indexer
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = MagicMock()
    indexer.index_topk = 8
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        seq_lens=torch.ones(1, device="cuda", dtype=torch.int32),
        attn_cp_metadata=object(),
    )

    with pytest.raises(
        RuntimeError, match="SM89 DSA indexer.*prefill context parallel"
    ):
        dsa_indexer._select_sm89_topk(
            indexer,
            _make_q(1)[:, 0],
            _make_weights(1).unsqueeze(-1),
            forward_batch,
            MagicMock(),
            layer_id=7,
        )

    indexer._get_topk_paged.assert_not_called()
    indexer.forward_indexer.assert_not_called()


@pytest.mark.parametrize("breakable", [False, True], ids=["pcg", "bcg"])
def test_sm89_graph_prefill_bypasses_split_op_and_reports_unsupported(breakable):
    from sglang.srt.layers.attention.dsa import dsa_indexer, triton_kernel
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    indexer = object.__new__(dsa_indexer.Indexer)
    torch.nn.Module.__init__(indexer)
    indexer.alt_stream = None
    indexer.block_size = 128
    indexer.scale_fmt = None
    indexer.dsa_enable_prefill_cp = False
    indexer.use_dsa_indexer_fusion = False
    indexer.n_heads = NUM_HEADS
    indexer.softmax_scale = HEAD_DIM**-0.5
    indexer.index_topk = 8
    indexer.num_init_tokens = 0
    indexer.num_local_tokens = 0
    indexer.paged_mqa_logits_backend = DSAPagedMQALogitsBackend.SM89
    indexer.weights_proj = SimpleNamespace(
        set_lora=False, weight=torch.empty(1, device="cuda")
    )

    q_bf16 = torch.ones(1, NUM_HEADS, HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    q_fp8 = _make_q(1)[:, 0]
    q_scale = torch.ones(1, NUM_HEADS, 1, device="cuda", dtype=torch.float32)
    weights = _make_weights(1).unsqueeze(-1)
    indexer._get_q_k_bf16 = MagicMock(
        return_value=(q_bf16, torch.empty_like(q_bf16), None)
    )
    forward_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        attn_cp_metadata=None,
    )

    with (
        patch.object(
            dsa_indexer, "_is_in_piecewise_or_breakable_cuda_graph", return_value=True
        ),
        patch.object(
            dsa_indexer,
            "is_in_breakable_cuda_graph",
            return_value=breakable,
        ),
        patch.object(
            dsa_indexer,
            "is_graph_dsa_split_op_surface",
            return_value=True,
        ),
        patch.object(triton_kernel, "act_quant", return_value=(q_fp8, q_scale)),
        patch.object(dsa_indexer, "logits_head_gate_graph", return_value=weights),
        patch.object(dsa_indexer, "pcg_dsa_indexer_prefill_split") as pcg_split,
        patch.object(dsa_indexer, "bcg_dsa_indexer_prefill_split") as bcg_split,
    ):
        with pytest.raises(RuntimeError, match="SM89 DSA indexer.*CUDA graph prefill"):
            indexer.forward_cuda(
                torch.empty(1, 1, device="cuda"),
                torch.empty(1, 1, device="cuda"),
                torch.zeros(1, device="cuda", dtype=torch.int64),
                forward_batch,
                layer_id=7,
            )

    pcg_split.assert_not_called()
    bcg_split.assert_not_called()
