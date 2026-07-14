from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from sglang.srt.layers.attention import dsa_backend


def _resolve_glm_sm89_backends(*, architectures=("GlmMoeDsaForCausalLM",), **overrides):
    from sglang.srt.arg_groups.overrides import (
        ResolvedView,
        _dsa_split_backend_resolution,
    )

    hf_config = SimpleNamespace(architectures=architectures)
    values = dict(
        kv_cache_dtype="fp8_e4m3",
        dsa_prefill_backend=None,
        dsa_decode_backend=None,
        dsa_paged_mqa_logits_backend="auto",
        dsa_topk_backend="sgl-kernel",
        enable_hisparse=False,
    )
    values.update(overrides)
    view = ResolvedView(
        SimpleNamespace(
            get_model_config=lambda: SimpleNamespace(hf_config=hf_config), **values
        )
    )
    with (
        patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
        patch("sglang.srt.arg_groups.overrides.is_npu", return_value=False),
        patch("sglang.srt.arg_groups.overrides.is_xpu", return_value=False),
        patch("sglang.srt.arg_groups.overrides.is_hip", return_value=False),
        patch("torch.cuda.get_device_capability", return_value=(8, 9)),
    ):
        return _dsa_split_backend_resolution(view)


def test_glm_sm89_fp8_defaults_resolve_complete_backend_set():
    assert _resolve_glm_sm89_backends() == {
        "dsa_prefill_backend": "sm89_triton",
        "dsa_decode_backend": "sm89_cuda",
        "dsa_paged_mqa_logits_backend": "sm89",
        "dsa_topk_backend": "torch",
    }


@pytest.mark.parametrize("architectures", [None, []])
def test_glm_sm89_resolver_ignores_missing_or_empty_architectures(architectures):
    assert _resolve_glm_sm89_backends(architectures=architectures) == {}


def test_glm_sm89_resolver_ignores_absent_architectures_attribute():
    from sglang.srt.arg_groups.overrides import (
        ResolvedView,
        _dsa_split_backend_resolution,
    )

    view = ResolvedView(
        SimpleNamespace(
            get_model_config=lambda: SimpleNamespace(hf_config=SimpleNamespace()),
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend=None,
            dsa_decode_backend=None,
            dsa_paged_mqa_logits_backend="auto",
            dsa_topk_backend="sgl-kernel",
            enable_hisparse=False,
        )
    )
    with (
        patch("sglang.srt.configs.model_config.is_deepseek_dsa") as is_deepseek_dsa,
        patch("sglang.srt.arg_groups.overrides.is_npu", return_value=False),
        patch("sglang.srt.arg_groups.overrides.is_xpu", return_value=False),
        patch("sglang.srt.arg_groups.overrides.is_hip", return_value=False),
        patch("torch.cuda.get_device_capability", return_value=(8, 9)),
    ):
        assert _dsa_split_backend_resolution(view) == {}

    is_deepseek_dsa.assert_not_called()


def test_glm_sm89_preserves_explicit_attention_backends():
    resolved = _resolve_glm_sm89_backends(
        dsa_prefill_backend="fa3", dsa_decode_backend="fa3"
    )
    assert "dsa_prefill_backend" not in resolved
    assert "dsa_decode_backend" not in resolved
    assert "dsa_paged_mqa_logits_backend" not in resolved
    assert "dsa_topk_backend" not in resolved


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        (
            {"dsa_prefill_backend": "sm89_triton"},
            {
                "dsa_decode_backend": "sm89_cuda",
                "dsa_paged_mqa_logits_backend": "sm89",
                "dsa_topk_backend": "torch",
            },
        ),
        (
            {"dsa_decode_backend": "sm89_cuda"},
            {
                "dsa_prefill_backend": "sm89_triton",
                "dsa_paged_mqa_logits_backend": "sm89",
                "dsa_topk_backend": "torch",
            },
        ),
        (
            {"dsa_paged_mqa_logits_backend": "sm89"},
            {
                "dsa_prefill_backend": "sm89_triton",
                "dsa_decode_backend": "sm89_cuda",
                "dsa_topk_backend": "torch",
            },
        ),
    ],
)
def test_glm_sm89_resolver_completes_an_explicit_partial_backend_set(
    overrides, expected
):
    assert _resolve_glm_sm89_backends(**overrides) == expected


@pytest.mark.parametrize(
    "overrides",
    [
        {"dsa_prefill_backend": "sm89_triton", "dsa_decode_backend": "fa3"},
        {"dsa_prefill_backend": "fa3", "dsa_decode_backend": "sm89_cuda"},
        {
            "dsa_prefill_backend": "sm89_triton",
            "dsa_paged_mqa_logits_backend": "deepgemm",
        },
        {
            "dsa_prefill_backend": "fa3",
            "dsa_paged_mqa_logits_backend": "sm89",
        },
    ],
)
def test_glm_sm89_resolver_rejects_mixed_backend_set(overrides):
    with pytest.raises(ValueError, match="complete SM89 DSA backend set"):
        _resolve_glm_sm89_backends(**overrides)


def _make_backend(
    *,
    model_arch="GlmMoeDsaForCausalLM",
    capability=(8, 9),
    kv_cache_dtype=torch.float8_e4m3fn,
    prefill="sm89_triton",
    decode="sm89_cuda",
    paged_indexer="sm89",
    speculative_algorithm=None,
):
    backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
    backend.model_arch = model_arch
    backend.device_capability = capability
    backend.kv_cache_dtype = kv_cache_dtype
    backend.dsa_prefill_impl = prefill
    backend.dsa_decode_impl = decode
    backend.dsa_paged_mqa_logits_backend = paged_indexer
    backend.speculative_algorithm = speculative_algorithm
    return backend


def test_accepts_glm_sm89_fp8_backend_contract():
    backend = _make_backend()
    backend._validate_sm89_backend_config()


def test_sm89_unfused_topk_does_not_build_topk_v2_plan():
    backend = _make_backend()
    backend.use_fused_topk = False
    seqlens = torch.tensor([1], dtype=torch.int32)

    with (
        patch.object(dsa_backend.envs.SGLANG_OPT_USE_TOPK_V2, "get", return_value=True),
        patch("sglang.jit_kernel.dsv4.topk.plan_topk_v2") as plan_topk_v2,
    ):
        assert backend._build_topk_v2_plan(seqlens) is None

    plan_topk_v2.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_arch", "DeepseekV3ForCausalLM", "GLM DSA"),
        ("device_capability", (9, 0), "SM89"),
        ("kv_cache_dtype", torch.bfloat16, "FP8 E4M3"),
        ("dsa_paged_mqa_logits_backend", "deepgemm", "paged MQA"),
        ("dsa_prefill_impl", "sm89_cuda", "prefill"),
        ("dsa_decode_impl", "sm89_triton", "decode"),
        ("speculative_algorithm", "EAGLE", "speculative"),
    ],
)
def test_rejects_invalid_sm89_backend_contract(field, value, message):
    backend = _make_backend()
    setattr(backend, field, value)
    with pytest.raises(ValueError, match=message):
        backend._validate_sm89_backend_config()


@pytest.mark.parametrize(
    ("prefill", "decode", "paged_indexer"),
    [
        ("fa3", "sm89_cuda", "sm89"),
        ("sm89_triton", "fa3", "sm89"),
        ("fa3", "fa3", "sm89"),
    ],
)
def test_backend_rejects_incomplete_sm89_backend_set(prefill, decode, paged_indexer):
    backend = _make_backend(prefill=prefill, decode=decode, paged_indexer=paged_indexer)

    with pytest.raises(ValueError, match="complete SM89 DSA backend set"):
        backend._validate_sm89_backend_config()


def test_sm89_prefill_helper_calls_sparse_mla_facade():
    backend = _make_backend()
    q_nope = torch.empty(3, 32, 512)
    q_rope = torch.empty(3, 32, 64)
    kv_cache = torch.empty(5, 1, 656, dtype=torch.float8_e4m3fn)
    page_table = torch.empty(3, 2048, dtype=torch.int32)
    cache_seqlens = torch.empty(3, dtype=torch.int32)
    expected = torch.empty_like(q_nope)

    with patch(
        "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
        "sm89_sparse_mla_prefill_triton",
        return_value=expected,
    ) as op:
        actual = backend._forward_sm89_triton(
            q_rope=q_rope,
            kv_cache=kv_cache,
            v_head_dim=512,
            q_nope=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=0.125,
            logit_cap=30.0,
            page_size=1,
        )

    assert actual is expected
    op.assert_called_once_with(
        q_nope=q_nope,
        q_rope=q_rope,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        sm_scale=0.125,
        logit_cap=30.0,
        v_head_dim=512,
    )


def test_sm89_decode_helper_calls_sparse_mla_facade():
    backend = _make_backend()
    q_nope = torch.empty(1, 32, 512)
    q_rope = torch.empty(1, 32, 64)
    kv_cache = torch.empty(8, 1, 656, dtype=torch.float8_e4m3fn)
    page_table = torch.empty(1, 2048, dtype=torch.int32)
    cache_seqlens = torch.empty(1, dtype=torch.int32)
    expected = torch.empty_like(q_nope)

    with patch(
        "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
        "sm89_sparse_mla_decode_cuda",
        return_value=expected,
    ) as op:
        actual = backend._forward_sm89_cuda_decode(
            q_rope=q_rope,
            kv_cache=kv_cache,
            v_head_dim=512,
            q_nope=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=0.125,
            logit_cap=30.0,
            page_size=1,
        )

    assert actual is expected
    op.assert_called_once_with(
        q_nope=q_nope,
        q_rope=q_rope,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        sm_scale=0.125,
        logit_cap=30.0,
        v_head_dim=512,
    )


def test_forward_decode_routes_physical_topk_table_to_sm89_cuda():
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    backend = _make_backend()
    backend.hisparse_coordinator = None
    backend.use_fused_topk = False
    backend._pad_topk_indices = lambda indices, _rows: indices
    logical_page_table = torch.tensor([[5, 6, 7]], dtype=torch.int32)
    physical_page_table = torch.tensor([[41, 17, -1]], dtype=torch.int32)
    cache_seqlens = torch.tensor([3], dtype=torch.int32)
    kv_cache = torch.empty(64, 1, 656, dtype=torch.float8_e4m3fn)
    backend.forward_metadata = SimpleNamespace(
        page_table_1=logical_page_table,
        dsa_cache_seqlens_int32=cache_seqlens,
    )
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _layer_id: kv_cache
    )
    expected = object()
    backend._forward_sm89_cuda_decode = MagicMock(return_value=expected)
    q_nope = torch.empty(1, 32, 512)
    q_rope = torch.empty(1, 32, 64)
    topk_indices = torch.tensor([[2, 1, 0]], dtype=torch.int32)
    layer = SimpleNamespace(
        is_cross_attention=False,
        tp_q_head_num=32,
        v_head_dim=512,
        head_dim=576,
        scaling=0.125,
        logit_cap=30.0,
        layer_id=7,
    )
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)

    with patch.object(
        dsa_backend,
        "transform_index_page_table_decode",
        return_value=physical_page_table,
    ) as transform:
        actual = backend.forward_decode(
            q=q_nope,
            k=None,
            v=None,
            layer=layer,
            forward_batch=forward_batch,
            save_kv_cache=False,
            q_rope=q_rope,
            topk_indices=topk_indices,
        )

    assert actual is expected
    transform.assert_called_once_with(
        page_table=logical_page_table,
        topk_indices=topk_indices,
        page_size=1,
    )
    backend._forward_sm89_cuda_decode.assert_called_once()
    call = backend._forward_sm89_cuda_decode.call_args.kwargs
    assert call["q_rope"].data_ptr() == q_rope.data_ptr()
    assert call["q_nope"].data_ptr() == q_nope.data_ptr()
    assert call["kv_cache"] is kv_cache
    assert call["page_table"] is physical_page_table
    assert call["cache_seqlens"] is cache_seqlens
    assert call["v_head_dim"] == 512
    assert call["sm_scale"] == 0.125
    assert call["logit_cap"] == 30.0
    assert call["page_size"] == 1
