import gc
import hashlib
import inspect
import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


_SPLITK_SCALE_FORMULA_VERSION = "token5-group3-mod7-v1"
_SPLITK_SCALE_SEED = 20260712


def _splitk_host_partial_state(logits, values, split_tokens):
    partial_lse = []
    partial_o = []
    for start in range(0, logits.numel(), split_tokens):
        split_logits = logits[start : start + split_tokens]
        split_values = values[start : start + split_tokens]
        finite = torch.isfinite(split_logits)
        if not finite.any():
            partial_lse.append(torch.tensor(float("-inf"), dtype=torch.float32))
            partial_o.append(torch.zeros(values.shape[1], dtype=torch.float32))
            continue
        valid_logits = split_logits[finite].float()
        lse = torch.logsumexp(valid_logits, dim=0)
        probabilities = torch.exp(valid_logits - lse).to(torch.bfloat16).float()
        rounded_values = split_values[finite].to(torch.bfloat16).float()
        partial_lse.append(lse)
        partial_o.append(probabilities @ rounded_values)
    return torch.stack(partial_lse), torch.stack(partial_o)


def _splitk_host_combine(partial_lse, partial_o):
    finite = torch.isfinite(partial_lse)
    if not finite.any():
        return torch.zeros(partial_o.shape[-1], dtype=torch.float32)
    global_lse = torch.logsumexp(partial_lse[finite], dim=0)
    weights = torch.zeros_like(partial_lse)
    weights[finite] = torch.exp(partial_lse[finite] - global_lse)
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


class TestSm89SparseMlaSplitKHost(unittest.TestCase):
    def test_splitk_host_ownership_maps(self):
        for split_tokens in (32, 64):
            qk_warp_count = split_tokens // 16
            for split in range(2048 // split_tokens):
                split_start = split * split_tokens
                qk_tokens = [
                    split_start + 16 * warp + token_local
                    for warp in range(qk_warp_count)
                    for token_local in range(16)
                ]
                self.assertEqual(
                    sorted(qk_tokens),
                    list(range(split_start, split_start + split_tokens)),
                )
            pv_dimensions = [
                16 * (4 * warp + fragment) + dim_local
                for warp in range(8)
                for fragment in range(4)
                for dim_local in range(16)
            ]
            self.assertEqual(sorted(pv_dimensions), list(range(512)))

    def test_splitk_host_softmax_state_oracle(self):
        logits = torch.tensor(
            [float("-inf")] * 4
            + [-1.5, 0.25, float("-inf"), 1.0]
            + [float("-inf")] * 4,
            dtype=torch.float32,
        )
        values = torch.arange(12 * 7, dtype=torch.float32).reshape(12, 7) / 17
        partial_lse, partial_o = _splitk_host_partial_state(logits, values, 4)
        merged = _splitk_host_combine(partial_lse, partial_o)
        finite = torch.isfinite(logits)
        full_lse = torch.logsumexp(logits[finite], dim=0)
        full_prob = torch.exp(logits[finite] - full_lse).to(torch.bfloat16).float()
        expected = full_prob @ values[finite].to(torch.bfloat16).float()

        self.assertTrue(torch.isneginf(partial_lse[[0, 2]]).all())
        self.assertTrue(torch.equal(partial_o[[0, 2]], torch.zeros_like(partial_o[[0, 2]])))
        torch.testing.assert_close(merged, expected, atol=0, rtol=0)

        empty_lse = torch.full((3,), float("-inf"), dtype=torch.float32)
        empty_o = torch.zeros(3, 7, dtype=torch.float32)
        all_empty = _splitk_host_combine(empty_lse, empty_o)
        self.assertTrue(torch.isfinite(all_empty).all())
        self.assertTrue(torch.equal(all_empty, torch.zeros_like(all_empty)))


def test_decode_kernel_selector(monkeypatch):
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        select_sm89_sparse_mla_decode_kernel,
    )

    monkeypatch.delenv("SGLANG_GLM_DSA_SM89_DECODE_KERNEL", raising=False)
    assert select_sm89_sparse_mla_decode_kernel() == "three_stage"
    assert select_sm89_sparse_mla_decode_kernel(" splitK32 ") == "splitk32"
    assert select_sm89_sparse_mla_decode_kernel("splitk64") == "splitk64"
    with pytest.raises(ValueError, match="three_stage.*splitk32.*splitk64"):
        select_sm89_sparse_mla_decode_kernel("tensorcore")


def test_splitk_workspace():
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        splitk_workspace_schema,
    )

    assert splitk_workspace_schema(1, 32, 2048, 32) == {
        "splits": 64,
        "partial_o_bytes": 4_194_304,
        "partial_lse_bytes": 8_192,
        "workspace_bytes": 4_202_496,
    }
    assert splitk_workspace_schema(1, 32, 2048, 64)["workspace_bytes"] == 2_101_248


def test_splitk_partial_only_wrapper_uses_exact_preallocated_workspace(monkeypatch):
    from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

    q_nope = object()
    q_rope = object()
    kv_cache = object()
    page_table = object()
    cache_seqlens = object()
    partial_o = object()
    partial_lse = object()
    allocation_calls = []

    def fake_empty(shape, *, device, dtype):
        allocation_calls.append((shape, device, dtype))
        return partial_o if len(allocation_calls) == 1 else partial_lse

    q_descriptor = SimpleNamespace(is_cuda=True, device=torch.device("cuda:0"))
    monkeypatch.setattr(torch, "empty", fake_empty)
    monkeypatch.setattr(
        sm89_sparse_mla_cuda,
        "_validate_sm89_sparse_mla_decode_cuda_splitk_workspace",
        MagicMock(),
    )
    allocated = (
        sm89_sparse_mla_cuda.allocate_sm89_sparse_mla_decode_cuda_splitk_workspace(
            q_descriptor, 32
        )
    )
    assert allocated == (partial_o, partial_lse)
    assert allocation_calls == [
        ((64, 1, 32, 512), torch.device("cuda:0"), torch.float32),
        ((64, 1, 32), torch.device("cuda:0"), torch.float32),
    ]

    partial_op = MagicMock(return_value=None)
    extension = SimpleNamespace(
        sm89_sparse_mla_decode_cuda_splitk_partial=partial_op
    )
    monkeypatch.setattr(
        sm89_sparse_mla_cuda,
        "_validate_sm89_sparse_mla_decode_cuda_splitk",
        MagicMock(return_value=0.125),
    )
    monkeypatch.setattr(
        sm89_sparse_mla_cuda,
        "_load_sm89_sparse_mla_cuda_ext",
        MagicMock(return_value=extension),
    )

    result = sm89_sparse_mla_cuda.sm89_sparse_mla_decode_cuda_splitk_partial(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        partial_o,
        partial_lse,
        0.125,
        0.0,
        512,
        32,
    )
    assert result is None
    partial_op.assert_called_once_with(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        partial_o,
        partial_lse,
        0.125,
        0.0,
        512,
        32,
    )


def test_splitk_partial_only_cuda_source_contract():
    from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

    source_path = (
        Path(sm89_sparse_mla_cuda.__file__).resolve().parent
        / "csrc"
        / "sm89_sparse_mla_cuda.cu"
    )
    source = source_path.read_text(encoding="utf-8")
    python_validation = inspect.getsource(
        sm89_sparse_mla_cuda._validate_sm89_sparse_mla_decode_cuda_splitk_workspace
    )
    partial_entry = _cuda_kernel_source_body(
        source, "sm89_sparse_mla_decode_cuda_splitk_partial"
    )
    shared_partial_launcher = _cuda_kernel_source_body(
        source, "launch_sm89_sparse_mla_splitk_partial"
    )
    full_launcher = _cuda_kernel_source_body(
        source, "Sm89SparseMlaSplitKResult launch_sm89_sparse_mla_splitk("
    )

    assert "validate_sm89_sparse_mla_splitk(" in partial_entry
    assert "validate_sm89_sparse_mla_splitk_workspace(" in partial_entry
    assert "launch_sm89_sparse_mla_splitk_partial(" in partial_entry
    assert "torch::empty" not in partial_entry
    assert "combine" not in partial_entry
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK();" in shared_partial_launcher
    assert "launch_sm89_sparse_mla_splitk_partial(" in full_launcher
    assert "sm89_sparse_mla_splitk_combine_kernel" in full_launcher
    assert (
        'm.def("sm89_sparse_mla_decode_cuda_splitk_partial", '
        "&sm89_sparse_mla_decode_cuda_splitk_partial"
    ) in source
    for marker in (
        "partial_o.scalar_type() == at::kFloat",
        "partial_lse.scalar_type() == at::kFloat",
        "partial_o.is_contiguous()",
        "partial_lse.is_contiguous()",
        "partial_o.storage().nbytes() > 0",
        "partial_lse.storage().nbytes() > 0",
        "reinterpret_cast<uintptr_t>(partial_o.data_ptr()) % 32 == 0",
    ):
        assert marker in source
    for marker in (
        "partial_o.device != q_nope.device",
        "partial_lse.device != q_nope.device",
        "partial_o.dtype != torch.float32",
        "partial_lse.dtype != torch.float32",
        "partial_o.shape != (splits, 1, 32, 512)",
        "partial_lse.shape != (splits, 1, 32)",
        "partial_o.is_contiguous()",
        "partial_lse.is_contiguous()",
        "partial_o.untyped_storage().nbytes() <= 0",
        "partial_lse.untyped_storage().nbytes() <= 0",
        "partial_o.data_ptr() % 32",
    ):
        assert marker in python_validation


class TestSm89SparseMlaBackendLogging(unittest.TestCase):
    def test_glm_sm89_fa3_logs_effective_torch_mla_only_once(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_glm_sm89_dsa_fallback = True
        backend.device_sm_major = 8
        backend._logged_glm_sm89_effective_torch_mla = False

        sentinel = object()
        backend._forward_torch_mla = MagicMock(return_value=sentinel)

        q_rope = torch.zeros(1, 1, 2)
        q_nope = torch.zeros(1, 1, 2)
        kv_cache = torch.zeros(1, 1, 4)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        cache_seqlens = torch.ones(1, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        with patch.object(dsa_backend.logger, "warning") as mock_warning, patch(
            "sglang.srt.layers.attention.dsa_backend.flash_attn_with_kvcache"
        ) as mock_fa3:
            first = backend._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=2,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
            )
            second = backend._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=2,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
            )

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        backend._forward_torch_mla.assert_called()
        mock_fa3.assert_not_called()
        mock_warning.assert_called_once()
        self.assertIn("GLM DSA SM89 effective backend is torch_mla", mock_warning.call_args.args[0])


class TestSm89SparseMlaDebugHelpers(unittest.TestCase):
    def test_glm_dsa_sm89_profile_enabled_reflects_env_var(self):
        from sglang.srt.layers.attention.dsa.sm89_debug import (
            glm_dsa_sm89_profile_enabled,
        )

        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(glm_dsa_sm89_profile_enabled())
        with patch.dict(
            "os.environ", {"SGLANG_GLM_DSA_SM89_PROFILE": "0"}, clear=True
        ):
            self.assertFalse(glm_dsa_sm89_profile_enabled())
        with patch.dict(
            "os.environ", {"SGLANG_GLM_DSA_SM89_PROFILE": "1"}, clear=True
        ):
            self.assertTrue(glm_dsa_sm89_profile_enabled())

    def test_cuda_timer_disabled_skips_cuda_events_and_print(self):
        from sglang.srt.layers.attention.dsa.sm89_debug import cuda_timer

        with patch("torch.cuda.Event") as mock_event, patch(
            "builtins.print"
        ) as mock_print:
            with cuda_timer("torch_mla.total", enabled=False):
                pass

        mock_event.assert_not_called()
        mock_print.assert_not_called()

    def test_profile_region_disabled_skips_nvtx_and_timer(self):
        from sglang.srt.layers.attention.dsa import sm89_debug

        with patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_PROFILE": "0"}, clear=True
        ), patch.object(sm89_debug, "nvtx_range") as mock_nvtx, patch.object(
            sm89_debug, "cuda_timer"
        ) as mock_timer:
            with sm89_debug.profile_region("moe.test"):
                pass

        mock_nvtx.assert_not_called()
        mock_timer.assert_not_called()

    def test_profile_region_enabled_wraps_nvtx_and_timer(self):
        from sglang.srt.layers.attention.dsa import sm89_debug

        events = []

        @contextmanager
        def fake_nvtx_range(name):
            events.append(("nvtx_enter", name))
            try:
                yield
            finally:
                events.append(("nvtx_exit", name))

        @contextmanager
        def fake_cuda_timer(name, enabled):
            events.append(("timer_enter", name, enabled))
            try:
                yield
            finally:
                events.append(("timer_exit", name, enabled))

        with patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_PROFILE": "1"}, clear=True
        ), patch.object(
            sm89_debug, "nvtx_range", side_effect=fake_nvtx_range
        ), patch.object(
            sm89_debug, "cuda_timer", side_effect=fake_cuda_timer
        ):
            with sm89_debug.profile_region("moe.test"):
                pass

        self.assertEqual(
            events,
            [
                ("nvtx_enter", "moe.test"),
                ("timer_enter", "moe.test", True),
                ("timer_exit", "moe.test", True),
                ("nvtx_exit", "moe.test"),
            ],
        )


class TestKTEPWrapperProfiling(unittest.TestCase):
    def test_all_cpu_apply_profiles_submit_and_sync(self):
        from sglang.srt.layers.moe import kt_ep_wrapper

        method = object.__new__(kt_ep_wrapper.KTEPWrapperMethod)
        method.tp_rank = 0
        method.wrapper = object()
        method.num_gpu_experts = 0
        method.moe_runner_config = SimpleNamespace(activation="silu")
        method.submit = MagicMock()

        x = torch.zeros(2, 4)
        cpu_out = torch.ones_like(x)
        method.sync = MagicMock(return_value=cpu_out)
        dispatch_output = SimpleNamespace(hidden_states=x, topk_output=object())

        events = []

        @contextmanager
        def fake_profile_region(name, enabled=None):
            events.append(("enter", name, enabled))
            try:
                yield
            finally:
                events.append(("exit", name, enabled))

        with patch.object(
            kt_ep_wrapper,
            "profile_region",
            side_effect=fake_profile_region,
            create=True,
        ):
            output = method.apply(
                layer=SimpleNamespace(), dispatch_output=dispatch_output
            )

        self.assertIs(output.hidden_states, cpu_out)
        method.submit.assert_called_once()
        method.sync.assert_called_once_with(x)
        self.assertIn(("enter", "kt_ep.apply.total", None), events)
        self.assertIn(("enter", "kt_ep.submit", None), events)
        self.assertIn(("enter", "kt_ep.sync", None), events)


class TestSm89SparseMlaForwardProfiling(unittest.TestCase):
    def test_forward_raw_is_profiled_when_profile_env_enabled(self):
        from sglang.srt.model_executor import model_runner

        runner = object.__new__(model_runner.ModelRunner)
        runner.device = "cuda"
        runner.decode_cuda_graph_runner = None
        runner.prefill_cuda_graph_runner = None
        runner.eager_runner = SimpleNamespace(execute=MagicMock(return_value="logits"))
        runner._prepare_eager_forward_batch = MagicMock()
        runner.pp_group = SimpleNamespace(is_last_rank=False)
        runner.attn_backend = object()

        forward_mode = SimpleNamespace(
            is_cpu_graph=lambda: False,
            is_cuda_graph=lambda: False,
            is_decode=lambda: False,
            is_split_prefill=lambda: False,
            is_extend=lambda include_draft_extend_v2=False: True,
        )
        forward_batch = SimpleNamespace(
            forward_mode=forward_mode,
            global_num_tokens_cpu=None,
        )
        events = []

        @contextmanager
        def fake_nvtx_range(name):
            events.append(("nvtx_enter", name))
            try:
                yield
            finally:
                events.append(("nvtx_exit", name))

        @contextmanager
        def fake_cuda_timer(name, enabled):
            events.append(("timer_enter", name, enabled))
            try:
                yield
            finally:
                events.append(("timer_exit", name, enabled))

        with patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_PROFILE": "1"}, clear=True
        ), patch.object(
            model_runner, "has_forward_context", return_value=True
        ), patch.object(
            model_runner, "nvtx_range", side_effect=fake_nvtx_range
        ), patch.object(
            model_runner, "cuda_timer", side_effect=fake_cuda_timer
        ):
            output = model_runner.ModelRunner._forward_raw(
                runner,
                forward_batch=forward_batch,
                pp_proxy_tensors=None,
            )

        self.assertEqual(output.logits_output, "logits")
        self.assertIn(("nvtx_enter", "model_runner.forward_raw.total"), events)
        self.assertIn(("nvtx_exit", "model_runner.forward_raw.total"), events)
        self.assertIn(
            ("timer_enter", "model_runner.forward_raw.total", True), events
        )
        self.assertIn(("timer_exit", "model_runner.forward_raw.total", True), events)


class TestSm89SparseMlaTimingInstrumentation(unittest.TestCase):
    def test_forward_torch_mla_wraps_required_regions(self):
        from sglang.srt.layers.attention import dsa_backend

        events = []

        @contextmanager
        def fake_nvtx_range(name):
            events.append(("nvtx_enter", name))
            try:
                yield
            finally:
                events.append(("nvtx_exit", name))

        @contextmanager
        def fake_cuda_timer(name, enabled):
            events.append(("timer_enter", name, enabled))
            try:
                yield
            finally:
                events.append(("timer_exit", name, enabled))

        q_rope = torch.zeros(2, 1, 4)
        q_nope = torch.zeros(2, 1, 4)
        kv_cache = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8)
        page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)

        with patch.object(
            dsa_backend, "glm_dsa_sm89_profile_enabled", return_value=True
        ), patch.object(dsa_backend, "nvtx_range", side_effect=fake_nvtx_range), patch.object(
            dsa_backend, "cuda_timer", side_effect=fake_cuda_timer
        ):
            out = dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                object.__new__(dsa_backend.DeepseekSparseAttnBackend),
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=4,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
            )

        self.assertEqual(out.shape, q_nope.shape)
        for region in (
            "torch_mla.total",
            "torch_mla.dequant",
            "torch_mla.cat_qkv",
            "torch_mla.loop",
            "torch_mla.gather",
            "torch_mla.sdpa_or_matmul",
        ):
            self.assertIn(("nvtx_enter", region), events)
            self.assertIn(("nvtx_exit", region), events)
            self.assertIn(("timer_enter", region, True), events)
            self.assertIn(("timer_exit", region, True), events)


class TestSm89SparseMlaShapeDump(unittest.TestCase):
    def test_forward_torch_mla_does_not_dump_shapes_when_env_disabled(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend._dumped_glm_sm89_torch_mla_shapes = False

        q_rope = torch.zeros(2, 1, 4)
        q_nope = torch.zeros(2, 1, 4)
        kv_cache = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8)
        page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)

        with patch.dict("os.environ", {}, clear=True), patch(
            "builtins.print"
        ) as mock_print:
            out = dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                backend,
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=4,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
                layer_id=7,
            )

        self.assertEqual(out.shape, q_nope.shape)
        mock_print.assert_not_called()
        self.assertFalse(backend._dumped_glm_sm89_torch_mla_shapes)

    def test_forward_torch_mla_skips_q_token_one_probe_before_dumping(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend._dumped_glm_sm89_torch_mla_shapes = False

        q_rope = torch.zeros(1, 1, 4)
        q_nope = torch.zeros(1, 1, 4)
        kv_cache = torch.zeros(1, 1, 8, dtype=torch.float32)
        page_table = torch.tensor([[0, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([1], dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            shapes_path = os.path.join(tmpdir, "shapes.jsonl")
            with patch.dict(
                "os.environ",
                {
                    "SGLANG_GLM_DSA_SM89_DUMP_SHAPES": "1",
                    "SGLANG_GLM_DSA_SM89_SHAPES_PATH": shapes_path,
                },
                clear=True,
            ), patch("builtins.print") as mock_print:
                out = dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                    backend,
                    q_rope=q_rope,
                    kv_cache=kv_cache,
                    v_head_dim=4,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    sm_scale=1.0,
                    logit_cap=0.0,
                    page_size=1,
                )
                file_exists = os.path.exists(shapes_path)
        self.assertEqual(out.shape, q_nope.shape)
        mock_print.assert_not_called()
        self.assertFalse(file_exists)
        self.assertFalse(backend._dumped_glm_sm89_torch_mla_shapes)

    def test_forward_torch_mla_dumps_raw_and_dequant_shapes_once_when_env_enabled(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend._dumped_glm_sm89_torch_mla_shapes = False

        q_rope = torch.zeros(2, 1, 4)
        q_nope = torch.zeros(2, 1, 4)
        raw_kv_cache = torch.zeros(3, 1, 8, dtype=torch.float8_e4m3fn)
        dequant_kv_cache = torch.arange(24).reshape(3, 1, 8).to(torch.float32)
        page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            shapes_path = os.path.join(tmpdir, "shapes.jsonl")
            with patch.dict(
                "os.environ",
                {
                    "SGLANG_GLM_DSA_SM89_DUMP_SHAPES": "1",
                    "SGLANG_GLM_DSA_SM89_SHAPES_PATH": shapes_path,
                },
                clear=True,
            ), patch.object(
                dsa_backend, "dequantize_k_cache", return_value=dequant_kv_cache
            ), patch("builtins.print") as mock_print:
                first = dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                    backend,
                    q_rope=q_rope,
                    kv_cache=raw_kv_cache,
                    v_head_dim=4,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    sm_scale=1.0,
                    logit_cap=0.0,
                    page_size=1,
                    layer_id=7,
                )
                with open(shapes_path, "r", encoding="utf-8") as fh:
                    file_lines = fh.read().splitlines()
                self.assertEqual(len(file_lines), 1)

                second = dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                    backend,
                    q_rope=q_rope,
                    kv_cache=raw_kv_cache,
                    v_head_dim=4,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    sm_scale=1.0,
                    logit_cap=0.0,
                    page_size=1,
                    layer_id=11,
                )
                with open(shapes_path, "r", encoding="utf-8") as fh:
                    file_lines = fh.read().splitlines()
                self.assertEqual(len(file_lines), 1)

        self.assertEqual(first.shape, q_nope.shape)
        self.assertEqual(second.shape, q_nope.shape)
        mock_print.assert_called_once()
        self.assertEqual(len(mock_print.call_args.args), 1)
        line = mock_print.call_args.args[0]
        dump = json.loads(line)
        file_dump = json.loads(file_lines[0])
        self.assertEqual(dump["event"], "GLM_DSA_SM89_SHAPES")
        self.assertEqual(file_dump, dump)
        self.assertEqual(dump["layer_id"], 7)
        self.assertEqual(dump["q_nope_shape"], [2, 1, 4])
        self.assertEqual(dump["q_rope_shape"], [2, 1, 4])
        self.assertEqual(dump["kv_cache_shape"], [3, 1, 8])
        self.assertEqual(dump["kv_cache_dtype"], "torch.float8_e4m3fn")
        self.assertEqual(dump["dequant_kv_cache_shape"], [3, 1, 8])
        self.assertEqual(dump["dequant_kv_cache_dtype"], "torch.float32")
        self.assertEqual(dump["page_table_shape"], [2, 3])
        self.assertEqual(dump["cache_seqlens_shape"], [2])
        self.assertEqual(dump["cache_seqlens_max"], 2)
        self.assertEqual(dump["v_head_dim"], 4)
        self.assertEqual(dump["page_size"], 1)
        self.assertEqual(dump["q_nope_stride"], list(q_nope.stride()))
        self.assertEqual(dump["q_rope_stride"], list(q_rope.stride()))
        self.assertEqual(dump["kv_cache_stride"], list(raw_kv_cache.stride()))
        self.assertEqual(
            dump["dequant_kv_cache_stride"], list(dequant_kv_cache.stride())
        )
        self.assertEqual(dump["page_table_stride"], list(page_table.stride()))
        self.assertEqual(
            dump["cache_seqlens_stride"], list(cache_seqlens.stride())
        )
        self.assertTrue(backend._dumped_glm_sm89_torch_mla_shapes)

    def test_forward_torch_mla_invalid_min_q_tokens_falls_back_to_two(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend._dumped_glm_sm89_torch_mla_shapes = False

        q_rope = torch.zeros(1, 1, 4)
        q_nope = torch.zeros(1, 1, 4)
        kv_cache = torch.zeros(1, 1, 8, dtype=torch.float32)
        page_table = torch.tensor([[0, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([1], dtype=torch.int32)

        with tempfile.TemporaryDirectory() as tmpdir:
            shapes_path = os.path.join(tmpdir, "shapes.jsonl")
            with patch.dict(
                "os.environ",
                {
                    "SGLANG_GLM_DSA_SM89_DUMP_SHAPES": "1",
                    "SGLANG_GLM_DSA_SM89_DUMP_MIN_Q_TOKENS": "invalid",
                    "SGLANG_GLM_DSA_SM89_SHAPES_PATH": shapes_path,
                },
                clear=True,
            ), patch("builtins.print") as mock_print:
                dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                    backend,
                    q_rope=q_rope,
                    kv_cache=kv_cache,
                    v_head_dim=4,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    sm_scale=1.0,
                    logit_cap=0.0,
                    page_size=1,
                    layer_id=9,
                )
                self.assertFalse(os.path.exists(shapes_path))

                q_rope = torch.zeros(2, 1, 4)
                q_nope = torch.zeros(2, 1, 4)
                raw_kv_cache = torch.zeros(3, 1, 8, dtype=torch.float32)
                page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
                cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)
                dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                    backend,
                    q_rope=q_rope,
                    kv_cache=raw_kv_cache,
                    v_head_dim=4,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    sm_scale=1.0,
                    logit_cap=0.0,
                    page_size=1,
                    layer_id=10,
                )
                with open(shapes_path, "r", encoding="utf-8") as fh:
                    file_lines = fh.read().splitlines()

        self.assertEqual(len(file_lines), 1)
        mock_print.assert_called_once()
        self.assertEqual(len(mock_print.call_args.args), 1)
        dump = json.loads(mock_print.call_args.args[0])
        self.assertEqual(dump["event"], "GLM_DSA_SM89_SHAPES")
        self.assertEqual(dump["q_nope_shape"], [2, 1, 4])
        self.assertTrue(backend._dumped_glm_sm89_torch_mla_shapes)

    def test_forward_fa3_passes_layer_id_to_torch_mla_fallback(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_glm_sm89_dsa_fallback = True
        backend.device_sm_major = 8
        backend._logged_glm_sm89_effective_torch_mla = False
        backend._forward_torch_mla = MagicMock(return_value=torch.zeros(1, 1, 2))

        layer = type("Layer", (), {"layer_id": 13})()
        q_rope = torch.zeros(1, 1, 2)
        q_nope = torch.zeros(1, 1, 2)
        kv_cache = torch.zeros(1, 1, 4)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        cache_seqlens = torch.ones(1, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        backend._forward_fa3(
            q_rope=q_rope,
            kv_cache=kv_cache,
            v_head_dim=2,
            q_nope=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=1,
            sm_scale=1.0,
            logit_cap=0.0,
            page_size=1,
            layer_id=layer.layer_id,
        )

        self.assertEqual(
            backend._forward_torch_mla.call_args.kwargs["layer_id"], layer.layer_id
        )


class TestSm89SparseMlaReference(unittest.TestCase):
    def _make_reference_case(self, total_q, topk, row_lens_kind, dtype):
        torch.manual_seed(total_q * 1009 + topk)
        num_heads = 2
        v_head_dim = 8
        rope_dim = 4
        q_shape = (total_q, num_heads, v_head_dim)
        rope_shape = (total_q, num_heads, rope_dim)
        kv_tokens = max(topk + total_q + 7, 16)

        q_nope = torch.randn(q_shape, dtype=torch.float32).to(torch.bfloat16) * 0.125
        q_rope = torch.randn(rope_shape, dtype=torch.float32).to(torch.bfloat16) * 0.125
        dequant_kv_cache = (
            torch.randn(kv_tokens, 1, v_head_dim + rope_dim, dtype=torch.float32).to(
                torch.bfloat16
            )
            * 0.125
        )

        row_lens = []
        for row in range(total_q):
            if row_lens_kind == "all_full":
                row_len = topk
            elif row_lens_kind == "alternating":
                choices = (topk, max(topk // 2, 1), 1)
                row_len = choices[row % len(choices)]
            elif row_lens_kind == "one_empty":
                row_len = 0 if row == 0 else max(topk - (row % 5), 1)
            else:
                raise AssertionError(f"unknown row_lens_kind={row_lens_kind}")
            row_lens.append(row_len)

        page_table = torch.full((total_q, topk), -1, dtype=torch.int32)
        for row, row_len in enumerate(row_lens):
            if row_len == 0:
                continue
            token_ids = (torch.arange(row_len, dtype=torch.int32) * 3 + row * 7) % (
                kv_tokens
            )
            page_table[row, :row_len] = token_ids

        kv_cache = dequant_kv_cache
        if dtype == torch.float8_e4m3fn:
            kv_cache = torch.empty(kv_tokens, 1, 656, dtype=torch.float8_e4m3fn)

        return {
            "q_nope": q_nope,
            "q_rope": q_rope,
            "kv_cache": kv_cache,
            "dequant_kv_cache": dequant_kv_cache,
            "page_table": page_table,
            "cache_seqlens": torch.tensor(row_lens, dtype=torch.int32),
            "sm_scale": 1.0 / (v_head_dim + rope_dim) ** 0.5,
            "logit_cap": 0.0,
            "v_head_dim": v_head_dim,
        }

    def test_reference_synthetic_shapes_are_finite(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
        )

        for total_q in (1, 17, 128, 257):
            for topk in (64, 256, 2048):
                for row_lens_kind in ("all_full", "alternating", "one_empty"):
                    for dtype in (torch.bfloat16, torch.float8_e4m3fn):
                        for logit_cap in (0.0, 30.0):
                            with self.subTest(
                                total_q=total_q,
                                topk=topk,
                                row_lens_kind=row_lens_kind,
                                dtype=dtype,
                                logit_cap=logit_cap,
                            ):
                                case = self._make_reference_case(
                                    total_q, topk, row_lens_kind, dtype
                                )
                                if dtype == torch.float8_e4m3fn:
                                    patch_target = (
                                        "sglang.srt.layers.attention.dsa."
                                        "dequant_k_cache.dequantize_k_cache"
                                    )
                                    with patch(
                                        patch_target,
                                        return_value=case["dequant_kv_cache"],
                                    ) as mock_dequant:
                                        out = sm89_sparse_mla_prefill_reference(
                                            case["q_nope"],
                                            case["q_rope"],
                                            case["kv_cache"],
                                            case["page_table"],
                                            case["cache_seqlens"],
                                            case["sm_scale"],
                                            logit_cap,
                                            case["v_head_dim"],
                                        )
                                    mock_dequant.assert_called_once_with(
                                        case["kv_cache"]
                                    )
                                else:
                                    out = sm89_sparse_mla_prefill_reference(
                                        case["q_nope"],
                                        case["q_rope"],
                                        case["kv_cache"],
                                        case["page_table"],
                                        case["cache_seqlens"],
                                        case["sm_scale"],
                                        logit_cap,
                                        case["v_head_dim"],
                                    )

                                self.assertEqual(out.shape, case["q_nope"].shape)
                                self.assertFalse(torch.isnan(out).any().item())
                                self.assertFalse(torch.isinf(out).any().item())
                                empty_rows = case["cache_seqlens"] == 0
                                if empty_rows.any():
                                    self.assertTrue(
                                        torch.equal(
                                            out[empty_rows],
                                            torch.zeros_like(out[empty_rows]),
                                        )
                                    )

    def test_reference_matches_torch_mla_fallback(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
        )

        case = self._make_reference_case(17, 64, "one_empty", torch.bfloat16)
        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend._dumped_glm_sm89_torch_mla_shapes = False

        with patch.dict("os.environ", {}, clear=True):
            for logit_cap in (0.0, 30.0):
                with self.subTest(logit_cap=logit_cap):
                    reference = sm89_sparse_mla_prefill_reference(
                        case["q_nope"],
                        case["q_rope"],
                        case["kv_cache"],
                        case["page_table"],
                        case["cache_seqlens"],
                        case["sm_scale"],
                        logit_cap,
                        case["v_head_dim"],
                    )
                    fallback = dsa_backend.DeepseekSparseAttnBackend._forward_torch_mla(
                        backend,
                        q_rope=case["q_rope"],
                        kv_cache=case["kv_cache"],
                        v_head_dim=case["v_head_dim"],
                        q_nope=case["q_nope"],
                        page_table=case["page_table"],
                        cache_seqlens=case["cache_seqlens"],
                        sm_scale=case["sm_scale"],
                        logit_cap=logit_cap,
                        page_size=1,
                    )

                    self.assertTrue(torch.allclose(reference, fallback, atol=0, rtol=0))


class TestSm89SparseMlaBackendDispatch(unittest.TestCase):
    def _make_backend(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.model_arch = "GlmMoeDsaForCausalLM"
        backend.device_capability = (8, 9)
        backend._dumped_glm_sm89_torch_mla_shapes = False
        return backend

    def _make_inputs(self):
        q_nope = torch.zeros(2, 1, 512, dtype=torch.bfloat16)
        q_rope = torch.zeros(2, 1, 64, dtype=torch.bfloat16)
        kv_cache = torch.zeros(4, 1, 656, dtype=torch.float8_e4m3fn)
        page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)
        return q_nope, q_rope, kv_cache, page_table, cache_seqlens

    def test_forward_decode_routes_sm89_cuda_physical_metadata(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = self._make_backend()
        backend.dsa_decode_impl = "sm89_cuda"
        backend.hisparse_coordinator = None
        backend._use_dsa_fuse_topk = MagicMock(return_value=False)

        q_nope, q_rope, kv_cache, logical_page_table, cache_seqlens = (
            self._make_inputs()
        )
        physical_page_table = logical_page_table + 17
        topk_indices = torch.tensor([[1, 0, -1], [0, -1, -1]], dtype=torch.int32)
        backend.forward_metadata = SimpleNamespace(
            page_table_1=logical_page_table,
            dsa_cache_seqlens_int32=cache_seqlens,
        )
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=MagicMock(return_value=kv_cache)
        )
        sentinel = object()
        backend._forward_sm89_cuda_decode = MagicMock(return_value=sentinel)
        layer = SimpleNamespace(
            is_cross_attention=False,
            tp_q_head_num=1,
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
        ) as mock_transform:
            out = dsa_backend.DeepseekSparseAttnBackend.forward_decode(
                backend,
                q=q_nope,
                k=None,
                v=None,
                layer=layer,
                forward_batch=forward_batch,
                save_kv_cache=False,
                q_rope=q_rope,
                topk_indices=topk_indices,
            )

        self.assertIs(out, sentinel)
        mock_transform.assert_called_once_with(
            page_table=logical_page_table,
            topk_indices=topk_indices,
            page_size=1,
        )
        backend._forward_sm89_cuda_decode.assert_called_once()
        call = backend._forward_sm89_cuda_decode.call_args.kwargs
        self.assertTrue(torch.equal(call["q_rope"], q_rope))
        self.assertIs(call["kv_cache"], kv_cache)
        self.assertEqual(call["v_head_dim"], 512)
        self.assertTrue(torch.equal(call["q_nope"], q_nope))
        self.assertIs(call["page_table"], physical_page_table)
        self.assertIs(call["cache_seqlens"], cache_seqlens)
        self.assertEqual(call["sm_scale"], 0.125)
        self.assertEqual(call["logit_cap"], 30.0)
        self.assertEqual(call["page_size"], 1)

    def test_forward_sm89_cuda_calls_kernel_and_logs_once(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = self._make_backend()
        backend._logged_glm_sm89_effective_sm89_cuda = False
        q_nope, q_rope, kv_cache, page_table, cache_seqlens = self._make_inputs()
        sentinel = torch.ones_like(q_nope)

        with patch.object(dsa_backend.logger, "info") as mock_info, patch(
            "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
            "sm89_sparse_mla_decode_cuda",
            return_value=sentinel,
        ) as mock_kernel:
            first = dsa_backend.DeepseekSparseAttnBackend._forward_sm89_cuda_decode(
                backend,
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
            second = dsa_backend.DeepseekSparseAttnBackend._forward_sm89_cuda_decode(
                backend,
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

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(mock_kernel.call_count, 2)
        mock_kernel.assert_called_with(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=0.125,
            logit_cap=30.0,
            v_head_dim=512,
        )
        mock_info.assert_called_once_with(
            "GLM DSA SM89 effective decode backend is sm89_cuda."
        )

    def test_forward_sm89_cuda_rejects_invalid_runtime_contract(self):
        from sglang.srt.layers.attention import dsa_backend

        q_nope, q_rope, kv_cache, page_table, cache_seqlens = self._make_inputs()
        cases = (
            ("model", {"model_arch": "OtherModel"}, {}, "GLM DSA"),
            ("device", {"device_capability": (9, 0)}, {}, "SM89"),
            ("kv", {}, {"kv_cache": kv_cache.to(torch.bfloat16)}, "FP8 E4M3"),
            ("page_size", {}, {"page_size": 64}, "page_size=1"),
            ("v_head_dim", {}, {"v_head_dim": 256}, "GLM dimensions"),
            ("q_nope", {}, {"q_nope": q_nope[..., :256]}, "GLM dimensions"),
            ("q_rope", {}, {"q_rope": q_rope[..., :32]}, "GLM dimensions"),
        )

        with patch(
            "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
            "sm89_sparse_mla_decode_cuda"
        ) as mock_kernel:
            for name, backend_overrides, call_overrides, message in cases:
                with self.subTest(name=name):
                    backend = self._make_backend()
                    for attr, value in backend_overrides.items():
                        setattr(backend, attr, value)
                    kwargs = {
                        "q_rope": q_rope,
                        "kv_cache": kv_cache,
                        "v_head_dim": 512,
                        "q_nope": q_nope,
                        "page_table": page_table,
                        "cache_seqlens": cache_seqlens,
                        "sm_scale": 0.125,
                        "logit_cap": 30.0,
                        "page_size": 1,
                    }
                    kwargs.update(call_overrides)

                    with self.assertRaisesRegex(ValueError, message):
                        dsa_backend.DeepseekSparseAttnBackend._forward_sm89_cuda_decode(
                            backend, **kwargs
                        )

        mock_kernel.assert_not_called()

    def test_forward_sm89_cuda_kernel_error_propagates_without_torch_fallback(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = self._make_backend()
        backend._logged_glm_sm89_effective_sm89_cuda = False
        backend._forward_torch_mla = MagicMock()
        q_nope, q_rope, kv_cache, page_table, cache_seqlens = self._make_inputs()
        error = RuntimeError("decode kernel failed")

        with patch(
            "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
            "sm89_sparse_mla_decode_cuda",
            side_effect=error,
        ):
            with self.assertRaises(RuntimeError) as raised:
                dsa_backend.DeepseekSparseAttnBackend._forward_sm89_cuda_decode(
                    backend,
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

        self.assertIs(raised.exception, error)
        backend._forward_torch_mla.assert_not_called()

    def test_forward_sm89_triton_calls_kernel(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = self._make_backend()
        q_nope, q_rope, kv_cache, page_table, cache_seqlens = self._make_inputs()
        sentinel = torch.zeros_like(q_nope)

        with patch(
            "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
            "sm89_sparse_mla_prefill_triton",
            return_value=sentinel,
        ) as mock_kernel:
            out = dsa_backend.DeepseekSparseAttnBackend._forward_sm89_triton(
                backend,
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=512,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
                layer_id=3,
            )

        self.assertIs(out, sentinel)
        mock_kernel.assert_called_once_with(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=1.0,
            logit_cap=0.0,
            v_head_dim=512,
        )

    def test_forward_sm89_triton_kernel_error_requires_explicit_fallback(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = self._make_backend()
        q_nope, q_rope, kv_cache, page_table, cache_seqlens = self._make_inputs()

        with patch.dict("os.environ", {}, clear=True), patch(
            "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
            "sm89_sparse_mla_prefill_triton",
            side_effect=RuntimeError("kernel failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                dsa_backend.DeepseekSparseAttnBackend._forward_sm89_triton(
                    backend,
                    q_rope=q_rope,
                    kv_cache=kv_cache,
                    v_head_dim=512,
                    q_nope=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    sm_scale=1.0,
                    logit_cap=0.0,
                    page_size=1,
                    layer_id=3,
                )

    def test_forward_sm89_triton_allows_env_gated_torch_fallback(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = self._make_backend()
        q_nope, q_rope, kv_cache, page_table, cache_seqlens = self._make_inputs()
        sentinel = torch.ones_like(q_nope)
        backend._forward_torch_mla = MagicMock(return_value=sentinel)

        with patch.dict(
            "os.environ",
            {"SGLANG_GLM_DSA_SM89_ALLOW_TORCH_FALLBACK": "1"},
            clear=True,
        ), patch(
            "sglang.srt.layers.attention.dsa.sm89_sparse_mla."
            "sm89_sparse_mla_prefill_triton",
            side_effect=RuntimeError("kernel failed"),
        ):
            out = dsa_backend.DeepseekSparseAttnBackend._forward_sm89_triton(
                backend,
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=512,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
                layer_id=3,
            )

        self.assertIs(out, sentinel)
        backend._forward_torch_mla.assert_called_once()


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestSm89SparseMlaDecodeCuda(unittest.TestCase):
    _SM_SCALE = 1.0 / (512 + 64) ** 0.5

    def _make_exact_splitk_case(
        self,
        pool_size=4096,
        seed=_SPLITK_SCALE_SEED,
        page_table=None,
        cache_seqlen=2048,
        q_layout="noncontiguous",
    ):
        device = torch.device("cuda")
        generator = torch.Generator(device=device).manual_seed(seed)
        q_nope_storage = torch.randn(
            1, 32, 1024, device=device, dtype=torch.float32, generator=generator
        ).to(torch.bfloat16)
        q_rope_storage = torch.randn(
            1, 32, 128, device=device, dtype=torch.float32, generator=generator
        ).to(torch.bfloat16)
        if q_layout == "noncontiguous":
            q_nope = q_nope_storage[..., ::2]
            q_rope = q_rope_storage[..., ::2]
        elif q_layout == "contiguous":
            q_nope = q_nope_storage[..., ::2].contiguous()
            q_rope = q_rope_storage[..., ::2].contiguous()
        else:
            raise ValueError(f"unsupported Q layout: {q_layout}")

        packed_bytes = torch.empty(
            pool_size, 1, 656, device=device, dtype=torch.uint8
        )
        kv_cache = packed_bytes.view(torch.float8_e4m3fn)
        dim = torch.arange(512, device=device, dtype=torch.int64).unsqueeze(0)
        for start in range(0, pool_size, 8192):
            end = min(start + 8192, pool_size)
            token = torch.arange(
                start, end, device=device, dtype=torch.int64
            ).unsqueeze(1)
            quantized = (((token * 13 + dim * 7) % 29) - 14).float() / 8
            kv_cache[start:end, 0, :512].copy_(
                quantized.to(torch.float8_e4m3fn)
            )
            rope = torch.randn(
                end - start,
                64,
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
            rope = torch.where(rope == 0, torch.full_like(rope, 0.125), rope)
            packed_bytes[start:end, 0, 528:].view(torch.bfloat16).copy_(
                rope.to(torch.bfloat16)
            )

        token = torch.arange(pool_size, device=device, dtype=torch.int64).unsqueeze(1)
        group = torch.arange(4, device=device, dtype=torch.int64).unsqueeze(0)
        scales = 0.03125 * (1 + ((token * 5 + group * 3) % 7)).float()
        packed_bytes[:, 0, 512:528].view(torch.float32).copy_(scales)

        if page_table is None:
            selected = (
                torch.arange(2048, device=device, dtype=torch.int64)
                * (pool_size - 1)
                // 2047
            ).to(torch.int32)
            page_table = selected.unsqueeze(0)
        else:
            page_table = page_table.to(device=device, dtype=torch.int32)
            if page_table.ndim == 1:
                page_table = page_table.unsqueeze(0)
        return {
            "q_nope": q_nope,
            "q_rope": q_rope,
            "kv_cache": kv_cache,
            "page_table": page_table,
            "cache_seqlens": torch.tensor(
                [cache_seqlen], device=device, dtype=torch.int32
            ),
        }

    def _assert_exact_splitk_scales(self, case):
        selected = case["page_table"][0]
        valid = selected[(selected >= 0) & (selected < case["kv_cache"].shape[0])]
        quantile_positions = torch.linspace(
            0, valid.numel() - 1, 9, device=valid.device
        ).round().long()
        tokens = valid[quantile_positions].long()
        packed_bytes = case["kv_cache"].view(torch.uint8)
        actual = packed_bytes[tokens, 0, 512:528].contiguous().view(torch.float32)
        group = torch.arange(4, device=tokens.device, dtype=torch.int64).unsqueeze(0)
        expected = 0.03125 * (1 + ((tokens.unsqueeze(1) * 5 + group * 3) % 7)).float()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        for group_id in range(4):
            self.assertGreaterEqual(torch.unique(actual[:, group_id]).numel(), 2)
        self.assertGreaterEqual(actual.min().item(), 0.03125)
        self.assertLessEqual(actual.max().item(), 0.21875)

    def _dequant_selected_splitk(self, kv_cache, token_ids):
        packed = kv_cache.view(torch.uint8)[token_ids.long(), 0]
        quantized = packed[:, :512].contiguous().view(torch.float8_e4m3fn)
        scales = packed[:, 512:528].contiguous().view(torch.float32)
        nope = (quantized.float() * scales.repeat_interleave(128, dim=1)).to(
            torch.bfloat16
        )
        rope = packed[:, 528:].contiguous().view(torch.bfloat16)
        return nope, rope

    def _splitk_reference(self, case):
        row_len = max(0, min(int(case["cache_seqlens"][0].item()), 2048))
        selected = case["page_table"][0, :row_len]
        valid = (selected >= 0) & (selected < case["kv_cache"].shape[0])
        if not valid.any():
            return torch.zeros_like(case["q_nope"])
        nope, rope = self._dequant_selected_splitk(
            case["kv_cache"], selected[valid]
        )
        q = torch.cat([case["q_nope"][0], case["q_rope"][0]], dim=-1)
        k = torch.cat([nope, rope], dim=-1)
        logits = (q.float() @ k.float().transpose(0, 1)) * self._SM_SCALE
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        probabilities = torch.exp(logits - lse).to(torch.bfloat16).float()
        return (probabilities @ nope.float()).to(torch.bfloat16).unsqueeze(0)

    def _splitk_partial_reference(self, case, split_tokens):
        splits = 2048 // split_tokens
        partial_o = torch.zeros(
            splits, 1, 32, 512, device="cuda", dtype=torch.float32
        )
        partial_lse = torch.full(
            (splits, 1, 32), float("-inf"), device="cuda", dtype=torch.float32
        )
        split_valid = torch.zeros(splits, device="cuda", dtype=torch.int32)
        row_len = max(0, min(int(case["cache_seqlens"][0].item()), 2048))
        q = torch.cat([case["q_nope"][0], case["q_rope"][0]], dim=-1)
        for split in range(splits):
            start = split * split_tokens
            stop = start + split_tokens
            selected = case["page_table"][0, start:stop]
            positions = torch.arange(start, stop, device="cuda")
            valid = (
                (positions < row_len)
                & (selected >= 0)
                & (selected < case["kv_cache"].shape[0])
            )
            if not valid.any():
                continue
            split_valid[split] = 1
            nope, rope = self._dequant_selected_splitk(
                case["kv_cache"], selected[valid]
            )
            k = torch.cat([nope, rope], dim=-1)
            logits = (q.float() @ k.float().transpose(0, 1)) * self._SM_SCALE
            lse = torch.logsumexp(logits, dim=-1)
            probability = torch.exp(logits - lse.unsqueeze(1)).to(torch.bfloat16)
            partial_lse[split, 0] = lse
            partial_o[split, 0] = probability.float() @ nope.float()
        return partial_o, partial_lse, split_valid

    def _assert_splitk_numeric_gates(self, actual, expected):
        self.assertTrue(torch.isfinite(actual).all())
        if not torch.count_nonzero(expected):
            self.assertTrue(torch.equal(actual, torch.zeros_like(actual)))
            return
        diff = (actual.float() - expected.float()).abs()
        self.assertLessEqual(diff.max().item(), 5e-2)
        self.assertLessEqual(diff.mean().item(), 5e-3)
        cosine = torch.nn.functional.cosine_similarity(
            actual.float().flatten(), expected.float().flatten(), dim=0
        ).item()
        self.assertGreaterEqual(cosine, 0.995)

    def _splitk_extension_identity(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        ext = sm89_sparse_mla_cuda._load_sm89_sparse_mla_cuda_ext()
        extension_path = Path(ext.__file__).resolve()
        source_path = (
            Path(sm89_sparse_mla_cuda.__file__).resolve().parent
            / "csrc"
            / "sm89_sparse_mla_cuda.cu"
        )
        identity = {
            "extension_path": str(extension_path),
            "extension_sha256": hashlib.sha256(extension_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        }
        print("SPLITK_EXTENSION_IDENTITY=" + json.dumps(identity, sort_keys=True))
        return ext, identity

    def _splitk_metrics(self, actual, expected):
        diff = (actual.float() - expected.float()).abs()
        if torch.count_nonzero(expected):
            cosine = torch.nn.functional.cosine_similarity(
                actual.float().flatten(), expected.float().flatten(), dim=0
            ).item()
        else:
            cosine = 1.0 if not torch.count_nonzero(actual) else 0.0
        return {
            "max_abs": diff.max().item(),
            "mean_abs": diff.mean().item(),
            "cosine": cosine,
        }

    def _three_stage_splitk_reference(self, case):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_prefill_cuda,
        )

        return sm89_sparse_mla_prefill_cuda(
            **case,
            sm_scale=self._SM_SCALE,
            logit_cap=0.0,
            v_head_dim=512,
            block_n=32,
            cuda_impl="tensorcore",
        )

    def test_splitk_direct_entry_matches_reference(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_decode_cuda_splitk,
            sm89_sparse_mla_prefill_cuda,
        )

        case = self._make_exact_splitk_case(pool_size=4096)
        self._assert_exact_splitk_scales(case)
        expected = self._splitk_reference(case)
        three_stage = sm89_sparse_mla_prefill_cuda(
            **case,
            sm_scale=self._SM_SCALE,
            logit_cap=0.0,
            v_head_dim=512,
            block_n=32,
            cuda_impl="tensorcore",
        )
        self._assert_splitk_numeric_gates(three_stage, expected)
        for split_tokens in (32, 64):
            with self.subTest(split_tokens=split_tokens):
                actual = sm89_sparse_mla_decode_cuda_splitk(
                    **case,
                    sm_scale=self._SM_SCALE,
                    logit_cap=0.0,
                    v_head_dim=512,
                    split_tokens=split_tokens,
                )
                self._assert_splitk_numeric_gates(actual, expected)
                self._assert_splitk_numeric_gates(actual, three_stage)

    def test_splitk_debug_partial_state_handles_empty_and_mixed_splits(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        for split_tokens in (32, 64):
            pages = torch.full((2048,), -1, dtype=torch.int32)
            pages[:7] = torch.arange(7, dtype=torch.int32)
            second_start = 2 * split_tokens + 3
            pages[second_start : second_start + 9] = torch.arange(
                100, 109, dtype=torch.int32
            )
            case = self._make_exact_splitk_case(page_table=pages)
            expected_o, expected_lse, expected_valid = (
                self._splitk_partial_reference(case, split_tokens)
            )
            ext = sm89_sparse_mla_cuda._load_sm89_sparse_mla_cuda_ext()
            out, partial_o, partial_lse, split_valid = (
                ext.sm89_sparse_mla_decode_cuda_splitk_debug(
                    case["q_nope"],
                    case["q_rope"],
                    case["kv_cache"],
                    case["page_table"],
                    case["cache_seqlens"],
                    self._SM_SCALE,
                    0.0,
                    512,
                    split_tokens,
                )
            )
            self._assert_splitk_numeric_gates(out, self._splitk_reference(case))
            self.assertEqual(partial_o.shape, expected_o.shape)
            self.assertEqual(partial_lse.shape, expected_lse.shape)
            self.assertEqual(split_valid.shape, (2048 // split_tokens, 1, 2))
            self.assertTrue(torch.equal(split_valid[:, 0, 0], split_valid[:, 0, 1]))
            self.assertTrue(torch.equal(split_valid[:, 0, 0], expected_valid))
            finite = torch.isfinite(expected_lse)
            torch.testing.assert_close(
                partial_lse[finite], expected_lse[finite], atol=5e-3, rtol=5e-3
            )
            self.assertTrue(torch.isneginf(partial_lse[~finite]).all())
            diff = (partial_o - expected_o).abs()
            self.assertLessEqual(diff.max().item(), 5e-2)
            self.assertLessEqual(diff.mean().item(), 5e-3)
            empty = ~expected_valid.bool()
            self.assertTrue(
                torch.equal(partial_o[empty], torch.zeros_like(partial_o[empty]))
            )

    def test_splitk_all_empty_is_finite_and_bit_exact_zero(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_decode_cuda_splitk,
        )

        pages = torch.full((2048,), -1, dtype=torch.int32)
        case = self._make_exact_splitk_case(
            page_table=pages, cache_seqlen=2048
        )
        for split_tokens in (32, 64):
            out = sm89_sparse_mla_decode_cuda_splitk(
                **case,
                sm_scale=self._SM_SCALE,
                logit_cap=0.0,
                v_head_dim=512,
                split_tokens=split_tokens,
            )
            self.assertEqual(out.shape, (1, 32, 512))
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertTrue(torch.isfinite(out).all())
            self.assertTrue(torch.equal(out, torch.zeros_like(out)))

    def test_splitk_source_orders_token_barrier_before_qk(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        source_path = (
            Path(sm89_sparse_mla_cuda.__file__).resolve().parent
            / "csrc"
            / "sm89_sparse_mla_cuda.cu"
        )
        source = source_path.read_text(encoding="utf-8")
        kernel = _cuda_kernel_source_body(
            source, "sm89_sparse_mla_splitk_partial_kernel"
        )
        publication = kernel.index("token_ids[tid] = token")
        barrier = kernel.index("__syncthreads_or(token_valid)")
        first_token_read = kernel.index("token_ids[", publication + 1)
        first_kv_load = min(
            index
            for marker in ("load_fp8_scaled(", "load_rope_bf16(")
            if (index := kernel.find(marker)) >= 0
        )
        first_wmma = kernel.index("wmma::load_matrix_sync")
        self.assertLess(publication, barrier)
        self.assertLess(barrier, first_token_read)
        self.assertLess(barrier, first_kv_load)
        self.assertLess(barrier, first_wmma)

    def test_splitk_source_enforces_wmma_pointer_alignment(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        source_path = (
            Path(sm89_sparse_mla_cuda.__file__).resolve().parent
            / "csrc"
            / "sm89_sparse_mla_cuda.cu"
        )
        source = source_path.read_text(encoding="utf-8")
        kernel = _cuda_kernel_source_body(
            source, "sm89_sparse_mla_splitk_partial_kernel"
        )
        for operand in (
            "q_tile",
            "k_tiles",
            "score_tile",
            "prob_tile",
            "value_tiles",
        ):
            with self.subTest(operand=operand):
                self.assertRegex(
                    kernel,
                    rf"__shared__\s+__align__\(32\)[^;]*\b{operand}\[",
                )
        workspace_validation = _cuda_kernel_source_body(
            source, "validate_sm89_sparse_mla_splitk_workspace"
        )
        full_launcher = _cuda_kernel_source_body(
            source, "Sm89SparseMlaSplitKResult launch_sm89_sparse_mla_splitk("
        )
        allocation = full_launcher.index("auto partial_o = torch::empty(")
        alignment_check = workspace_validation.index(
            "reinterpret_cast<uintptr_t>(partial_o.data_ptr()) % 32 == 0"
        )
        validation_call = full_launcher.index(
            "validate_sm89_sparse_mla_splitk_workspace(", allocation
        )
        launch_call = full_launcher.index(
            "launch_sm89_sparse_mla_splitk_partial(", validation_call
        )
        self.assertGreaterEqual(alignment_check, 0)
        self.assertLess(allocation, validation_call)
        self.assertLess(validation_call, launch_call)

    def test_splitk_source_guards_nonfinite_lse_without_barrier(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        source_path = (
            Path(sm89_sparse_mla_cuda.__file__).resolve().parent
            / "csrc"
            / "sm89_sparse_mla_cuda.cu"
        )
        source = source_path.read_text(encoding="utf-8")
        kernel = _cuda_kernel_source_body(
            source, "sm89_sparse_mla_splitk_partial_kernel"
        )
        finite_lse_block = _cuda_kernel_source_body(
            kernel, "if (isfinite(row_lse))"
        )
        self.assertIn("score_tile[probability_head][token_local] - row_lse", finite_lse_block)
        self.assertNotIn("__syncthreads", finite_lse_block)

    def test_splitk_direct_wrapper_rejects_before_extension_load(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        case = self._make_exact_splitk_case()
        pool_size = case["kv_cache"].shape[0]
        bad_kv_inner_stride = torch.empty(
            pool_size, 1, 1312, device="cuda", dtype=torch.float8_e4m3fn
        ).as_strided((pool_size, 1, 656), (1312, 1312, 2))
        bad_kv_row_stride = torch.empty(
            pool_size, 1, 658, device="cuda", dtype=torch.float8_e4m3fn
        )[..., :656]
        bad_kv_base_alignment = torch.empty(
            pool_size, 1, 660, device="cuda", dtype=torch.float8_e4m3fn
        )[..., 1:657]
        zero_stride_length = torch.zeros(
            1, device="cuda", dtype=torch.int32
        ).expand(2)[:1]
        invalid_cases = {
            "q_dtype": (case | {"q_nope": case["q_nope"].float()}, {}),
            "q_shape": (case | {"q_nope": case["q_nope"][:, :, :511]}, {}),
            "q_stride": (
                case
                | {
                    "q_nope": case["q_nope"].as_strided(
                        case["q_nope"].shape,
                        (0, case["q_nope"].stride(1), case["q_nope"].stride(2)),
                    )
                },
                {},
            ),
            "rope_dtype": (case | {"q_rope": case["q_rope"].float()}, {}),
            "rope_stride": (
                case
                | {
                    "q_rope": case["q_rope"].as_strided(
                        case["q_rope"].shape,
                        (0, case["q_rope"].stride(1), case["q_rope"].stride(2)),
                    )
                },
                {},
            ),
            "page_dtype": (case | {"page_table": case["page_table"].long()}, {}),
            "page_shape": (case | {"page_table": case["page_table"][:, :1024]}, {}),
            "page_stride": (
                case
                | {
                    "page_table": case["page_table"].as_strided(
                        case["page_table"].shape, (0, case["page_table"].stride(1))
                    )
                },
                {},
            ),
            "length_dtype": (
                case | {"cache_seqlens": case["cache_seqlens"].long()},
                {},
            ),
            "length_stride": (case | {"cache_seqlens": zero_stride_length}, {}),
            "kv_dtype": (case | {"kv_cache": case["kv_cache"].bfloat16()}, {}),
            "kv_empty": (
                case
                | {
                    "kv_cache": torch.empty(
                        0, 1, 656, device="cuda", dtype=torch.float8_e4m3fn
                    )
                },
                {},
            ),
            "kv_inner_stride": (case | {"kv_cache": bad_kv_inner_stride}, {}),
            "kv_row_stride": (case | {"kv_cache": bad_kv_row_stride}, {}),
            "kv_base_alignment": (
                case | {"kv_cache": bad_kv_base_alignment},
                {},
            ),
            "device": (case | {"q_rope": case["q_rope"].cpu()}, {}),
            "sm_scale_nan": (case, {"sm_scale": float("nan")}),
            "sm_scale_inf": (case, {"sm_scale": float("inf")}),
            "cap_positive": (case, {"logit_cap": 1.0}),
            "cap_nan": (case, {"logit_cap": float("nan")}),
            "value_dim": (case, {"v_head_dim": 256}),
            "split_width": (case, {"split_tokens": 16}),
        }
        ext = SimpleNamespace(sm89_sparse_mla_decode_cuda_splitk=MagicMock())
        with patch.object(
            sm89_sparse_mla_cuda,
            "_load_sm89_sparse_mla_cuda_ext",
            return_value=ext,
        ) as load_ext:
            for name, (invalid_case, overrides) in invalid_cases.items():
                kwargs = {
                    "sm_scale": self._SM_SCALE,
                    "logit_cap": 0.0,
                    "v_head_dim": 512,
                    "split_tokens": 32,
                }
                kwargs.update(overrides)
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "split-K"
                ):
                    sm89_sparse_mla_cuda.sm89_sparse_mla_decode_cuda_splitk(
                        **invalid_case, **kwargs
                    )
        load_ext.assert_not_called()
        ext.sm89_sparse_mla_decode_cuda_splitk.assert_not_called()

    def test_splitk_wrapper_rejects_sm_scale_fp32_overflow_before_load(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        case = self._make_exact_splitk_case()
        ext = SimpleNamespace(sm89_sparse_mla_decode_cuda_splitk=MagicMock())
        with patch.object(
            sm89_sparse_mla_cuda,
            "_load_sm89_sparse_mla_cuda_ext",
            return_value=ext,
        ) as load_ext, self.assertRaisesRegex(ValueError, "FP32 sm_scale"):
            sm89_sparse_mla_cuda.sm89_sparse_mla_decode_cuda_splitk(
                **case,
                sm_scale=1e300,
                logit_cap=0.0,
                v_head_dim=512,
                split_tokens=32,
            )
        load_ext.assert_not_called()
        ext.sm89_sparse_mla_decode_cuda_splitk.assert_not_called()

    def test_splitk_direct_extension_rejects_sm_scale_fp32_overflow(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla_cuda

        ext, _ = self._splitk_extension_identity()
        case = self._make_exact_splitk_case()
        with self.assertRaisesRegex(RuntimeError, "FP32 sm_scale"):
            ext.sm89_sparse_mla_decode_cuda_splitk(
                case["q_nope"],
                case["q_rope"],
                case["kv_cache"],
                case["page_table"],
                case["cache_seqlens"],
                1e300,
                0.0,
                512,
                32,
            )

        source_path = (
            Path(sm89_sparse_mla_cuda.__file__).resolve().parent
            / "csrc"
            / "sm89_sparse_mla_cuda.cu"
        )
        source = source_path.read_text(encoding="utf-8")
        launcher = _cuda_kernel_source_body(
            source, "Sm89SparseMlaSplitKResult launch_sm89_sparse_mla_splitk("
        )
        validation = launcher.index(
            "const float effective_sm_scale = validate_sm89_sparse_mla_splitk("
        )
        allocation = launcher.index("auto partial_o = torch::empty(")
        launch = launcher.index("launch_sm89_sparse_mla_splitk_partial(", allocation)
        self.assertLess(validation, allocation)
        self.assertLess(validation, launch)

    def test_splitk_direct_extension_repeats_layout_checks(self):
        ext, _ = self._splitk_extension_identity()
        case = self._make_exact_splitk_case()
        pool_size = case["kv_cache"].shape[0]
        invalid = (
            (
                "q_stride",
                case
                | {
                    "q_nope": case["q_nope"].as_strided(
                        case["q_nope"].shape,
                        (0, case["q_nope"].stride(1), case["q_nope"].stride(2)),
                    )
                },
            ),
            (
                "q_rope_stride",
                case
                | {
                    "q_rope": case["q_rope"].as_strided(
                        case["q_rope"].shape,
                        (0, case["q_rope"].stride(1), case["q_rope"].stride(2)),
                    )
                },
            ),
            (
                "page_stride",
                case
                | {
                    "page_table": case["page_table"].as_strided(
                        case["page_table"].shape, (0, case["page_table"].stride(1))
                    )
                },
            ),
            (
                "length_stride",
                case
                | {
                    "cache_seqlens": torch.zeros(
                        1, device="cuda", dtype=torch.int32
                    ).expand(2)[:1]
                },
            ),
            (
                "kv_empty",
                case
                | {
                    "kv_cache": torch.empty(
                        0, 1, 656, device="cuda", dtype=torch.float8_e4m3fn
                    )
                },
            ),
            (
                "kv_inner_stride",
                case
                | {
                    "kv_cache": torch.empty(
                        pool_size,
                        1,
                        1312,
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    ).as_strided((pool_size, 1, 656), (1312, 1312, 2))
                },
            ),
            (
                "kv_row_stride",
                case
                | {
                    "kv_cache": torch.empty(
                        pool_size,
                        1,
                        658,
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    )[..., :656]
                },
            ),
            (
                "kv_base_alignment",
                case
                | {
                    "kv_cache": torch.empty(
                        pool_size,
                        1,
                        660,
                        device="cuda",
                        dtype=torch.float8_e4m3fn,
                    )[..., 1:657]
                },
            ),
        )
        for name, invalid_case in invalid:
            with self.subTest(name=name), self.assertRaisesRegex(
                RuntimeError, "split-K"
            ):
                ext.sm89_sparse_mla_decode_cuda_splitk(
                    invalid_case["q_nope"],
                    invalid_case["q_rope"],
                    invalid_case["kv_cache"],
                    invalid_case["page_table"],
                    invalid_case["cache_seqlens"],
                    self._SM_SCALE,
                    0.0,
                    512,
                    32,
                )

    def test_splitk_complete_physical_pool_matrix(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_decode_cuda_splitk,
        )

        self._splitk_extension_identity()
        for split_tokens in (32, 64):
            for pool_size in (32896, 131200, 262272):
                with self.subTest(
                    split_tokens=split_tokens, pool_size=pool_size
                ):
                    case = self._make_exact_splitk_case(
                        pool_size=pool_size, seed=_SPLITK_SCALE_SEED
                    )
                    self._assert_exact_splitk_scales(case)
                    expected = self._splitk_reference(case)
                    three_stage = self._three_stage_splitk_reference(case)
                    actual = sm89_sparse_mla_decode_cuda_splitk(
                        **case,
                        sm_scale=self._SM_SCALE,
                        logit_cap=0.0,
                        v_head_dim=512,
                        split_tokens=split_tokens,
                    )
                    self._assert_splitk_numeric_gates(actual, expected)
                    self._assert_splitk_numeric_gates(actual, three_stage)
                    print(
                        "SPLITK_CORRECTNESS="
                        + json.dumps(
                            {
                                "split_tokens": split_tokens,
                                "physical_pool_tokens": pool_size,
                                "seed": _SPLITK_SCALE_SEED,
                                "scale_formula": _SPLITK_SCALE_FORMULA_VERSION,
                                "scale_min": 0.03125,
                                "scale_max": 0.21875,
                                "reference": self._splitk_metrics(actual, expected),
                                "three_stage": self._splitk_metrics(
                                    actual, three_stage
                                ),
                            },
                            sort_keys=True,
                        )
                    )
                    del actual, three_stage, expected, case
                    torch.cuda.empty_cache()

    def test_splitk_length_page_and_stride_matrix(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_decode_cuda_splitk,
        )

        base = self._make_exact_splitk_case(pool_size=4096, q_layout="contiguous")
        self.assertEqual(base["q_nope"].stride(), (16384, 512, 1))
        self.assertEqual(base["q_rope"].stride(), (2048, 64, 1))
        for split_tokens in (32, 64):
            for row_len in (0, 1, 31, 32, 33, 63, 64, 65, 2047, 2048):
                case = base | {
                    "cache_seqlens": torch.tensor(
                        [row_len], device="cuda", dtype=torch.int32
                    )
                }
                with self.subTest(split_tokens=split_tokens, row_len=row_len):
                    expected = self._splitk_reference(case)
                    three_stage = self._three_stage_splitk_reference(case)
                    actual = sm89_sparse_mla_decode_cuda_splitk(
                        **case,
                        sm_scale=self._SM_SCALE,
                        logit_cap=0.0,
                        v_head_dim=512,
                        split_tokens=split_tokens,
                    )
                    self._assert_splitk_numeric_gates(actual, expected)
                    self._assert_splitk_numeric_gates(actual, three_stage)

            pages = base["page_table"].clone()
            pages[0, 1] = -7
            pages[0, 2] = base["kv_cache"].shape[0] + 17
            pages[0, 100] = base["kv_cache"].shape[0] + 23
            pages[0, 2047] = base["kv_cache"].shape[0] + 29
            semantic_cases = {
                "contiguous_q_rope": base,
                "mixed": base
                | {
                    "page_table": pages,
                    "cache_seqlens": torch.tensor(
                        [65], device="cuda", dtype=torch.int32
                    ),
                },
                "oversized_length": base
                | {
                    "page_table": pages,
                    "cache_seqlens": torch.tensor(
                        [4096], device="cuda", dtype=torch.int32
                    ),
                },
                "negative_length": base
                | {
                    "page_table": pages,
                    "cache_seqlens": torch.tensor(
                        [-9], device="cuda", dtype=torch.int32
                    ),
                },
            }
            page_storage = torch.empty(
                1, 4096, device="cuda", dtype=torch.int32
            )
            page_storage[:, ::2].copy_(base["page_table"])
            semantic_cases["noncontiguous_page"] = base | {
                "page_table": page_storage[:, ::2]
            }
            q_nope_storage = torch.empty(
                1, 32, 1024, device="cuda", dtype=torch.bfloat16
            )
            q_rope_storage = torch.empty(
                1, 32, 128, device="cuda", dtype=torch.bfloat16
            )
            q_nope_storage[..., ::2].copy_(base["q_nope"])
            q_rope_storage[..., ::2].copy_(base["q_rope"])
            noncontiguous_q_nope = q_nope_storage[..., ::2]
            noncontiguous_q_rope = q_rope_storage[..., ::2]
            self.assertEqual(noncontiguous_q_nope.stride(), (32768, 1024, 2))
            self.assertEqual(noncontiguous_q_rope.stride(), (4096, 128, 2))
            semantic_cases["noncontiguous_q_rope"] = base | {
                "q_nope": noncontiguous_q_nope,
                "q_rope": noncontiguous_q_rope,
            }
            for name, case in semantic_cases.items():
                with self.subTest(split_tokens=split_tokens, semantics=name):
                    expected = self._splitk_reference(case)
                    three_stage = self._three_stage_splitk_reference(case)
                    actual = sm89_sparse_mla_decode_cuda_splitk(
                        **case,
                        sm_scale=self._SM_SCALE,
                        logit_cap=0.0,
                        v_head_dim=512,
                        split_tokens=split_tokens,
                    )
                    self._assert_splitk_numeric_gates(actual, expected)
                    self._assert_splitk_numeric_gates(actual, three_stage)

    def test_splitk_back_to_back_nondefault_stream(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_decode_cuda_splitk,
        )

        for split_tokens in (32, 64):
            case_a = self._make_exact_splitk_case(seed=_SPLITK_SCALE_SEED + 1)
            case_b = self._make_exact_splitk_case(seed=_SPLITK_SCALE_SEED + 2)
            case_b["page_table"] = torch.flip(case_b["page_table"], dims=(1,))
            expected_a = self._splitk_reference(case_a)
            expected_b = self._splitk_reference(case_b)
            ready = torch.cuda.Event()
            ready.record()
            done = torch.cuda.Event()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                actual_a = sm89_sparse_mla_decode_cuda_splitk(
                    **case_a,
                    sm_scale=self._SM_SCALE,
                    logit_cap=0.0,
                    v_head_dim=512,
                    split_tokens=split_tokens,
                )
                actual_b = sm89_sparse_mla_decode_cuda_splitk(
                    **case_b,
                    sm_scale=self._SM_SCALE,
                    logit_cap=0.0,
                    v_head_dim=512,
                    split_tokens=split_tokens,
                )
                done.record(stream)
            torch.cuda.current_stream().wait_event(done)
            torch.cuda.synchronize()
            self._assert_splitk_numeric_gates(actual_a, expected_a)
            self._assert_splitk_numeric_gates(actual_b, expected_b)
            self.assertNotEqual(
                actual_a.untyped_storage().data_ptr(),
                actual_b.untyped_storage().data_ptr(),
            )

    def test_splitk_invalid_pages_memcheck(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
            sm89_sparse_mla_decode_cuda_splitk,
        )

        self._splitk_extension_identity()
        pool_size = 4096
        pages = torch.arange(2048, dtype=torch.int32)
        pages[1] = -11
        pages[2] = pool_size + 13
        pages[3] = pool_size + 17
        pages[127] = pool_size + 19
        case = self._make_exact_splitk_case(
            pool_size=pool_size, page_table=pages, cache_seqlen=64
        )
        for split_tokens in (32, 64):
            out = sm89_sparse_mla_decode_cuda_splitk(
                **case,
                sm_scale=self._SM_SCALE,
                logit_cap=0.0,
                v_head_dim=512,
                split_tokens=split_tokens,
            )
            self.assertEqual(out.shape, (1, 32, 512))
            self.assertEqual(out.dtype, torch.bfloat16)
        torch.cuda.synchronize()
        del out, case, pages
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _make_decode_case(self, batch, row_lens, seed):
        self.assertEqual(len(row_lens), batch)
        torch.manual_seed(seed)
        device = torch.device("cuda")
        topk = 2048
        pool_size = 4096

        q_nope = torch.randn(
            batch, 32, 1024, device=device, dtype=torch.float32
        ).to(torch.bfloat16)[..., ::2]
        q_rope = torch.randn(
            batch, 32, 128, device=device, dtype=torch.float32
        ).to(torch.bfloat16)[..., ::2]
        dequant_kv_cache = torch.randn(
            pool_size, 1, 576, device=device, dtype=torch.float32
        ).to(torch.bfloat16)
        page_table = torch.full(
            (batch, topk), -1, device=device, dtype=torch.int32
        )
        shuffled_slots = torch.randperm(
            pool_size, device=device, dtype=torch.int32
        ).repeat(2)
        offset = 0
        for row, row_len in enumerate(row_lens):
            if row_len:
                page_table[row, :row_len] = shuffled_slots[
                    offset : offset + row_len
                ]
                offset = (offset + row_len) % pool_size

        from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

        kv_cache = quantize_k_cache(
            dequant_kv_cache.view(pool_size, 1, 1, -1)
        ).view(pool_size, 1, 656)
        return {
            "q_nope": q_nope,
            "q_rope": q_rope,
            "kv_cache": kv_cache,
            "page_table": page_table,
            "cache_seqlens": torch.tensor(
                row_lens, device=device, dtype=torch.int32
            ),
        }

    def _assert_matches_reference(self, case, logit_cap):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_decode_cuda,
            sm89_sparse_mla_prefill_reference,
        )

        expected = sm89_sparse_mla_prefill_reference(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            self._SM_SCALE,
            logit_cap,
            512,
        )
        actual = sm89_sparse_mla_decode_cuda(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            self._SM_SCALE,
            logit_cap,
            512,
        )
        self.assertEqual(actual.shape, case["q_nope"].shape)
        diff = (actual.float() - expected.float()).abs()
        nonempty = case["cache_seqlens"] > 0
        self.assertLessEqual(diff[nonempty].max().item(), 5e-2)
        self.assertLessEqual(diff[nonempty].mean().item(), 5e-3)
        cosine = torch.nn.functional.cosine_similarity(
            actual[nonempty].float().flatten(),
            expected[nonempty].float().flatten(),
            dim=0,
        ).item()
        self.assertGreaterEqual(cosine, 0.995)
        empty = ~nonempty
        if empty.any():
            self.assertTrue(
                torch.equal(actual[empty], torch.zeros_like(actual[empty]))
            )

    def test_decode_cuda_wrapper_forces_tensorcore(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla

        q_nope = torch.zeros(2, 32, 512, device="cuda", dtype=torch.bfloat16)
        q_rope = torch.zeros(2, 32, 64, device="cuda", dtype=torch.bfloat16)
        kv = torch.zeros(8, 1, 656, device="cuda", dtype=torch.float8_e4m3fn)
        pages = torch.zeros(2, 4, device="cuda", dtype=torch.int32)
        lengths = torch.full((2,), 4, device="cuda", dtype=torch.int32)
        sentinel = torch.ones_like(q_nope)
        with patch.object(
            sm89_sparse_mla, "sm89_sparse_mla_prefill_cuda", return_value=sentinel
        ) as op:
            out = sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                q_nope, q_rope, kv, pages, lengths, 0.0416666667, 0.0, 512
            )
        self.assertIs(out, sentinel)
        self.assertEqual(
            op.call_args.kwargs,
            {
                "q_nope": q_nope,
                "q_rope": q_rope,
                "kv_cache": kv,
                "page_table": pages,
                "cache_seqlens": lengths,
                "sm_scale": 0.0416666667,
                "logit_cap": 0.0,
                "v_head_dim": 512,
                "block_n": 32,
                "cuda_impl": "tensorcore",
            },
        )

    def test_decode_kernel_routes_explicit_split_width(self):
        from sglang.srt.layers.attention.dsa import (
            sm89_sparse_mla,
            sm89_sparse_mla_cuda,
        )

        three_stage_cases = (
            self._make_decode_case(1, [64], seed=401),
            self._make_decode_case(2, [1, 64], seed=402),
        )
        with patch.object(
            sm89_sparse_mla,
            "sm89_sparse_mla_prefill_cuda",
            side_effect=lambda **kwargs: torch.ones_like(kwargs["q_nope"]),
        ) as three_stage:
            for case in three_stage_cases:
                out = sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                    **case,
                    sm_scale=self._SM_SCALE,
                    logit_cap=30.0,
                    v_head_dim=512,
                    decode_kernel="three_stage",
                )
                self.assertEqual(out.shape, case["q_nope"].shape)

        self.assertEqual(three_stage.call_count, 2)
        for call in three_stage.call_args_list:
            self.assertEqual(call.kwargs["block_n"], 32)
            self.assertEqual(call.kwargs["cuda_impl"], "tensorcore")
            self.assertEqual(call.kwargs["logit_cap"], 30.0)

        split_case = self._make_decode_case(1, [64], seed=403)
        self.assertTrue(all(stride > 0 for stride in split_case["q_nope"].stride()))
        self.assertTrue(all(stride > 0 for stride in split_case["q_rope"].stride()))
        self.assertTrue(
            all(stride > 0 for stride in split_case["page_table"].stride())
        )
        self.assertEqual(split_case["cache_seqlens"].stride(0), 1)
        self.assertEqual(split_case["kv_cache"].stride(2), 1)
        self.assertGreaterEqual(split_case["kv_cache"].stride(0), 656)
        self.assertEqual(split_case["kv_cache"].stride(0) % 4, 0)
        self.assertEqual(split_case["kv_cache"].data_ptr() % 16, 0)

        for selector, split_tokens in (("splitk32", 32), ("splitk64", 64)):
            sentinel = torch.ones_like(split_case["q_nope"])
            split_op = MagicMock(return_value=sentinel)
            ext = SimpleNamespace(sm89_sparse_mla_decode_cuda_splitk=split_op)
            with patch.object(
                sm89_sparse_mla_cuda,
                "_load_sm89_sparse_mla_cuda_ext",
                return_value=ext,
            ) as load_ext:
                out = sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                    **split_case,
                    sm_scale=self._SM_SCALE,
                    logit_cap=0.0,
                    v_head_dim=512,
                    decode_kernel=selector,
                )

            self.assertIs(out, sentinel)
            load_ext.assert_called_once_with()
            split_op.assert_called_once_with(
                split_case["q_nope"],
                split_case["q_rope"],
                split_case["kv_cache"],
                split_case["page_table"],
                split_case["cache_seqlens"],
                torch.tensor(self._SM_SCALE, dtype=torch.float32).item(),
                0.0,
                512,
                split_tokens,
            )

    def test_decode_kernel_rejects_split_contract_before_extension_load(self):
        from sglang.srt.layers.attention.dsa import (
            sm89_sparse_mla,
            sm89_sparse_mla_cuda,
        )

        case = self._make_decode_case(1, [64], seed=404)
        pool_size = case["kv_cache"].shape[0]
        bad_kv_inner_stride = torch.empty(
            pool_size, 1, 1312, device="cuda", dtype=torch.float8_e4m3fn
        ).as_strided((pool_size, 1, 656), (1312, 1312, 2))
        bad_kv_row_stride = torch.empty(
            pool_size, 1, 658, device="cuda", dtype=torch.float8_e4m3fn
        )[..., :656]
        bad_kv_base_alignment = torch.empty(
            pool_size, 1, 660, device="cuda", dtype=torch.float8_e4m3fn
        )[..., 1:657]
        invalid_cases = {
            "batch": self._make_decode_case(2, [1, 64], seed=405),
            "heads": case
            | {
                "q_nope": case["q_nope"][:, :31],
                "q_rope": case["q_rope"][:, :31],
            },
            "topk": case | {"page_table": case["page_table"][:, :1024]},
            "q_nope_dtype": case | {"q_nope": case["q_nope"].float()},
            "q_nope_rank": case
            | {"q_nope": case["q_nope"].unsqueeze(0)},
            "q_rope_dtype": case | {"q_rope": case["q_rope"].float()},
            "page_table_dtype": case | {"page_table": case["page_table"].long()},
            "page_table_rank": case
            | {"page_table": case["page_table"].unsqueeze(0)},
            "cache_seqlens_dtype": case
            | {"cache_seqlens": case["cache_seqlens"].long()},
            "cache_seqlens_shape": case
            | {"cache_seqlens": case["cache_seqlens"].unsqueeze(1)},
            "kv_cache_dtype": case | {"kv_cache": case["kv_cache"].bfloat16()},
            "kv_cache_rank": case | {"kv_cache": case["kv_cache"].unsqueeze(0)},
            "device": case | {"q_rope": case["q_rope"].cpu()},
            "q_stride": case
            | {
                "q_nope": case["q_nope"].as_strided(
                    case["q_nope"].shape,
                    (0, case["q_nope"].stride(1), case["q_nope"].stride(2)),
                )
            },
            "q_rope_stride": case
            | {
                "q_rope": case["q_rope"].as_strided(
                    case["q_rope"].shape,
                    (0, case["q_rope"].stride(1), case["q_rope"].stride(2)),
                )
            },
            "page_stride": case
            | {
                "page_table": case["page_table"].as_strided(
                    case["page_table"].shape, (0, case["page_table"].stride(1))
                )
            },
            "cache_stride": case
            | {
                "cache_seqlens": torch.zeros(
                    1, device="cuda", dtype=torch.int32
                ).expand(2)[:1]
            },
            "kv_inner_stride": case | {"kv_cache": bad_kv_inner_stride},
            "kv_row_stride": case | {"kv_cache": bad_kv_row_stride},
            "kv_base_alignment": case | {"kv_cache": bad_kv_base_alignment},
        }
        split_op = MagicMock()
        ext = SimpleNamespace(sm89_sparse_mla_decode_cuda_splitk=split_op)
        with patch.object(
            sm89_sparse_mla_cuda,
            "_load_sm89_sparse_mla_cuda_ext",
            return_value=ext,
        ) as load_ext:
            for name, invalid_case in invalid_cases.items():
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "sm89_cuda decode"
                ):
                    sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                        **invalid_case,
                        sm_scale=self._SM_SCALE,
                        logit_cap=0.0,
                        v_head_dim=512,
                        decode_kernel="splitk32",
                    )
            with patch.object(
                sm89_sparse_mla_cuda,
                "splitk_workspace_schema",
                return_value={"workspace_bytes": 32 * 1024 * 1024},
            ), self.assertRaisesRegex(ValueError, "workspace"):
                sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                    **case,
                    sm_scale=self._SM_SCALE,
                    logit_cap=0.0,
                    v_head_dim=512,
                    decode_kernel="splitk32",
                )

        load_ext.assert_not_called()
        split_op.assert_not_called()

    def test_decode_kernel_rejects_nonfinite_and_negative_caps(self):
        from sglang.srt.layers.attention.dsa import (
            sm89_sparse_mla,
            sm89_sparse_mla_cuda,
        )

        case = self._make_decode_case(1, [64], seed=406)
        split_op = MagicMock()
        ext = SimpleNamespace(sm89_sparse_mla_decode_cuda_splitk=split_op)
        with patch.object(
            sm89_sparse_mla,
            "sm89_sparse_mla_prefill_cuda",
            return_value=torch.ones_like(case["q_nope"]),
        ) as three_stage, patch.object(
            sm89_sparse_mla_cuda,
            "_load_sm89_sparse_mla_cuda_ext",
            return_value=ext,
        ) as load_ext:
            for selector in ("three_stage", "splitk32", "splitk64"):
                for logit_cap in (-1.0, float("nan"), float("inf"), float("-inf")):
                    with self.subTest(selector=selector, logit_cap=logit_cap), self.assertRaisesRegex(
                        ValueError, "finite and nonnegative"
                    ):
                        sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                            **case,
                            sm_scale=self._SM_SCALE,
                            logit_cap=logit_cap,
                            v_head_dim=512,
                            decode_kernel=selector,
                        )

            sentinel = torch.ones_like(case["q_nope"])
            three_stage.return_value = sentinel
            self.assertIs(
                sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                    **case,
                    sm_scale=self._SM_SCALE,
                    logit_cap=30.0,
                    v_head_dim=512,
                    decode_kernel="three_stage",
                ),
                sentinel,
            )
            for selector in ("splitk32", "splitk64"):
                with self.subTest(selector=selector), self.assertRaisesRegex(
                    ValueError, "logit_cap=0"
                ):
                    sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                        **case,
                        sm_scale=self._SM_SCALE,
                        logit_cap=30.0,
                        v_head_dim=512,
                        decode_kernel=selector,
                    )

        self.assertEqual(three_stage.call_count, 1)
        load_ext.assert_not_called()
        split_op.assert_not_called()

    def test_decode_kernel_logs_exact_selector_once(self):
        from sglang.srt.layers.attention.dsa import (
            sm89_sparse_mla,
            sm89_sparse_mla_cuda,
        )

        case = self._make_decode_case(1, [64], seed=407)
        ext = SimpleNamespace(
            sm89_sparse_mla_decode_cuda_splitk=MagicMock(
                return_value=torch.ones_like(case["q_nope"])
            )
        )
        sm89_sparse_mla._logged_decode_kernels.clear()
        try:
            with patch.object(
                sm89_sparse_mla_cuda,
                "_load_sm89_sparse_mla_cuda_ext",
                return_value=ext,
            ), patch.object(sm89_sparse_mla.logger, "info") as mock_info:
                for selector in ("splitk32", "splitk32", "splitk64"):
                    sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                        **case,
                        sm_scale=self._SM_SCALE,
                        logit_cap=0.0,
                        v_head_dim=512,
                        decode_kernel=selector,
                    )

            self.assertEqual(mock_info.call_count, 2)
            self.assertEqual(
                mock_info.call_args_list[0].args,
                (
                    "GLM DSA SM89 effective decode kernel is %s selected by %s.",
                    "splitk32",
                    "SGLANG_GLM_DSA_SM89_DECODE_KERNEL",
                ),
            )
            self.assertEqual(
                mock_info.call_args_list[1].args,
                (
                    "GLM DSA SM89 effective decode kernel is %s selected by %s.",
                    "splitk64",
                    "SGLANG_GLM_DSA_SM89_DECODE_KERNEL",
                ),
            )
        finally:
            sm89_sparse_mla._logged_decode_kernels.clear()

    def test_decode_cuda_matches_reference_for_batch_shapes_and_softcaps(self):
        row_lens_by_batch = {
            1: [1],
            2: [0, 64],
            8: [0, 1, 64, 2048, 1, 64, 2048, 1],
        }
        for batch, row_lens in row_lens_by_batch.items():
            case = self._make_decode_case(batch, row_lens, seed=1000 + batch)
            self.assertFalse(case["q_nope"].is_contiguous())
            for logit_cap in (0.0, 30.0):
                with self.subTest(batch=batch, logit_cap=logit_cap):
                    self._assert_matches_reference(case, logit_cap)
        torch.cuda.synchronize()

    def test_decode_cuda_generic_path_is_stream_safe_and_non_aliasing(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_decode_cuda,
            sm89_sparse_mla_prefill_reference,
        )

        case_a = self._make_decode_case(2, [0, 2048], seed=201)
        case_b = self._make_decode_case(2, [1, 64], seed=202)
        inputs_ready = torch.cuda.Event()
        inputs_ready.record()
        done_a = torch.cuda.Event()
        done_b = torch.cuda.Event()
        stream_a = torch.cuda.Stream()
        stream_b = torch.cuda.Stream()

        with torch.cuda.stream(stream_a):
            stream_a.wait_event(inputs_ready)
            expected_a = sm89_sparse_mla_prefill_reference(
                **case_a,
                sm_scale=self._SM_SCALE,
                logit_cap=30.0,
                v_head_dim=512,
            )
            actual_a = sm89_sparse_mla_decode_cuda(
                **case_a,
                sm_scale=self._SM_SCALE,
                logit_cap=30.0,
                v_head_dim=512,
            )
            done_a.record(stream_a)
        with torch.cuda.stream(stream_b):
            stream_b.wait_event(inputs_ready)
            expected_b = sm89_sparse_mla_prefill_reference(
                **case_b,
                sm_scale=self._SM_SCALE,
                logit_cap=0.0,
                v_head_dim=512,
            )
            actual_b = sm89_sparse_mla_decode_cuda(
                **case_b,
                sm_scale=self._SM_SCALE,
                logit_cap=0.0,
                v_head_dim=512,
            )
            done_b.record(stream_b)
        torch.cuda.current_stream().wait_event(done_a)
        torch.cuda.current_stream().wait_event(done_b)
        torch.cuda.synchronize()

        for actual, expected, case in (
            (actual_a, expected_a, case_a),
            (actual_b, expected_b, case_b),
        ):
            nonempty = case["cache_seqlens"] > 0
            actual_nonempty = actual[nonempty].float()
            expected_nonempty = expected[nonempty].float()
            diff = (actual_nonempty - expected_nonempty).abs()
            self.assertLessEqual(diff.max().item(), 5e-2)
            self.assertLessEqual(diff.mean().item(), 5e-3)
            cosine = torch.nn.functional.cosine_similarity(
                actual_nonempty.flatten(), expected_nonempty.flatten(), dim=0
            ).item()
            self.assertGreaterEqual(cosine, 0.995)
            empty = ~nonempty
            if empty.any():
                self.assertTrue(
                    torch.equal(actual[empty], torch.zeros_like(actual[empty]))
                )
        self.assertNotEqual(
            actual_a.untyped_storage().data_ptr(),
            actual_b.untyped_storage().data_ptr(),
        )

    def test_decode_cuda_rejects_invalid_contracts_before_kernel_call(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla

        case = self._make_decode_case(2, [1, 64], seed=300)
        invalid_cases = {
            "q_nope_dtype": {"q_nope": case["q_nope"].float()},
            "q_rope_dtype": {"q_rope": case["q_rope"].float()},
            "q_nope_rank": {"q_nope": case["q_nope"].unsqueeze(0)},
            "page_table_dtype": {"page_table": case["page_table"].long()},
            "page_table_rank": {"page_table": case["page_table"].unsqueeze(0)},
            "cache_seqlens_dtype": {
                "cache_seqlens": case["cache_seqlens"].long()
            },
            "kv_cache_dtype": {"kv_cache": case["kv_cache"].bfloat16()},
            "kv_cache_rank": {"kv_cache": case["kv_cache"].unsqueeze(0)},
            "device": {"q_rope": case["q_rope"].cpu()},
            "head_count": {
                "q_nope": torch.zeros(2, 65, 512, device="cuda", dtype=torch.bfloat16),
                "q_rope": torch.zeros(2, 65, 64, device="cuda", dtype=torch.bfloat16),
            },
            "topk_width": {
                "page_table": torch.zeros(2, 4097, device="cuda", dtype=torch.int32)
            },
            "kv_width": {
                "kv_cache": torch.zeros(
                    4096, 1, 655, device="cuda", dtype=torch.float8_e4m3fn
                )
            },
            "row_length_shape": {
                "cache_seqlens": torch.ones(2, 1, device="cuda", dtype=torch.int32)
            },
        }
        with patch.object(sm89_sparse_mla, "sm89_sparse_mla_prefill_cuda") as op:
            for name, replacements in invalid_cases.items():
                args = case | replacements
                with self.subTest(name=name), self.assertRaisesRegex(
                    ValueError, "sm89_cuda decode"
                ):
                    sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
                        **args,
                        sm_scale=self._SM_SCALE,
                        logit_cap=0.0,
                        v_head_dim=512,
                    )
        op.assert_not_called()


@unittest.skipIf(not torch.cuda.is_available(), "CUDA is required")
class TestSm89SparseMlaTriton(unittest.TestCase):
    def _make_triton_case(self, total_q, topk):
        torch.manual_seed(total_q * 4099 + topk)
        device = torch.device("cuda")
        num_heads = 1
        v_head_dim = 512
        rope_dim = 64
        kv_tokens = topk + total_q + 17

        q_nope = (
            torch.randn(
                total_q, num_heads, v_head_dim, device=device, dtype=torch.float32
            ).to(torch.bfloat16)
            * 0.125
        )
        q_rope = (
            torch.randn(
                total_q, num_heads, rope_dim, device=device, dtype=torch.float32
            ).to(torch.bfloat16)
            * 0.125
        )
        dequant_kv_cache = (
            torch.randn(kv_tokens, 1, v_head_dim + rope_dim, device=device).to(
                torch.bfloat16
            )
            * 0.125
        )

        page_table = torch.full((total_q, topk), -1, device=device, dtype=torch.int32)
        row_lens = []
        for row in range(total_q):
            if row == 0:
                row_len = 0
            elif row % 3 == 0:
                row_len = max(topk // 2, 1)
            else:
                row_len = topk
            row_lens.append(row_len)
            if row_len > 0:
                token_ids = (torch.arange(row_len, device=device, dtype=torch.int32) * 5 + row * 11) % (
                    kv_tokens
                )
                page_table[row, :row_len] = token_ids

        from sglang.srt.layers.attention.dsa.quant_k_cache import quantize_k_cache

        kv_cache = quantize_k_cache(dequant_kv_cache.view(kv_tokens, 1, 1, -1)).view(
            kv_tokens, 1, 656
        )

        return {
            "q_nope": q_nope,
            "q_rope": q_rope,
            "kv_cache": kv_cache,
            "page_table": page_table,
            "cache_seqlens": torch.tensor(row_lens, device=device, dtype=torch.int32),
            "sm_scale": 1.0 / (v_head_dim + rope_dim) ** 0.5,
            "v_head_dim": v_head_dim,
        }

    def test_triton_matches_reference(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
            sm89_sparse_mla_prefill_triton,
        )

        for total_q in (16, 64, 128):
            for topk in (64, 256):
                for logit_cap in (0.0, 30.0):
                    with self.subTest(
                        total_q=total_q, topk=topk, logit_cap=logit_cap
                    ):
                        case = self._make_triton_case(total_q, topk)
                        expected = sm89_sparse_mla_prefill_reference(
                            case["q_nope"],
                            case["q_rope"],
                            case["kv_cache"],
                            case["page_table"],
                            case["cache_seqlens"],
                            case["sm_scale"],
                            logit_cap,
                            case["v_head_dim"],
                        )
                        actual = sm89_sparse_mla_prefill_triton(
                            case["q_nope"],
                            case["q_rope"],
                            case["kv_cache"],
                            case["page_table"],
                            case["cache_seqlens"],
                            case["sm_scale"],
                            logit_cap,
                            case["v_head_dim"],
                        )
                        torch.cuda.synchronize()

                        expected_f = expected.float()
                        actual_f = actual.float()
                        diff = (actual_f - expected_f).abs()
                        max_abs = diff.max().item()
                        mean_abs = diff.mean().item()
                        cosine = torch.nn.functional.cosine_similarity(
                            actual_f.flatten(), expected_f.flatten(), dim=0
                        ).item()

                        self.assertLessEqual(max_abs, 5e-2)
                        self.assertLessEqual(mean_abs, 5e-3)
                        self.assertGreaterEqual(cosine, 0.995)
                        empty_rows = case["cache_seqlens"] == 0
                        self.assertTrue(
                            torch.equal(
                                actual[empty_rows],
                                torch.zeros_like(actual[empty_rows]),
                            )
                        )

    def test_cuda_backend_matches_reference_small_shape(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_cuda,
            sm89_sparse_mla_prefill_reference,
        )

        case = self._make_triton_case(total_q=8, topk=32)
        expected = sm89_sparse_mla_prefill_reference(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            case["sm_scale"],
            30.0,
            case["v_head_dim"],
        )
        actual = sm89_sparse_mla_prefill_cuda(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            case["sm_scale"],
            30.0,
            case["v_head_dim"],
            block_n=32,
        )
        torch.cuda.synchronize()

        diff = (actual.float() - expected.float()).abs()
        self.assertLessEqual(diff.max().item(), 5e-2)
        self.assertLessEqual(diff.mean().item(), 5e-3)
        empty_rows = case["cache_seqlens"] == 0
        self.assertTrue(
            torch.equal(actual[empty_rows], torch.zeros_like(actual[empty_rows]))
        )

    def test_triton_entry_can_select_cuda_backend_with_env(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla

        q_nope = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
        q_rope = torch.zeros(1, 1, 64, device="cuda", dtype=torch.bfloat16)
        kv_cache = torch.zeros(1, 1, 656, device="cuda", dtype=torch.float8_e4m3fn)
        page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
        sentinel = torch.ones_like(q_nope)

        with patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_KERNEL": "cuda"}, clear=True
        ), patch.object(
            sm89_sparse_mla,
            "sm89_sparse_mla_prefill_cuda",
            return_value=sentinel,
        ) as mock_cuda, patch.object(sm89_sparse_mla.logger, "info") as mock_info:
            sm89_sparse_mla._logged_kernel_impls.clear()
            actual = sm89_sparse_mla.sm89_sparse_mla_prefill_triton(
                q_nope,
                q_rope,
                kv_cache,
                page_table,
                cache_seqlens,
                1.0,
                30.0,
                512,
                block_n=32,
            )
            second = sm89_sparse_mla.sm89_sparse_mla_prefill_triton(
                q_nope,
                q_rope,
                kv_cache,
                page_table,
                cache_seqlens,
                1.0,
                30.0,
                512,
                block_n=32,
            )

        self.assertIs(actual, sentinel)
        self.assertIs(second, sentinel)
        self.assertEqual(mock_cuda.call_count, 2)
        mock_cuda.assert_called_with(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=1.0,
            logit_cap=30.0,
            v_head_dim=512,
            block_n=32,
        )
        mock_info.assert_called_once()
        self.assertIn("uses %s implementation", mock_info.call_args.args[0])
        self.assertEqual(mock_info.call_args.args[1], "cuda")
        sm89_sparse_mla._logged_kernel_impls.clear()

    def test_cuda_backend_is_profiled_when_profile_env_enabled(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla

        q_nope = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
        q_rope = torch.zeros(1, 1, 64, device="cuda", dtype=torch.bfloat16)
        kv_cache = torch.zeros(1, 1, 656, device="cuda", dtype=torch.float8_e4m3fn)
        page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
        sentinel = torch.ones_like(q_nope)
        events = []

        @contextmanager
        def fake_nvtx_range(name):
            events.append(("nvtx_enter", name))
            try:
                yield
            finally:
                events.append(("nvtx_exit", name))

        @contextmanager
        def fake_cuda_timer(name, enabled):
            events.append(("timer_enter", name, enabled))
            try:
                yield
            finally:
                events.append(("timer_exit", name, enabled))

        with patch.dict(
            os.environ,
            {
                "SGLANG_GLM_DSA_SM89_KERNEL": "cuda",
                "SGLANG_GLM_DSA_SM89_PROFILE": "1",
            },
            clear=True,
        ), patch.object(
            sm89_sparse_mla,
            "sm89_sparse_mla_prefill_cuda",
            return_value=sentinel,
        ), patch.object(
            sm89_sparse_mla, "nvtx_range", side_effect=fake_nvtx_range
        ), patch.object(
            sm89_sparse_mla, "cuda_timer", side_effect=fake_cuda_timer
        ):
            actual = sm89_sparse_mla.sm89_sparse_mla_prefill_triton(
                q_nope,
                q_rope,
                kv_cache,
                page_table,
                cache_seqlens,
                1.0,
                30.0,
                512,
                block_n=32,
            )

        self.assertIs(actual, sentinel)
        self.assertIn(("nvtx_enter", "sm89_sparse_mla.cuda.total"), events)
        self.assertIn(("nvtx_exit", "sm89_sparse_mla.cuda.total"), events)
        self.assertIn(("timer_enter", "sm89_sparse_mla.cuda.total", True), events)
        self.assertIn(("timer_exit", "sm89_sparse_mla.cuda.total", True), events)

    def test_cuda_backend_skips_profile_context_when_profile_env_disabled(self):
        from sglang.srt.layers.attention.dsa import sm89_sparse_mla

        q_nope = torch.zeros(1, 1, 512, device="cuda", dtype=torch.bfloat16)
        q_rope = torch.zeros(1, 1, 64, device="cuda", dtype=torch.bfloat16)
        kv_cache = torch.zeros(1, 1, 656, device="cuda", dtype=torch.float8_e4m3fn)
        page_table = torch.zeros(1, 1, device="cuda", dtype=torch.int32)
        cache_seqlens = torch.ones(1, device="cuda", dtype=torch.int32)
        sentinel = torch.ones_like(q_nope)

        with patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_KERNEL": "cuda"}, clear=True
        ), patch.object(
            sm89_sparse_mla,
            "sm89_sparse_mla_prefill_cuda",
            return_value=sentinel,
        ), patch.object(
            sm89_sparse_mla, "nvtx_range"
        ) as mock_nvtx, patch.object(
            sm89_sparse_mla, "cuda_timer"
        ) as mock_timer:
            actual = sm89_sparse_mla.sm89_sparse_mla_prefill_triton(
                q_nope,
                q_rope,
                kv_cache,
                page_table,
                cache_seqlens,
                1.0,
                30.0,
                512,
                block_n=32,
            )

        self.assertIs(actual, sentinel)
        mock_nvtx.assert_not_called()
        mock_timer.assert_not_called()

    def test_select_v_block_from_env(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_v_block,
        )

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_sm89_sparse_mla_v_block(None), 64)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_V_BLOCK": "128"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_v_block(None), 128)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_V_BLOCK": "256"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_v_block(None), 256)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_V_BLOCK": "256"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_v_block(128), 128)

    def test_select_v_block_rejects_unsupported_values(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_v_block,
        )

        with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_V_BLOCK"):
            select_sm89_sparse_mla_v_block(96)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_V_BLOCK": "invalid"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_V_BLOCK"):
                select_sm89_sparse_mla_v_block(None)

    def test_select_block_n_from_env(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_block_n,
        )

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_sm89_sparse_mla_block_n(None), 64)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_N": "32"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_block_n(None), 32)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_N": "128"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_block_n(None), 128)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_N": "128"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_block_n(32), 32)

    def test_select_block_n_rejects_unsupported_values(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_block_n,
        )

        with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_BLOCK_N"):
            select_sm89_sparse_mla_block_n(96)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_N": "invalid"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_BLOCK_N"):
                select_sm89_sparse_mla_block_n(None)

    def test_select_block_m_from_env(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_block_m,
        )

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_sm89_sparse_mla_block_m(None), 1)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_M": "2"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_block_m(None), 2)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_M": "4"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_block_m(None), 4)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_M": "4"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_block_m(2), 2)

    def test_select_block_m_rejects_unsupported_values(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_block_m,
        )

        with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_BLOCK_M"):
            select_sm89_sparse_mla_block_m(3)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_BLOCK_M": "invalid"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_BLOCK_M"):
                select_sm89_sparse_mla_block_m(None)

    def test_select_split_k_from_env(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_split_k,
        )

        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(select_sm89_sparse_mla_split_k(None), 1)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_SPLIT_K": "4"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_split_k(None), 4)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_SPLIT_K": "8"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_split_k(None), 8)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_SPLIT_K": "8"}, clear=True
        ):
            self.assertEqual(select_sm89_sparse_mla_split_k(4), 4)

    def test_select_split_k_rejects_unsupported_values(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
            select_sm89_sparse_mla_split_k,
        )

        with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_SPLIT_K"):
            select_sm89_sparse_mla_split_k(3)

        with unittest.mock.patch.dict(
            os.environ, {"SGLANG_GLM_DSA_SM89_SPLIT_K": "invalid"}, clear=True
        ):
            with self.assertRaisesRegex(ValueError, "SGLANG_GLM_DSA_SM89_SPLIT_K"):
                select_sm89_sparse_mla_split_k(None)

    def test_triton_v_block_variants_match_reference(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
            sm89_sparse_mla_prefill_triton,
        )

        case = self._make_triton_case(total_q=16, topk=64)
        expected = sm89_sparse_mla_prefill_reference(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            case["sm_scale"],
            0.0,
            case["v_head_dim"],
        )

        for v_block in (64, 128, 256):
            with self.subTest(v_block=v_block):
                actual = sm89_sparse_mla_prefill_triton(
                    case["q_nope"],
                    case["q_rope"],
                    case["kv_cache"],
                    case["page_table"],
                    case["cache_seqlens"],
                    case["sm_scale"],
                    0.0,
                    case["v_head_dim"],
                    v_block=v_block,
                )
                torch.cuda.synchronize()
                diff = (actual.float() - expected.float()).abs()
                self.assertLessEqual(diff.max().item(), 5e-2)
                self.assertLessEqual(diff.mean().item(), 5e-3)

    def test_triton_block_n_variants_match_reference(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
            sm89_sparse_mla_prefill_triton,
        )

        case = self._make_triton_case(total_q=16, topk=128)
        expected = sm89_sparse_mla_prefill_reference(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            case["sm_scale"],
            30.0,
            case["v_head_dim"],
        )

        for block_n in (32, 64, 128):
            with self.subTest(block_n=block_n):
                actual = sm89_sparse_mla_prefill_triton(
                    case["q_nope"],
                    case["q_rope"],
                    case["kv_cache"],
                    case["page_table"],
                    case["cache_seqlens"],
                    case["sm_scale"],
                    30.0,
                    case["v_head_dim"],
                    v_block=128,
                    block_n=block_n,
                )
                torch.cuda.synchronize()
                diff = (actual.float() - expected.float()).abs()
                self.assertLessEqual(diff.max().item(), 5e-2)
                self.assertLessEqual(diff.mean().item(), 5e-3)

    def test_triton_block_m_variants_match_reference(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
            sm89_sparse_mla_prefill_triton,
        )

        case = self._make_triton_case(total_q=16, topk=64)
        expected = sm89_sparse_mla_prefill_reference(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            case["sm_scale"],
            30.0,
            case["v_head_dim"],
        )

        for block_m in (1, 2, 4):
            with self.subTest(block_m=block_m):
                actual = sm89_sparse_mla_prefill_triton(
                    case["q_nope"],
                    case["q_rope"],
                    case["kv_cache"],
                    case["page_table"],
                    case["cache_seqlens"],
                    case["sm_scale"],
                    30.0,
                    case["v_head_dim"],
                    v_block=64,
                    block_n=32,
                    block_m=block_m,
                )
                torch.cuda.synchronize()
                diff = (actual.float() - expected.float()).abs()
                self.assertLessEqual(diff.max().item(), 5e-2)
                self.assertLessEqual(diff.mean().item(), 5e-3)

    def test_triton_split_k_variants_match_reference(self):
        from sglang.srt.layers.attention.dsa.sm89_sparse_mla import (
            sm89_sparse_mla_prefill_reference,
            sm89_sparse_mla_prefill_triton,
        )

        case = self._make_triton_case(total_q=16, topk=128)
        expected = sm89_sparse_mla_prefill_reference(
            case["q_nope"],
            case["q_rope"],
            case["kv_cache"],
            case["page_table"],
            case["cache_seqlens"],
            case["sm_scale"],
            30.0,
            case["v_head_dim"],
        )

        for split_k in (1, 4, 8):
            with self.subTest(split_k=split_k):
                actual = sm89_sparse_mla_prefill_triton(
                    case["q_nope"],
                    case["q_rope"],
                    case["kv_cache"],
                    case["page_table"],
                    case["cache_seqlens"],
                    case["sm_scale"],
                    30.0,
                    case["v_head_dim"],
                    v_block=128,
                    block_n=32,
                    block_m=1,
                    split_k=split_k,
                )
                torch.cuda.synchronize()
                diff = (actual.float() - expected.float()).abs()
                self.assertLessEqual(diff.max().item(), 5e-2)
                self.assertLessEqual(diff.mean().item(), 5e-3)


if __name__ == "__main__":
    unittest.main()
