import json
import os
import tempfile
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch


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
            self.assertTrue(
                torch.allclose(
                    actual[nonempty], expected[nonempty], atol=5e-2, rtol=5e-3
                )
            )
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
            "cache_seqlens_dtype": {
                "cache_seqlens": case["cache_seqlens"].long()
            },
            "kv_cache_dtype": {"kv_cache": case["kv_cache"].bfloat16()},
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
