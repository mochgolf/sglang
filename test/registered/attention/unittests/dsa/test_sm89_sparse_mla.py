import json
import unittest
from contextlib import contextmanager
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

    def test_forward_torch_mla_dumps_shapes_once_when_env_enabled(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend._dumped_glm_sm89_torch_mla_shapes = False

        q_rope = torch.zeros(2, 1, 4)
        q_nope = torch.zeros(2, 1, 4)
        kv_cache = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8)
        page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)

        with patch.dict(
            "os.environ", {"SGLANG_GLM_DSA_SM89_DUMP_SHAPES": "1"}, clear=True
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
                layer_id=7,
            )
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
                layer_id=11,
            )

        mock_print.assert_called_once()
        self.assertEqual(len(mock_print.call_args.args), 1)
        line = mock_print.call_args.args[0]
        dump = json.loads(line)
        self.assertEqual(dump["event"], "GLM_DSA_SM89_SHAPES")
        self.assertEqual(dump["layer_id"], 7)
        self.assertEqual(dump["q_nope_shape"], [2, 1, 4])
        self.assertEqual(dump["q_rope_shape"], [2, 1, 4])
        self.assertEqual(dump["kv_cache_shape"], [3, 1, 8])
        self.assertEqual(dump["kv_cache_dtype"], "torch.float32")
        self.assertEqual(dump["page_table_shape"], [2, 3])
        self.assertEqual(dump["cache_seqlens_shape"], [2])
        self.assertEqual(dump["cache_seqlens_max"], 2)
        self.assertEqual(dump["v_head_dim"], 4)
        self.assertEqual(dump["page_size"], 1)
        self.assertEqual(dump["q_nope_stride"], list(q_nope.stride()))
        self.assertEqual(dump["q_rope_stride"], list(q_rope.stride()))
        self.assertEqual(dump["kv_cache_stride"], list(kv_cache.stride()))
        self.assertEqual(dump["page_table_stride"], list(page_table.stride()))
        self.assertEqual(
            dump["cache_seqlens_stride"], list(cache_seqlens.stride())
        )
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


if __name__ == "__main__":
    unittest.main()
