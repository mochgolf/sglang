import importlib
import inspect
import json
import os
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

import sglang.srt.server_args as server_args_module
from sglang.srt.environ import envs
from sglang.srt.arg_groups.speculative_hook import handle_speculative_decoding
from sglang.srt.layers.cp.base import is_cp_enabled, is_interleave
from sglang.srt.model_executor.cuda_graph_config import (
    Backend,
    CudaGraphConfig,
    PhaseConfig,
)
from sglang.srt.server_args import PortArgs, ServerArgs, prepare_server_args
from sglang.srt.server_args_config_parser import ConfigArgumentMerger
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
    CustomTestCase,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=12, suite="base-b-test-cpu")

# Mock get_device() so all tests run on CPU-only CI runners
_mock_device = patch("sglang.srt.server_args.get_device", return_value="cuda")
_mock_device.start()


class TestPrepareServerArgs(CustomTestCase):
    def test_prepare_server_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--json-model-override-args",
                '{"rope_scaling": {"factor": 2.0, "rope_type": "linear"}}',
            ]
        )
        self.assertEqual(server_args.model_path, DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN)
        self.assertEqual(
            json.loads(server_args.json_model_override_args),
            {"rope_scaling": {"factor": 2.0, "rope_type": "linear"}},
        )

    def test_config_nested_dict_args_are_json(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("mm-process-config:\n  image:\n    resize: 128\n")
            config_file = f.name

        try:
            parser = server_args_module.argparse.ArgumentParser()
            ServerArgs.add_cli_args(parser)
            merged = ConfigArgumentMerger(parser).merge_config_with_args(
                [
                    "--config",
                    config_file,
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                ]
            )
            value = merged[merged.index("--mm-process-config") + 1]
            parsed = parser.parse_args(merged)

            self.assertEqual(json.loads(value), {"image": {"resize": 128}})
            self.assertEqual(parsed.mm_process_config, {"image": {"resize": 128}})
        finally:
            os.unlink(config_file)


class TestLoadBalanceMethod(unittest.TestCase):
    def test_non_pd_defaults_to_round_robin(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="null")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_prefill_defaults_to_follow_bootstrap_room(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="prefill")
        self.assertEqual(server_args.load_balance_method, "follow_bootstrap_room")

    def test_pd_decode_defaults_to_round_robin(self):
        server_args = ServerArgs(model_path="dummy", disaggregation_mode="decode")
        self.assertEqual(server_args.load_balance_method, "round_robin")

    def test_pd_decode_radix_cache_rejects_hisparse(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="nixl",
                enable_hisparse=True,
            )

        self.assertIn(
            "--disaggregation-decode-enable-radix-cache is incompatible with "
            "--enable-hisparse",
            str(context.exception),
        )

    def test_pd_decode_radix_cache_allows_mooncake(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake",
        )

        self.assertFalse(server_args.disable_radix_cache)

    def test_pd_decode_radix_cache_rejects_fake_backend(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(
                model_path="dummy",
                disaggregation_mode="decode",
                disaggregation_decode_enable_radix_cache=True,
                disaggregation_transfer_backend="fake",
            )

        self.assertIn(
            "--disaggregation-decode-enable-radix-cache is incompatible "
            "with --disaggregation-transfer-backend fake",
            str(context.exception),
        )

    def test_pd_decode_radix_cache_allows_ascend(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="ascend",
        )

        self.assertFalse(server_args.disable_radix_cache)

    def test_pd_decode_radix_cache_allows_mooncake_tcp(self):
        server_args = ServerArgs(
            model_path="dummy",
            disaggregation_mode="decode",
            disaggregation_decode_enable_radix_cache=True,
            disaggregation_transfer_backend="mooncake_tcp",
        )

        self.assertFalse(server_args.disable_radix_cache)
        self.assertEqual(server_args.disaggregation_transfer_backend, "mooncake")


class TestHiSparseDsaBackendPolicy(unittest.TestCase):
    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_defaults_to_flashmla_sparse_on_cuda_bfloat16(self, _mock_is_hip):
        server_args = ServerArgs(model_path="dummy", enable_hisparse=True)

        server_args._set_default_dsa_backends(kv_cache_dtype="bfloat16", major=9)

        self.assertEqual(server_args.dsa_prefill_backend, "flashmla_sparse")
        self.assertEqual(server_args.dsa_decode_backend, "flashmla_sparse")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_defaults_to_flashmla_kv_on_cuda_fp8(self, _mock_is_hip):
        server_args = ServerArgs(model_path="dummy", enable_hisparse=True)

        server_args._set_default_dsa_backends(kv_cache_dtype="fp8_e4m3", major=9)

        self.assertEqual(server_args.dsa_prefill_backend, "flashmla_kv")
        self.assertEqual(server_args.dsa_decode_backend, "flashmla_kv")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_defaults_to_tilelang_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(model_path="dummy", enable_hisparse=True)

        server_args._set_default_dsa_backends(kv_cache_dtype="bfloat16", major=9)

        self.assertEqual(server_args.dsa_prefill_backend, "tilelang")
        self.assertEqual(server_args.dsa_decode_backend, "tilelang")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_preserves_rocm_user_backend_and_defaults_missing_side(
        self, _mock_is_hip
    ):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            dsa_prefill_backend="tilelang",
        )

        server_args._set_default_dsa_backends(kv_cache_dtype="bfloat16", major=9)

        self.assertEqual(server_args.dsa_prefill_backend, "tilelang")
        self.assertEqual(server_args.dsa_decode_backend, "tilelang")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_accepts_aiter_backend_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="aiter",
            dsa_decode_backend="aiter",
        )

        server_args._validate_hisparse_dsa_backend("dsa_prefill_backend", "prefill")
        server_args._validate_hisparse_dsa_backend("dsa_decode_backend", "decode")

    @patch("sglang.srt.server_args.is_hip", return_value=True)
    def test_hisparse_rejects_cuda_backend_on_rocm(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="flashmla_sparse",
        )

        with self.assertRaisesRegex(ValueError, "tilelang"):
            server_args._validate_hisparse_dsa_backend("dsa_prefill_backend", "prefill")

    @patch("sglang.srt.server_args.is_hip", return_value=False)
    def test_hisparse_rejects_rocm_backend_on_cuda(self, _mock_is_hip):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="tilelang",
        )

        with self.assertRaisesRegex(ValueError, "flashmla_sparse"):
            server_args._validate_hisparse_dsa_backend("dsa_decode_backend", "decode")

    def test_hisparse_accepts_bfloat16_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="bfloat16",
        )

        server_args._validate_hisparse_kv_cache_dtype()

    def test_hisparse_accepts_fp8_e4m3_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="fp8_e4m3",
        )

        server_args._validate_hisparse_kv_cache_dtype()

    def test_hisparse_rejects_unsupported_kv_cache_dtype(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            kv_cache_dtype="float16",
        )

        with self.assertRaisesRegex(ValueError, r"fp8_e4m3"):
            server_args._validate_hisparse_kv_cache_dtype()


class TestGlmSm89DsaFallback(CustomTestCase):
    def test_glm_sm89_fallback_target_predicate(self):
        self.assertTrue(
            server_args_module.is_glm_dsa_sm89_fallback_target(
                "GlmMoeDsaForCausalLM", capability=(8, 9)
            )
        )
        self.assertFalse(
            server_args_module.is_glm_dsa_sm89_fallback_target(
                "GlmMoeDsaForCausalLM", capability=(9, 0)
            )
        )
        self.assertFalse(
            server_args_module.is_glm_dsa_sm89_fallback_target(
                "OtherModel", capability=(8, 9)
            )
        )

    def test_glm_sm89_fallback_defaults_avoid_flashmla_and_sgl_kernel(self):
        server_args = ServerArgs(model_path="dummy", kv_cache_dtype="fp8_e4m3")

        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
            stack.enter_context(envs.SGLANG_OPT_USE_TOPK_V2.override(True))
            stack.enter_context(envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False))

            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

            self.assertEqual(server_args.dsa_prefill_backend, "fa3")
            self.assertEqual(server_args.dsa_decode_backend, "fa3")
            self.assertEqual(server_args.dsa_topk_backend, "torch")
            self.assertTrue(envs.SGLANG_DSA_FUSE_TOPK.get())
            self.assertTrue(envs.SGLANG_OPT_USE_TOPK_V2.get())
            self.assertFalse(envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get())
            self.assertTrue(server_args.enable_glm_dsa_sm89_fallback)

    def test_dsa_choices_include_sm89_triton(self):
        self.assertIn("sm89_triton", server_args_module.DSA_CHOICES)

    def test_dsa_choices_include_sm89_cuda(self):
        self.assertIn("sm89_cuda", server_args_module.DSA_CHOICES)

    def test_glm_sm89_accepts_explicit_sm89_cuda_decode(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="sm89_cuda",
        )

        server_args._set_default_dsa_backends(
            kv_cache_dtype="fp8_e4m3",
            major=8,
            minor=9,
            model_arch="GlmMoeDsaForCausalLM",
        )

        self.assertEqual(server_args.dsa_prefill_backend, "fa3")
        self.assertEqual(server_args.dsa_decode_backend, "sm89_cuda")
        self.assertTrue(server_args.enable_glm_dsa_sm89_fallback)

    def test_sm89_cuda_rejects_prefill(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="sm89_cuda",
        )

        with self.assertRaisesRegex(ValueError, "decode only"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_sm89_cuda_rejects_non_glm_target(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="sm89_cuda",
        )

        with self.assertRaisesRegex(ValueError, "GLM DSA on SM89"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="OtherModel",
            )

    def test_sm89_cuda_rejects_non_sm89_target(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="sm89_cuda",
        )

        with self.assertRaisesRegex(ValueError, "GLM DSA on SM89"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=9,
                minor=0,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_sm89_cuda_requires_fp8_kv(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="bfloat16",
            dsa_decode_backend="sm89_cuda",
        )

        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="bfloat16",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_sm89_cuda_rejects_all_speculative_algorithms(self):
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        algorithms = [
            algorithm.name
            for algorithm in SpeculativeAlgorithm
            if algorithm is not SpeculativeAlgorithm.NONE
        ] + ["NEXTN"]
        for algorithm in algorithms:
            with self.subTest(algorithm=algorithm):
                server_args = ServerArgs(
                    model_path="dummy",
                    kv_cache_dtype="fp8_e4m3",
                    dsa_decode_backend="sm89_cuda",
                    speculative_algorithm=algorithm,
                )

                with self.assertRaisesRegex(ValueError, "speculative/MTP"):
                    server_args._set_default_dsa_backends(
                        kv_cache_dtype="fp8_e4m3",
                        major=8,
                        minor=9,
                        model_arch="GlmMoeDsaForCausalLM",
                    )

    def test_glm_sm89_fallback_accepts_explicit_sm89_triton_prefill(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="sm89_triton",
        )

        server_args._set_default_dsa_backends(
            kv_cache_dtype="fp8_e4m3",
            major=8,
            minor=9,
            model_arch="GlmMoeDsaForCausalLM",
        )

        self.assertEqual(server_args.dsa_prefill_backend, "sm89_triton")
        self.assertEqual(server_args.dsa_decode_backend, "fa3")
        self.assertEqual(server_args.dsa_topk_backend, "torch")
        self.assertTrue(server_args.enable_glm_dsa_sm89_fallback)

    def test_glm_sm89_fallback_rejects_explicit_sm89_triton_decode(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_decode_backend="sm89_triton",
        )

        with self.assertRaisesRegex(ValueError, "prefill only"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_sm89_triton_rejects_non_glm_sm89_target(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="sm89_triton",
        )

        with self.assertRaisesRegex(ValueError, "GLM DSA on SM89"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=9,
                minor=0,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_sm89_triton_requires_fp8_kv(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="bfloat16",
            dsa_prefill_backend="sm89_triton",
        )

        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="bfloat16",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_glm_sm89_fallback_does_not_leak_env_to_non_target_args(self):
        with ExitStack() as stack:
            stack.enter_context(envs.SGLANG_DSA_FUSE_TOPK.override(True))
            stack.enter_context(envs.SGLANG_OPT_USE_TOPK_V2.override(True))
            stack.enter_context(envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.override(True))
            stack.enter_context(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.override(True))
            stack.enter_context(envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.override(False))

            target_args = ServerArgs(model_path="dummy", kv_cache_dtype="fp8_e4m3")
            target_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

            non_target_args = ServerArgs(model_path="dummy", kv_cache_dtype="fp8_e4m3")
            non_target_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=9,
                minor=0,
                model_arch="GlmMoeDsaForCausalLM",
            )

            self.assertTrue(envs.SGLANG_DSA_FUSE_TOPK.get())
            self.assertTrue(envs.SGLANG_OPT_USE_TOPK_V2.get())
            self.assertTrue(envs.SGLANG_OPT_USE_TILELANG_MHC_PRE.get())
            self.assertTrue(envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get())
            self.assertFalse(envs.SGLANG_FP8_PAGED_MQA_LOGITS_TORCH.get())
            self.assertTrue(target_args.enable_glm_dsa_sm89_fallback)
            self.assertFalse(non_target_args.enable_glm_dsa_sm89_fallback)

    def test_glm_sm89_fallback_rejects_explicit_flashmla_backend(self):
        server_args = ServerArgs(
            model_path="dummy",
            kv_cache_dtype="fp8_e4m3",
            dsa_prefill_backend="flashmla_kv",
        )

        with self.assertRaisesRegex(ValueError, "flashmla_kv"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )

    def test_glm_sm89_fallback_marks_indexer_deepgemm_disabled(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.dsa import dsa_indexer

        server_args = ServerArgs(model_path="dummy", kv_cache_dtype="fp8_e4m3")
        server_args._set_default_dsa_backends(
            kv_cache_dtype="fp8_e4m3",
            major=8,
            minor=9,
            model_arch="GlmMoeDsaForCausalLM",
        )

        fake_decode_mode = SimpleNamespace(
            is_decode_or_idle=lambda: True,
            is_target_verify=lambda: False,
            is_draft_extend_v2=lambda: False,
        )
        self.assertTrue(
            dsa_indexer._server_args_enable_glm_sm89_dsa_fallback(server_args)
        )
        self.assertFalse(
            dsa_backend._should_build_deep_gemm_mqa_metadata(
                server_args.enable_glm_dsa_sm89_fallback, fake_decode_mode
            )
        )

    def test_glm_sm89_fallback_disables_fused_topk_per_backend_instance(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        with envs.SGLANG_DSA_FUSE_TOPK.override(True):
            backend.use_glm_sm89_dsa_fallback = True
            self.assertFalse(backend._use_dsa_fuse_topk())

            backend.use_glm_sm89_dsa_fallback = False
            self.assertTrue(backend._use_dsa_fuse_topk())

    def test_glm_sm89_fallback_indexer_metadata_forces_unfused_topk(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_glm_sm89_dsa_fallback = True
        backend.hisparse_coordinator = None
        backend.forward_metadata = SimpleNamespace(
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )
        backend.dsa_topk_backend = DSATopKBackend.TORCH
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_prefill_impl = "fa3"
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with envs.SGLANG_DSA_FUSE_TOPK.override(True):
            metadata = backend.get_indexer_metadata(0, forward_batch)

        self.assertEqual(metadata.topk_backend, DSATopKBackend.TORCH)
        self.assertTrue(metadata.force_unfused_topk)

    def test_glm_sm89_fallback_fa3_dequantizes_fp8_kv_cache(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.num_splits = 0

        q_rope = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
        q_nope = torch.zeros(1, 1, 512, dtype=torch.bfloat16)
        fp8_kv_cache = torch.empty(1, 1, 656, dtype=torch.float8_e4m3fn)
        dequantized_kv_cache = torch.zeros(1, 1, 576, dtype=torch.bfloat16)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        cache_seqlens = torch.ones(1, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        def fake_flash_attn_with_kvcache(**kwargs):
            self.assertEqual(kwargs["q"].dtype, torch.bfloat16)
            self.assertEqual(kwargs["qv"].dtype, torch.bfloat16)
            self.assertEqual(kwargs["k_cache"].dtype, torch.bfloat16)
            self.assertEqual(kwargs["v_cache"].dtype, torch.bfloat16)
            self.assertEqual(kwargs["k_cache"].shape[-1], 64)
            self.assertEqual(kwargs["v_cache"].shape[-1], 512)
            return torch.zeros_like(kwargs["qv"])

        with (
            patch(
                "sglang.srt.layers.attention.dsa_backend.dequantize_k_cache",
                return_value=dequantized_kv_cache,
            ) as mock_dequantize,
            patch(
                "sglang.srt.layers.attention.dsa_backend.flash_attn_with_kvcache",
                side_effect=fake_flash_attn_with_kvcache,
            ),
        ):
            backend._forward_fa3(
                q_rope=q_rope,
                kv_cache=fp8_kv_cache,
                v_head_dim=512,
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

        mock_dequantize.assert_called_once_with(fp8_kv_cache)

    def test_glm_sm89_fallback_fa3_routes_to_torch_mla(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_glm_sm89_dsa_fallback = True
        backend.device_sm_major = 8
        sentinel = object()
        backend._forward_torch_mla = MagicMock(return_value=sentinel)

        q_rope = torch.zeros(1, 1, 2)
        q_nope = torch.zeros(1, 1, 2)
        kv_cache = torch.zeros(1, 1, 4)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        cache_seqlens = torch.ones(1, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        with patch(
            "sglang.srt.layers.attention.dsa_backend.flash_attn_with_kvcache"
        ) as mock_fa3:
            result = backend._forward_fa3(
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

        self.assertIs(result, sentinel)
        backend._forward_torch_mla.assert_called_once()
        mock_fa3.assert_not_called()

    def test_glm_sm89_torch_mla_matches_manual_attention(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        q_nope = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        q_rope = torch.tensor([[[0.5, 0.0]], [[0.0, 0.5]]])
        kv_cache = torch.tensor(
            [
                [[1.0, 0.0, 1.0, 0.0]],
                [[0.0, 1.0, 0.0, 1.0]],
                [[1.0, 1.0, 0.5, 0.5]],
            ]
        )
        page_table = torch.tensor([[0, 1, -1], [2, -1, -1]], dtype=torch.int32)
        cache_seqlens = torch.tensor([2, 1], dtype=torch.int32)

        actual = backend._forward_torch_mla(
            q_rope=q_rope,
            kv_cache=kv_cache,
            v_head_dim=2,
            q_nope=q_nope,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=1.0,
            logit_cap=0.0,
            page_size=1,
        )

        expected_rows = []
        q_all = torch.cat([q_nope, q_rope], dim=-1)
        for i in range(q_all.shape[0]):
            indices = page_table[i, : cache_seqlens[i].item()].to(torch.long)
            selected = kv_cache[indices].squeeze(1)
            scores = torch.matmul(q_all[i], selected.transpose(0, 1))
            probs = torch.softmax(scores, dim=-1)
            expected_rows.append(torch.matmul(probs, selected[:, :2]))
        expected = torch.stack(expected_rows, dim=0)

        torch.testing.assert_close(actual, expected)

    def test_regular_dsa_indexer_metadata_keeps_fused_topk_enabled(self):
        from sglang.srt.layers.attention import dsa_backend
        from sglang.srt.layers.attention.dsa.dsa_topk_backend import DSATopKBackend
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_glm_sm89_dsa_fallback = False
        backend.hisparse_coordinator = None
        backend.forward_metadata = SimpleNamespace(
            paged_mqa_schedule_metadata=None,
            paged_mqa_ctx_lens_2d=None,
        )
        backend.dsa_topk_backend = DSATopKBackend.SGL_KERNEL
        backend.dsa_kv_cache_store_fp8 = True
        backend.dsa_prefill_impl = "fa3"
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        with envs.SGLANG_DSA_FUSE_TOPK.override(True):
            metadata = backend.get_indexer_metadata(0, forward_batch)

        self.assertEqual(metadata.topk_backend, DSATopKBackend.SGL_KERNEL)
        self.assertFalse(metadata.force_unfused_topk)

    def _call_glm_sm89_fallback_selector(
        self,
        dsa_indexer,
        fake_indexer,
        fake_q,
        fake_weights,
        fake_forward_batch,
        fake_metadata,
        layer_id,
    ):
        kwargs = {"layer_id": layer_id}
        if "metadata" in inspect.signature(
            dsa_indexer._select_glm_sm89_fallback_topk
        ).parameters:
            kwargs["metadata"] = fake_metadata
        return dsa_indexer._select_glm_sm89_fallback_topk(
            fake_indexer,
            fake_q,
            fake_weights,
            fake_forward_batch,
            **kwargs,
        )

    def test_glm_sm89_forward_cuda_passes_resolved_metadata_to_selector(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        indexer = object.__new__(dsa_indexer.Indexer)
        torch.nn.Module.__init__(indexer)
        indexer.alt_stream = None
        indexer.dsa_enable_prefill_cp = False
        indexer.use_glm_sm89_dsa_fallback = True
        indexer.block_size = 128
        indexer.scale_fmt = None
        indexer.n_heads = 32
        indexer.weights_proj = SimpleNamespace()

        query = object()
        key = object()
        q_fp8 = object()
        q_scale = object()
        weights = object()
        indexer._get_q_k_bf16 = MagicMock(return_value=(query, key))
        indexer._store_index_k_cache = MagicMock()
        indexer._get_logits_head_gate = MagicMock(return_value=weights)

        x = torch.empty(1, 16)
        q_lora = torch.empty(1, 16)
        positions = torch.zeros(1, dtype=torch.int64)
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.DECODE)
        metadata = object()
        attn_backend = SimpleNamespace(
            get_indexer_metadata=MagicMock(return_value=metadata)
        )
        expected = object()

        with (
            patch.object(
                dsa_indexer, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            patch.object(dsa_indexer, "get_attn_backend", return_value=attn_backend),
            patch(
                "sglang.srt.layers.attention.dsa.triton_kernel.act_quant",
                return_value=(q_fp8, q_scale),
            ),
            patch.object(
                dsa_indexer,
                "_select_glm_sm89_fallback_topk",
                return_value=expected,
            ) as mock_select,
        ):
            result = indexer.forward_cuda(
                x,
                q_lora,
                positions,
                forward_batch,
                layer_id=7,
            )

        self.assertIs(result, expected)
        mock_select.assert_called_once_with(
            indexer,
            q_fp8,
            weights,
            forward_batch,
            metadata,
            7,
        )

    def test_glm_sm89_fallback_indexer_routes_decode_like_modes_to_paged(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        for forward_mode in (
            ForwardMode.DECODE,
            ForwardMode.IDLE,
            ForwardMode.TARGET_VERIFY,
            ForwardMode.DRAFT_EXTEND_V2,
        ):
            with self.subTest(forward_mode=forward_mode):
                paged_topk = object()
                eager_topk = object()
                fake_q = MagicMock()
                fake_q.contiguous.return_value = "q-contiguous"
                fake_weights = object()
                fake_forward_batch = SimpleNamespace(forward_mode=forward_mode)
                fake_metadata = object()
                fake_indexer = SimpleNamespace(
                    index_topk=64,
                    _get_topk_paged=MagicMock(return_value=paged_topk),
                    forward_indexer=MagicMock(return_value=eager_topk),
                )

                with (
                    patch.object(
                        dsa_indexer,
                        "is_in_tc_piecewise_cuda_graph",
                        return_value=False,
                    ),
                    patch.object(
                        dsa_indexer,
                        "_broadcast_indexer_topk_from_rank0",
                        side_effect=lambda topk: topk,
                    ),
                    patch.object(
                        dsa_indexer,
                        "maybe_capture_indexer_topk",
                        side_effect=lambda _, topk: topk,
                    ),
                ):
                    result = self._call_glm_sm89_fallback_selector(
                        dsa_indexer,
                        fake_indexer,
                        fake_q,
                        fake_weights,
                        fake_forward_batch,
                        fake_metadata,
                        layer_id=7,
                    )

                self.assertIs(result, paged_topk)
                fake_indexer._get_topk_paged.assert_called_once_with(
                    fake_forward_batch,
                    7,
                    fake_q,
                    fake_weights,
                    fake_metadata,
                )
                fake_indexer.forward_indexer.assert_not_called()

    def test_glm_sm89_fallback_indexer_keeps_eager_prefill(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        eager_topk = object()
        fake_q = MagicMock()
        fake_q.contiguous.return_value = "q-contiguous"
        fake_weights = object()
        fake_forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        fake_metadata = object()
        fake_indexer = SimpleNamespace(
            index_topk=64,
            _get_topk_paged=MagicMock(),
            forward_indexer=MagicMock(return_value=eager_topk),
        )

        with (
            patch.object(
                dsa_indexer, "is_in_tc_piecewise_cuda_graph", return_value=False
            ),
            patch.object(
                dsa_indexer,
                "_broadcast_indexer_topk_from_rank0",
                side_effect=lambda topk: topk,
            ),
            patch.object(
                dsa_indexer,
                "maybe_capture_indexer_topk",
                side_effect=lambda _, topk: topk,
            ),
        ):
            result = self._call_glm_sm89_fallback_selector(
                dsa_indexer,
                fake_indexer,
                fake_q,
                fake_weights,
                fake_forward_batch,
                fake_metadata,
                layer_id=7,
            )

        self.assertIs(result, eager_topk)
        fake_indexer.forward_indexer.assert_called_once_with(
            "q-contiguous",
            fake_weights,
            fake_forward_batch,
            topk=64,
            layer_id=7,
        )
        fake_indexer._get_topk_paged.assert_not_called()

    def test_glm_sm89_fallback_indexer_rejects_piecewise_cuda_graph(self):
        from sglang.srt.layers.attention.dsa import dsa_indexer

        fake_indexer = object()

        with patch.object(
            dsa_indexer, "is_in_tc_piecewise_cuda_graph", return_value=True
        ), self.assertRaisesRegex(RuntimeError, "TC-piecewise CUDA graph"):
            self._call_glm_sm89_fallback_selector(
                dsa_indexer,
                fake_indexer,
                object(),
                object(),
                object(),
                None,
                layer_id=0,
            )

    def test_glm_sm89_hisparse_fails_fast(self):
        server_args = ServerArgs(
            model_path="dummy",
            enable_hisparse=True,
            disable_radix_cache=True,
            kv_cache_dtype="fp8_e4m3",
        )

        with self.assertRaisesRegex(ValueError, "SM89"):
            server_args._set_default_dsa_backends(
                kv_cache_dtype="fp8_e4m3",
                major=8,
                minor=9,
                model_arch="GlmMoeDsaForCausalLM",
            )


class TestContextParallelServerArgs(CustomTestCase):
    def setUp(self):
        self.parser = server_args_module.argparse.ArgumentParser()
        ServerArgs.add_cli_args(self.parser)

    def _new_cp_args(self, **overrides):
        server_args = object.__new__(ServerArgs)
        defaults = dict(
            enable_prefill_context_parallel=False,
            enable_dsa_prefill_context_parallel=False,
            enable_prefill_cp=False,
            cp_strategy=None,
            model_path="instance://127.0.0.1:8000/dummy",
            dsa_prefill_cp_mode="round-robin-split",
            prefill_cp_mode="in-seq-split",
            attn_cp_size=1,
            tp_size=1,
            dp_size=1,
            moe_dp_size=1,
            ep_size=1,
            pp_size=1,
            enable_aiter_allreduce_fusion=False,
        )
        defaults.update(overrides)
        for key, value in defaults.items():
            setattr(server_args, key, value)
        return server_args

    def test_canonical_prefill_cp_cli_sets_unified_fields(self):
        args = self.parser.parse_args(
            ["--model", "dummy", "--enable-prefill-cp", "--cp-strategy", "interleave"]
        )

        self.assertTrue(args.enable_prefill_cp)
        self.assertEqual(args.cp_strategy, "interleave")

    def test_canonical_prefill_cp_requires_strategy(self):
        args = self.parser.parse_args(["--model", "dummy", "--enable-prefill-cp"])

        self.assertTrue(args.enable_prefill_cp)
        self.assertIsNone(args.cp_strategy)

        server_args = self._new_cp_args(
            enable_prefill_cp=args.enable_prefill_cp,
            cp_strategy=args.cp_strategy,
        )
        with self.assertRaisesRegex(ValueError, "--cp-strategy"):
            server_args._handle_context_parallelism()

    def test_deprecated_dsa_cp_mode_maps_to_unified_strategy(self):
        args = self.parser.parse_args(
            [
                "--model",
                "dummy",
                "--enable-dsa-prefill-context-parallel",
                "--dsa-prefill-cp-mode",
                "round-robin-split",
            ]
        )
        server_args = self._new_cp_args(
            enable_dsa_prefill_context_parallel=(
                args.enable_dsa_prefill_context_parallel
            ),
            dsa_prefill_cp_mode=args.dsa_prefill_cp_mode,
        )

        server_args._handle_legacy_cp_arguments()

        self.assertTrue(server_args.enable_prefill_cp)
        self.assertEqual(server_args.cp_strategy, "interleave")
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")

    def test_canonical_interleave_cp_mirrors_to_dsa_runtime_aliases(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attention_backend="dsa",
        )

        server_args._handle_legacy_cp_arguments()
        server_args._handle_context_parallelism()

        self.assertTrue(server_args.enable_dsa_prefill_context_parallel)
        self.assertFalse(server_args.enable_prefill_context_parallel)
        self.assertEqual(server_args.dsa_prefill_cp_mode, "round-robin-split")
        self.assertEqual(server_args.prefill_cp_mode, "round-robin-split")

    def test_context_parallel_handler_initializes_cp_strategy(self):
        server_args = self._new_cp_args(
            enable_prefill_cp=True,
            cp_strategy="interleave",
            attn_cp_size=2,
            tp_size=2,
        )

        server_args._handle_context_parallelism()

        self.assertTrue(is_cp_enabled())
        self.assertTrue(is_interleave())

    def test_registered_cp_legacy_args_map_to_unified_strategy(self):
        cases = [
            (
                "deepseek_v3_mla_cp",
                dict(enable_prefill_context_parallel=True),
                "zigzag",
                "in-seq-split",
                False,
                True,
            ),
            (
                "qwen3_gqa_cp",
                dict(
                    enable_prefill_context_parallel=True,
                    tp_size=4,
                    attn_cp_size=2,
                ),
                "zigzag",
                "in-seq-split",
                False,
                True,
            ),
            (
                "deepseek_v32_dsa_in_seq_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    dsa_prefill_cp_mode="in-seq-split",
                    tp_size=8,
                    dp_size=2,
                    attn_cp_size=4,
                ),
                "zigzag",
                "in-seq-split",
                True,
                False,
            ),
            (
                "deepseek_v32_dsa_round_robin_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    tp_size=8,
                    attn_cp_size=8,
                ),
                "interleave",
                "round-robin-split",
                True,
                False,
            ),
            (
                "deepseek_v4_flash_fp4_b200_dsa_round_robin_split",
                dict(
                    enable_dsa_prefill_context_parallel=True,
                    dsa_prefill_cp_mode="round-robin-split",
                    tp_size=4,
                    attn_cp_size=4,
                ),
                "interleave",
                "round-robin-split",
                True,
                False,
            ),
        ]

        for name, overrides, strategy, mode, expect_dsa, expect_generic in cases:
            with self.subTest(name=name):
                server_args = self._new_cp_args(**overrides)

                server_args._handle_legacy_cp_arguments()
                server_args._handle_context_parallelism()

                self.assertTrue(server_args.enable_prefill_cp)
                self.assertEqual(server_args.cp_strategy, strategy)
                self.assertEqual(server_args.dsa_prefill_cp_mode, mode)
                self.assertEqual(server_args.prefill_cp_mode, mode)
                self.assertEqual(
                    server_args.enable_dsa_prefill_context_parallel, expect_dsa
                )
                self.assertEqual(
                    server_args.enable_prefill_context_parallel, expect_generic
                )


class TestPortArgs(unittest.TestCase):
    @patch("sglang.srt.server_args.get_free_port")
    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_with_nccl_port_none(self, mock_temp_file, mock_get_free_port):
        """Test that get_free_port() is called when nccl_port is None"""
        mock_temp_file.return_value.name = "temp_file"
        mock_get_free_port.return_value = 45678  # Mock ephemeral port

        # Use MagicMock here to verify get_free_port is called
        server_args = MagicMock()
        server_args.nccl_port = None
        server_args.enable_dp_attention = False
        server_args.tokenizer_worker_num = 1

        port_args = PortArgs.init_new(server_args)

        # Verify get_free_port was called
        mock_get_free_port.assert_called_once()

        # Verify the returned port is used
        self.assertEqual(port_args.nccl_port, 45678)

    @patch("sglang.srt.server_args.tempfile.NamedTemporaryFile")
    def test_init_new_standard_case(self, mock_temp_file):
        mock_temp_file.return_value.name = "temp_file"

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = False

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.scheduler_input_ipc_name.startswith("ipc://"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("ipc://"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_single_node_dp_attention(self):

        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = None

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://127.0.0.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://127.0.0.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_dp_rank(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None
        server_args.enable_dp_attention = True
        server_args.nnodes = 1
        server_args.dist_init_addr = "192.168.1.1:25000"

        worker_ports = [25006, 25007, 25008, 25009]
        port_args = PortArgs.init_new(server_args, dp_rank=2, worker_ports=worker_ports)

        self.assertTrue(port_args.scheduler_input_ipc_name.endswith(":25008"))

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:25000"

        port_args = PortArgs.init_new(server_args)

        self.assertTrue(port_args.tokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertTrue(
            port_args.scheduler_input_ipc_name.startswith("tcp://192.168.1.1:")
        )
        self.assertTrue(port_args.detokenizer_ipc_name.startswith("tcp://192.168.1.1:"))
        self.assertIsInstance(port_args.nccl_port, int)

    def test_init_new_with_malformed_ipv4_address(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1"

        with self.assertRaises(ValueError) as context:
            PortArgs.init_new(server_args)

        self.assertIn("Missing port", str(context.exception))

    def test_init_new_with_malformed_ipv4_address_invalid_port(self):
        server_args = ServerArgs(model_path="dummy")
        server_args.port = 30000
        server_args.nccl_port = None

        server_args.enable_dp_attention = True
        server_args.nnodes = 2
        server_args.dist_init_addr = "192.168.1.1:abc"

        with self.assertRaises(ValueError):
            PortArgs.init_new(server_args)


class TestSSLArgs(unittest.TestCase):
    def test_default_ssl_fields_are_none(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertIsNone(server_args.ssl_keyfile)
        self.assertIsNone(server_args.ssl_certfile)
        self.assertIsNone(server_args.ssl_ca_certs)
        self.assertIsNone(server_args.ssl_keyfile_password)

    def test_ssl_keyfile_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(model_path="dummy", ssl_keyfile="key.pem")
        self.assertIn("--ssl-certfile", str(context.exception))

    def test_ssl_certfile_without_keyfile_raises(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(model_path="dummy", ssl_certfile="cert.pem")
        self.assertIn("--ssl-keyfile", str(context.exception))

    @patch("os.path.isfile", return_value=True)
    def test_ssl_both_keyfile_and_certfile_accepted(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy", ssl_keyfile="key.pem", ssl_certfile="cert.pem"
        )
        self.assertEqual(server_args.ssl_keyfile, "key.pem")
        self.assertEqual(server_args.ssl_certfile, "cert.pem")

    def test_url_returns_http_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertTrue(server_args.url().startswith("http://"))

    def test_url_rewrites_all_interfaces_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="0.0.0.0")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")

    def test_url_rewrites_empty_host_to_loopback(self):
        server_args = ServerArgs(model_path="dummy", host="")
        self.assertEqual(server_args.url(), "http://127.0.0.1:30000")

    @patch("os.path.isfile", return_value=True)
    def test_url_returns_https_with_ssl(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy", ssl_keyfile="key.pem", ssl_certfile="cert.pem"
        )
        self.assertTrue(server_args.url().startswith("https://"))

    @patch("os.path.isfile", return_value=True)
    def test_ssl_cli_args_parsed(self, _mock_isfile):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--ssl-keyfile",
                "key.pem",
                "--ssl-certfile",
                "cert.pem",
                "--ssl-ca-certs",
                "ca.pem",
                "--ssl-keyfile-password",
                "secret",
            ]
        )
        self.assertEqual(server_args.ssl_keyfile, "key.pem")
        self.assertEqual(server_args.ssl_certfile, "cert.pem")
        self.assertEqual(server_args.ssl_ca_certs, "ca.pem")
        self.assertEqual(server_args.ssl_keyfile_password, "secret")

    def test_ssl_verify_without_ssl(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertIs(server_args.ssl_verify(), True)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_no_ca(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy", ssl_keyfile="key.pem", ssl_certfile="cert.pem"
        )
        self.assertIs(server_args.ssl_verify(), False)

    @patch("os.path.isfile", return_value=True)
    def test_ssl_verify_with_ssl_and_ca(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy",
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            ssl_ca_certs="ca.pem",
        )
        self.assertEqual(server_args.ssl_verify(), "ca.pem")

    def test_ssl_ca_certs_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(model_path="dummy", ssl_ca_certs="ca.pem")
        self.assertIn("--ssl-ca-certs", str(context.exception))

    def test_ssl_keyfile_password_without_certfile_raises(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(model_path="dummy", ssl_keyfile_password="secret")
        self.assertIn("--ssl-keyfile-password", str(context.exception))

    def test_ssl_keyfile_not_found_raises(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(
                model_path="dummy",
                ssl_keyfile="/nonexistent/key.pem",
                ssl_certfile="/nonexistent/cert.pem",
            )
        self.assertIn("not found", str(context.exception))

    def test_ssl_certfile_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with self.assertRaises(ValueError) as context:
                ServerArgs(
                    model_path="dummy",
                    ssl_keyfile=keyfile.name,
                    ssl_certfile="/nonexistent/cert.pem",
                )
            self.assertIn("SSL certificate file not found", str(context.exception))

    def test_ssl_ca_certs_not_found_raises(self):
        with tempfile.NamedTemporaryFile(suffix=".pem") as keyfile:
            with tempfile.NamedTemporaryFile(suffix=".pem") as certfile:
                with self.assertRaises(ValueError) as context:
                    ServerArgs(
                        model_path="dummy",
                        ssl_keyfile=keyfile.name,
                        ssl_certfile=certfile.name,
                        ssl_ca_certs="/nonexistent/ca.pem",
                    )
                self.assertIn(
                    "SSL CA certificates file not found", str(context.exception)
                )

    def test_enable_ssl_refresh_default_false(self):
        server_args = ServerArgs(model_path="dummy")
        self.assertFalse(server_args.enable_ssl_refresh)

    def test_enable_ssl_refresh_without_ssl_raises(self):
        with self.assertRaises(ValueError) as context:
            ServerArgs(model_path="dummy", enable_ssl_refresh=True)
        self.assertIn("--enable-ssl-refresh", str(context.exception))
        self.assertIn("--ssl-certfile", str(context.exception))

    @patch("os.path.isfile", return_value=True)
    def test_enable_ssl_refresh_with_ssl_accepted(self, _mock_isfile):
        server_args = ServerArgs(
            model_path="dummy",
            ssl_keyfile="key.pem",
            ssl_certfile="cert.pem",
            enable_ssl_refresh=True,
        )
        self.assertTrue(server_args.enable_ssl_refresh)

    @patch("os.path.isfile", return_value=True)
    def test_enable_ssl_refresh_cli_flag(self, _mock_isfile):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--ssl-keyfile",
                "key.pem",
                "--ssl-certfile",
                "cert.pem",
                "--enable-ssl-refresh",
            ]
        )
        self.assertTrue(server_args.enable_ssl_refresh)


class TestHiCacheArgs(unittest.TestCase):
    def _make_args(self, **overrides) -> ServerArgs:
        args = ServerArgs(model_path="dummy")
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def _assert_hicache_fields(
        self,
        args: ServerArgs,
        *,
        expected_io_backend: str,
        expected_mem_layout: str,
        expected_decode_backend: str | None = None,
    ):
        self.assertEqual(args.hicache_io_backend, expected_io_backend)
        self.assertEqual(args.hicache_mem_layout, expected_mem_layout)
        if expected_decode_backend is not None:
            self.assertEqual(args.decode_attention_backend, expected_decode_backend)

    def test_hicache_io_backend_and_mem_layout_compatibility(self):
        cases = [
            {
                "name": "default_kernel_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                },
                "expected_io_backend": "kernel",
                "expected_mem_layout": "page_first",
            },
            {
                "name": "kernel_with_page_first_direct",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first_direct",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "direct_with_page_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "page_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "mooncake_with_layer_first",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_storage_backend": "mooncake",
                    "hicache_io_backend": "direct",
                    "hicache_mem_layout": "layer_first",
                },
                "expected_io_backend": "direct",
                "expected_mem_layout": "page_first_direct",
            },
            {
                "name": "fa3_kernel_with_explicit_decode_backend",
                "overrides": {
                    "enable_hierarchical_cache": True,
                    "hicache_io_backend": "kernel",
                    "hicache_mem_layout": "page_first",
                    "attention_backend": "triton",
                    "decode_attention_backend": "fa3",
                },
                "expected_io_backend": "kernel",
                "expected_mem_layout": "page_first",
                "expected_decode_backend": "fa3",
            },
        ]

        for case in cases:
            with self.subTest(case=case["name"]):
                args = self._make_args(**case["overrides"])
                args._handle_hicache()
                self._assert_hicache_fields(
                    args,
                    expected_io_backend=case["expected_io_backend"],
                    expected_mem_layout=case["expected_mem_layout"],
                    expected_decode_backend=case.get("expected_decode_backend"),
                )

    def test_hicache_kernel_keeps_implicit_fa3_decode_backend(self):
        args = self._make_args(
            enable_hierarchical_cache=True,
            hicache_io_backend="kernel",
            attention_backend="fa3",
            decode_attention_backend=None,
        )

        args._handle_hicache()

        self.assertEqual(args.hicache_io_backend, "kernel")
        self.assertEqual(args.hicache_mem_layout, "page_first")
        self.assertIsNone(args.decode_attention_backend)


class TestNgramExternalSamArgs(CustomTestCase):
    def test_prepare_server_args_parses_external_sam_args(self):
        server_args = prepare_server_args(
            [
                "--model-path",
                "dummy",
                "--speculative-algorithm",
                "NGRAM",
                "--speculative-ngram-external-corpus-path",
                "/tmp/ngram-corpus.jsonl",
                "--speculative-ngram-external-sam-budget",
                "4",
                "--speculative-ngram-external-corpus-max-tokens",
                "128",
            ]
        )
        self.assertEqual(
            server_args.speculative_ngram_external_corpus_path,
            "/tmp/ngram-corpus.jsonl",
        )
        self.assertEqual(server_args.speculative_ngram_external_sam_budget, 4)
        self.assertEqual(server_args.speculative_ngram_external_corpus_max_tokens, 128)

    def _make_dummy_ngram_args(self, **overrides):
        args = ServerArgs(model_path="dummy")
        args.speculative_algorithm = "NGRAM"
        args.speculative_num_draft_tokens = 12
        args.device = "cuda"
        for key, value in overrides.items():
            setattr(args, key, value)
        return args

    def test_external_sam_budget_must_fit_draft_budget(self):
        args = self._make_dummy_ngram_args(
            speculative_num_draft_tokens=4,
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=4,
        )
        with self.assertRaises(ValueError) as context:
            handle_speculative_decoding(args)
        self.assertIn("speculative_num_draft_tokens - 1", str(context.exception))

    def test_external_corpus_max_tokens_must_be_positive(self):
        args = self._make_dummy_ngram_args(
            speculative_ngram_external_corpus_path="/tmp/ngram-corpus.jsonl",
            speculative_ngram_external_sam_budget=2,
            speculative_ngram_external_corpus_max_tokens=0,
        )
        with self.assertRaises(ValueError) as context:
            handle_speculative_decoding(args)
        self.assertIn("external-corpus-max-tokens", str(context.exception))


class TestAdaptiveSpecArgs(CustomTestCase):
    def test_adaptive_defaults_to_config_step_when_spec_params_omitted(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json") as f:
            json.dump(
                {
                    "1": {"candidate_steps": [1, 3, 5]},
                    "8": {"candidate_steps": [1]},
                },
                f,
            )
            f.flush()

            args = ServerArgs(model_path="dummy")
            args.speculative_algorithm = "EAGLE"
            args.speculative_adaptive = True
            args.speculative_adaptive_config = f.name
            args.device = "cuda"
            args.get_model_config = lambda: SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["LlamaForCausalLM"],
                    get_text_config=lambda: SimpleNamespace(),
                )
            )

            handle_speculative_decoding(args)

        self.assertTrue(args.speculative_adaptive)
        self.assertEqual(args.speculative_eagle_topk, 1)
        self.assertEqual(args.speculative_num_steps, 3)
        self.assertEqual(args.speculative_num_draft_tokens, 4)


class TestDeepEPWaterfillArgs(CustomTestCase):
    def test_waterfill_enforces_shared_experts_fusion(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_deepep_waterfill=True,
            disable_shared_experts_fusion=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertFalse(server_args.disable_shared_experts_fusion)
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_overrides_moe_a2a_backend_to_deepep(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="none",
            enable_deepep_waterfill=True,
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertEqual(server_args.moe_a2a_backend, "deepep")
        self.assertTrue(server_args.enforce_shared_experts_fusion)

    def test_waterfill_supports_deepep_low_latency_mode(self):
        server_args = ServerArgs(
            model_path="dummy",
            moe_a2a_backend="deepep",
            enable_deepep_waterfill=True,
            deepep_mode="low_latency",
        )
        # dummy-model path short-circuits __post_init__; invoke the handler directly.
        server_args._handle_a2a_moe()

        self.assertEqual(server_args.deepep_mode, "low_latency")
        self.assertFalse(server_args.disable_cuda_graph)
        self.assertTrue(server_args.enforce_shared_experts_fusion)


class TestPrefillOnlyDisableKvCache(unittest.TestCase):
    """Validation for --prefill-only-disable-kv-cache.

    The flag wires NoOpMHATokenToKVPool, which is only safe when:
      - the engine is in embedding mode (fa_skip_kv_cache active in FA backend),
      - chunked_prefill_size == -1 (no inter-chunk K/V reuse),
      - disable_radix_cache (radix cache otherwise indexes empty pool slots),
      - no context-parallel attention (CP writes to the pool via set_kv_buffer),
      - no HiSparse (uses a different pool family),
      - kv_cache_dtype != fp4_e2m1 (FP4 pool is a separate allocation path).
    All other configurations must be rejected at __post_init__ time so users
    get a clear error before model load.
    """

    def _base_kwargs(self, **overrides):
        kwargs = dict(
            model_path="dummy",
            is_embedding=True,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            prefill_only_disable_kv_cache=True,
        )
        kwargs.update(overrides)
        return kwargs

    def test_valid_minimal_config_constructs(self):
        sa = ServerArgs(**self._base_kwargs())
        self.assertTrue(sa.prefill_only_disable_kv_cache)

    def test_rejects_when_not_embedding(self):
        with self.assertRaisesRegex(ValueError, "requires --is-embedding"):
            ServerArgs(**self._base_kwargs(is_embedding=False))

    def test_rejects_when_chunked_prefill_size_not_minus_one(self):
        with self.assertRaisesRegex(ValueError, "--chunked-prefill-size=-1"):
            ServerArgs(**self._base_kwargs(chunked_prefill_size=8192))

    def test_rejects_when_radix_cache_enabled(self):
        with self.assertRaisesRegex(ValueError, "--disable-radix-cache"):
            ServerArgs(**self._base_kwargs(disable_radix_cache=False))

    def test_rejects_attn_cp_size_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "--attn-cp-size"):
            ServerArgs(**self._base_kwargs(attn_cp_size=2, tp_size=2))

    def test_rejects_prefill_context_parallel(self):
        with self.assertRaisesRegex(ValueError, "--enable-prefill-cp"):
            ServerArgs(**self._base_kwargs(enable_prefill_context_parallel=True))

    def test_rejects_hisparse(self):
        with self.assertRaisesRegex(ValueError, "--enable-hisparse"):
            ServerArgs(**self._base_kwargs(enable_hisparse=True))

    def test_rejects_fp4_kv_cache(self):
        with self.assertRaisesRegex(ValueError, "fp4_e2m1"):
            ServerArgs(**self._base_kwargs(kv_cache_dtype="fp4_e2m1"))


class TestCudaGraphConfigDataclassAccess(CustomTestCase):
    @patch(
        "sglang.srt.model_executor.runner_backend."
        "tc_piecewise_cuda_graph_backend.get_moe_a2a_backend"
    )
    def test_tc_piecewise_build_config_reads_phase_config_dataclass(
        self, mock_get_moe_a2a_backend
    ):
        from sglang.srt.model_executor.runner_backend.tc_piecewise_cuda_graph_backend import (
            TcPiecewiseCudaGraphBackend,
        )

        mock_backend = mock_get_moe_a2a_backend.return_value
        mock_backend.is_deepep.return_value = False
        mock_backend.is_mooncake.return_value = False
        server_args = SimpleNamespace(
            cuda_graph_config=CudaGraphConfig(
                prefill=PhaseConfig(
                    backend=Backend.TC_PIECEWISE,
                    bs=[32, 64],
                    tc_compiler="eager",
                )
            ),
            enable_torch_compile_debug_mode=False,
        )

        config = TcPiecewiseCudaGraphBackend.build_compilation_config(server_args)

        self.assertEqual(config.get_capture_sizes(), [32, 64])
        self.assertEqual(config.compiler, "eager")


class TestCutedslMoeMaxNumTokens(CustomTestCase):
    """The shared CuteDSL MoE per-forward token bound. Fields are set directly
    to exercise the math independently of __post_init__ resolution.

    cg-refactor: the legacy disable_piecewise_cuda_graph /
    piecewise_cuda_graph_max_tokens / cuda_graph_max_bs fields were
    consolidated into cuda_graph_config; the helper accepts the legacy
    kwarg names for test readability and translates them to the per-phase
    dataclasses.
    """

    def _args(self, **overrides):
        server_args = ServerArgs(model_path="dummy")
        fields = dict(
            speculative_algorithm=None,
            speculative_num_draft_tokens=None,
            max_prefill_tokens=16384,
            disable_piecewise_cuda_graph=False,
            piecewise_cuda_graph_max_tokens=2048,
            cuda_graph_max_bs=512,
        )
        fields.update(overrides)
        disable_piecewise = fields.pop("disable_piecewise_cuda_graph")
        piecewise_max = fields.pop("piecewise_cuda_graph_max_tokens")
        cg_max_bs = fields.pop("cuda_graph_max_bs")
        for key, value in fields.items():
            setattr(server_args, key, value)
        server_args.cuda_graph_config = CudaGraphConfig(
            decode=PhaseConfig(backend=Backend.FULL, max_bs=cg_max_bs),
            prefill=PhaseConfig(
                backend=(
                    Backend.DISABLED if disable_piecewise else Backend.TC_PIECEWISE
                ),
                max_bs=piecewise_max,
                tc_compiler="eager",
            ),
        )
        return server_args

    def test_prefill_dominates_in_default_config(self):
        self.assertEqual(self._args().cutedsl_moe_max_num_tokens(), 16384)

    def test_speculative_decoding_scales_decode_bound(self):
        # decode bound 512 * 8 dominates the small prefill/piecewise bounds
        args = self._args(
            max_prefill_tokens=512,
            piecewise_cuda_graph_max_tokens=512,
            speculative_algorithm="EAGLE",
            speculative_num_draft_tokens=8,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 4096)

    def test_piecewise_bound_excluded_when_disabled(self):
        args = self._args(
            max_prefill_tokens=512,
            disable_piecewise_cuda_graph=True,
            cuda_graph_max_bs=64,
        )
        self.assertEqual(args.cutedsl_moe_max_num_tokens(), 512)


class TestSamplingBackendTokenOracleEnvGate(CustomTestCase):
    """The 'token_oracle' choice is gated on SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE.

    The choice set is built once at server_args.py import time, so each subtest
    reloads the module with the env var set to the desired value.
    """

    def _reload_server_args_with_env(self, *, enabled: bool):
        previous = os.environ.get("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE")
        os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = "1" if enabled else "0"
        try:
            return importlib.reload(server_args_module)
        finally:
            if previous is None:
                os.environ.pop("SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE", None)
            else:
                os.environ["SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE"] = previous

    def test_token_oracle_rejected_when_env_disabled(self):
        reloaded = self._reload_server_args_with_env(enabled=False)
        self.assertNotIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)

        with self.assertRaises(SystemExit):
            reloaded.prepare_server_args(
                [
                    "--model-path",
                    DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                    "--sampling-backend",
                    "token_oracle",
                ]
            )

    def test_token_oracle_accepted_when_env_enabled(self):
        reloaded = self._reload_server_args_with_env(enabled=True)
        self.assertIn("token_oracle", reloaded.SAMPLING_BACKEND_CHOICES)

        parsed = reloaded.prepare_server_args(
            [
                "--model-path",
                DEFAULT_SMALL_MODEL_NAME_FOR_TEST_QWEN,
                "--sampling-backend",
                "token_oracle",
                # Explicit device so ServerArgs.__post_init__ does not call
                # get_device() (fails on CPU-only CI runners) and does not run
                # _handle_cpu_backends (which would override sampling_backend
                # to "pytorch", masking what we want to verify).
                "--device",
                "cuda",
            ]
        )
        self.assertEqual(parsed.sampling_backend, "token_oracle")


if __name__ == "__main__":
    unittest.main()
