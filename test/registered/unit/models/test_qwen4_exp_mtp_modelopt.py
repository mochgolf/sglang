"""Qwen4/Qwen3.5 MTP ModelOpt routing uses the real module prefix."""

import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.hyperconnection import GatedResidual
from sglang.srt.layers.linear import ReplicatedLinear
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod
from sglang.srt.layers.quantization.modelopt_quant import (
    _Fp8LmHeadLinearMethod,
    ModelOptFp4Config,
    ModelOptMixedPrecisionConfig,
    ModelOptNvFp4FusedMoEMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.runner.eager_runner import _use_stable_prefill
from sglang.srt.models import qwen4_exp
from sglang.srt.models.qwen3_5_mtp import (
    _MTP_ROUTED_EXPERTS_PREFIX,
    _mtp_quant_config,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase


register_cpu_ci(est_time=5, suite="base-a-test-cpu")


_PACKED_MODULES_MAPPING = {
    "qkv_proj": ["q_proj", "k_proj", "v_proj"],
    "gate_up_proj": ["gate_proj", "up_proj"],
}


def _serialized_fp4_config(exclude_modules):
    return ModelOptFp4Config.from_config(
        {
            "quantization": {
                "quant_algo": "NVFP4",
                "kv_cache_quant_algo": None,
                "exclude_modules": exclude_modules,
            },
            "group_size": 16,
            "packed_modules_mapping": _PACKED_MODULES_MAPPING,
        }
    )


def _mixed_config(exclude_modules, *, with_mtp_experts):
    quantized_layers = {}
    if with_mtp_experts:
        for projection in ("gate_proj", "up_proj", "down_proj"):
            quantized_layers[
                f"{_MTP_ROUTED_EXPERTS_PREFIX}.0.{projection}"
            ] = {"quant_algo": "W4A16_NVFP4", "group_size": 16}
    return ModelOptMixedPrecisionConfig.from_config(
        {
            "quantization": {
                "quant_algo": "MIXED_PRECISION",
                "kv_cache_quant_algo": None,
                "exclude_modules": exclude_modules,
                "quantized_layers": quantized_layers
                or {
                    "model.layers.0.mlp.gate_proj": {"quant_algo": "NVFP4"}
                },
            },
            "packed_modules_mapping": _PACKED_MODULES_MAPPING,
        }
    )


class TestQwen4ExpMtpModelOpt(CustomTestCase):
    def test_stable_hc_mix_falls_back_when_fused_path_is_unsupported(self):
        layer = GatedResidual.__new__(GatedResidual)
        torch.nn.Module.__init__(layer)
        layer.hc_count = 2
        layer.hidden_size = 4
        layer.params_dtype = torch.float32
        layer.config = SimpleNamespace(hc_per_branch_norm=True)
        layer.hc_norm = torch.nn.Identity()
        layer.input_mix_weight_down = torch.nn.Linear(8, 2, bias=False)
        layer.input_mix_weight_up = torch.nn.Linear(2, 8, bias=False)
        layer._jit_mix_ok = False
        expected = torch.ones((3, 4))
        layer._mix_compute = Mock(return_value=expected)

        with patch(
            "sglang.srt.layers.hyperconnection.fused_hc_mix_supported",
            return_value=False,
        ):
            got, _ = layer.mix(torch.ones((3, 8)), stable=True)
        self.assertIs(got, expected)
        layer._mix_compute.assert_called_once()

    def test_stable_prefill_uses_forward_scope(self):
        batch = SimpleNamespace()
        self.assertFalse(qwen4_exp._stable_prefill_hc(batch))
        with forward_context(
            ForwardContext(attn_backend=SimpleNamespace(), stable_prefill=True)
        ):
            self.assertTrue(qwen4_exp._stable_prefill_hc(batch))

    def test_stable_prefill_uses_original_target_mode(self):
        target = SimpleNamespace(is_draft_worker=False)
        with patch.object(
            envs.SGLANG_STABLE_PREFILL, "get", return_value=True
        ):
            self.assertTrue(
                _use_stable_prefill(
                    SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        _original_forward_mode=None,
                    ),
                    target,
                )
            )
            for original_mode in (
                ForwardMode.DECODE,
                ForwardMode.TARGET_VERIFY,
                ForwardMode.DRAFT_EXTEND_V2,
            ):
                with self.subTest(original_mode=original_mode):
                    self.assertFalse(
                        _use_stable_prefill(
                            SimpleNamespace(
                                forward_mode=ForwardMode.EXTEND,
                                _original_forward_mode=original_mode,
                            ),
                            target,
                        )
                    )
            self.assertFalse(
                _use_stable_prefill(
                    SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        _original_forward_mode=None,
                    ),
                    SimpleNamespace(is_draft_worker=True),
                )
            )
            self.assertFalse(
                _use_stable_prefill(
                    SimpleNamespace(
                        forward_mode=ForwardMode.EXTEND,
                        _original_forward_mode=None,
                    ),
                    target,
                    cp_v2_active=True,
                )
            )

    def test_online_fp8_lm_head_forces_marlin_without_global_override(self):
        config = _serialized_fp4_config(["lm_head"])
        layer = ParallelLMHead.__new__(ParallelLMHead)

        def init_without_marlin(method, quant_config):
            method.quant_config = quant_config
            method.use_marlin = False

        with (
            patch.dict(
                os.environ,
                {
                    "SGLANG_QWEN38_FP8_LM_HEAD": "1",
                    "SGLANG_FORCE_FP8_MARLIN": "0",
                },
            ),
            patch.object(Fp8LinearMethod, "__init__", init_without_marlin),
        ):
            method = config.get_quant_method(layer, "lm_head")

        self.assertIsInstance(method, _Fp8LmHeadLinearMethod)
        self.assertTrue(method.use_marlin)

    def test_stock_broad_mtp_ignore_stays_unquantized(self):
        config = _mixed_config(["mtp*", "mtp.layers.0*"], with_mtp_experts=False)

        self.assertIsNone(_mtp_quant_config(config))

    def test_v1_keeps_modelopt_for_routed_experts_only(self):
        config = _mixed_config(
            [
                "mtp.layers.0.self_attn.*",
                "mtp.layers.0.mlp.gate",
                "mtp.layers.0.mlp.shared_expert*",
                "model.shared_head.head",
            ],
            with_mtp_experts=True,
        )

        self.assertIs(_mtp_quant_config(config), config)
        self.assertFalse(config.is_layer_excluded(_MTP_ROUTED_EXPERTS_PREFIX))

        for prefix in (
            "mtp.layers.0.self_attn.qkv_proj",
            "mtp.layers.0.mlp.gate",
            "model.shared_head.head",
        ):
            with self.subTest(prefix=prefix):
                layer = (
                    ParallelLMHead.__new__(ParallelLMHead)
                    if prefix == "model.shared_head.head"
                    else ReplicatedLinear.__new__(ReplicatedLinear)
                )
                self.assertIsInstance(
                    config.get_quant_method(layer, prefix), UnquantizedLinearMethod
                )

    def test_serialized_stock_broad_mtp_ignore_stays_unquantized(self):
        config = _serialized_fp4_config(["mtp*", "mtp.layers.0*"])

        self.assertTrue(config.is_checkpoint_nvfp4_serialized)
        self.assertIsNone(_mtp_quant_config(config))

    def test_serialized_v1_keeps_expert_quantization_only(self):
        config = _serialized_fp4_config(
            [
                "mtp.layers.0.self_attn.*",
                "mtp.layers.0.mlp.gate",
                "mtp.layers.0.mlp.shared_expert*",
                "model.shared_head.head",
            ]
        )

        self.assertIs(_mtp_quant_config(config), config)
        self.assertFalse(config.is_layer_excluded(_MTP_ROUTED_EXPERTS_PREFIX))
        for prefix in (
            "mtp.layers.0.self_attn.qkv_proj",
            "mtp.layers.0.mlp.gate",
            "model.shared_head.head",
        ):
            with self.subTest(prefix=prefix):
                layer = (
                    ParallelLMHead.__new__(ParallelLMHead)
                    if prefix == "model.shared_head.head"
                    else ReplicatedLinear.__new__(ReplicatedLinear)
                )
                self.assertIsInstance(
                    config.get_quant_method(layer, prefix), UnquantizedLinearMethod
                )

    def test_exact_mtp_fused_moe_prefix_selects_modelopt_nvfp4(self):
        config = _mixed_config([], with_mtp_experts=True)
        layer = FusedMoE.__new__(FusedMoE)

        # The constructor checks the active CUDA/Marlin backend; this unit only
        # needs to prove dispatch selects the ModelOpt method for the exact path.
        with patch.object(ModelOptNvFp4FusedMoEMethod, "__init__", return_value=None):
            method = config.get_quant_method(layer, _MTP_ROUTED_EXPERTS_PREFIX)

        self.assertIsInstance(method, ModelOptNvFp4FusedMoEMethod)

    def test_serialized_exact_mtp_fused_moe_prefix_selects_modelopt_nvfp4(self):
        config = _serialized_fp4_config(
            [
                "mtp.layers.0.self_attn.*",
                "mtp.layers.0.mlp.gate",
                "mtp.layers.0.mlp.shared_expert*",
                "model.shared_head.head",
            ]
        )
        layer = FusedMoE.__new__(FusedMoE)

        with patch.object(ModelOptNvFp4FusedMoEMethod, "__init__", return_value=None):
            method = config.get_quant_method(layer, _MTP_ROUTED_EXPERTS_PREFIX)

        self.assertIsInstance(method, ModelOptNvFp4FusedMoEMethod)


if __name__ == "__main__":
    unittest.main()
