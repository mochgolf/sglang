"""Qwen4/Qwen3.5 MTP ModelOpt routing uses the real module prefix."""

import os
import unittest
from unittest.mock import patch

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
