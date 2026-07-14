import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear, QKVParallelLinear
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.parameter import PerTensorScaleParameter
from sglang.srt.layers.quantization.modelopt_quant import ModelOptNvFp4FusedMoEMethod
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestModelOptNvfp4(CustomTestCase):
    def _make_layer(self):
        return MergedColumnParallelLinear(
            input_size=16,
            output_sizes=[16, 16],
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def _make_qkv_layer(self):
        return QKVParallelLinear(
            hidden_size=16,
            head_size=8,
            total_num_heads=2,
            total_num_kv_heads=2,
            bias=False,
            tp_rank=0,
            tp_size=1,
        )

    def test_fused_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.25]))

    def test_fused_scalar_scale_load_rejects_non_scalar(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        with self.assertRaisesRegex(ValueError, "Expected scalar scale"):
            layer.weight_loader_v2(scale, torch.tensor([0.25, 0.5]))

    def test_fused_qkv_scalar_scale_load_fills_all_logical_slots(self):
        layer = self._make_qkv_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(3, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.125, dtype=torch.float32))

        torch.testing.assert_close(scale, torch.tensor([0.125, 0.125, 0.125]))

    def test_explicit_shard_scale_loads_stay_independent(self):
        layer = self._make_layer()
        scale = PerTensorScaleParameter(
            data=torch.empty(2, dtype=torch.float32),
            weight_loader=layer.weight_loader_v2,
        )

        layer.weight_loader_v2(scale, torch.tensor(0.25, dtype=torch.float32), 0)
        layer.weight_loader_v2(scale, torch.tensor(0.5, dtype=torch.float32), 1)

        torch.testing.assert_close(scale, torch.tensor([0.25, 0.5]))

    def test_fused_moe_uses_requested_zero_expert_count_for_all_weights(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_local_experts = 8
                self.num_experts = 8
                self.moe_runner_config = SimpleNamespace(is_gated=True)

        method = object.__new__(ModelOptNvFp4FusedMoEMethod)
        method.num_gpu_experts = 0
        method.quant_config = SimpleNamespace(
            is_checkpoint_nvfp4_serialized=True,
            is_nvfp4_online=False,
            group_size=16,
        )
        method.enable_flashinfer_trtllm_moe = True
        layer = Layer()

        method.create_weights(
            layer=layer,
            num_experts=0,
            hidden_size=32,
            intermediate_size_per_partition=32,
            params_dtype=torch.bfloat16,
        )

        assert layer.w13_weight.shape == (0, 64, 16)
        assert layer.w2_weight.shape == (0, 32, 16)
        assert layer.w13_weight_scale.shape == (0, 64, 2)
        assert layer.w2_weight_scale.shape == (0, 32, 2)
        assert layer.w13_weight_scale_2.shape == (0, 2)
        assert layer.w2_weight_scale_2.shape == (0,)
        assert layer.w13_input_scale.shape == (0, 2)
        assert layer.w2_input_scale.shape == (0,)

    def test_fused_moe_input_scales_keep_global_expert_slots_for_ep_loading(self):
        class Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.num_local_experts = 2
                self.num_experts = 8
                self.moe_runner_config = SimpleNamespace(is_gated=True)

        method = object.__new__(ModelOptNvFp4FusedMoEMethod)
        method.quant_config = SimpleNamespace(
            is_checkpoint_nvfp4_serialized=True,
            is_nvfp4_online=False,
            group_size=16,
        )
        method.enable_flashinfer_trtllm_moe = True
        layer = Layer()

        method.create_weights(
            layer=layer,
            num_experts=layer.num_local_experts,
            hidden_size=32,
            intermediate_size_per_partition=32,
            params_dtype=torch.bfloat16,
        )

        assert layer.w13_weight.shape[0] == layer.num_local_experts
        assert layer.w2_weight.shape[0] == layer.num_local_experts
        assert layer.w13_input_scale.shape == (layer.num_experts, 2)
        assert layer.w2_input_scale.shape == (layer.num_experts,)

        highest_global_expert_id = layer.num_experts - 1
        fused_moe = object.__new__(FusedMoE)
        fused_moe.quant_config = SimpleNamespace(get_name=lambda: "modelopt_fp4")
        fused_moe._weight_loader_impl = lambda **kwargs: FusedMoE._load_single_value(
            fused_moe,
            kwargs["param"],
            kwargs["loaded_weight"],
            kwargs["expert_id"],
        )
        with patch(
            "sglang.srt.layers.moe.fused_moe_triton.layer.get_global_expert_location_metadata",
            return_value=None,
        ):
            FusedMoE.weight_loader(
                fused_moe,
                layer.w13_input_scale,
                torch.tensor([0.25, 0.5]),
                "w13_input_scale",
                "w1",
                highest_global_expert_id,
            )
            FusedMoE.weight_loader(
                fused_moe,
                layer.w2_input_scale,
                torch.tensor(0.75),
                "w2_input_scale",
                "w2",
                highest_global_expert_id,
            )
        torch.testing.assert_close(
            layer.w13_input_scale[highest_global_expert_id], torch.tensor([0.25, 0.5])
        )
        torch.testing.assert_close(
            layer.w2_input_scale[highest_global_expert_id], torch.tensor(0.75)
        )


if __name__ == "__main__":
    unittest.main()
