from types import SimpleNamespace
import unittest

import torch

import sglang.srt.layers.moe.kt_ep_wrapper as kt_ep_wrapper
from sglang.srt.layers.moe.token_dispatcher.standard import (
    StandardDispatchOutput,
    StandardTopKOutput,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-b-test-cpu")


class _DummyGpuMethod:
    def __init__(self):
        self.num_gpu_experts = None
        self.created_num_experts = None
        self.processed_weights = False

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        self.created_num_experts = num_experts

    def process_weights_after_loading(self, layer):
        self.processed_weights = True
        raise AssertionError("GPU postprocess should not run without GPU experts")


class _DummyForwardGpuMethod:
    def __init__(self):
        self.apply_call_count = 0

    def apply(self, layer, dispatch_output):
        self.apply_call_count += 1
        raise AssertionError("GPU apply should not run without GPU experts")

    def create_weights(
        self,
        layer,
        num_experts,
        hidden_size,
        intermediate_size_per_partition,
        params_dtype,
        **extra_weight_attrs,
    ):
        return None


class _DummyLayer:
    top_k = 8
    intermediate_size_per_partition = 1024
    moe_tp_size = 2


class TestKTEPWrapper(unittest.TestCase):
    def test_kt_wrapper_passes_gpu_experts_mask_to_kt_kernel(self):
        captured = {}

        class FakeKTMoEWrapper:
            def __init__(
                self,
                *,
                layer_idx,
                num_experts,
                num_experts_per_tok,
                hidden_size,
                moe_intermediate_size,
                gpu_experts_mask,
                cpuinfer_threads,
                threadpool_count,
                weight_path,
                chunked_prefill_size,
                method,
                max_deferred_experts_per_token,
                num_gpu_experts=None,
            ):
                captured.update(locals())

        old_kt_moe_wrapper = kt_ep_wrapper.KTMoEWrapper
        old_available = kt_ep_wrapper.KTRANSFORMERS_AVAILABLE
        old_get_parallel = kt_ep_wrapper.get_parallel
        try:
            kt_ep_wrapper.KTMoEWrapper = FakeKTMoEWrapper
            kt_ep_wrapper.KTRANSFORMERS_AVAILABLE = True
            kt_ep_wrapper.get_parallel = lambda: SimpleNamespace(tp_rank=0)

            gpu_method = _DummyGpuMethod()
            wrapper = kt_ep_wrapper.KTEPWrapperMethod(
                gpu_method,
                kt_ep_wrapper.KTConfig(
                    layer_idx=3,
                    num_gpu_experts=0,
                    cpuinfer_threads=76,
                    threadpool_count=1,
                    weight_path="/tmp/weights",
                    chunked_prefill_size=2048,
                    max_deferred_experts_per_token=None,
                    method="NVFP4",
                ),
            )

            wrapper.create_weights(
                layer=_DummyLayer(),
                num_experts=168,
                hidden_size=6144,
                intermediate_size_per_partition=1024,
                params_dtype=torch.bfloat16,
            )
        finally:
            kt_ep_wrapper.KTMoEWrapper = old_kt_moe_wrapper
            kt_ep_wrapper.KTRANSFORMERS_AVAILABLE = old_available
            kt_ep_wrapper.get_parallel = old_get_parallel

        self.assertEqual(gpu_method.created_num_experts, 0)
        gpu_experts_mask = captured["gpu_experts_mask"]
        self.assertEqual(gpu_experts_mask.shape, (168,))
        self.assertIs(gpu_experts_mask.dtype, torch.bool)
        self.assertEqual(gpu_experts_mask.device.type, "cpu")
        self.assertEqual(gpu_experts_mask.sum().item(), 0)

    def test_all_cpu_experts_skip_gpu_weight_postprocess(self):
        old_available = kt_ep_wrapper.KTRANSFORMERS_AVAILABLE
        old_get_parallel = kt_ep_wrapper.get_parallel
        try:
            kt_ep_wrapper.KTRANSFORMERS_AVAILABLE = True
            kt_ep_wrapper.get_parallel = lambda: SimpleNamespace(tp_rank=1)

            gpu_method = _DummyGpuMethod()
            wrapper = kt_ep_wrapper.KTEPWrapperMethod(
                gpu_method,
                kt_ep_wrapper.KTConfig(
                    layer_idx=3,
                    num_gpu_experts=0,
                    cpuinfer_threads=76,
                    threadpool_count=1,
                    weight_path="/tmp/weights",
                    chunked_prefill_size=2048,
                    max_deferred_experts_per_token=None,
                    method="NVFP4",
                ),
            )

            wrapper.process_weights_after_loading(torch.nn.Module())
        finally:
            kt_ep_wrapper.KTRANSFORMERS_AVAILABLE = old_available
            kt_ep_wrapper.get_parallel = old_get_parallel

        self.assertFalse(gpu_method.processed_weights)

    def test_all_cpu_experts_skip_gpu_forward_apply(self):
        old_available = kt_ep_wrapper.KTRANSFORMERS_AVAILABLE
        old_get_parallel = kt_ep_wrapper.get_parallel
        try:
            kt_ep_wrapper.KTRANSFORMERS_AVAILABLE = True
            kt_ep_wrapper.get_parallel = lambda: SimpleNamespace(tp_rank=0)

            gpu_method = _DummyForwardGpuMethod()
            wrapper = kt_ep_wrapper.KTEPWrapperMethod(
                gpu_method,
                kt_ep_wrapper.KTConfig(
                    layer_idx=3,
                    num_gpu_experts=0,
                    cpuinfer_threads=76,
                    threadpool_count=1,
                    weight_path="/tmp/weights",
                    chunked_prefill_size=2048,
                    max_deferred_experts_per_token=None,
                    method="NVFP4",
                ),
            )

            dispatch_output = StandardDispatchOutput(
                hidden_states=torch.ones((2, 2), dtype=torch.float32),
                hidden_states_scale=None,
                topk_output=StandardTopKOutput(
                    topk_weights=torch.full((2, 2), 0.5, dtype=torch.float32),
                    topk_ids=torch.zeros((2, 2), dtype=torch.int32),
                    router_logits=torch.zeros((2, 2), dtype=torch.float32),
                ),
            )

            submit_called = []
            sync_called = []
            fake_cpu_output = torch.full((2, 2), 9.0, dtype=torch.float32)

            def fake_submit(layer, output):
                submit_called.append(output)

            def fake_sync(x):
                sync_called.append(x)
                return fake_cpu_output

            wrapper.submit = fake_submit
            wrapper.sync = fake_sync

            output = wrapper.apply(layer=_DummyLayer(), dispatch_output=dispatch_output)
        finally:
            kt_ep_wrapper.KTRANSFORMERS_AVAILABLE = old_available
            kt_ep_wrapper.get_parallel = old_get_parallel

        self.assertEqual(gpu_method.apply_call_count, 0)
        self.assertEqual(len(submit_called), 1)
        self.assertEqual(len(sync_called), 1)
        self.assertTrue(torch.equal(output.hidden_states, fake_cpu_output))


if __name__ == "__main__":
    unittest.main()
