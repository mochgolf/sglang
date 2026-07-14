from types import SimpleNamespace

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
        self.apply_call_count = 0

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

    def apply(self, layer, dispatch_output):
        self.apply_call_count += 1
        raise AssertionError("GPU apply should not run without GPU experts")


class _DummyLayer:
    top_k = 8
    intermediate_size_per_partition = 1024
    moe_tp_size = 2


def _kt_config(num_gpu_experts):
    return kt_ep_wrapper.KTConfig(
        layer_idx=3,
        num_gpu_experts=num_gpu_experts,
        cpuinfer_threads=76,
        threadpool_count=1,
        weight_path="/tmp/weights",
        chunked_prefill_size=2048,
        max_deferred_experts_per_token=None,
        method="NVFP4",
    )


def test_none_gpu_experts_normalizes_to_zero_and_passes_cpu_mask(monkeypatch):
    captured = {}

    class FakeKTMoEWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(kt_ep_wrapper, "KTMoEWrapper", FakeKTMoEWrapper, raising=False)
    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        kt_ep_wrapper, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )

    gpu_method = _DummyGpuMethod()
    wrapper = kt_ep_wrapper.KTEPWrapperMethod(gpu_method, _kt_config(None))
    wrapper.create_weights(
        layer=_DummyLayer(),
        num_experts=168,
        hidden_size=6144,
        intermediate_size_per_partition=1024,
        params_dtype=torch.bfloat16,
    )

    assert wrapper.num_gpu_experts == 0
    assert gpu_method.created_num_experts == 0
    assert captured["num_gpu_experts"] == 0
    gpu_experts_mask = captured["gpu_experts_mask"]
    assert gpu_experts_mask.shape == (168,)
    assert gpu_experts_mask.dtype is torch.bool
    assert gpu_experts_mask.device.type == "cpu"
    assert gpu_experts_mask.sum().item() == 0


def test_gpu_experts_mask_marks_the_prefix(monkeypatch):
    captured = {}

    class FakeKTMoEWrapper:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(kt_ep_wrapper, "KTMoEWrapper", FakeKTMoEWrapper, raising=False)
    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        kt_ep_wrapper, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )

    wrapper = kt_ep_wrapper.KTEPWrapperMethod(_DummyGpuMethod(), _kt_config(3))
    wrapper.create_weights(
        layer=_DummyLayer(),
        num_experts=5,
        hidden_size=6144,
        intermediate_size_per_partition=1024,
        params_dtype=torch.bfloat16,
    )

    assert torch.equal(
        captured["gpu_experts_mask"], torch.tensor([True, True, True, False, False])
    )


def test_all_cpu_experts_skip_gpu_methods_and_return_cpu_output(monkeypatch):
    monkeypatch.setattr(kt_ep_wrapper, "KTRANSFORMERS_AVAILABLE", True)
    monkeypatch.setattr(
        kt_ep_wrapper, "get_parallel", lambda: SimpleNamespace(tp_rank=0)
    )

    gpu_method = _DummyGpuMethod()
    wrapper = kt_ep_wrapper.KTEPWrapperMethod(gpu_method, _kt_config(0))
    wrapper.process_weights_after_loading(torch.nn.Module())

    dispatch_output = StandardDispatchOutput(
        hidden_states=torch.ones((2, 2), dtype=torch.float32),
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=torch.full((2, 2), 0.5, dtype=torch.float32),
            topk_ids=torch.zeros((2, 2), dtype=torch.int32),
            router_logits=torch.zeros((2, 2), dtype=torch.float32),
        ),
    )
    cpu_output = torch.full((2, 2), 9.0, dtype=torch.float32)
    submitted = []
    synced = []
    wrapper.submit = lambda layer, output: submitted.append(output)
    wrapper.sync = lambda x: synced.append(x) or cpu_output

    output = wrapper.apply(layer=_DummyLayer(), dispatch_output=dispatch_output)

    assert not gpu_method.processed_weights
    assert gpu_method.apply_call_count == 0
    assert submitted == [dispatch_output]
    assert synced == [dispatch_output.hidden_states]
    assert torch.equal(output.hidden_states, cpu_output)
