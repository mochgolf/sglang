import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[4]


class FakeCudaEvent:
    def __init__(self, events, *, enable_timing):
        events.append(("event", enable_timing))
        self.events = events

    def record(self):
        self.events.append("record")

    def elapsed_time(self, _other):
        return 1.0


@pytest.mark.parametrize(
    ("legacy", "nsys", "expected_pushes", "expected_event_count", "expected_syncs"),
    [
        ("0", "0", [], 0, 0),
        ("1", "0", ["test.region"], 2, 2),
        ("0", "1", ["test.region"], 0, 0),
    ],
)
def test_profile_region_separates_legacy_timer_from_nsys_nvtx(
    monkeypatch,
    capsys,
    legacy,
    nsys,
    expected_pushes,
    expected_event_count,
    expected_syncs,
):
    from sglang.srt.layers.attention.dsa import sm89_debug

    events = []
    pushes = []
    pops = []
    monkeypatch.setenv("SGLANG_GLM_DSA_SM89_PROFILE", legacy)
    monkeypatch.setenv("SGLANG_SM89_DECODE_NSYS_PROFILE", nsys)
    monkeypatch.setattr(sm89_debug.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sm89_debug.torch.cuda,
        "Event",
        lambda **kwargs: FakeCudaEvent(events, **kwargs),
    )
    monkeypatch.setattr(
        sm89_debug.torch.cuda, "synchronize", lambda: events.append("synchronize")
    )
    monkeypatch.setattr(
        sm89_debug.torch.cuda.nvtx, "range_push", lambda name: pushes.append(name)
    )
    monkeypatch.setattr(sm89_debug.torch.cuda.nvtx, "range_pop", lambda: pops.append(1))

    with sm89_debug.profile_region("test.region"):
        pass

    assert pushes == expected_pushes
    assert len(pops) == len(expected_pushes)
    assert sum(item[0] == "event" for item in events if isinstance(item, tuple)) == (
        expected_event_count
    )
    assert events.count("synchronize") == expected_syncs
    if nsys == "1":
        assert capsys.readouterr().out == ""


def test_glm_dsa_sm89_nsys_enabled_reflects_environment(monkeypatch):
    from sglang.srt.layers.attention.dsa.sm89_debug import (
        glm_dsa_sm89_nsys_enabled,
    )

    monkeypatch.delenv("SGLANG_SM89_DECODE_NSYS_PROFILE", raising=False)
    assert not glm_dsa_sm89_nsys_enabled()
    monkeypatch.setenv("SGLANG_SM89_DECODE_NSYS_PROFILE", "0")
    assert not glm_dsa_sm89_nsys_enabled()
    monkeypatch.setenv("SGLANG_SM89_DECODE_NSYS_PROFILE", "1")
    assert glm_dsa_sm89_nsys_enabled()


def test_profile_range_falls_back_to_balanced_torch_nvtx_on_exception(monkeypatch):
    from sglang.srt.utils import nvtx_utils

    pushes = []
    pops = []
    monkeypatch.setattr(nvtx_utils, "_nvtx_module", None)
    monkeypatch.setattr(nvtx_utils.torch.autograd, "_profiler_enabled", lambda: False)
    monkeypatch.setattr(
        nvtx_utils.torch.cuda.nvtx, "range_push", lambda name: pushes.append(name)
    )
    monkeypatch.setattr(nvtx_utils.torch.cuda.nvtx, "range_pop", lambda: pops.append(1))

    with pytest.raises(RuntimeError, match="inside range"):
        with nvtx_utils.profile_range("fallback.range", nvtx_enabled=True):
            raise RuntimeError("inside range")

    assert pushes == ["fallback.range"]
    assert pops == [1]


def _method_ast(relative_path, class_name, method_name):
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    class_node = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    return next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == method_name
    )


def _function_ast(relative_path, function_name):
    tree = ast.parse((REPO_ROOT / relative_path).read_text())
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )


def _named_calls(nodes, name):
    calls = []
    for root in nodes:
        for node in ast.walk(root):
            if not isinstance(node, ast.Call):
                continue
            function_name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr if isinstance(node.func, ast.Attribute) else None
            )
            if function_name == name:
                calls.append(node)
    return calls


def _assert_exact_indexed_fstring(node, prefix, index_name):
    assert isinstance(node, ast.JoinedStr)
    assert len(node.values) == 2
    assert isinstance(node.values[0], ast.Constant)
    assert node.values[0].value == prefix
    assert isinstance(node.values[1], ast.FormattedValue)
    assert isinstance(node.values[1].value, ast.Name)
    assert node.values[1].value.id == index_name


def test_deepseek_tc_piecewise_branch_keeps_original_nullcontext_only():
    method = _method_ast(
        "python/sglang/srt/models/deepseek_v2.py", "DeepseekV2Model", "forward"
    )
    layer_loops = [
        node
        for node in ast.walk(method)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "i"
        and _named_calls(node.body, "check_cuda_graph_backend")
    ]
    assert len(layer_loops) == 1
    layer_loop = layer_loops[0]
    tc_guards = [
        node
        for node in layer_loop.body
        if isinstance(node, ast.If)
        and _named_calls([node.test], "check_cuda_graph_backend")
    ]
    assert len(tc_guards) == 1
    tc_guard = tc_guards[0]

    assert len(_named_calls(tc_guard.body, "nullcontext")) == 1
    assert not _named_calls(tc_guard.body, "profile_region")
    assert not _named_calls(tc_guard.body, "with_current_layer")
    assert len(_named_calls(tc_guard.orelse, "_deepseek_layer_profile_context")) == 1
    assert not _named_calls(tc_guard.orelse, "profile_region")
    assert not _named_calls(tc_guard.orelse, "with_current_layer")

    helper = _function_ast(
        "python/sglang/srt/models/deepseek_v2.py",
        "_deepseek_layer_profile_context",
    )
    stack_contexts = [node for node in helper.body if isinstance(node, ast.With)]
    assert len(stack_contexts) == 1
    assert len(_named_calls(stack_contexts[0].items, "ExitStack")) == 1
    assert len(_named_calls(stack_contexts[0].body, "with_current_layer")) == 1
    profile_calls = _named_calls(stack_contexts[0].body, "profile_region")
    assert len(profile_calls) == 1
    _assert_exact_indexed_fstring(
        profile_calls[0].args[0], "model.layer.", "layer_id"
    )

    layer_contexts = [node for node in layer_loop.body if isinstance(node, ast.With)]
    assert len(layer_contexts) == 1
    assert len(layer_contexts[0].items) == 1
    assert isinstance(layer_contexts[0].items[0].context_expr, ast.Name)
    assert layer_contexts[0].items[0].context_expr.id == "ctx"


def test_deepseek_layer_profile_context_unwinds_recorder_on_profile_enter_error(
    monkeypatch,
):
    from sglang.srt.models import deepseek_v2

    events = []
    body_entered = False

    @contextmanager
    def recorder_context():
        events.append("recorder_enter")
        try:
            yield
        finally:
            events.append("recorder_exit")

    class RaisingProfileContext:
        def __enter__(self):
            events.append("profile_enter")
            raise RuntimeError("profile enter failed")

        def __exit__(self, *_args):
            events.append("profile_exit")

    recorder = SimpleNamespace(
        with_current_layer=lambda layer_id: (
            events.append(("layer_id", layer_id)) or recorder_context()
        )
    )
    monkeypatch.setattr(
        deepseek_v2, "get_global_expert_distribution_recorder", lambda: recorder
    )
    monkeypatch.setattr(
        deepseek_v2,
        "profile_region",
        lambda name: (
            events.append(("profile_name", name)) or RaisingProfileContext()
        ),
    )

    with pytest.raises(RuntimeError, match="profile enter failed"):
        with deepseek_v2._deepseek_layer_profile_context(23):
            body_entered = True

    assert not body_entered
    assert events == [
        ("layer_id", 23),
        "recorder_enter",
        ("profile_name", "model.layer.23"),
        "profile_enter",
        "recorder_exit",
    ]


def test_model_step_names_are_exact_for_both_tp_ranks_and_live_callsite():
    from sglang.srt.model_executor import model_runner

    forward_batch = SimpleNamespace(
        forward_mode=model_runner.ForwardMode.DECODE,
        batch_size=1,
        extend_num_tokens=0,
    )
    assert model_runner._build_step_span_name(forward_batch, 0) == (
        "model.step.tp0.decode"
    )
    assert model_runner._build_step_span_name(forward_batch, 1) == (
        "model.step.tp1.decode"
    )

    method = _method_ast(
        "python/sglang/srt/model_executor/model_runner.py", "ModelRunner", "forward"
    )
    operation_calls = _named_calls(method.body, "operations_nvtx_range")
    assert len(operation_calls) == 1
    assert ast.unparse(operation_calls[0].args[0]) == (
        "_build_step_span_name(forward_batch, self.tp_rank)"
    )


def test_indexer_and_sparse_mla_emit_exact_scoped_names(monkeypatch):
    from sglang.srt.layers.attention.dsa import dsa_indexer, sm89_sparse_mla
    from sglang.srt.model_executor.forward_batch_info import ForwardMode

    events = []

    @contextmanager
    def fake_profile_region(name, enabled=None):
        events.append(("enter", name, enabled))
        try:
            yield
        finally:
            events.append(("exit", name, enabled))

    monkeypatch.setattr(dsa_indexer, "profile_region", fake_profile_region)
    monkeypatch.setattr(dsa_indexer, "is_in_tc_piecewise_cuda_graph", lambda: False)
    monkeypatch.setattr(
        dsa_indexer, "_broadcast_indexer_topk_from_rank0", lambda value: value
    )
    monkeypatch.setattr(
        dsa_indexer, "maybe_capture_indexer_topk", lambda _layer_id, value: value
    )
    q_contiguous = object()
    q_fp8 = SimpleNamespace(contiguous=lambda: q_contiguous)
    indexer = SimpleNamespace(
        index_topk=8,
        forward_indexer=lambda *args, **kwargs: (args, kwargs),
    )
    metadata = object()
    forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    dsa_indexer._select_glm_sm89_fallback_topk(
        indexer, q_fp8, "weights", forward_batch, metadata, 17
    )

    sentinel = object()
    monkeypatch.setattr(sm89_sparse_mla, "profile_region", fake_profile_region)
    monkeypatch.setattr(
        sm89_sparse_mla, "_sm89_sparse_mla_decode_cuda", lambda **_kwargs: sentinel
    )
    result = sm89_sparse_mla.sm89_sparse_mla_decode_cuda(
        q_nope=object(),
        q_rope=object(),
        kv_cache=object(),
        page_table=object(),
        cache_seqlens=object(),
        sm_scale=1.0,
        logit_cap=0.0,
        v_head_dim=512,
    )

    assert result is sentinel
    assert events == [
        ("enter", "dsa.indexer.layer.17", None),
        ("exit", "dsa.indexer.layer.17", None),
        ("enter", "sm89_sparse_mla.decode.cuda.total", None),
        ("exit", "sm89_sparse_mla.decode.cuda.total", None),
    ]


@pytest.mark.parametrize(
    ("nsys_enabled", "expected_ranges"),
    [
        (False, []),
        (
            True,
            [
                "fused_moe.dispatch",
                "fused_moe.run_moe_core",
                "fused_moe.combine",
                "fused_moe.all_reduce",
            ],
        ),
    ],
)
def test_fused_moe_default_off_direct_path_and_nsys_only_ranges(
    monkeypatch, nsys_enabled, expected_ranges
):
    from sglang.srt.layers.moe.fused_moe_triton import layer as fused_moe_layer

    ranges = []
    operations = []

    @contextmanager
    def fake_profile_region(name, enabled=None):
        ranges.append((name, enabled))
        yield

    def dispatch(**_kwargs):
        operations.append("dispatch")
        return "dispatch_output"

    def run_moe_core(**_kwargs):
        operations.append("run_moe_core")
        return "combine_input"

    def combine(**_kwargs):
        operations.append("combine")
        return torch.ones(1, 4)

    def all_reduce(value):
        operations.append("all_reduce")
        return value

    monkeypatch.setattr(
        fused_moe_layer, "glm_dsa_sm89_profile_enabled", lambda: False
    )
    monkeypatch.setattr(
        fused_moe_layer,
        "glm_dsa_sm89_nsys_enabled",
        lambda: nsys_enabled,
        raising=False,
    )
    monkeypatch.setattr(fused_moe_layer, "profile_region", fake_profile_region)
    monkeypatch.setattr(
        fused_moe_layer, "use_symmetric_memory", lambda *_args, **_kwargs: nullcontext()
    )
    monkeypatch.setattr(fused_moe_layer, "get_tp_group", lambda: object())
    monkeypatch.setattr(fused_moe_layer, "is_allocation_symmetric", lambda: False)
    monkeypatch.setattr(fused_moe_layer, "tensor_model_parallel_all_reduce", all_reduce)

    fake_layer = SimpleNamespace(
        quant_method=object(),
        dispatcher=SimpleNamespace(dispatch=dispatch, combine=combine),
        run_moe_core=run_moe_core,
        reduce_results=True,
        moe_tp_size=2,
        moe_ep_size=1,
    )
    result = fused_moe_layer.FusedMoE.forward_impl(
        fake_layer, torch.zeros(1, 4), object()
    )

    assert result.shape == (1, 4)
    assert operations == ["dispatch", "run_moe_core", "combine", "all_reduce"]
    assert ranges == [(name, False) for name in expected_ranges]


def test_explicit_false_profile_region_still_emits_nsys_nvtx(monkeypatch):
    from sglang.srt.layers.attention.dsa import sm89_debug

    pushes = []
    pops = []
    monkeypatch.setenv("SGLANG_GLM_DSA_SM89_PROFILE", "0")
    monkeypatch.setenv("SGLANG_SM89_DECODE_NSYS_PROFILE", "1")
    monkeypatch.setattr(sm89_debug.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        sm89_debug.torch.cuda.nvtx, "range_push", lambda name: pushes.append(name)
    )
    monkeypatch.setattr(sm89_debug.torch.cuda.nvtx, "range_pop", lambda: pops.append(1))
    monkeypatch.setattr(
        sm89_debug.torch.cuda,
        "Event",
        lambda **_kwargs: pytest.fail("nsys mode created a CUDA event"),
    )
    monkeypatch.setattr(
        sm89_debug.torch.cuda,
        "synchronize",
        lambda: pytest.fail("nsys mode synchronized CUDA"),
    )

    with sm89_debug.profile_region("explicit.false", enabled=False):
        pass

    assert pushes == ["explicit.false"]
    assert pops == [1]
