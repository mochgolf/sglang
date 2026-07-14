from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from types import ModuleType

import pytest
import torch

import sglang.srt.managers.scheduler as scheduler_module
import sglang.srt.managers.scheduler_components.batch_result_processor as result_processor_module
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm


PROFILE_RID = "residual:task2-test"
PROFILE_ENVIRONMENT = {
    "SGLANG_GLM_DSA_SM89_PROFILE": "0",
    "SGLANG_SM89_DECODE_AGGREGATE_PROFILE": "0",
}


def marker_state_type():
    assert hasattr(
        scheduler_module, "_SM89DecodeResidualMarkerState"
    ), "scheduler residual marker state is missing"
    return scheduler_module._SM89DecodeResidualMarkerState


def marker_state_factory():
    assert hasattr(
        scheduler_module, "_create_sm89_decode_residual_marker_state"
    ), "scheduler residual marker state factory is missing"
    return scheduler_module._create_sm89_decode_residual_marker_state


def layer_deriver():
    assert hasattr(
        scheduler_module, "_derive_sm89_decode_residual_layer_ids"
    ), "scheduler residual layer derivation is missing"
    return scheduler_module._derive_sm89_decode_residual_layer_ids


def new_state(emitted: list[str]):
    return marker_state_type()(emit=emitted.append)


_AGGREGATE_STAGE_GROUPS = {
    "cpuinfer": ("submit_callback", "task", "sync_callback", "sync_wait"),
    "tp_moe": ("total", "merge"),
    "amx_m1": (
        "setup",
        "q_input",
        "gate_up",
        "activation",
        "q_down",
        "down",
        "weighted_sum",
        "total",
    ),
}


def aggregate_for_forwards(forwards: int) -> dict[str, dict[str, dict[str, dict[str, int]]]]:
    first_count = 75 * min(forwards, 64)
    second_count = 75 * max(forwards - 64, 0)
    return {
        group: {
            stage: {
                "first": {"ns": first_count + 1, "count": first_count},
                "second": {"ns": second_count + 1, "count": second_count},
                "total": {
                    "ns": first_count + second_count + 2,
                    "count": first_count + second_count,
                },
            }
            for stage in stages
        }
        for group, stages in _AGGREGATE_STAGE_GROUPS.items()
    }


def new_aggregate_state(emitted: list[str], begin, end):
    environment = dict(PROFILE_ENVIRONMENT)
    environment["SGLANG_SM89_DECODE_AGGREGATE_PROFILE"] = "1"
    return marker_state_factory()(
        environment,
        tp_rank=0,
        emit=emitted.append,
        begin_decode_aggregate=begin,
        end_decode_aggregate=end,
    )


def launch_forwards(state, count: int) -> None:
    for forward_id in range(count):
        assert state.record_launch(
            forward_id=forward_id,
            rid=PROFILE_RID,
            completion_tokens=0 if forward_id == 0 else 1,
            max_new_tokens=128,
        )


def account_exact_completion(state, launched: int) -> None:
    for forward_id in range(127):
        state.record_result(
            forward_id=forward_id,
            rid=PROFILE_RID,
            accepted=True,
            completion_tokens=forward_id + 2,
        )
    for forward_id in range(127, launched):
        state.record_result(
            forward_id=forward_id,
            rid=PROFILE_RID,
            accepted=False,
            completion_tokens=128,
        )


def scheduler_with_state(emitted: list[str]):
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler._sm89_decode_residual_marker_state = new_state(emitted)
    return scheduler


def target_req():
    return SimpleNamespace(
        rid=PROFILE_RID,
        output_ids=[1],
        sampling_params=SimpleNamespace(max_new_tokens=128),
        is_retracted=False,
        finished=lambda: False,
    )


def target_batch(req, *, forward_id: int):
    return ScheduleBatch(
        reqs=[req],
        forward_mode=ForwardMode.DECODE,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        forward_iter=forward_id,
    )


def test_derives_exact_glm52_kt_and_active_indexer_layers() -> None:
    expected_indexer_layers = (0, 1, 2, *range(6, 75, 4))
    indexer_types = tuple(
        "full" if layer_id in expected_indexer_layers else "shared"
        for layer_id in range(78)
    )
    kt_layers, indexer_layers = layer_deriver()(
        num_hidden_layers=78,
        first_k_dense_replace=3,
        index_topk_freq=4,
        index_skip_topk_offset=3,
        indexer_types=indexer_types,
    )

    assert kt_layers == tuple(range(3, 78))
    assert len(kt_layers) == 75
    assert indexer_layers == expected_indexer_layers
    assert len(indexer_layers) == 21
    assert scheduler_module._SM89_DECODE_RESIDUAL_EXPECTED_KT_LAYER_IDS == kt_layers
    assert (
        scheduler_module._SM89_DECODE_RESIDUAL_EXPECTED_INDEXER_LAYER_IDS
        == indexer_layers
    )


def test_layer_derivation_rejects_types_that_disagree_with_offset() -> None:
    indexer_types = ("full", "full", "shared", *("shared",) * 75)

    with pytest.raises(ValueError, match="indexer_types disagree"):
        layer_deriver()(
            num_hidden_layers=78,
            first_k_dense_replace=3,
            index_topk_freq=4,
            index_skip_topk_offset=3,
            indexer_types=indexer_types,
        )


@pytest.mark.parametrize("aggregate", ("0", "1"))
def test_factory_requires_exact_profile_flags_and_rank_zero(aggregate: str) -> None:
    environment = dict(PROFILE_ENVIRONMENT)
    environment["SGLANG_SM89_DECODE_AGGREGATE_PROFILE"] = aggregate
    aggregate_abi = (
        {
            "begin_decode_aggregate": lambda nonce: None,
            "end_decode_aggregate": lambda nonce: aggregate_for_forwards(0),
        }
        if aggregate == "1"
        else {}
    )

    assert (
        marker_state_factory()(
            environment, tp_rank=0, emit=lambda _: None, **aggregate_abi
        )
        is not None
    )
    assert (
        marker_state_factory()(
            environment, tp_rank=1, emit=lambda _: None, **aggregate_abi
        )
        is None
    )
    assert (
        marker_state_factory()({}, tp_rank=0, emit=lambda _: None, **aggregate_abi)
        is None
    )
    assert (
        marker_state_factory()(
            {**environment, "SGLANG_GLM_DSA_SM89_PROFILE": "1"},
            tp_rank=0,
            emit=lambda _: None,
            **aggregate_abi,
        )
        is None
    )


def test_aggregate_factory_rejects_installed_none_symbols_before_markers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    kt_kernel = ModuleType("kt_kernel")
    extension = ModuleType("kt_kernel.kt_kernel_ext")
    extension.begin_decode_aggregate = None
    extension.end_decode_aggregate = None
    kt_kernel.kt_kernel_ext = extension
    monkeypatch.setitem(sys.modules, "kt_kernel", kt_kernel)
    monkeypatch.setitem(sys.modules, "kt_kernel.kt_kernel_ext", extension)

    with pytest.raises(RuntimeError, match="aggregate ABI is unavailable"):
        marker_state_factory()(
            {**PROFILE_ENVIRONMENT, "SGLANG_SM89_DECODE_AGGREGATE_PROFILE": "1"},
            tp_rank=0,
            emit=emitted.append,
        )

    assert emitted == []


@pytest.mark.parametrize("begin, end", ((0, 1), (object(), object())))
def test_aggregate_factory_rejects_noncallable_injected_symbols_before_markers(
    begin: object,
    end: object,
) -> None:
    emitted: list[str] = []

    with pytest.raises(RuntimeError, match="aggregate ABI is unavailable"):
        marker_state_factory()(
            {**PROFILE_ENVIRONMENT, "SGLANG_SM89_DECODE_AGGREGATE_PROFILE": "1"},
            tp_rank=0,
            emit=emitted.append,
            begin_decode_aggregate=begin,
            end_decode_aggregate=end,
        )

    assert emitted == []


def test_enabled_state_ignores_normal_request_ids_without_markers() -> None:
    emitted: list[str] = []
    state = new_state(emitted)

    assert not state.record_launch(
        forward_id=0,
        rid="normal-request",
        completion_tokens=1,
        max_new_tokens=128,
    )
    state.assert_drained()

    assert emitted == []
    assert state.decode_forward_count == 0


def test_first_launch_rejects_postprocessed_baseline_before_begin() -> None:
    emitted: list[str] = []
    state = new_state(emitted)

    with pytest.raises(RuntimeError, match="completion baseline must be 0"):
        state.record_launch(
            forward_id=0,
            rid=PROFILE_RID,
            completion_tokens=1,
            max_new_tokens=128,
        )

    assert state.decode_forward_count == 0
    assert emitted == []


def test_state_locks_the_first_exact_residual_rid() -> None:
    emitted: list[str] = []
    state = new_state(emitted)
    state.record_launch(
        forward_id=0,
        rid=PROFILE_RID,
        completion_tokens=0,
        max_new_tokens=128,
    )

    with pytest.raises(RuntimeError, match="target RID changed"):
        state.record_launch(
            forward_id=1,
            rid="residual:different-target",
            completion_tokens=1,
            max_new_tokens=128,
        )

    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]


def test_production_run_batch_begins_aggregate_before_model_forward_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    emitted: list[str] = []
    scheduler = scheduler_module.Scheduler.__new__(scheduler_module.Scheduler)
    scheduler._sm89_decode_residual_marker_state = new_aggregate_state(
        emitted,
        lambda nonce: events.append(("begin", nonce)),
        lambda nonce: aggregate_for_forwards(2),
    )
    scheduler.forward_ct = 0
    scheduler.scripted_scheduler_hook = None
    scheduler.profiler_manager = SimpleNamespace(
        _profile_batch_predicate=lambda batch: None
    )
    scheduler.forward_sleep_time = None
    scheduler.is_generation = True
    scheduler.enable_overlap = False
    scheduler.enable_pdmux = False
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.future_map = SimpleNamespace()
    scheduler.model_worker = SimpleNamespace(
        forward_batch_generation=lambda batch, **kwargs: events.append(
            ("model_forward", batch.forward_iter)
        )
        or GenerationBatchResult()
    )
    scheduler.update_cache_from_scheduler = lambda batch, result: None
    scheduler._maybe_report_active_ranks = lambda: None
    monkeypatch.setattr(
        scheduler_module, "resolve_forward_inputs", lambda batch, future_map: None
    )

    req = target_req()
    req.output_ids.clear()
    batch = target_batch(req, forward_id=0)
    scheduler.run_batch(batch)
    req.output_ids.append(1)
    scheduler.run_batch(batch)

    assert events == [
        ("begin", 1),
        ("model_forward", 1),
        ("model_forward", 2),
    ]
    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]


def test_event_loop_overlap_launches_decode_before_prefill_postprocessing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    emitted: list[str] = []
    scheduler = scheduler_with_state(emitted)
    req = target_req()
    req.output_ids.clear()
    req.inflight_middle_chunks = 0
    req.time_stats = SimpleNamespace(
        set_prefill_finished_time=lambda: None,
        set_last_decode_finish_time=lambda: None,
        set_completion_time=lambda: None,
    )
    req.update_finish_state = lambda new_accept_len=1: None
    req.require_reasoning = False
    req.return_routed_experts = False
    req.return_logprob = False
    req.return_hidden_states = False
    req.grammar = None
    req.mamba_ping_pong_track_buffer = None
    req.multimodal_inputs = None
    req.session = None

    prefill_batch = ScheduleBatch(
        reqs=[req],
        forward_mode=ForwardMode.EXTEND,
        spec_algorithm=SpeculativeAlgorithm.NONE,
        return_logprob=False,
    )
    empty_batch = ScheduleBatch(reqs=[], batch_is_full=False)
    prefill_result = GenerationBatchResult(next_token_ids=torch.tensor([1]))
    decode_result = GenerationBatchResult(next_token_ids=torch.tensor([2]))
    events: list[tuple] = []

    metrics = SimpleNamespace(
        num_generated_tokens=0,
        forward_ct_decode=0,
        report_prefill_stats=lambda **kwargs: None,
        report_decode_stats=lambda *args, **kwargs: None,
        update_spec_metrics=lambda *args, **kwargs: None,
        log_batch_result_stats=lambda *args, **kwargs: None,
        update_device_timer=lambda: None,
    )
    output_streamer = SimpleNamespace(
        stream_output=lambda *args, **kwargs: events.append(
            ("processed", len(req.output_ids))
        )
    )
    scheduler.batch_result_processor = SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=DisaggregationMode.NULL,
        enable_overlap=True,
        enable_overlap_mlx=False,
        server_args=SimpleNamespace(
            enable_hisparse=False,
            enable_metrics=False,
            disaggregation_decode_enable_offload_kvcache=False,
        ),
        model_config=SimpleNamespace(think_end_id=None),
        token_to_kv_pool_allocator=SimpleNamespace(
            free_group_begin=lambda: None,
            free_group_end=lambda: None,
        ),
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=SimpleNamespace(),
        metrics_reporter=metrics,
        draft_worker=None,
        model_worker=SimpleNamespace(),
        logprob_result_processor=None,
        output_streamer=output_streamer,
        abort_request=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        result_processor_module,
        "maybe_cache_unfinished_req",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        scheduler_module, "set_schedule_time_batch", lambda batch: None
    )

    scheduler.gracefully_exit = False
    scheduler._engine_paused = False
    scheduler.request_receiver = SimpleNamespace(recv_requests=lambda: [])
    scheduler.process_input_requests = lambda recv_reqs: None
    scheduler._apply_war_barrier = lambda: None
    scheduler.enable_fpm = False
    scheduler.dllm_config = None
    scheduler.chunked_req = None
    scheduler.enable_hisparse = False
    scheduler.running_batch = empty_batch
    scheduler.last_batch = None
    scheduler.require_mlp_sync = False
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.dp_attn_adapter = SimpleNamespace(
        maybe_prepare_mlp_sync_batch=lambda batch, **kwargs: batch
    )
    scheduler._maybe_prepare_ngram_embedding = lambda batch: batch
    scheduler.is_generation = True
    scheduler.launch_batch_sample_if_needed = lambda result: None
    scheduler.publish_load_snapshot = lambda **kwargs: None
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.metrics_reporter = metrics
    scheduler._maybe_clear_mm_inputs = lambda batch: None
    scheduler.maybe_send_health_check_signal = lambda: None
    scheduler.forward_ct = 0
    scheduler._abort_on_waiting_timeout = lambda: None
    scheduler._abort_on_running_timeout = lambda: None
    scheduler.process_pending_chunked_abort = lambda: None

    prefill_offered = False

    def get_new_batch_prefill():
        nonlocal prefill_offered
        if not prefill_offered:
            prefill_offered = True
            return prefill_batch
        return None

    update_calls = 0

    def update_running_batch(batch):
        nonlocal update_calls
        update_calls += 1
        if update_calls == 1:
            batch.forward_mode = ForwardMode.DECODE
            events.append(("get_next_decode", batch.reqs[0].rid))
            return batch
        scheduler.gracefully_exit = True
        return empty_batch

    def run_batch_without_model(batch, pp_proxy_tensors=None):
        scheduler.forward_ct += 1
        batch.forward_iter = scheduler.forward_ct
        if batch.forward_mode.is_decode():
            queued_batch, queued_result = scheduler.result_queue[0]
            events.append(
                (
                    "decode_launch",
                    len(req.output_ids),
                    queued_batch.forward_mode,
                    queued_batch.reqs[0].rid,
                    queued_result is prefill_result,
                )
            )
        scheduler._record_sm89_decode_residual_launch(batch)
        return decode_result if batch.forward_mode.is_decode() else prefill_result

    scheduler.get_new_batch_prefill = get_new_batch_prefill
    scheduler.update_running_batch = update_running_batch
    scheduler.run_batch = run_batch_without_model

    scheduler.event_loop_overlap()

    state = scheduler._sm89_decode_residual_marker_state
    assert events == [
        ("get_next_decode", PROFILE_RID),
        ("decode_launch", 0, ForwardMode.EXTEND, PROFILE_RID, True),
        ("processed", 1),
        ("processed", 2),
    ]
    assert len(scheduler.result_queue) == 0
    assert req.output_ids == [1, 2]
    assert state.decode_forward_count == 1
    assert state.accepted_decode_results == 1
    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]


def test_scheduler_hooks_classify_finished_overlap_result_as_discarded() -> None:
    emitted: list[str] = []
    scheduler = scheduler_with_state(emitted)
    req = target_req()
    req.output_ids = list(range(128))
    req.finished = lambda: True
    batch = target_batch(req, forward_id=128)
    state = scheduler._sm89_decode_residual_marker_state
    state.rid = PROFILE_RID
    state.completion_tokens = 128
    state.accepted_decode_results = 127
    state.decode_forward_count = 128
    state._accounted_forward_ids.update(range(1, 128))
    state._pending_forward_ids.add(128)

    observation = scheduler._prepare_sm89_decode_residual_result(batch)
    scheduler._complete_sm89_decode_residual_result(observation)

    assert state.discarded_decode_results == 1
    assert emitted[-1] == f"PROFILE_END rid={PROFILE_RID}"


@pytest.mark.parametrize("launched", (127, 128, 129))
def test_dynamic_overlap_accounting_does_not_assume_launched_count(
    launched: int,
) -> None:
    emitted: list[str] = []
    state = new_state(emitted)

    launch_forwards(state, launched)
    account_exact_completion(state, launched)
    state.assert_drained()

    discarded = launched - 127
    assert state.accepted_decode_results == 127
    assert state.discarded_decode_results == discarded
    assert state.decode_forward_count == launched
    assert state.accepted_decode_results + state.discarded_decode_results == launched
    assert emitted == [
        f"PROFILE_BEGIN rid={PROFILE_RID}",
        (
            f"PROFILE_ACCOUNTING rid={PROFILE_RID} completion_tokens=128 "
            f"accepted_intervals=127 decode_forward_count={launched} "
            f"discarded_overlap_forward_count={discarded}"
        ),
        f"PROFILE_END rid={PROFILE_RID}",
    ]


def test_end_waits_until_every_launched_target_result_is_discarded() -> None:
    emitted: list[str] = []
    state = new_state(emitted)
    launch_forwards(state, 128)

    for forward_id in range(127):
        state.record_result(
            forward_id=forward_id,
            rid=PROFILE_RID,
            accepted=True,
            completion_tokens=forward_id + 2,
        )

    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]
    with pytest.raises(RuntimeError, match="unaccounted"):
        state.assert_drained()
    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]


@pytest.mark.parametrize("duplicate_kind", ("launch", "result"))
def test_duplicate_accounting_fails_closed_before_accounting_or_end(
    duplicate_kind: str,
) -> None:
    emitted: list[str] = []
    state = new_state(emitted)
    state.record_launch(
        forward_id=0,
        rid=PROFILE_RID,
        completion_tokens=0,
        max_new_tokens=128,
    )
    if duplicate_kind == "result":
        state.record_result(
            forward_id=0,
            rid=PROFILE_RID,
            accepted=True,
            completion_tokens=2,
        )

    with pytest.raises(RuntimeError, match="duplicate"):
        if duplicate_kind == "launch":
            state.record_launch(
                forward_id=0,
                rid=PROFILE_RID,
                completion_tokens=0,
                max_new_tokens=128,
            )
        else:
            state.record_result(
                forward_id=0,
                rid=PROFILE_RID,
                accepted=True,
                completion_tokens=2,
            )

    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]


def test_missing_or_unknown_result_fails_closed() -> None:
    emitted: list[str] = []
    state = new_state(emitted)
    launch_forwards(state, 128)

    with pytest.raises(RuntimeError, match="unknown"):
        state.record_result(
            forward_id=999,
            rid=PROFILE_RID,
            accepted=False,
            completion_tokens=1,
        )

    assert emitted == [f"PROFILE_BEGIN rid={PROFILE_RID}"]


def test_late_append_after_profile_end_fails_closed() -> None:
    emitted: list[str] = []
    state = new_state(emitted)
    launch_forwards(state, 127)
    account_exact_completion(state, 127)

    with pytest.raises(RuntimeError, match="after PROFILE_END"):
        state.record_launch(
            forward_id=128,
            rid=PROFILE_RID,
            completion_tokens=128,
            max_new_tokens=128,
        )

    assert [line.split()[0] for line in emitted] == [
        "PROFILE_BEGIN",
        "PROFILE_ACCOUNTING",
        "PROFILE_END",
    ]


def test_marker_stage_never_resolves_or_calls_aggregate_abi() -> None:
    emitted: list[str] = []

    def unavailable(*args, **kwargs):
        raise AssertionError("marker stage touched aggregate ABI")

    state = marker_state_factory()(
        PROFILE_ENVIRONMENT,
        tp_rank=0,
        emit=emitted.append,
        begin_decode_aggregate=unavailable,
        end_decode_aggregate=unavailable,
    )
    assert state is not None
    launch_forwards(state, 127)
    account_exact_completion(state, 127)

    assert [line.split()[0] for line in emitted] == [
        "PROFILE_BEGIN",
        "PROFILE_ACCOUNTING",
        "PROFILE_END",
    ]


@pytest.mark.parametrize("forwards", (127, 128))
def test_aggregate_lifecycle_uses_scheduler_counts_and_compact_order(
    forwards: int,
) -> None:
    events: list[object] = []
    emitted: list[str] = []

    def begin(nonce: int) -> None:
        events.append(("begin", nonce))

    def end(nonce: int):
        events.append(("end", nonce))
        return aggregate_for_forwards(forwards)

    state = new_aggregate_state(emitted, begin, end)
    launch_forwards(state, forwards)
    account_exact_completion(state, forwards)

    aggregate_markers = [line for line in emitted if line.startswith("PROFILE_KT ")]
    assert events == [("begin", 0), ("end", 0)]
    assert len(aggregate_markers) == 1
    assert aggregate_markers[0] == (
        f"PROFILE_KT rid={PROFILE_RID} aggregate="
        f"{json.dumps(aggregate_for_forwards(forwards), sort_keys=True, separators=(',', ':'))}"
    )
    assert json.loads(aggregate_markers[0].split(" aggregate=", 1)[1]) == aggregate_for_forwards(
        forwards
    )
    assert [line.split()[0] for line in emitted] == [
        "PROFILE_BEGIN",
        "PROFILE_KT",
        "PROFILE_ACCOUNTING",
        "PROFILE_END",
    ]


def test_aggregate_end_waits_until_every_target_result_drains() -> None:
    emitted: list[str] = []
    end_calls: list[int] = []
    state = new_aggregate_state(
        emitted,
        lambda nonce: None,
        lambda nonce: end_calls.append(nonce) or aggregate_for_forwards(128),
    )
    launch_forwards(state, 128)

    for forward_id in range(127):
        state.record_result(
            forward_id=forward_id,
            rid=PROFILE_RID,
            accepted=True,
            completion_tokens=forward_id + 2,
        )

    assert end_calls == []
    assert [line.split()[0] for line in emitted] == ["PROFILE_BEGIN"]
    state.record_result(
        forward_id=127,
        rid=PROFILE_RID,
        accepted=False,
        completion_tokens=128,
    )
    assert end_calls == [0]


def test_aggregate_begin_failure_fails_closed_without_profile_begin() -> None:
    emitted: list[str] = []
    state = new_aggregate_state(
        emitted,
        lambda nonce: (_ for _ in ()).throw(RuntimeError("begin failed")),
        lambda nonce: aggregate_for_forwards(127),
    )

    with pytest.raises(RuntimeError, match="begin failed"):
        state.record_launch(
            forward_id=0,
            rid=PROFILE_RID,
            completion_tokens=0,
            max_new_tokens=128,
        )

    assert emitted == []
    with pytest.raises(RuntimeError, match="failed closed"):
        state.assert_drained()


@pytest.mark.parametrize("forward_id", (-1, True))
def test_aggregate_rejects_invalid_scheduler_nonce_before_begin(
    forward_id: int,
) -> None:
    emitted: list[str] = []
    begin_calls: list[int] = []
    state = new_aggregate_state(
        emitted,
        begin_calls.append,
        lambda nonce: aggregate_for_forwards(127),
    )

    with pytest.raises(RuntimeError, match="session nonce must be a nonnegative integer"):
        state.record_launch(
            forward_id=forward_id,
            rid=PROFILE_RID,
            completion_tokens=0,
            max_new_tokens=128,
        )

    assert begin_calls == []
    assert emitted == []


def test_aggregate_end_failure_fails_closed_without_completion_markers() -> None:
    emitted: list[str] = []
    state = new_aggregate_state(
        emitted,
        lambda nonce: None,
        lambda nonce: (_ for _ in ()).throw(RuntimeError("end failed")),
    )
    launch_forwards(state, 127)

    with pytest.raises(RuntimeError, match="end failed"):
        account_exact_completion(state, 127)

    assert [line.split()[0] for line in emitted] == ["PROFILE_BEGIN"]


@pytest.mark.parametrize(
    "kind",
    (
        "none",
        "missing_group",
        "extra_group",
        "missing_value",
        "extra_value",
        "bool",
        "negative",
        "scheduler_mismatch",
    ),
)
def test_aggregate_schema_and_scheduler_count_fail_closed(kind: str) -> None:
    emitted: list[str] = []
    aggregate = None if kind == "none" else aggregate_for_forwards(127)
    if kind == "missing_group":
        aggregate.pop("amx_m1")
    elif kind == "extra_group":
        aggregate["extra"] = {}
    elif kind == "missing_value":
        aggregate["cpuinfer"]["task"].pop("total")
    elif kind == "extra_value":
        aggregate["cpuinfer"]["task"]["total"]["extra"] = 0
    elif kind == "bool":
        aggregate["cpuinfer"]["task"]["total"]["count"] = True
    elif kind == "negative":
        aggregate["cpuinfer"]["task"]["total"]["ns"] = -1
    elif kind == "scheduler_mismatch":
        aggregate["cpuinfer"]["task"]["total"]["count"] = 75 * 127 - 1
    state = new_aggregate_state(emitted, lambda nonce: None, lambda nonce: aggregate)
    launch_forwards(state, 127)

    with pytest.raises(RuntimeError, match="aggregate"):
        account_exact_completion(state, 127)

    assert [line.split()[0] for line in emitted] == ["PROFILE_BEGIN"]
