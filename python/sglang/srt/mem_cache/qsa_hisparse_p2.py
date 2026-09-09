"""Guarded B2 eager QSA: one prefill arena, two persistent decode leases."""

import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.qsa_hisparse_slots import QSAHiSparseSlots
from sglang.srt.mem_cache.qsa_hisparse_v3 import (
    QSAHiSparseV3,
    pack_c4,
    stage_short_prefix,
    unpack_index,
    validate_configuration,
)


class _RequestCache(QSAHiSparseV3):
    """Reuse V3's selected/refetch/writeback algorithms on request-private views.

    Construction, handoff, batch metadata and release belong to the P2 adapter.
    The inherited single-owner lifecycle methods are never used here.
    """

    def __init__(self, adapter, lease):
        self.adapter, self.lease = adapter, lease
        self.pool, self.device = adapter.pool, adapter.device
        self.strict, self.capacity = adapter.strict, adapter.capacity
        self.generation = lease.generation
        self.copy_stream = adapter.copy_stream
        self.seq_len = self.decode_steps = 0
        self.offloaded = False
        self.states = []
        self.host = None if adapter.host_slabs is None else adapter.host_slabs[lease.slot]
        ring = adapter.slots.ring_slice(lease)
        self.full = SimpleNamespace(
            k_buffer=[k[ring] for k in adapter.full.k_buffer],
            v_buffer=[v[ring] for v in adapter.full.v_buffer],
        )
        if self.host is not None:
            self.compact_base = lease.slot * 2052
            self.compact = adapter.compact[:, self.compact_base:self.compact_base + 2052]
            self.compact_table = adapter.compact_table[lease.slot:lease.slot + 1]
            self.indices, self.gathered, self.unpacked = adapter.indices, adapter.gathered, adapter.unpacked
            self.zero_req, self.real = adapter.zero_req, adapter.real
            self.compressed_len = adapter.compressed_lens[lease.slot:lease.slot + 1]
            self.compact.zero_()
            self.compact_table.zero_()
            for ring_k, ring_v in zip(self.full.k_buffer, self.full.v_buffer):
                ring_k.zero_()
                ring_v.zero_()
        self.handoff_event = None

    def record(self, event, **extra):
        self.adapter.record(event, self.lease, seq_len=self.seq_len,
                            decode_steps=self.decode_steps, **extra)


class QSAHiSparseP2:
    is_qsa_p2 = True

    def __init__(self, runner, mode):
        if mode not in ("p2-offload", "p2-resident"):
            raise ValueError("QSA P2 requires an explicit resident/offload arm")
        self.runner, self.mode = runner, mode
        self.pool = runner.token_to_kv_pool
        self.full = self.pool.full_kv_pool
        validate_configuration(runner.server_args, self.pool, p2=True)
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
        from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
        from sglang.srt.model_executor.cuda_graph_config import cuda_graph_fully_disabled

        if (not str(self.pool.device).startswith("cuda")
                or not cuda_graph_fully_disabled()
                or type(self.full) is not MHATokenToKVPool
                or type(runner.token_to_kv_pool_allocator) is not PagedTokenToKVPoolAllocator
                or getattr(runner, "_unified_memory_pool", None) is not None
                or getattr(self.pool, "qsa_hisparse_v3", None) is not None):
            raise ValueError("QSA P2 needs one eager backend and independent static pools")
        if (getattr(runner.server_args, "enable_mixed_chunk", False)
                or getattr(runner.server_args, "enable_priority_preemption", False)):
            raise ValueError("QSA P2 forbids mixed prefill/decode and preemption")
        self.observe = os.environ.get("SGLANG_QSA_HISPARSE_V3_OBSERVE", "strict")
        if self.observe not in ("strict", "light") or os.environ.get("SGLANG_QSA_HISPARSE_V3_CAPTURE"):
            raise ValueError("QSA P2 requires strict/light observation without V3 capture")
        self.strict = self.observe == "strict"
        self.capacity = 262144
        self.slots = QSAHiSparseSlots(self.capacity, 64, 2)
        self.req_pool = runner.req_to_token_pool
        self.req_table = self.req_pool.req_to_token
        if self.req_table.shape[1] < self.capacity:
            raise ValueError("QSA P2 request table is smaller than a context")
        expected_raw = self.slots.raw_pool_size if mode == "p2-offload" else self.pool.size
        if (self.full.size != expected_raw or self.full.page_size != 64
                or self.full.dtype != self.pool.dtype or self.full.device != self.pool.device
                or self.full.head_num != 1 or self.full.head_dim != 256
                or self.full.layer_num != 12 or self.full.kv_cache_layout.lower() != "nhd"
                or any(tuple(t.shape) != (expected_raw + 64, 1, 256)
                       for t in self.full.k_buffer + self.full.v_buffer)):
            raise ValueError("QSA P2 raw backing geometry/capacity does not match the lease layout")
        self.device, self.rank = self.pool.device, runner.ps.tp_rank
        self.layer_ids = list(self.pool.full_attention_layer_id_mapping)
        self.raw_ptrs = [t.data_ptr() for t in self.full.k_buffer + self.full.v_buffer]
        self.index_ptr = self.pool.qsa_compressed_flat.data_ptr()
        self.copy_stream = torch.cuda.Stream(device=self.device)
        self.producer_stream = None
        self.requests = {}
        self.pending_releases = []
        self.batch_requests = []
        self.offloaded = False
        self.forward_id = 0
        self.raw_write_locs = None
        directory = os.environ.get("SGLANG_QSA_HISPARSE_V3_EVENTS")
        self.path = None
        if directory:
            Path(directory).mkdir(parents=True, exist_ok=True)
            self.path = Path(directory) / f"rank-{self.rank}.jsonl"
        start = time.monotonic()
        self.host_slabs = None
        host_alloc_wall_ms = 0.0
        self.workspace = []
        if mode == "p2-offload":
            self.host_slabs = torch.empty((2, 12, self.capacity // 4, 2048), dtype=torch.uint8, pin_memory=True)
            host_alloc_wall_ms = (time.monotonic() - start) * 1000
            self.indices = unpack_index(self.device)
            self.gathered = torch.empty((512, 2048), dtype=torch.uint8, device=self.device)
            self.unpacked = torch.empty((2, 2048, 1, 256), dtype=torch.uint8, device=self.device)
            self.compact = torch.zeros((2, 2 * 2052, 1, 256), dtype=torch.uint8, device=self.device)
            self.compact_table = torch.zeros((2, self.capacity), dtype=torch.int32, device=self.device)
            self.zero_req = torch.zeros(1, dtype=torch.int32, device=self.device)
            self.real = torch.ones(1, dtype=torch.int32, device=self.device)
            self.compressed_lens = torch.zeros(2, dtype=torch.int32, device=self.device)
            self.workspace = [self.indices, self.gathered, self.unpacked, self.compact,
                              self.compact_table, self.zero_req, self.real, self.compressed_lens]
        self.record("init", allocation_phase="startup", host_alloc_wall_ms=host_alloc_wall_ms)

    def record(self, event, lease=None, **extra):
        if self.path is None:
            return
        reserved = 0 if self.host_slabs is None else self.host_slabs.nbytes
        active = sum(0 if s.host is None else s.host.nbytes for s in self.requests.values())
        _, mamba_sizes, _ = self.req_pool.mamba_pool.get_contiguous_buf_infos()
        row = {
            "event": event, "time_ns": time.time_ns(), "rank": self.rank,
            "mode": self.mode, "observe": self.observe, "forward_id": self.forward_id,
            "req_pool_idx": None if lease is None else lease.req_pool_idx,
            "generation": None if lease is None else lease.generation,
            "rid": None if lease is None else lease.rid,
            "lease_slot": None if lease is None else lease.slot,
            "phase": None if lease is None else self.slots.phases.get(lease.req_pool_idx),
            **self.slots.snapshot(),
            "logical_capacity": self.pool.size,
            "logical_available": self.runner.token_to_kv_pool_allocator.available_size(),
            "raw_bytes": sum(t.nbytes for t in self.full.k_buffer + self.full.v_buffer),
            "raw_backing_size_tokens": self.full.size,
            "raw_ptrs": self.raw_ptrs, "index_ptr": self.index_ptr,
            "raw_storage_unchanged": self.raw_ptrs == [t.data_ptr() for t in self.full.k_buffer + self.full.v_buffer],
            "index_bytes": self.pool.qsa_compressed_flat.nbytes,
            "index_storage_unchanged": self.index_ptr == self.pool.qsa_compressed_flat.data_ptr(),
            "host_reserved_bytes": reserved, "host_bytes": active, "host_free_bytes": reserved - active,
            "pending_release_count": sum(self.slots.active.get(x.req_pool_idx) == x
                                         for x in self.pending_releases),
            "hot_bytes": sum(t["hot"].nbytes for s in self.requests.values() for t in s.states),
            "workspace_bytes": sum(t.nbytes for t in self.workspace),
            "mamba_bytes": sum(mamba_sizes),
            "mamba_available": self.req_pool.mamba_allocator.available_size(),
            "leases": [{"req_pool_idx": s.lease.req_pool_idx, "generation": s.lease.generation,
                        "rid": s.lease.rid, "slot": s.lease.slot,
                        "phase": self.slots.phases[s.lease.req_pool_idx],
                        "host_ptr": None if s.host is None else s.host.data_ptr(),
                        "hot_ptrs": [x["hot"].data_ptr() for x in s.states],
                        "ring_k_ptrs": [x.data_ptr() for x in s.full.k_buffer],
                        "ring_v_ptrs": [x.data_ptr() for x in s.full.v_buffer]}
                       for s in self.requests.values()],
            "cuda_allocated": torch.cuda.memory_allocated(self.device),
            "cuda_reserved": torch.cuda.memory_reserved(self.device),
            **extra,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def _request(self, req_idx, rid):
        state = self.requests[req_idx]
        lease = state.lease
        self.slots.require(lease)
        if lease.rid != rid or lease.generation != int(self.req_pool.req_generation[req_idx]):
            raise RuntimeError("QSA P2 request identity/generation changed")
        return state

    def begin_batch(self, batch):
        if batch.forward_mode.is_idle():
            self.batch_requests = []
            return
        decode = batch.forward_mode.is_decode()
        if not decode and not batch.forward_mode.is_extend():
            raise RuntimeError("QSA P2 supports plain extend/decode only")
        if (not 1 <= batch.batch_size <= 2 or (not decode and batch.batch_size != 1)
                or batch.req_pool_indices_cpu is None or batch.seq_lens_cpu is None
                or not batch.rids or len(batch.rids) != batch.batch_size):
            raise RuntimeError("QSA P2 requires real eager B1/B2 rows and CPU metadata")
        req_indices = batch.req_pool_indices_cpu.tolist()
        seq_lens = batch.seq_lens_cpu.tolist()
        if len(req_indices) != batch.batch_size or len(seq_lens) != batch.batch_size or len(set(req_indices)) != len(req_indices):
            raise RuntimeError("QSA P2 batch metadata aliases or pads requests")
        if self.strict and (batch.req_pool_indices.cpu().tolist() != req_indices
                            or batch.seq_lens.cpu().tolist() != seq_lens):
            raise RuntimeError("QSA P2 CPU/GPU row identities or lengths differ")
        states = []
        for req_idx, rid, seq in zip(req_indices, batch.rids, seq_lens):
            if not 1 <= seq <= self.capacity:
                raise ValueError("QSA P2 request exceeds context capacity")
            if req_idx not in self.requests:
                if decode:
                    raise RuntimeError("QSA P2 decode without an admitted lease")
                lease = self.slots.acquire(req_idx, int(self.req_pool.req_generation[req_idx]), rid)
                state = _RequestCache(self, lease)
                self.requests[req_idx] = state
                self.record("begin_prefill", lease)
            state = self._request(req_idx, rid)
            self.slots.require(state.lease, "decode" if decode else "prefill")
            states.append(state)
        # Validate the entire batch before advancing either request's decode state.
        self.forward_id += 1
        self.batch_requests = states
        for state, seq in zip(states, seq_lens):
            state.seq_len = seq
            if decode:
                state.decode_steps += 1
                if state.host is not None:
                    state.compressed_len.fill_(seq // 4)
        self.offloaded = decode and self.mode == "p2-offload"
        if self.mode == "p2-offload":
            if decode:
                locations = [self.slots.ring_write_location(s.lease, s.seq_len) for s in self.batch_requests]
                self.raw_write_locs = torch.tensor(locations, dtype=torch.int64, device=self.device)
                self.row_slots = torch.tensor([s.lease.slot for s in self.batch_requests], dtype=torch.int32, device=self.device)
            else:
                state = self.batch_requests[0]
                extend = batch.extend_seq_lens_cpu[0]
                segment = self.slots.staging_slice(state.lease, state.seq_len - extend, state.seq_len)
                self.raw_write_locs = torch.arange(segment.start, segment.stop, dtype=torch.int64, device=self.device)
        if decode:
            self.record("decode_batch", batch_size=len(self.batch_requests), rows=[{
                "req_pool_idx": s.lease.req_pool_idx, "generation": s.lease.generation,
                "rid": s.lease.rid, "lease_slot": s.lease.slot,
                "seq_len": s.seq_len, "tail": s.seq_len % 4,
                "compressed_len": s.seq_len // 4, "closes_c4": s.seq_len % 4 == 0,
                "ring_location": self.slots.ring_write_location(s.lease, s.seq_len),
            } for s in self.batch_requests])

    def write_locations(self, logical_locs):
        if self.mode == "p2-resident":
            return logical_locs
        if self.raw_write_locs is None or self.raw_write_locs.numel() != logical_locs.numel():
            raise RuntimeError("QSA P2 raw writer does not match current batch")
        return self.raw_write_locs

    def prefill_slots(self, req_idx, seq_len):
        if self.mode == "p2-resident":
            return self.req_table[req_idx, :seq_len].long()
        state = self.requests[req_idx]
        segment = self.slots.staging_slice(state.lease, 0, seq_len)
        return torch.arange(segment.start, segment.stop, dtype=torch.int64, device=self.device)

    def handoff(self, req):
        state = self._request(req.kv.req_pool_idx, req.rid)
        try:
            return self._handoff(req)
        except Exception:
            # Even a failed submission may have queued work before recording its event.
            try:
                self.copy_stream.synchronize()
            finally:
                self.slots.phases[state.lease.req_pool_idx] = "failed"
                self.record("handoff_failed", state.lease)
            raise

    def _handoff(self, req):
        state = self._request(req.kv.req_pool_idx, req.rid)
        lease, prompt_len = state.lease, state.seq_len
        if prompt_len < 2048:
            raise ValueError("QSA P2 requires at least 512 complete prefill C4 blocks")
        before = self.runner.token_to_kv_pool_allocator.available_size()
        self.slots.begin_handoff(lease)
        self.record("handoff_begin", lease, prompt_len=prompt_len)
        start = time.monotonic()
        if self.mode == "p2-offload":
            blocks, tail = divmod(prompt_len, 4)
            for li, lid in enumerate(self.layer_ids):
                k, v = self.full.k_buffer[li], self.full.v_buffer[li]
                for offset in range(0, blocks * 4, 4096):
                    stop = min(offset + 4096, blocks * 4)
                    segment = self.slots.staging_slice(lease, offset, stop)
                    kb, vb = k[segment].view(torch.uint8), v[segment].view(torch.uint8)
                    packed = pack_c4(kb, vb)
                    producer, done = torch.cuda.Event(), torch.cuda.Event()
                    producer.record()
                    with torch.cuda.stream(self.copy_stream):
                        self.copy_stream.wait_event(producer)
                        dst = state.host[li, offset // 4:stop // 4]
                        dst.copy_(packed, non_blocking=True)
                        done.record(self.copy_stream)
                    state.handoff_event = done
                    # Preserve the accepted bounded handoff's buffer lifetime.
                    done.synchronize()
                    if self.strict and (not torch.equal(dst[:, :1024].reshape(-1, 1, 256), kb.cpu())
                                        or not torch.equal(dst[:, 1024:].reshape(-1, 1, 256), vb.cpu())):
                        raise AssertionError("QSA P2 handoff K/V bytes differ")
                if tail:
                    segment = self.slots.staging_slice(lease, prompt_len - tail, prompt_len)
                    state.full.k_buffer[li][1:1 + tail].copy_(k[segment])
                    state.full.v_buffer[li][1:1 + tail].copy_(v[segment])
                layer_state = state.make_state()
                stage_short_prefix(layer_state["hot"], layer_state["tokens"], state.host[li, :blocks])
                layer_state["hot"][2048].copy_(state.host[li, blocks - 1], non_blocking=True)
                layer_state["tokens"][0, 2048] = blocks - 1
                state.states.append(layer_state)
                self.record("handoff_layer", lease, layer=lid, d2h_bytes=blocks * 2048, bytes_checked=self.strict)
            state.offloaded = True
        done = torch.cuda.Event()
        done.record()
        done.synchronize()
        state.handoff_event = done
        self.slots.finish_handoff(lease, done)
        if before != self.runner.token_to_kv_pool_allocator.available_size():
            raise AssertionError("QSA P2 handoff changed logical index ownership")
        self.record("handoff_complete", lease, prompt_len=prompt_len,
                    wall_ms=(time.monotonic() - start) * 1000,
                    host_slab_ptr=None if state.host is None else state.host.data_ptr())
        return lease, done

    def after_store(self, layer):
        if self.offloaded:
            for state in self.batch_requests:
                self.slots.require(state.lease, "decode")
                state.after_store(layer)

    def selected(self, layer, raw_indices):
        if raw_indices.shape != (len(self.batch_requests), 2051) or not self.offloaded:
            raise RuntimeError("QSA P2 selection rows do not match current decode batch")
        # ponytail: B2 eager resolves rows serially; measure before batching launches in P3.
        for row, state in enumerate(self.batch_requests):
            self.slots.require(state.lease, "decode")
            state.selected(layer, raw_indices[row:row + 1])
        return (self.compact[0].view(self.pool.dtype), self.compact[1].view(self.pool.dtype),
                self.compact_table, self.row_slots)

    def capture_decode(self, *args, **kwargs):
        # The V3 per-token tensor capture is explicitly rejected at P2 startup.
        return

    def release(self, req_idx, rid):
        if req_idx not in self.requests:
            if req_idx in self.slots.active:
                raise RuntimeError("QSA P2 cannot release a partially initialized lease")
            return None  # Allocated, but no model forward and no physical lease yet.
        state = self._request(req_idx, rid)
        self.record("release_begin", state.lease)
        terminal = torch.cuda.Event()
        if self.producer_stream is None:
            raise RuntimeError("QSA P2 release has no registered producer stream")
        terminal.record(self.producer_stream)
        events = [s["done"] for s in state.states if s["done"] is not None]
        if state.handoff_event is not None:
            events.append(state.handoff_event)
        self.slots.drain(state.lease, terminal, events)
        self.record("release_drained", state.lease, pending_events=0)
        return state.lease

    def after_release(self, lease):
        if lease is None:
            return
        self.slots.require(lease, "drained", "logical_flushed")
        allocator = self.runner.token_to_kv_pool_allocator
        if allocator.free_group is not None:
            self.slots.require(lease, "drained")
            if lease in self.pending_releases:
                raise RuntimeError("duplicate QSA P2 deferred release")
            self.pending_releases.append(lease)
            self.record("logical_release_pending", lease)
            return
        if self.slots.phases[lease.req_pool_idx] == "drained":
            self.slots.logical_flushed(lease, allocator)
        state = self.requests[lease.req_pool_idx]
        state.states.clear()
        state.host = None
        self.batch_requests = [s for s in self.batch_requests if s.lease != lease]
        self.slots.commit_release(lease)
        del self.requests[lease.req_pool_idx]
        self.record("logical_release_complete", lease)

    def after_logical_flush(self):
        while self.pending_releases:
            lease = self.pending_releases[0]
            try:
                self.after_release(lease)
            finally:
                # Keep unfinished leases reachable on failure. A ledger failure
                # after commit must not replay the already released generation.
                if lease.req_pool_idx not in self.slots.active:
                    self.pending_releases.pop(0)


class QSAHiSparseCoordinator:
    """QSA implementation of the existing staging/decode scheduler hooks."""

    is_qsa_p2 = True

    def __init__(self, adapter, tp_group):
        self.adapter, self.tp_group = adapter, tp_group
        self.ack_staging_queue = []
        initial_pair = os.environ.get("SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR", "0")
        if initial_pair not in ("0", "1"):
            raise ValueError("QSA P2 initial-pair validation flag must be 0 or 1")
        # Validation only: hold the first request until both prefills finish so
        # resident/offload controls have identical first-decode membership.
        self.wait_initial_pair = initial_pair == "1"
        self.num_real_reqs = torch.ones(1, dtype=torch.int32, device=adapter.device)

    def set_decode_producer_stream(self, stream):
        if stream is None:
            raise ValueError("QSA coordinator requires the real forward producer stream")
        self.adapter.producer_stream = stream

    def admit_request_into_staging(self, req):
        if getattr(req, "beam_group", None) is not None:
            raise ValueError("QSA P2 does not support beam/prefix ownership sharing")
        if self.adapter.producer_stream is None:
            raise RuntimeError("QSA handoff has no prefill producer stream")
        producer = torch.cuda.Event()
        producer.record(self.adapter.producer_stream)
        torch.cuda.current_stream(self.adapter.device).wait_event(producer)
        lease, event = self.adapter.handoff(req)
        req.hisparse_staging = True
        self.ack_staging_queue.append(SimpleNamespace(req=req, lease=lease, event=event))
        self.adapter.record("ready_parked", lease)

    def has_ongoing_staging(self):
        return bool(self.ack_staging_queue)

    def collect_ready_reqs(self):
        # An empty local queue must still rendezvous: the peer may have an item.
        total = torch.tensor(len(self.ack_staging_queue), dtype=torch.int64)
        torch.distributed.all_reduce(total, group=self.tp_group)
        if int(total) == 0:
            return []
        signatures = [(x.lease.req_pool_idx, x.lease.generation, x.lease.rid, x.lease.slot)
                      for x in self.ack_staging_queue]
        count = 0
        for item in self.ack_staging_queue:
            self.adapter.slots.require(item.lease, "host_ready")
            self.adapter._request(item.lease.req_pool_idx, item.lease.rid)
            if not item.event.query():
                break
            count += 1
        # Admission is rare; compare identities as well as the ready prefix count.
        world = torch.distributed.get_world_size(self.tp_group)
        votes = [None] * world
        torch.distributed.all_gather_object(
            votes, (signatures, count, self.wait_initial_pair), group=self.tp_group)
        if any(ids != signatures or barrier != self.wait_initial_pair
               for ids, _, barrier in votes):
            raise RuntimeError("QSA TP ranks disagree on staging lease identities")
        count = min(n for _, n, _ in votes)
        if self.wait_initial_pair:
            if count < 2:
                return []
            self.adapter.record("initial_pair_released", ordered_leases=signatures)
            self.wait_initial_pair = False
        ready, self.ack_staging_queue = self.ack_staging_queue[:count], self.ack_staging_queue[count:]
        for item in ready:
            self.adapter.slots.admit_decode(item.lease)
            item.req.hisparse_staging = False
            self.adapter.record("ready_converged", item.lease)
        return [item.req for item in ready]

    def map_last_loc_to_buffer(self, seq_lens, out_cache_loc, req_pool_indices,
                               seq_lens_cpu, req_pool_indices_cpu):
        # Keep out_cache_loc logical: index-K consumes it later in the model.
        for req_idx in req_pool_indices_cpu.tolist():
            state = self.adapter.requests[req_idx]
            self.adapter._request(req_idx, state.lease.rid)
            self.adapter.slots.require(state.lease, "decode")

    def wait_for_pending_backup(self):
        # Each request/layer waits its own writeback in the shared V3 algorithms.
        return

    def request_finished(self, req):
        # common.release_kv_cache owns the single drain -> logical flush -> release.
        self.adapter._request(req.kv.req_pool_idx, req.rid)

    def retract_req(self, req):
        raise RuntimeError("QSA P2 cannot retract an offloaded request into prefill")

    def destroy(self):
        if self.adapter.producer_stream is not None:
            self.adapter.producer_stream.synchronize()
        self.adapter.copy_stream.synchronize()
