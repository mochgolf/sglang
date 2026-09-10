"""Scaled CPU runtime checks. CUDA events/writer/resolver are substitutes.

The real adapter, coordinator, backend addressing, pools and postflush hooks run;
this is neither a DMA/kernel check nor live TP2/service acceptance.
"""

from contextlib import nullcontext
import os
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.mem_cache.qsa_hisparse_p2 import QSAHiSparseP2, QSAHiSparseCoordinator
from sglang.srt.mem_cache.qsa_hisparse_slots import QSAHiSparseSlots
from sglang.srt.mem_cache.qsa_kv_pool import QSATokenToKVPool
from sglang.srt.mem_cache.qsa_hisparse_v3 import unpack_index


class Event:
    def __init__(self, **kwargs):
        self.recorded = False

    def record(self, stream=None):
        self.recorded = True

    def query(self):
        return self.recorded

    def synchronize(self):
        assert self.recorded, "cannot drain an unrecorded event"

    def elapsed_time(self, other):
        return 0.0


class TestQSAHiSparseP2(unittest.TestCase):
    def test_ready_batch_preserves_position_metadata(self):
        from array import array

        from sglang.srt.managers.schedule_batch import Req
        from sglang.srt.managers.scheduler import Scheduler
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        reqs = [Req(str(i), "", array("q", range(n)), SamplingParams(max_new_tokens=4))
                for i, n in enumerate((7, 10, 13), 1)]
        for i, req in enumerate(reqs, 1):
            req.kv.req_pool_idx = i
            req.output_ids = [42]
        scheduler = SimpleNamespace(
            device="cpu", req_to_token_pool=SimpleNamespace(device="cpu"),
            token_to_kv_pool_allocator=None, tree_cache=None,
            model_config=SimpleNamespace(vocab_size=128, is_encoder_decoder=False),
            enable_overlap=False, spec_algorithm=SpeculativeAlgorithm.NONE,
            future_map=Mock(),
        )

        def build(rows):
            # Sampling and token relay are unrelated to metadata propagation.
            with patch("sglang.srt.managers.scheduler.SamplingBatchInfo.from_schedule_batch",
                       return_value=Mock()), patch(
                    "sglang.srt.managers.schedule_batch.get_spec",
                    return_value=SimpleNamespace(speculative_algorithm=None)):
                batch = Scheduler._build_hisparse_decode_batch(scheduler, rows)
            self.assertEqual(len(batch.multimodal_inputs), len(rows))
            for actual, req in zip(batch.multimodal_inputs, rows):
                self.assertIs(actual, req.multimodal_inputs)
            return batch

        def positions(batch, expected):
            forward = ForwardBatch.__new__(ForwardBatch)
            # First decode consumes the prefill output at the next sequence length.
            forward.seq_lens = batch.seq_lens + 1
            forward.seq_lens_cpu = batch.seq_lens_cpu + 1
            with patch("sglang.srt.model_executor.forward_batch_info.get_exec",
                       return_value=SimpleNamespace(
                           deterministic=SimpleNamespace(rl_on_policy_target=None))):
                forward._compute_mrope_positions_decode(scheduler, batch)
            self.assertTrue(torch.equal(
                forward.mrope_positions, torch.tensor([expected] * 3)))

        positions(build(reqs[:1]), [7])
        positions(build([reqs[1], reqs[0]]), [10, 7])
        # Distinct metadata proves row/object preservation, not just None padding.
        reqs[1].multimodal_inputs = SimpleNamespace(
            mrope_positions=None, mrope_position_delta=torch.tensor(5))
        batch = build([reqs[1], reqs[0]])
        positions(batch, [15, 7])
        batch.filter_batch(keep_indices=[0])
        self.assertIs(batch.multimodal_inputs[0], reqs[1].multimodal_inputs)
        batch.merge_batch(build([reqs[2]]))
        self.assertEqual(batch.reqs, [reqs[1], reqs[2]])
        self.assertIs(batch.multimodal_inputs[0], reqs[1].multimodal_inputs)
        self.assertIsNone(batch.multimodal_inputs[1])
        positions(batch, [15, 13])

    def test_batch_isolation_and_real_postflush(self):
        a = QSAHiSparseP2.__new__(QSAHiSparseP2)
        a.mode, a.device, a.rank, a.strict, a.observe = "p2-offload", "cpu", 0, True, "strict"
        a.capacity = 2112
        a.slots = QSAHiSparseSlots(a.capacity, 64, 2)
        a.full = MHATokenToKVPool(
            a.slots.raw_pool_size, 64, torch.float8_e4m3fn, 1, 256, 1,
            "cpu", False, enable_alt_stream=False,
        )
        a.req_pool = ReqToTokenPool(2, a.capacity, "cpu", False)
        # Native mapping-shaped CPU tensors; this scenario checks observation,
        # not Mamba allocation or state numerics (those require the live model).
        a.req_pool.enable_mamba_extra_buffer = False
        a.req_pool.req_index_to_mamba_index_mapping = torch.tensor([0, 7, 9])
        a.req_table = a.req_pool.req_to_token
        a.pool = QSATokenToKVPool(
            size=2 * a.capacity, dtype=torch.float8_e4m3fn, page_size=64,
            head_num=1, head_dim=256, full_attention_layer_ids=[3], device="cpu",
            mamba_pool=SimpleNamespace(), qsa_index_kv_heads=1, qsa_index_head_dim=128,
            qsa_compress_ratio=4, qsa_token_topk=2048, num_request_slots=3,
            full_kv_pool=a.full,
        )
        a.pool.qsa_hisparse_v3 = a
        logical = PagedTokenToKVPoolAllocator(2 * a.capacity, 64, a.pool.dtype, "cpu", a.pool, False)
        a.runner = SimpleNamespace(token_to_kv_pool_allocator=logical)
        a.copy_stream, a.producer_stream = Mock(), Mock()
        a.requests, a.pending_releases, a.batch_requests = {}, [], []
        a.offloaded, a.forward_id, a.raw_write_locs = False, 0, None
        a.path, a.layer_ids = None, [3]
        a.host_slabs = torch.empty((2, 1, a.capacity // 4, 2048), dtype=torch.uint8)
        a.indices = unpack_index("cpu")
        a.gathered = torch.empty((512, 2048), dtype=torch.uint8)
        a.unpacked = torch.empty((2, 2048, 1, 256), dtype=torch.uint8)
        a.compact = torch.zeros((2, 4104, 1, 256), dtype=torch.uint8)
        a.compact_table = torch.zeros((2, a.capacity), dtype=torch.int32)
        a.zero_req, a.real = torch.zeros(1, dtype=torch.int32), torch.ones(1, dtype=torch.int32)
        a.compressed_lens = torch.zeros(2, dtype=torch.int32)
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend.hisparse_v3, backend.token_to_kv_pool = a, a.pool
        layer = SimpleNamespace(layer_id=3)

        def cpu_writer(layer, loc, k, v):
            a.full.k_buffer[0].view(torch.uint8)[loc] = k
            a.full.v_buffer[0].view(torch.uint8)[loc] = v

        def row(req, seq):
            return SimpleNamespace(req=req, seq=seq)

        def batch(rows, decode, extend=None):
            indices = torch.tensor([r.req.kv.req_pool_idx for r in rows])
            return SimpleNamespace(
                batch_size=len(rows), req_pool_indices_cpu=indices,
                req_pool_indices=indices, rids=[r.req.rid for r in rows],
                seq_lens_cpu=torch.tensor([r.seq for r in rows]),
                seq_lens=torch.tensor([r.seq for r in rows]),
                extend_seq_lens_cpu=[extend] if extend is not None else None,
                forward_mode=SimpleNamespace(is_idle=lambda: False,
                    is_decode=lambda: decode, is_extend=lambda: not decode),
            )

        def claim(rid):
            req_idx = a.req_pool.alloc_rows(1)[0]
            rows = logical.alloc(a.capacity)
            a.req_table[req_idx] = rows.to(torch.int32)
            return SimpleNamespace(rid=rid, kv=SimpleNamespace(req_pool_idx=req_idx)), rows

        def votes(out, vote, **kwargs):
            out[:] = [vote, vote]

        def resolver(*args):
            # Only the short fully resident prefix path is modeled on CPU.
            blocks, tokens, host_locs, device_locs, host, hot, out = args[:7]
            assert torch.equal(tokens[0, blocks[0].long()], blocks[0])
            out.copy_(blocks)

        with patch.object(torch.cuda, "Event", Event), \
                patch.object(torch.cuda, "stream", side_effect=lambda _: nullcontext()), \
                patch.object(torch.cuda, "current_stream", return_value=Mock()), \
                patch.object(a.pool, "set_kv_buffer", side_effect=cpu_writer), \
                patch.object(torch.distributed, "get_world_size", return_value=2), \
                patch.object(torch.distributed, "all_reduce"), \
                patch.object(torch.distributed, "all_gather_object", side_effect=votes), \
                patch("sglang.kernels.ops.kvcache.hisparse.load_cache_to_device_buffer_mla", side_effect=resolver):
            coord = QSAHiSparseCoordinator(a, None)
            self.assertFalse(coord.wait_initial_pair)
            with patch.dict(os.environ, {"SGLANG_QSA_P2_VALIDATE_INITIAL_PAIR": "1"}):
                coord = QSAHiSparseCoordinator(a, None)
            coord.set_decode_producer_stream(a.producer_stream)
            def peer_nonempty(out, vote, **kwargs):
                out[:] = [vote, ([(1, 1, "peer-only", 0)], 1, vote[2])]
            with patch.object(torch.distributed, "all_reduce", side_effect=lambda total, **kw: total.fill_(1)), \
                    patch.object(torch.distributed, "all_gather_object", side_effect=peer_nonempty), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            req_a, ar = claim("A")
            req_b = None
            saved_a = None
            for req, rows, length, kbyte, vbyte in ((req_a, ar, 2049, 7, 19),):
                for end, extend in ((2048, 2048), (length, length - 2048)):
                    a.begin_batch(batch([row(req, end)], False, extend))
                    backend._store_kv(layer, rows[end - extend:end],
                        torch.full((extend, 1, 256), kbyte, dtype=torch.uint8),
                        torch.full((extend, 1, 256), vbyte, dtype=torch.uint8))
                coord.admit_request_into_staging(req)
                # A remote rank that is not ready cannot admit local-ready A.
                def slow(out, vote, **kwargs):
                    out[:] = [vote, (vote[0], 0, vote[2])]
                with patch.object(torch.distributed, "all_gather_object", side_effect=slow):
                    self.assertEqual(coord.collect_ready_reqs(), [])
                self.assertEqual(coord.collect_ready_reqs(), [])
                self.assertEqual(a.slots.phases[req.kv.req_pool_idx], "host_ready")
                self.assertIsNone(a.slots.prefill_owner)
                saved_a = a.requests[req.kv.req_pool_idx].host.clone()

            req_b, br = claim("B")
            a.begin_batch(batch([row(req_b, 2050)], False, 2050))
            backend._store_kv(layer, br[:2050], torch.full((2050, 1, 256), 31, dtype=torch.uint8),
                              torch.full((2050, 1, 256), 43, dtype=torch.uint8))
            physical = a.prefill_slots(req_b.kv.req_pool_idx, 2050)
            self.assertTrue(torch.all(a.full.k_buffer[0].view(torch.uint8)[physical] == 31))
            # B's first logical slots numerically fit, but alias ring storage.
            wrong = a.full.k_buffer[0].view(torch.uint8)[br[:4]]
            self.assertFalse(torch.all(wrong == 31))
            self.assertTrue(torch.equal(a.requests[req_a.kv.req_pool_idx].host, saved_a))
            coord.admit_request_into_staging(req_b)
            def mismatch(out, vote, **kwargs):
                out[:] = [vote, ([(999, 1, "wrong", 0)], 1, vote[2])]
            with patch.object(torch.distributed, "all_gather_object", side_effect=mismatch), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            def barrier_mismatch(out, vote, **kwargs):
                out[:] = [vote, (vote[0], vote[1], not vote[2])]
            with patch.object(torch.distributed, "all_gather_object", side_effect=barrier_mismatch), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            def slot_mismatch(out, vote, **kwargs):
                ids = [(*identity[:3], 1 - identity[3]) for identity in vote[0]]
                out[:] = [vote, (ids, vote[1], vote[2])]
            with patch.object(torch.distributed, "all_gather_object", side_effect=slot_mismatch), \
                    self.assertRaisesRegex(RuntimeError, "identities"):
                coord.collect_ready_reqs()
            self.assertEqual(coord.collect_ready_reqs(), [req_a, req_b])
            self.assertFalse(coord.wait_initial_pair)

            state_b = a.requests[req_b.kv.req_pool_idx]
            a.req_pool.req_index_to_mamba_index_mapping[req_b.kv.req_pool_idx] = 9
            native = a.native_lease_snapshot(state_b, include_pages=True)
            self.assertEqual(native["logical_row_ptr"], a.req_table[req_b.kv.req_pool_idx].data_ptr())
            self.assertEqual(native["logical_page_ids"], (br[:2050:64] // 64).tolist())
            self.assertEqual(native["mamba_pool_idx"], 9)
            self.assertFalse(native["mamba_extra_buffer_enabled"])
            self.assertEqual(native["mamba_track_slots"], [])
            # Same backing pointer does not imply an unchanged native mapping.
            a.req_table[req_b.kv.req_pool_idx, 0] = ar[0]
            a.req_pool.req_index_to_mamba_index_mapping[req_b.kv.req_pool_idx] = 7
            changed = a.native_lease_snapshot(state_b, include_pages=True)
            self.assertEqual(changed["logical_row_ptr"], native["logical_row_ptr"])
            self.assertNotEqual(changed["logical_page_ids"], native["logical_page_ids"])
            self.assertNotEqual(changed["mamba_pool_idx"], native["mamba_pool_idx"])
            a.req_table[req_b.kv.req_pool_idx, 0] = br[0] + 1
            with self.assertRaisesRegex(RuntimeError, "not aligned"):
                a.native_lease_snapshot(state_b, include_pages=True)
            a.req_table[req_b.kv.req_pool_idx, 0] = br[0]
            a.req_pool.req_index_to_mamba_index_mapping[req_b.kv.req_pool_idx] = 9
            a.req_pool.enable_mamba_extra_buffer = True
            a.req_pool.req_index_to_mamba_ping_pong_track_buffer_mapping = torch.tensor([[0], [11], [13]])
            a.req_pool.req_index_to_mamba_ping_pong_track_buffer_mapping[req_b.kv.req_pool_idx] = 13
            self.assertEqual(a.native_lease_snapshot(state_b)["mamba_track_slots"], [13])
            a.req_pool.enable_mamba_extra_buffer = False

            a_steps = a.requests[req_a.kv.req_pool_idx].decode_steps
            a.req_pool.req_generation[req_b.kv.req_pool_idx] += 1
            with self.assertRaisesRegex(RuntimeError, "stale native"):
                a.native_lease_snapshot(state_b)
            with self.assertRaisesRegex(RuntimeError, "generation changed"):
                a.begin_batch(batch([row(req_a, 2050), row(req_b, 2051)], True))
            self.assertEqual(a.requests[req_a.kv.req_pool_idx].decode_steps, a_steps)
            a.req_pool.req_generation[req_b.kv.req_pool_idx] -= 1

            for seq_a, seq_b in ((2050, 2051), (2051, 2052), (2052, 2053), (2053, 2054)):
                rows = [row(req_b, seq_b), row(req_a, seq_a)]
                a.begin_batch(batch(rows, True))
                loc = torch.tensor([br[seq_b - 1], ar[seq_a - 1]])
                keep_logical = loc.clone()
                coord.map_last_loc_to_buffer(None, loc, None, None,
                                             torch.tensor([req_b.kv.req_pool_idx, req_a.kv.req_pool_idx]))
                self.assertTrue(torch.equal(loc, keep_logical))
                backend._store_kv(layer, loc, torch.tensor([31, 7], dtype=torch.uint8)[:, None, None].expand(2, 1, 256),
                                 torch.tensor([43, 19], dtype=torch.uint8)[:, None, None].expand(2, 1, 256))
                a.after_store(layer)
                topk = []
                for seq in (seq_b, seq_a):
                    ids = torch.cat((torch.arange(511), torch.tensor([seq // 4 - 1])))
                    tail = seq % 4
                    topk.append(torch.cat(((ids[:, None] * 4 + torch.arange(4)).flatten(),
                                          torch.arange(seq - tail, seq), torch.full((3 - tail,), -1))))
                topk = torch.stack(topk).to(torch.int32)
                k, v, table, physical_rows = a.selected(layer, topk)
                for r, seq, kb, vb in ((0, seq_b, 31, 43), (1, seq_a, 7, 19)):
                    locs = table[physical_rows[r], topk[r, :2048 + seq % 4].long()].long()
                    self.assertTrue(torch.all(k.view(torch.uint8)[locs] == kb))
                    self.assertTrue(torch.all(v.view(torch.uint8)[locs] == vb))
                self.assertNotEqual(int(physical_rows[0]), int(physical_rows[1]))

            state_b = a.requests[req_b.kv.req_pool_idx]
            host_b = state_b.host.clone()
            lease_a = a.release(req_a.kv.req_pool_idx, "A")
            logical.free_group_begin()
            logical.free(ar)
            a.after_release(lease_a)
            self.assertEqual(a.slots.phases[lease_a.req_pool_idx], "drained")
            self.assertEqual(len(a.slots.free_slots), 0)
            logical.free_group_end()  # Executes the real P2 allocator callback.
            a.req_pool.free_rows([req_a.kv.req_pool_idx])
            self.assertEqual(len(a.slots.free_slots), 1)
            self.assertIs(a.requests[req_b.kv.req_pool_idx], state_b)
            self.assertTrue(torch.equal(state_b.host, host_b))
            req_c, cr = claim("C")
            a.begin_batch(batch([row(req_c, 2048)], False, 2048))
            self.assertEqual(req_c.kv.req_pool_idx, req_a.kv.req_pool_idx)
            self.assertEqual(a.requests[req_c.kv.req_pool_idx].generation, lease_a.generation + 1)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                a.after_release(lease_a)
            logical.free_group_begin()
            for req, rows in ((req_c, cr), (req_b, br)):
                lease = a.release(req.kv.req_pool_idx, req.rid)
                logical.free(rows)
                a.after_release(lease)
            self.assertEqual(len(a.pending_releases), 2)
            first, second = a.pending_releases
            # A pre-commit failure retains both leases, including the current
            # logical_flushed lease; native pages have already flushed once.
            with patch.object(a.slots, "commit_release", side_effect=RuntimeError("commit fault")), \
                    self.assertRaisesRegex(RuntimeError, "commit fault"):
                logical.free_group_end()
            self.assertEqual(a.pending_releases, [first, second])
            self.assertEqual(a.slots.phases[first.req_pool_idx], "logical_flushed")
            self.assertEqual(a.slots.phases[second.req_pool_idx], "drained")
            self.assertEqual(a.slots.free_slots, [])
            self.assertEqual(logical.available_size(), 2 * a.capacity)

            def record_fault(event, *args, **kwargs):
                if event == "logical_release_complete":
                    raise RuntimeError("ledger fault")

            # A post-commit ledger failure removes only the completed head.
            with patch.object(a, "record", side_effect=record_fault), \
                    self.assertRaisesRegex(RuntimeError, "ledger fault"):
                a.after_logical_flush()
            self.assertEqual(a.pending_releases, [second])
            self.assertNotIn(first.req_pool_idx, a.requests)
            self.assertEqual(a.slots.active[second.req_pool_idx], second)
            self.assertEqual(a.slots.phases[second.req_pool_idx], "drained")
            self.assertEqual(len(a.slots.free_slots), 1)
            a.after_logical_flush()
            self.assertEqual(a.pending_releases, [])
            self.assertEqual(a.requests, {})
            self.assertEqual(logical.available_size(), 2 * a.capacity)
            self.assertEqual(len(a.slots.free_slots), 2)


if __name__ == "__main__":
    unittest.main()
