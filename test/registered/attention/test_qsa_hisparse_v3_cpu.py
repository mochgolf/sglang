"""CPU checks for the actual V3 byte-layout helpers and startup guards."""

import unittest
import tempfile
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.mem_cache.qsa_hisparse_v3 import (
    QSAHiSparseV3,
    pack_c4,
    stage_short_prefix,
    unpack_index,
    validate_configuration,
)


class TestQSAHiSparseV3(unittest.TestCase):
    def test_light_observation_keeps_writeback_dependencies(self):
        for strict in (False, True):
            adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
            adapter.strict, adapter.device = strict, "cpu"
            adapter.offloaded, adapter.seq_len, adapter.generation = True, 2052, 1
            adapter.pool = SimpleNamespace(_transfer_full_attention_id=lambda _: 0)
            k = torch.arange(5 * 256).reshape(5, 1, 256).to(torch.uint8)
            v = (k + 17).clone()
            adapter.full = SimpleNamespace(k_buffer=[k], v_buffer=[v])
            adapter.capacity = 2056
            state = adapter.make_state()
            previous = Mock(name="previous_writeback")
            state["done"] = previous
            adapter.states = [state]
            adapter.host = torch.zeros((1, 514, 2048), dtype=torch.uint8)
            adapter.copy_stream = Mock(name="copy_stream")
            current = Mock(name="current_stream")
            producer, copy_begin, done = Mock(), Mock(), Mock()
            events = [producer, copy_begin, done] if strict else [producer, done]
            with patch.object(torch.cuda, "Event", side_effect=events), \
                    patch.object(torch.cuda, "current_stream", return_value=current), \
                    patch.object(torch.cuda, "stream", return_value=nullcontext()):
                adapter.after_store(SimpleNamespace(layer_id=3))
            current.wait_event.assert_called_once_with(previous)
            producer.record.assert_called_once_with()
            adapter.copy_stream.wait_event.assert_called_once_with(producer)
            done.record.assert_called_once_with(adapter.copy_stream)
            done.synchronize.assert_not_called()
            self.assertIs(state["done"], done)
            self.assertEqual(state["writeback_bytes"], 2048)
            self.assertTrue(torch.equal(adapter.host[0, 512, :1024], k[1:5].flatten()))
            self.assertTrue(torch.equal(adapter.host[0, 512, 1024:], v[1:5].flatten()))
            # A light per-step record must not inspect pool/memory or open files.
            adapter.path = Path("must-not-write.jsonl")
            if not strict:
                for event in ("decode_step", "prefill_step", "selected_check"):
                    adapter.record(event)
            # Boundary evidence must remain active in either observation mode.
            with self.assertRaises(AttributeError):
                adapter.record("handoff_complete")

    def test_observation_rejects_invalid_or_contaminated_mode(self):
        # Stop at pool validation, before constructing CUDA resources.
        for env, expected in (({}, "strict"),
                              ({"SGLANG_QSA_HISPARSE_V3_OBSERVE": "light"}, "light")):
            adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
            with patch.dict("os.environ", env, clear=True), patch(
                "sglang.srt.mem_cache.qsa_hisparse_v3.validate_configuration",
                side_effect=RuntimeError("CPU configuration boundary"),
            ), self.assertRaisesRegex(RuntimeError, "CPU configuration boundary"):
                adapter.__init__(SimpleNamespace(token_to_kv_pool=None, server_args=None), "offload")
            self.assertEqual(adapter.observe, expected)
            self.assertEqual(adapter.strict, expected == "strict")
        for env in (
            {"SGLANG_QSA_HISPARSE_V3_OBSERVE": "typo"},
            {"SGLANG_QSA_HISPARSE_V3_OBSERVE": "light",
             "SGLANG_QSA_HISPARSE_V3_CAPTURE": "/tmp/capture"},
        ):
            with patch.dict("os.environ", env, clear=True), self.assertRaises(ValueError):
                QSAHiSparseV3(None, "offload")

    def test_decode_ring_preserves_writer_padding(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.failed = adapter.releasing = False
        adapter.owner, adapter.owner_rid = 0, "ring-check"
        adapter.mode, adapter.offloaded = "offload", True
        adapter.decode_steps = 0
        adapter.ring_loc = torch.zeros(1, dtype=torch.int64)
        adapter.compressed_len = torch.zeros(1, dtype=torch.int32)
        adapter.record = lambda *args, **kwargs: None
        mode = SimpleNamespace(is_idle=lambda: False, is_decode=lambda: True)
        ring = torch.zeros(5, dtype=torch.int64)
        for position in range(261120, 261128):
            batch = SimpleNamespace(forward_mode=mode, batch_size=1,
                                    req_pool_indices=torch.tensor([0]),
                                    rids=["ring-check"], seq_lens=torch.tensor([position + 1]))
            adapter.begin_batch(batch)
            slot = int(adapter.ring_loc[0])
            # The real CUDA writer's reserved_skip_index defaults to zero.
            self.assertIn(slot, (1, 2, 3, 4))
            ring[slot] = position
            tail = (position + 1) % 4
            count = tail or 4
            self.assertEqual(ring[1:1 + count].tolist(),
                             list(range(position + 1 - count, position + 1)))
            self.assertEqual(int(ring[0]), 0)
        self.assertEqual(adapter.decode_steps, 8)

    def test_capture_uses_actual_consumer_length(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.decode_steps, adapter.seq_len = 1, 2049
        adapter.layer_ids, adapter.capture_layers = [3], {}
        adapter.owner_rid, adapter.generation, adapter.rank = "capture-check", 1, 0
        q = torch.ones((1, 12, 256), dtype=torch.bfloat16)
        packed = torch.zeros((2051, 1, 256), dtype=torch.bfloat16)
        with tempfile.TemporaryDirectory() as directory:
            adapter.capture_dir = Path(directory)
            adapter.capture_decode(
                SimpleNamespace(layer_id=3), q, packed, packed + 1,
                torch.arange(2051)[None], q + 2, 0.5, 2.0,
                torch.tensor([3]), torch.tensor([0, 1]), torch.tensor([0, 3]),
            )
            paths = list(adapter.capture_dir.glob("*.pt"))
            saved = torch.load(paths[0], weights_only=True)["layers"][3]
            # Observe cu_k, even if a bug makes it disagree with sequence-derived counts.
            self.assertEqual(saved["k"].shape[0], 3)
            self.assertTrue(torch.equal(saved["output"], q + 2))
            self.assertEqual(saved["k_stride"], packed.stride())

    def test_layout_and_tail(self):
        # Independent raw source rows; neither expectation uses candidate indices.
        raw = torch.arange(2052 * 256, dtype=torch.int64).reshape(2052, 1, 256)
        k = (raw % 251).to(torch.uint8)
        v = ((raw * 7 + 13) % 251).to(torch.uint8)
        packed = pack_c4(k[:2048], v[:2048])
        for block in (0, 15, 16, 511):
            self.assertTrue(torch.equal(packed[block, :1024], k[block * 4:block * 4 + 4].flatten()))
            self.assertTrue(torch.equal(packed[block, 1024:], v[block * 4:block * 4 + 4].flatten()))
        got = packed.view(-1, 256).index_select(0, unpack_index("cpu")).view(2, 2048, 1, 256)
        self.assertTrue(torch.equal(got[0], k[:2048]))
        self.assertTrue(torch.equal(got[1], v[:2048]))
        for tail in range(4):
            compact = torch.zeros((2, 2052, 1, 256), dtype=torch.uint8)
            compact[:, 1:2049].copy_(got)
            compact[0, 2049:2049 + tail].copy_(k[2048:2048 + tail])
            compact[1, 2049:2049 + tail].copy_(v[2048:2048 + tail])
            self.assertTrue(torch.equal(compact[0, 1:2049 + tail], k[:2048 + tail]))
            self.assertTrue(torch.equal(compact[1, 1:2049 + tail], v[:2048 + tail]))
        with self.assertRaises(ValueError):
            pack_c4(k[:3], v[:3])
        # The real resolver bypasses H2D at <=2048 C4: its ordered rows must exist.
        records = torch.arange(2049 * 8).reshape(2049, 8)
        for count in (512, 2048, 2049):
            hot = torch.full((2112, 8), -1)
            tokens = torch.full((1, 2112), -1, dtype=torch.int32)
            stage_short_prefix(hot, tokens, records[:count])
            if count <= 2048:
                self.assertTrue(torch.equal(hot[:count], records[:count]))
                self.assertTrue(torch.equal(tokens[0, :count], torch.arange(count)))
            else:
                self.assertTrue(torch.all(hot == -1))

    def test_configuration(self):
        args = SimpleNamespace(max_running_requests=1, tp_size=2, pp_size=1,
                               disable_radix_cache=True, disable_overlap_schedule=True,
                               cuda_graph_backend_decode="disabled", cuda_graph_backend_prefill="disabled",
                               context_length=262144, max_total_tokens=262144,
                               chunked_prefill_size=2048, skip_server_warmup=True,
                               enable_deterministic_inference=True,
                               random_seed=147342228,
                               speculative_algorithm=None,
                               disaggregation_mode="null")
        full = SimpleNamespace(use_hnd=False, kv_cache_layout="NHD",
                               post_capture_active=False, is_quantized_kv_cache=False)
        pool = SimpleNamespace(full_kv_pool=full, size=262144, page_size=64, qsa_compress_ratio=4,
                               qsa_token_topk=2048, full_layer_nums=12, head_num=1,
                               head_dim=256, dtype=torch.float8_e4m3fn)
        validate_configuration(args, pool)
        for name, bad in (("max_running_requests", 2), ("disable_radix_cache", False),
                          ("disable_overlap_schedule", False), ("cuda_graph_backend_decode", "full"),
                          ("cuda_graph_backend_prefill", "full"), ("context_length", 8192),
                          ("max_total_tokens", 8192), ("chunked_prefill_size", 4096),
                          ("skip_server_warmup", False), ("enable_streaming_session", True),
                          ("enable_deterministic_inference", False),
                          ("random_seed", 42),
                          ("speculative_algorithm", "NEXTN"), ("enable_hisparse", True)):
            changed = SimpleNamespace(**vars(args))
            setattr(changed, name, bad)
            with self.assertRaises(ValueError):
                validate_configuration(changed, pool)

    def test_stale_release(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.owner, adapter.owner_rid, adapter.generation = 1, "B", 2
        adapter.releasing = False
        adapter.pending_release = None
        allocator = SimpleNamespace(free_group=None)
        adapter.runner = SimpleNamespace(token_to_kv_pool_allocator=allocator)
        with self.assertRaises(RuntimeError):
            adapter.release(1, "A")
        adapter.releasing = True
        with self.assertRaises(RuntimeError):
            adapter.after_release((1, "B", 1))
        self.assertEqual((adapter.owner, adapter.owner_rid, adapter.generation), (1, "B", 2))
        adapter.record = lambda *args, **kwargs: None
        adapter.after_release((1, "B", 2))
        self.assertIsNone(adapter.owner)
        self.assertFalse(adapter.releasing)
        # Exercise the real allocator flush: the lease stays held until free runs.
        from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator

        adapter.owner, adapter.owner_rid, adapter.generation = 1, "C", 3
        adapter.releasing = True
        allocator = PagedTokenToKVPoolAllocator(
            size=192, page_size=64, dtype=torch.uint8, device="cpu",
            kvcache=SimpleNamespace(qsa_hisparse_v3=adapter), need_sort=True,
        )
        adapter.runner.token_to_kv_pool_allocator = allocator
        events = []
        adapter.record = lambda event: events.append((event, allocator.available_size()))
        rows = allocator.alloc(128)
        allocator.free_group_begin()
        allocator.free(rows[:64])
        allocator.free_segment(rows[64:], start_pos=64)
        adapter.after_release((1, "C", 3))
        self.assertEqual(adapter.owner_rid, "C")
        self.assertEqual(allocator.available_size(), 64)
        allocator.free_group_end()
        self.assertEqual(events[-1], ("logical_release_complete", 192))
        self.assertIsNone(adapter.owner)


if __name__ == "__main__":
    unittest.main()
