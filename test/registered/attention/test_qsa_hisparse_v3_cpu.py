"""CPU checks for the actual V3 byte-layout helpers and startup guards."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.qsa_hisparse_v3 import (
    QSAHiSparseV3,
    pack_c4,
    stage_short_prefix,
    unpack_index,
    validate_configuration,
)


class TestQSAHiSparseV3(unittest.TestCase):
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
                          ("speculative_algorithm", "NEXTN"), ("enable_hisparse", True)):
            changed = SimpleNamespace(**vars(args))
            setattr(changed, name, bad)
            with self.assertRaises(ValueError):
                validate_configuration(changed, pool)

    def test_stale_release(self):
        adapter = QSAHiSparseV3.__new__(QSAHiSparseV3)
        adapter.owner, adapter.owner_rid, adapter.generation = 1, "B", 2
        adapter.releasing = False
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


if __name__ == "__main__":
    unittest.main()
