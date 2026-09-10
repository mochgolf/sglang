import sys
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa_backend
from sglang.srt.layers.attention.qwen_sparse_attn_backend import (
    QwenSparseAttnBackend,
    _resolve_flash_attn_varlen_func,
)


class TestQwenSparseFlashAttentionResolution(unittest.TestCase):
    def test_packed_view_matches_batch_and_preserves_backing_storage(self):
        # Exercise the production CUDA branch with CPU storage and substituted
        # kernels. This checks call shape/lifetime, not attention arithmetic.
        backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
        backend._fa2_scratch = {}
        backend._cuda_graph_max_tokens = 2
        backend.hisparse_v3 = None
        raw_k = torch.empty((1, 1, 256), dtype=torch.float16)
        raw_v = torch.empty_like(raw_k)
        backend.token_to_kv_pool = SimpleNamespace(
            get_key_buffer=lambda _: raw_k, get_value_buffer=lambda _: raw_v
        )
        backend.req_to_token_pool = SimpleNamespace(
            req_to_token=torch.zeros((2, 1), dtype=torch.int32)
        )
        backend._kv_descales = lambda *_: (1.0, 1.0)
        pointers = None
        with patch.object(qsa_backend, "_resolve_trtllm_sparse_decode", return_value=None), \
                patch.object(qsa_backend, "_resolve_flash_attn_varlen_func") as resolve, \
                patch.object(qsa_backend, "qwen_sparse_fa2_cu_seqlens_triton"), \
                patch.object(qsa_backend, "qwen_sparse_kv_extraction_compact_triton"), \
                patch.object(qsa_backend, "operations_nvtx_range", side_effect=lambda _: nullcontext()):
            for graph, count in ((True, 2), (True, 1), (False, 1), (False, 2)):
                metadata = SimpleNamespace(
                    is_cuda_graph=graph,
                    sequence_lengths=torch.full((count,), 261120, dtype=torch.int32),
                    row_req_pool_indices=torch.arange(count, dtype=torch.int32),
                    fa2_valid_counts=torch.empty(count, dtype=torch.int32),
                    fa2_cu_seqlens_k=torch.empty(count + 1, dtype=torch.int32),
                    fa2_cu_seqlens_q=torch.arange(count + 1, dtype=torch.int32),
                )
                backend._resolve_metadata = lambda _: metadata
                q = SimpleNamespace(is_cuda=True, device=raw_k.device,
                                    dtype=raw_k.dtype, shape=(count, 12, 256))
                resolve.return_value.return_value = torch.zeros(q.shape)
                backend._forward_paged_attention(
                    q, SimpleNamespace(layer_id=3, scaling=0.0625), None,
                    torch.zeros((count, 2051), dtype=torch.int32),
                )
                call = resolve.return_value.call_args.kwargs
                self.assertEqual(call["k"].shape, (count * 2051, 1, 256))
                self.assertEqual(call["v"].shape, call["k"].shape)
                current = (call["k"].data_ptr(), call["v"].data_ptr())
                if pointers is None:
                    pointers = current
                    self.assertNotEqual(*pointers)
                self.assertEqual(current, pointers)
                backing = next(iter(backend._fa2_scratch.values()))
                self.assertEqual([x.shape[0] for x in backing], [4102, 4102])
                self.assertEqual(call["max_seqlen_k"], 2051)

    def test_sm89_uses_vendored_flash_attention_without_classic_fa2(self):
        _resolve_flash_attn_varlen_func.cache_clear()
        with patch.dict(sys.modules, {"flash_attn": None}), patch(
            "torch.cuda.get_device_capability", return_value=(8, 9)
        ):
            resolved = _resolve_flash_attn_varlen_func()
        self.assertEqual(
            resolved.__module__, "sglang.kernels.ops.attention.flash_attention"
        )
        _resolve_flash_attn_varlen_func.cache_clear()


if __name__ == "__main__":
    unittest.main()
