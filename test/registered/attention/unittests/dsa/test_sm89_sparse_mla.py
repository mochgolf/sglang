import unittest
from unittest.mock import MagicMock, patch

import torch


class TestSm89SparseMlaBackendLogging(unittest.TestCase):
    def test_glm_sm89_fa3_logs_effective_torch_mla_only_once(self):
        from sglang.srt.layers.attention import dsa_backend

        backend = object.__new__(dsa_backend.DeepseekSparseAttnBackend)
        backend.use_glm_sm89_dsa_fallback = True
        backend.device_sm_major = 8
        backend._logged_glm_sm89_effective_torch_mla = False

        sentinel = object()
        backend._forward_torch_mla = MagicMock(return_value=sentinel)

        q_rope = torch.zeros(1, 1, 2)
        q_nope = torch.zeros(1, 1, 2)
        kv_cache = torch.zeros(1, 1, 4)
        page_table = torch.zeros(1, 1, dtype=torch.int32)
        cache_seqlens = torch.ones(1, dtype=torch.int32)
        cu_seqlens = torch.tensor([0, 1], dtype=torch.int32)

        with patch.object(dsa_backend.logger, "warning") as mock_warning, patch(
            "sglang.srt.layers.attention.dsa_backend.flash_attn_with_kvcache"
        ) as mock_fa3:
            first = backend._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=2,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
            )
            second = backend._forward_fa3(
                q_rope=q_rope,
                kv_cache=kv_cache,
                v_head_dim=2,
                q_nope=q_nope,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=1,
                sm_scale=1.0,
                logit_cap=0.0,
                page_size=1,
            )

        self.assertIs(first, sentinel)
        self.assertIs(second, sentinel)
        backend._forward_torch_mla.assert_called()
        mock_fa3.assert_not_called()
        mock_warning.assert_called_once()
        self.assertIn("GLM DSA SM89 effective backend is torch_mla", mock_warning.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
