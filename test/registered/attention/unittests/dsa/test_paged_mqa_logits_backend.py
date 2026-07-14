from unittest.mock import patch

import pytest

from sglang.srt.layers.attention.dsa import dsa_indexer, paged_mqa_logits_backend
from sglang.srt.layers.attention.dsa.paged_mqa_logits_backend import (
    DSAPagedMQALogitsBackend,
)


def test_resolves_explicit_sm89_backend_on_ada():
    with (
        patch.object(paged_mqa_logits_backend, "is_hip", return_value=False),
        patch.object(paged_mqa_logits_backend, "is_cuda", return_value=True),
        patch.object(
            paged_mqa_logits_backend,
            "get_device_capability",
            return_value=(8, 9),
        ),
    ):
        assert DSAPagedMQALogitsBackend.resolve("sm89") is DSAPagedMQALogitsBackend.SM89


@pytest.mark.parametrize(
    ("is_cuda", "capability"),
    [(False, (0, 0)), (True, (8, 6)), (True, (9, 0)), (True, (10, 0))],
)
def test_rejects_sm89_backend_on_other_targets(is_cuda, capability):
    with (
        patch.object(paged_mqa_logits_backend, "is_hip", return_value=False),
        patch.object(paged_mqa_logits_backend, "is_cuda", return_value=is_cuda),
        patch.object(
            paged_mqa_logits_backend,
            "get_device_capability",
            return_value=capability,
        ),
        pytest.raises(ValueError, match="requires CUDA SM89"),
    ):
        DSAPagedMQALogitsBackend.resolve("sm89")


def test_auto_cuda_resolution_remains_deepgemm():
    with (
        patch.object(paged_mqa_logits_backend, "is_hip", return_value=False),
        patch.object(paged_mqa_logits_backend, "is_cuda", return_value=True),
    ):
        assert (
            DSAPagedMQALogitsBackend.resolve("auto")
            is DSAPagedMQALogitsBackend.DEEPGEMM
        )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (DSAPagedMQALogitsBackend.SM89, False),
        (DSAPagedMQALogitsBackend.DEEPGEMM, True),
    ],
)
def test_sm89_paged_backend_disables_indexer_fusion(backend, expected):
    with (
        patch.object(dsa_indexer, "_is_cuda", True),
        patch.object(
            dsa_indexer.envs.SGLANG_DISABLE_DSA_INDEXER_FUSION,
            "get",
            return_value=False,
        ),
    ):
        assert (
            dsa_indexer._should_use_dsa_indexer_fusion(
                paged_mqa_logits_backend=backend,
                is_neox_style=False,
            )
            is expected
        )
