import pytest
import torch

from sglang.kernels.ops.elementwise.fast_topk import fast_topk
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _check_topk_values(score, lengths, indices, topk, row_starts):
    """The returned indices must select exactly the top-k value multiset.

    Order is unspecified and tie-breaking may differ from torch.topk, so we
    compare sorted score values, not index sets.
    """
    for b in range(score.shape[0]):
        start = int(row_starts[b]) if row_starts is not None else 0
        length = int(lengths[b])
        section = score[b, start : start + length]
        row = indices[b]
        if length <= topk:
            # naive path: identity indices, then -1 fill
            assert torch.equal(
                row[:length].cpu(), torch.arange(length, dtype=torch.int32)
            )
            assert (row[length:] == -1).all()
            continue
        assert (row >= 0).all(), "long rows must fill every slot"
        picked = section[row.long()]
        expected = torch.topk(section, topk).values
        assert torch.equal(
            picked.sort(descending=True).values, expected.sort(descending=True).values
        ), f"row {b}: top-{topk} value multiset mismatch"


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "batch,length",
    [
        (1, 4096),
        (7, 3000),
        (33, 32768),
        (128, 2050),
    ],
)
def test_fast_topk_long_rows(topk, batch, length):
    torch.manual_seed(0)
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_short_and_mixed_rows(topk):
    torch.manual_seed(0)
    max_len = topk + 128
    batch = 8
    score = torch.randn(batch, max_len, dtype=torch.float32, device="cuda")
    # rows shorter than k (naive path), exactly k, and longer than k
    lens = [1, topk // 3, topk - 1, topk, topk + 1, topk + 7, 17, max_len]
    lengths = torch.tensor(lens[:batch], dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_with_row_starts(topk):
    torch.manual_seed(0)
    batch, width = 16, 8192
    score = torch.randn(batch, width, dtype=torch.float32, device="cuda")
    row_starts = torch.randint(0, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.randint(1, 2048, (batch,), dtype=torch.int32, device="cuda")
    lengths = torch.minimum(lengths, width - row_starts).to(torch.int32)
    # ensure some rows are longer than k
    lengths[0] = min(width - int(row_starts[0]), topk + 100)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_ragged_ignores_larger_values_outside_window(topk):
    width = topk + 1024
    score = torch.randn(4, width, dtype=torch.float32, device="cuda")
    row_starts = torch.tensor([0, 17, 63, 100], dtype=torch.int32, device="cuda")
    lengths = torch.tensor(
        [0, 1, topk - 1, topk + 1], dtype=torch.int32, device="cuda"
    )
    positions = torch.arange(width, device="cuda").unsqueeze(0)
    valid = (positions >= row_starts[:, None]) & (
        positions < row_starts[:, None] + lengths[:, None]
    )
    score.masked_fill_(~valid, 1e30)

    indices = fast_topk(score, lengths, topk, row_starts=row_starts)
    _check_topk_values(score, lengths, indices, topk, row_starts)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_row_stride(topk):
    torch.manual_seed(0)
    batch, length = 8, 4096
    base = torch.randn(batch, 2 * length, dtype=torch.float32, device="cuda")
    score = base[:, :length]  # stride(0) == 2*length, stride(1) == 1
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    expected = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, expected, topk, None)
    for _ in range(3):
        indices = fast_topk(score, lengths, topk)
        _check_topk_values(score, lengths, indices, topk, None)
        assert torch.equal(indices, expected)


@pytest.mark.parametrize("topk", [512, 2048])
@pytest.mark.parametrize(
    "fill",
    [
        "binary",  # only 0s and 1s: extreme duplication at the threshold bin
        "few_levels",  # a handful of distinct levels incl. negatives
        "constant",  # whole rows of one value
    ],
)
def test_fast_topk_duplicate_heavy(topk, fill):
    torch.manual_seed(0)
    batch, length = 16, 8192
    if fill == "binary":
        score = torch.randint(0, 2, (batch, length), dtype=torch.float32, device="cuda")
    elif fill == "few_levels":
        levels = torch.tensor([-5.0, -1.0, 0.0, 0.5, 2.0], device="cuda")
        score = levels[torch.randint(0, 5, (batch, length), device="cuda")]
    else:
        score = torch.full((batch, length), 3.25, dtype=torch.float32, device="cuda")
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    expected = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, expected, topk, None)
    for _ in range(3):
        indices = fast_topk(score, lengths, topk)
        _check_topk_values(score, lengths, indices, topk, None)
        assert torch.equal(indices, expected)


@pytest.mark.parametrize("topk", [512, 2048])
def test_fast_topk_negative_and_zero(topk):
    torch.manual_seed(0)
    batch, length = 8, 16384
    score = torch.randn(batch, length, dtype=torch.float32, device="cuda") * 100
    score[:, : length // 3] = 0.0  # long zero prefix
    score[:, length // 3 : length // 2] = -1e30  # very negative block
    lengths = torch.full((batch,), length, dtype=torch.int32, device="cuda")

    indices = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, indices, topk, None)


@pytest.mark.parametrize(
    "topk,length,narrow",
    [
        (512, 30_938, True),
        (512, 65_504, True),
        (2048, 262_144, True),
    ],
)
def test_fast_topk_candidate_buffer_overflow(topk, length, narrow):
    """Concentrated threshold bins must not silently truncate candidates."""

    torch.manual_seed(0)
    score = torch.randn(4, length, dtype=torch.float32, device="cuda")
    if narrow:
        score = score.mul_(0.1).add_(10.0)
    lengths = torch.full((4,), length, dtype=torch.int32, device="cuda")

    expected = fast_topk(score, lengths, topk)
    _check_topk_values(score, lengths, expected, topk, None)
    for _ in range(3):
        actual = fast_topk(score, lengths, topk)
        _check_topk_values(score, lengths, actual, topk, None)
        assert torch.equal(actual, expected)


def test_fast_topk_cuda_graph_replay():
    topk, length = 512, 65_504
    score = torch.empty(4, length, dtype=torch.float32, device="cuda")
    lengths = torch.full((4,), length, dtype=torch.int32, device="cuda")

    score.normal_().mul_(0.1).add_(10.0)
    for _ in range(3):
        fast_topk(score, lengths, topk)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        indices = [fast_topk(score, lengths, topk) for _ in range(13)]

    for seed in range(1, 21):
        generator = torch.Generator(device="cuda").manual_seed(seed)
        score.copy_(
            torch.randn(
                score.shape,
                generator=generator,
                dtype=score.dtype,
                device=score.device,
            )
            .mul_(0.1)
            .add_(10.0)
        )
        graph.replay()
        for output in indices:
            _check_topk_values(score, lengths, output, topk, None)
            assert torch.equal(output, indices[0])


def test_fast_topk_unsupported_k():
    score = torch.randn(2, 4096, dtype=torch.float32, device="cuda")
    lengths = torch.full((2,), 4096, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError, match="topk"):
        fast_topk(score, lengths, 1024)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
