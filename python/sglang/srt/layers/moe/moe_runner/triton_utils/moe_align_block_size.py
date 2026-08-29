from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.environ import envs
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_xpu

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()

if _is_cuda or _is_hip or _is_xpu or _is_musa:
    from sglang.kernels.ops.moe import moe_align_block_size as sgl_moe_align_block_size

if _is_cuda:
    from sglang.kernels.ops.moe.moe_align_small_numel import (
        SMALL_NUMEL_LIMIT,
        moe_align_small_numel,
    )

# Where the CUDA kernel's own small-batch single-block path stops: its
# per-thread histogram costs 4 * (buckets + 1) ** 2 bytes of shared memory.
_CUDA_SMALL_BATCH_MAX_BUCKETS = 64
# M=12, top-k=10 is the production speculative-prefill shape.  The existing
# single-CTA kernel is faster than generic align + stable reorder at NP=128;
# keep the wider gate exclusive to callers that already require stable order.
_STABLE_SMALL_NUMEL_LIMIT = 128


@triton.jit
def _stable_reorder_kernel(
    topk_ids,
    sorted_ids,
    cumsum,
    numel,
    block_size,
    IGNORE_INVALID: tl.constexpr,
    SORT_BLOCK: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
):
    bucket = tl.program_id(0)
    previous_end = tl.load(cumsum + bucket - 1, mask=bucket > 0, other=0)
    bucket_start = ((previous_end + block_size - 1) // block_size) * block_size
    bucket_end = tl.load(cumsum + bucket)
    count = bucket_end - bucket_start

    if count <= SORT_BLOCK:
        sort_offsets = tl.arange(0, SORT_BLOCK)
        values = tl.load(
            sorted_ids + bucket_start + sort_offsets,
            mask=sort_offsets < count,
            other=numel,
        )
        values = tl.sort(values)
        tl.store(
            sorted_ids + bucket_start + sort_offsets,
            values,
            mask=sort_offsets < count,
        )
    else:
        # Highly skewed fallback: one ordered scan is slower but unbounded.
        scan_offsets = tl.arange(0, SCAN_BLOCK)
        cursor = bucket_start
        for start in tl.range(0, numel, SCAN_BLOCK):
            mask = start + scan_offsets < numel
            ids = tl.load(topk_ids + start + scan_offsets, mask=mask, other=-2)
            matches = mask & (ids + 1 == bucket)
            if IGNORE_INVALID:
                matches &= ids >= 0
            prefix = tl.cumsum(matches.to(tl.int32), axis=0) - matches
            tl.store(
                sorted_ids + cursor + prefix,
                start + scan_offsets,
                mask=matches,
            )
            cursor += tl.sum(matches.to(tl.int32), axis=0)


@triton.jit
def _stable_reorder_small_aot_kernel(
    topk_ids,
    sorted_ids,
    expert_ids,
    num_tokens_post_pad,
    numel,
    block_size,
    IGNORE_INVALID: tl.constexpr,
    TOPK_BLOCK: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
):
    """Canonicalize the CUDA single-CTA path, which has no cumsum output."""
    bucket = tl.program_id(0)
    block_offsets = tl.arange(0, MAX_BLOCKS)
    num_blocks = tl.load(num_tokens_post_pad) // block_size
    block_experts = tl.load(
        expert_ids + block_offsets,
        mask=block_offsets < num_blocks,
        other=-2,
    )
    first_block = tl.min(
        tl.where(block_experts == bucket - 1, block_offsets, MAX_BLOCKS)
    )

    offsets = tl.arange(0, TOPK_BLOCK)
    mask = offsets < numel
    ids = tl.load(topk_ids + offsets, mask=mask, other=-2)
    matches = mask & (ids + 1 == bucket)
    if IGNORE_INVALID:
        matches &= ids >= 0
    prefix = tl.cumsum(matches.to(tl.int32), axis=0) - matches
    tl.store(
        sorted_ids + first_block * block_size + prefix,
        offsets,
        mask=matches,
    )


def moe_align_block_size(
    topk_ids: torch.Tensor,
    block_size: int,
    num_experts: int,
    ignore_invalid_expert: bool = False,
    stable: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Aligns the token distribution across experts to be compatible with block
    size for matrix multiplication.

    Parameters:
    - topk_ids: A tensor of shape [total_tokens, top_k] representing the
        top-k expert indices for each token.
    - block_size: The block size used in block matrix multiplication.
    - num_experts: The total number of experts.
    - stable: Rewrite valid rows in ascending original pair order after the
        normal count/pad pass. This removes atomic scheduling order from
        numerically order-sensitive grouped GEMMs.

    Returns:
    - sorted_token_ids: A tensor containing the sorted token indices according
        to their allocated expert.
    - expert_ids: A tensor indicating the assigned expert index for each block.
    - num_tokens_post_padded: The total number of tokens after padding,
        ensuring divisibility by block_size.

    This function pads the number of tokens that each expert needs to process
    so that it is divisible by block_size.
    Padding ensures that during block matrix multiplication, the dimensions
    align correctly.

    Example:
    Given topk_ids = [[2, 3, 4], [1, 2, 4], [1, 3, 4], [1, 2, 3]],
    block_size = 4, and num_experts = 4:
    - We initially have 12 tokens (after repeating 'top_k' times) and 4 experts,
        with each expert needing to process 3 tokens.
    - As block_size is 4, we pad 1 token for each expert.
    - First, flatten topk_ids to [2, 3, 4, 1, 2, 4, 1, 3, 4, 1, 2, 3].
    - Then append padding tokens [12, 12, 12, 12] for each block.
    - After sorting by expert index, we obtain token_ids
        [3, 6, 9, 12, 0, 4, 10, 12, 1, 7, 11, 12, 2, 5, 8, 12].
        Tokens 12 are non-existent (padding) and are ignored in
        the subsequent matrix multiplication.
    - The padding ensures that the total number of tokens is now divisible
        by block_size for proper block matrix operations.
    """
    # ===== TO BE REFACTORED ====
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        if lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get() and num_experts <= 8191:
            from sglang.kernels.ops.moe.trtllm_lora_temp.virtual_experts import (
                _align_block_size_jit,
            )

            return _align_block_size_jit(topk_ids, block_size, num_experts)
    # ===== END TO BE REFACTORED ====

    if topk_ids.numel() < num_experts + 1:
        max_num_tokens_padded = topk_ids.numel() * block_size
    else:
        max_num_tokens_padded = topk_ids.numel() + (num_experts + 1) * (block_size - 1)

    sorted_ids = torch.empty(
        (max_num_tokens_padded,), dtype=torch.int32, device=topk_ids.device
    )
    max_num_m_blocks = triton.cdiv(max_num_tokens_padded, block_size)
    expert_ids = torch.empty(
        (max_num_m_blocks,), dtype=torch.int32, device=topk_ids.device
    )
    num_tokens_post_pad = torch.empty((1), dtype=torch.int32, device=topk_ids.device)

    # In EP, expert_ids for filtered experts are -1. We have num_experts + 1 ids in total.
    cumsum_buffer = torch.empty(
        (num_experts + 2,), dtype=torch.int32, device=topk_ids.device
    )
    cuda_small_batch = (
        _is_cuda
        and topk_ids.numel() < 1024
        and num_experts + 1 <= _CUDA_SMALL_BATCH_MAX_BUCKETS
    )

    # Tiny-batch fast path (bs=1 decode): one single-CTA triton launch replaces
    # the generic align + count_and_sort pair, covering the corner the CUDA
    # small-batch kernel cannot reach. Below that bucket limit the CUDA kernel
    # is already a single launch and does O(numel) work where this one does
    # O(numel ** 2) pairwise, so leave that side to it. ignore_invalid_expert is
    # a different contract from the "+1 offset" convention this kernel implements.
    if (
        _is_cuda
        and topk_ids.numel()
        <= (_STABLE_SMALL_NUMEL_LIMIT if stable else SMALL_NUMEL_LIMIT)
        and num_experts + 1 > _CUDA_SMALL_BATCH_MAX_BUCKETS
        and not ignore_invalid_expert
    ):
        moe_align_small_numel(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
        )
        return sorted_ids, expert_ids, num_tokens_post_pad

    # ===== TO BE REFACTORED ====
    use_jit_align = False
    if _SGLANG_EXPERIMENTAL_LORA_OPTI:
        from sglang.srt.lora.trtllm_lora_temp.environ import lora_envs

        use_jit_align = lora_envs.SGLANG_OPT_USE_JIT_KERNEL_MOE_ALIGN.get()
    if use_jit_align:
        from sglang.kernels.ops.moe.moe_align import (
            moe_align_block_size as jit_moe_align_block_size,
        )

        jit_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
        )
    # ===== END TO BE REFACTORED ====
    else:
        sgl_moe_align_block_size(
            topk_ids,
            num_experts + 1,
            block_size,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            cumsum_buffer,
            True,
            ignore_invalid_expert,
        )
    if stable:
        if cuda_small_batch:
            _stable_reorder_small_aot_kernel[(num_experts + 1,)](
                topk_ids,
                sorted_ids,
                expert_ids,
                num_tokens_post_pad,
                topk_ids.numel(),
                block_size,
                IGNORE_INVALID=ignore_invalid_expert,
                TOPK_BLOCK=triton.next_power_of_2(topk_ids.numel()),
                MAX_BLOCKS=triton.next_power_of_2(max_num_m_blocks),
                num_warps=4,
            )
        else:
            _stable_reorder_kernel[(num_experts + 1,)](
                topk_ids,
                sorted_ids,
                cumsum_buffer,
                topk_ids.numel(),
                block_size,
                IGNORE_INVALID=ignore_invalid_expert,
                SORT_BLOCK=min(256, triton.next_power_of_2(topk_ids.numel())),
                SCAN_BLOCK=256,
                num_warps=4,
            )
    return sorted_ids, expert_ids, num_tokens_post_pad
