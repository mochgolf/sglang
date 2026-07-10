from __future__ import annotations

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1


@triton.jit
def _sm89_sparse_mla_prefill_kernel(
    q_nope,
    q_rope,
    kv_nope,
    kv_scale,
    kv_rope,
    page_table,
    cache_seqlens,
    out,
    q_nope_stride_0: tl.constexpr,
    q_nope_stride_1: tl.constexpr,
    q_nope_stride_2: tl.constexpr,
    q_rope_stride_0: tl.constexpr,
    q_rope_stride_1: tl.constexpr,
    q_rope_stride_2: tl.constexpr,
    kv_nope_stride_0: tl.constexpr,
    kv_nope_stride_1: tl.constexpr,
    kv_scale_stride_0: tl.constexpr,
    kv_scale_stride_1: tl.constexpr,
    kv_rope_stride_0: tl.constexpr,
    kv_rope_stride_1: tl.constexpr,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    out_stride_2: tl.constexpr,
    sm_scale: tl.float32,
    logit_cap: tl.float32,
    topk: tl.int32,
    BLOCK_N: tl.constexpr,
    V_BLOCK: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    DIM_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    head_id = tl.program_id(1)
    v_block_id = tl.program_id(2)

    offs_n = tl.arange(0, BLOCK_N)
    offs_v = v_block_id * V_BLOCK + tl.arange(0, V_BLOCK)
    v_mask = offs_v < DIM_V

    row_len = tl.load(cache_seqlens + row_id).to(tl.int32)
    row_len = tl.minimum(row_len, topk)

    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([V_BLOCK], dtype=tl.float32)

    for n_start in range(0, topk, BLOCK_N):
        token_offsets = n_start + offs_n
        in_topk = token_offsets < topk
        raw_token_ids = tl.load(
            page_table
            + row_id * page_table_stride_0
            + token_offsets * page_table_stride_1,
            mask=in_topk,
            other=-1,
        )
        valid = in_topk & (token_offsets < row_len) & (raw_token_ids >= 0)
        token_ids = tl.where(valid, raw_token_ids, 0).to(tl.int64)

        scores = tl.zeros([BLOCK_N], dtype=tl.float32)

        for d_start in range(0, DIM_NOPE, GROUP_SIZE):
            offs_d = d_start + tl.arange(0, GROUP_SIZE)
            q_vals = tl.load(
                q_nope
                + row_id * q_nope_stride_0
                + head_id * q_nope_stride_1
                + offs_d * q_nope_stride_2,
                mask=offs_d < DIM_NOPE,
                other=0.0,
            ).to(tl.float32)
            k_fp8 = tl.load(
                kv_nope
                + token_ids[:, None] * kv_nope_stride_0
                + offs_d[None, :] * kv_nope_stride_1,
                mask=valid[:, None] & (offs_d[None, :] < DIM_NOPE),
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(
                kv_scale
                + token_ids * kv_scale_stride_0
                + (d_start // GROUP_SIZE) * kv_scale_stride_1,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            k_vals = k_fp8 * scale[:, None]
            scores += tl.sum(k_vals * q_vals[None, :], axis=1)

        offs_rope = tl.arange(0, DIM_ROPE)
        q_rope_vals = tl.load(
            q_rope
            + row_id * q_rope_stride_0
            + head_id * q_rope_stride_1
            + offs_rope * q_rope_stride_2,
            mask=offs_rope < DIM_ROPE,
            other=0.0,
        ).to(tl.float32)
        k_rope_vals = tl.load(
            kv_rope
            + token_ids[:, None] * kv_rope_stride_0
            + offs_rope[None, :] * kv_rope_stride_1,
            mask=valid[:, None] & (offs_rope[None, :] < DIM_ROPE),
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(k_rope_vals * q_rope_vals[None, :], axis=1)

        scores *= sm_scale
        if logit_cap > 0:
            scores = logit_cap * _tanh(scores / logit_cap)
        scores = tl.where(valid, scores, float("-inf"))

        tile_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, tile_max)
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_new)

        scale_group = offs_v // GROUP_SIZE
        v_fp8 = tl.load(
            kv_nope
            + token_ids[:, None] * kv_nope_stride_0
            + offs_v[None, :] * kv_nope_stride_1,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            kv_scale
            + token_ids[:, None] * kv_scale_stride_0
            + scale_group[None, :] * kv_scale_stride_1,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_vals = v_fp8 * v_scale

        acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out_vals = acc / l_i
    out_vals = tl.where(l_i > 0, out_vals, 0.0)
    tl.store(
        out
        + row_id * out_stride_0
        + head_id * out_stride_1
        + offs_v * out_stride_2,
        out_vals,
        mask=v_mask,
    )


@triton.jit
def _sm89_sparse_mla_prefill_kernel_mrow(
    q_nope,
    q_rope,
    kv_nope,
    kv_scale,
    kv_rope,
    page_table,
    cache_seqlens,
    out,
    q_nope_stride_0: tl.constexpr,
    q_nope_stride_1: tl.constexpr,
    q_nope_stride_2: tl.constexpr,
    q_rope_stride_0: tl.constexpr,
    q_rope_stride_1: tl.constexpr,
    q_rope_stride_2: tl.constexpr,
    kv_nope_stride_0: tl.constexpr,
    kv_nope_stride_1: tl.constexpr,
    kv_scale_stride_0: tl.constexpr,
    kv_scale_stride_1: tl.constexpr,
    kv_rope_stride_0: tl.constexpr,
    kv_rope_stride_1: tl.constexpr,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    out_stride_2: tl.constexpr,
    total_q: tl.int32,
    sm_scale: tl.float32,
    logit_cap: tl.float32,
    topk: tl.int32,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V_BLOCK: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    DIM_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row_block_id = tl.program_id(0)
    head_id = tl.program_id(1)
    v_block_id = tl.program_id(2)

    offs_m = tl.arange(0, BLOCK_M)
    row_ids = row_block_id * BLOCK_M + offs_m
    valid_rows = row_ids < total_q
    offs_n = tl.arange(0, BLOCK_N)
    offs_v = v_block_id * V_BLOCK + tl.arange(0, V_BLOCK)
    v_mask = offs_v < DIM_V

    row_len = tl.load(cache_seqlens + row_ids, mask=valid_rows, other=0).to(tl.int32)
    row_len = tl.minimum(row_len, topk)

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 0.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, V_BLOCK], dtype=tl.float32)

    for n_start in range(0, topk, BLOCK_N):
        token_offsets = n_start + offs_n
        in_topk = token_offsets < topk
        raw_token_ids = tl.load(
            page_table
            + row_ids[:, None] * page_table_stride_0
            + token_offsets[None, :] * page_table_stride_1,
            mask=valid_rows[:, None] & in_topk[None, :],
            other=-1,
        )
        valid = (
            valid_rows[:, None]
            & in_topk[None, :]
            & (token_offsets[None, :] < row_len[:, None])
            & (raw_token_ids >= 0)
        )
        token_ids = tl.where(valid, raw_token_ids, 0).to(tl.int64)

        scores = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

        for d_start in range(0, DIM_NOPE, GROUP_SIZE):
            offs_d = d_start + tl.arange(0, GROUP_SIZE)
            q_vals = tl.load(
                q_nope
                + row_ids[:, None] * q_nope_stride_0
                + head_id * q_nope_stride_1
                + offs_d[None, :] * q_nope_stride_2,
                mask=valid_rows[:, None] & (offs_d[None, :] < DIM_NOPE),
                other=0.0,
            ).to(tl.float32)
            k_fp8 = tl.load(
                kv_nope
                + token_ids[:, :, None] * kv_nope_stride_0
                + offs_d[None, None, :] * kv_nope_stride_1,
                mask=valid[:, :, None] & (offs_d[None, None, :] < DIM_NOPE),
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(
                kv_scale
                + token_ids * kv_scale_stride_0
                + (d_start // GROUP_SIZE) * kv_scale_stride_1,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            k_vals = k_fp8 * scale[:, :, None]
            scores += tl.sum(k_vals * q_vals[:, None, :], axis=2)

        offs_rope = tl.arange(0, DIM_ROPE)
        q_rope_vals = tl.load(
            q_rope
            + row_ids[:, None] * q_rope_stride_0
            + head_id * q_rope_stride_1
            + offs_rope[None, :] * q_rope_stride_2,
            mask=valid_rows[:, None] & (offs_rope[None, :] < DIM_ROPE),
            other=0.0,
        ).to(tl.float32)
        k_rope_vals = tl.load(
            kv_rope
            + token_ids[:, :, None] * kv_rope_stride_0
            + offs_rope[None, None, :] * kv_rope_stride_1,
            mask=valid[:, :, None] & (offs_rope[None, None, :] < DIM_ROPE),
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(k_rope_vals * q_rope_vals[:, None, :], axis=2)

        scores *= sm_scale
        if logit_cap > 0:
            scores = logit_cap * _tanh(scores / logit_cap)
        scores = tl.where(valid, scores, float("-inf"))

        tile_max = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, tile_max)
        p = tl.exp(scores - m_new[:, None])
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_new)

        scale_group = offs_v // GROUP_SIZE
        v_fp8 = tl.load(
            kv_nope
            + token_ids[:, :, None] * kv_nope_stride_0
            + offs_v[None, None, :] * kv_nope_stride_1,
            mask=valid[:, :, None] & v_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            kv_scale
            + token_ids[:, :, None] * kv_scale_stride_0
            + scale_group[None, None, :] * kv_scale_stride_1,
            mask=valid[:, :, None] & v_mask[None, None, :],
            other=0.0,
        ).to(tl.float32)
        v_vals = v_fp8 * v_scale

        acc = acc * alpha[:, None] + tl.sum(p[:, :, None] * v_vals, axis=1)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    out_vals = acc / l_i[:, None]
    out_vals = tl.where(l_i[:, None] > 0, out_vals, 0.0)
    tl.store(
        out
        + row_ids[:, None] * out_stride_0
        + head_id * out_stride_1
        + offs_v[None, :] * out_stride_2,
        out_vals,
        mask=valid_rows[:, None] & v_mask[None, :],
    )


@triton.jit
def _sm89_sparse_mla_prefill_split_kernel(
    q_nope,
    q_rope,
    kv_nope,
    kv_scale,
    kv_rope,
    page_table,
    cache_seqlens,
    partial_out,
    partial_lse,
    q_nope_stride_0: tl.constexpr,
    q_nope_stride_1: tl.constexpr,
    q_nope_stride_2: tl.constexpr,
    q_rope_stride_0: tl.constexpr,
    q_rope_stride_1: tl.constexpr,
    q_rope_stride_2: tl.constexpr,
    kv_nope_stride_0: tl.constexpr,
    kv_nope_stride_1: tl.constexpr,
    kv_scale_stride_0: tl.constexpr,
    kv_scale_stride_1: tl.constexpr,
    kv_rope_stride_0: tl.constexpr,
    kv_rope_stride_1: tl.constexpr,
    page_table_stride_0: tl.constexpr,
    page_table_stride_1: tl.constexpr,
    partial_out_stride_0: tl.constexpr,
    partial_out_stride_1: tl.constexpr,
    partial_out_stride_2: tl.constexpr,
    partial_out_stride_3: tl.constexpr,
    partial_lse_stride_0: tl.constexpr,
    partial_lse_stride_1: tl.constexpr,
    partial_lse_stride_2: tl.constexpr,
    sm_scale: tl.float32,
    logit_cap: tl.float32,
    topk: tl.int32,
    NUM_V_BLOCKS: tl.constexpr,
    SPLIT_K: tl.constexpr,
    SPLIT_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    V_BLOCK: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
    DIM_V: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
):
    row_id = tl.program_id(0)
    head_id = tl.program_id(1)
    split_v_id = tl.program_id(2)
    split_id = split_v_id // NUM_V_BLOCKS
    v_block_id = split_v_id - split_id * NUM_V_BLOCKS

    offs_n = tl.arange(0, BLOCK_N)
    offs_v = v_block_id * V_BLOCK + tl.arange(0, V_BLOCK)
    v_mask = offs_v < DIM_V
    split_start = split_id * SPLIT_SIZE
    split_end = tl.minimum(split_start + SPLIT_SIZE, topk)

    row_len = tl.load(cache_seqlens + row_id).to(tl.int32)
    row_len = tl.minimum(row_len, topk)

    m_i = tl.full((), float("-inf"), dtype=tl.float32)
    l_i = tl.full((), 0.0, dtype=tl.float32)
    acc = tl.zeros([V_BLOCK], dtype=tl.float32)

    for split_offset in range(0, SPLIT_SIZE, BLOCK_N):
        token_offsets = split_start + split_offset + offs_n
        in_split = token_offsets < split_end
        raw_token_ids = tl.load(
            page_table
            + row_id * page_table_stride_0
            + token_offsets * page_table_stride_1,
            mask=in_split,
            other=-1,
        )
        valid = in_split & (token_offsets < row_len) & (raw_token_ids >= 0)
        token_ids = tl.where(valid, raw_token_ids, 0).to(tl.int64)

        scores = tl.zeros([BLOCK_N], dtype=tl.float32)

        for d_start in range(0, DIM_NOPE, GROUP_SIZE):
            offs_d = d_start + tl.arange(0, GROUP_SIZE)
            q_vals = tl.load(
                q_nope
                + row_id * q_nope_stride_0
                + head_id * q_nope_stride_1
                + offs_d * q_nope_stride_2,
                mask=offs_d < DIM_NOPE,
                other=0.0,
            ).to(tl.float32)
            k_fp8 = tl.load(
                kv_nope
                + token_ids[:, None] * kv_nope_stride_0
                + offs_d[None, :] * kv_nope_stride_1,
                mask=valid[:, None] & (offs_d[None, :] < DIM_NOPE),
                other=0.0,
            ).to(tl.float32)
            scale = tl.load(
                kv_scale
                + token_ids * kv_scale_stride_0
                + (d_start // GROUP_SIZE) * kv_scale_stride_1,
                mask=valid,
                other=0.0,
            ).to(tl.float32)
            k_vals = k_fp8 * scale[:, None]
            scores += tl.sum(k_vals * q_vals[None, :], axis=1)

        offs_rope = tl.arange(0, DIM_ROPE)
        q_rope_vals = tl.load(
            q_rope
            + row_id * q_rope_stride_0
            + head_id * q_rope_stride_1
            + offs_rope * q_rope_stride_2,
            mask=offs_rope < DIM_ROPE,
            other=0.0,
        ).to(tl.float32)
        k_rope_vals = tl.load(
            kv_rope
            + token_ids[:, None] * kv_rope_stride_0
            + offs_rope[None, :] * kv_rope_stride_1,
            mask=valid[:, None] & (offs_rope[None, :] < DIM_ROPE),
            other=0.0,
        ).to(tl.float32)
        scores += tl.sum(k_rope_vals * q_rope_vals[None, :], axis=1)

        scores *= sm_scale
        if logit_cap > 0:
            scores = logit_cap * _tanh(scores / logit_cap)
        scores = tl.where(valid, scores, float("-inf"))

        tile_max = tl.max(scores, axis=0)
        m_new = tl.maximum(m_i, tile_max)
        p = tl.exp(scores - m_new)
        p = tl.where(valid, p, 0.0)
        alpha = tl.exp(m_i - m_new)

        scale_group = offs_v // GROUP_SIZE
        v_fp8 = tl.load(
            kv_nope
            + token_ids[:, None] * kv_nope_stride_0
            + offs_v[None, :] * kv_nope_stride_1,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_scale = tl.load(
            kv_scale
            + token_ids[:, None] * kv_scale_stride_0
            + scale_group[None, :] * kv_scale_stride_1,
            mask=valid[:, None] & v_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        v_vals = v_fp8 * v_scale

        acc = acc * alpha + tl.sum(p[:, None] * v_vals, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    out_vals = acc / l_i
    out_vals = tl.where(l_i > 0, out_vals, 0.0)
    split_lse = tl.where(l_i > 0, tl.log(l_i) + m_i, float("-inf"))
    tl.store(
        partial_out
        + split_id * partial_out_stride_0
        + row_id * partial_out_stride_1
        + head_id * partial_out_stride_2
        + offs_v * partial_out_stride_3,
        out_vals,
        mask=v_mask,
    )
    tl.store(
        partial_lse
        + split_id * partial_lse_stride_0
        + row_id * partial_lse_stride_1
        + head_id * partial_lse_stride_2,
        split_lse,
    )


@triton.jit
def _sm89_sparse_mla_merge_split_kernel(
    partial_out,
    partial_lse,
    out,
    partial_out_stride_0: tl.constexpr,
    partial_out_stride_1: tl.constexpr,
    partial_out_stride_2: tl.constexpr,
    partial_out_stride_3: tl.constexpr,
    partial_lse_stride_0: tl.constexpr,
    partial_lse_stride_1: tl.constexpr,
    partial_lse_stride_2: tl.constexpr,
    out_stride_0: tl.constexpr,
    out_stride_1: tl.constexpr,
    out_stride_2: tl.constexpr,
    SPLIT_K: tl.constexpr,
    V_BLOCK: tl.constexpr,
    DIM_V: tl.constexpr,
):
    row_id = tl.program_id(0)
    head_id = tl.program_id(1)
    v_block_id = tl.program_id(2)

    offs_s = tl.arange(0, SPLIT_K)
    offs_v = v_block_id * V_BLOCK + tl.arange(0, V_BLOCK)
    v_mask = offs_v < DIM_V

    split_lse = tl.load(
        partial_lse
        + offs_s * partial_lse_stride_0
        + row_id * partial_lse_stride_1
        + head_id * partial_lse_stride_2
    )
    max_lse = tl.max(split_lse, axis=0)
    weights = tl.exp(split_lse - max_lse)
    weights = tl.where(split_lse == float("-inf"), 0.0, weights)
    denom = tl.sum(weights, axis=0)

    partial_vals = tl.load(
        partial_out
        + offs_s[:, None] * partial_out_stride_0
        + row_id * partial_out_stride_1
        + head_id * partial_out_stride_2
        + offs_v[None, :] * partial_out_stride_3,
        mask=v_mask[None, :],
        other=0.0,
    ).to(tl.float32)
    merged = tl.sum(weights[:, None] * partial_vals, axis=0) / denom
    merged = tl.where(denom > 0, merged, 0.0)
    tl.store(
        out
        + row_id * out_stride_0
        + head_id * out_stride_1
        + offs_v * out_stride_2,
        merged,
        mask=v_mask,
    )


def select_sm89_sparse_mla_v_block(v_block: int | None = None) -> int:
    raw = (
        str(v_block)
        if v_block is not None
        else os.environ.get("SGLANG_GLM_DSA_SM89_V_BLOCK", "64")
    )
    try:
        selected = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_V_BLOCK must be one of 64, 128, or 256; "
            f"got {raw!r}."
        ) from exc
    if selected not in {64, 128, 256}:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_V_BLOCK must be one of 64, 128, or 256; "
            f"got {selected}."
        )
    return selected


def select_sm89_sparse_mla_block_n(block_n: int | None = None) -> int:
    raw = (
        str(block_n)
        if block_n is not None
        else os.environ.get("SGLANG_GLM_DSA_SM89_BLOCK_N", "64")
    )
    try:
        selected = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_BLOCK_N must be one of 32, 64, or 128; "
            f"got {raw!r}."
        ) from exc
    if selected not in {32, 64, 128}:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_BLOCK_N must be one of 32, 64, or 128; "
            f"got {selected}."
        )
    return selected


def select_sm89_sparse_mla_block_m(block_m: int | None = None) -> int:
    raw = (
        str(block_m)
        if block_m is not None
        else os.environ.get("SGLANG_GLM_DSA_SM89_BLOCK_M", "1")
    )
    try:
        selected = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_BLOCK_M must be one of 1, 2, or 4; "
            f"got {raw!r}."
        ) from exc
    if selected not in {1, 2, 4}:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_BLOCK_M must be one of 1, 2, or 4; "
            f"got {selected}."
        )
    return selected


def select_sm89_sparse_mla_split_k(split_k: int | None = None) -> int:
    raw = (
        str(split_k)
        if split_k is not None
        else os.environ.get("SGLANG_GLM_DSA_SM89_SPLIT_K", "1")
    )
    try:
        selected = int(raw)
    except ValueError as exc:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_SPLIT_K must be one of 1, 4, or 8; "
            f"got {raw!r}."
        ) from exc
    if selected not in {1, 4, 8}:
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_SPLIT_K must be one of 1, 4, or 8; "
            f"got {selected}."
        )
    return selected


def sm89_sparse_mla_prefill_triton(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    v_block: int | None = None,
    block_n: int | None = None,
    block_m: int | None = None,
    split_k: int | None = None,
) -> torch.Tensor:
    assert q_nope.is_cuda and q_rope.is_cuda and kv_cache.is_cuda
    assert page_table.is_cuda and cache_seqlens.is_cuda
    assert kv_cache.dtype == torch.float8_e4m3fn
    assert kv_cache.shape[-1] == 656
    assert kv_cache.stride(-1) == 1
    assert page_table.dtype in (torch.int32, torch.int64)
    assert v_head_dim == 512

    total_q, num_heads, dim_nope = q_nope.shape
    assert dim_nope == 512
    assert q_rope.shape == (total_q, num_heads, 64)
    assert page_table.shape[0] == total_q
    assert cache_seqlens.shape[0] == total_q

    kv_cache_2d = kv_cache.view(-1, kv_cache.shape[-1])
    kv_nope = kv_cache_2d[:, :512]
    kv_scale = kv_cache_2d[:, 512:528].view(torch.float32)
    kv_rope = kv_cache_2d[:, 528:].view(torch.bfloat16)

    selected_v_block = select_sm89_sparse_mla_v_block(v_block)
    selected_block_n = select_sm89_sparse_mla_block_n(block_n)
    selected_block_m = select_sm89_sparse_mla_block_m(block_m)
    selected_split_k = select_sm89_sparse_mla_split_k(split_k)
    if selected_split_k != 1 and selected_block_m != 1:
        raise ValueError("split_k > 1 currently requires BLOCK_M=1.")

    out = torch.empty_like(q_nope)
    if selected_split_k != 1:
        num_v_blocks = triton.cdiv(dim_nope, selected_v_block)
        partial_out = torch.empty(
            (selected_split_k, total_q, num_heads, dim_nope),
            dtype=q_nope.dtype,
            device=q_nope.device,
        )
        partial_lse = torch.empty(
            (selected_split_k, total_q, num_heads),
            dtype=torch.float32,
            device=q_nope.device,
        )
        split_size = triton.cdiv(page_table.shape[1], selected_split_k)
        _sm89_sparse_mla_prefill_split_kernel[
            (total_q, num_heads, num_v_blocks * selected_split_k)
        ](
            q_nope,
            q_rope,
            kv_nope,
            kv_scale,
            kv_rope,
            page_table,
            cache_seqlens,
            partial_out,
            partial_lse,
            q_nope.stride(0),
            q_nope.stride(1),
            q_nope.stride(2),
            q_rope.stride(0),
            q_rope.stride(1),
            q_rope.stride(2),
            kv_nope.stride(0),
            kv_nope.stride(1),
            kv_scale.stride(0),
            kv_scale.stride(1),
            kv_rope.stride(0),
            kv_rope.stride(1),
            page_table.stride(0),
            page_table.stride(1),
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            float(sm_scale),
            float(logit_cap),
            page_table.shape[1],
            NUM_V_BLOCKS=num_v_blocks,
            SPLIT_K=selected_split_k,
            SPLIT_SIZE=split_size,
            BLOCK_N=selected_block_n,
            V_BLOCK=selected_v_block,
            DIM_NOPE=512,
            DIM_ROPE=64,
            DIM_V=512,
            GROUP_SIZE=128,
            num_warps=8,
            num_stages=3,
        )
        _sm89_sparse_mla_merge_split_kernel[(total_q, num_heads, num_v_blocks)](
            partial_out,
            partial_lse,
            out,
            partial_out.stride(0),
            partial_out.stride(1),
            partial_out.stride(2),
            partial_out.stride(3),
            partial_lse.stride(0),
            partial_lse.stride(1),
            partial_lse.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            SPLIT_K=selected_split_k,
            V_BLOCK=selected_v_block,
            DIM_V=512,
            num_warps=8,
            num_stages=3,
        )
        return out

    grid = (
        triton.cdiv(total_q, selected_block_m),
        num_heads,
        triton.cdiv(dim_nope, selected_v_block),
    )
    kernel = (
        _sm89_sparse_mla_prefill_kernel
        if selected_block_m == 1
        else _sm89_sparse_mla_prefill_kernel_mrow
    )
    common_args = (
        q_nope,
        q_rope,
        kv_nope,
        kv_scale,
        kv_rope,
        page_table,
        cache_seqlens,
        out,
        q_nope.stride(0),
        q_nope.stride(1),
        q_nope.stride(2),
        q_rope.stride(0),
        q_rope.stride(1),
        q_rope.stride(2),
        kv_nope.stride(0),
        kv_nope.stride(1),
        kv_scale.stride(0),
        kv_scale.stride(1),
        kv_rope.stride(0),
        kv_rope.stride(1),
        page_table.stride(0),
        page_table.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    if selected_block_m == 1:
        kernel[grid](
            *common_args,
            float(sm_scale),
            float(logit_cap),
            page_table.shape[1],
            BLOCK_N=selected_block_n,
            V_BLOCK=selected_v_block,
            DIM_NOPE=512,
            DIM_ROPE=64,
            DIM_V=512,
            GROUP_SIZE=128,
            num_warps=8,
            num_stages=3,
        )
    else:
        kernel[grid](
            *common_args,
            total_q,
            float(sm_scale),
            float(logit_cap),
            page_table.shape[1],
            BLOCK_M=selected_block_m,
            BLOCK_N=selected_block_n,
            V_BLOCK=selected_v_block,
            DIM_NOPE=512,
            DIM_ROPE=64,
            DIM_V=512,
            GROUP_SIZE=128,
            num_warps=8,
            num_stages=3,
        )
    return out
