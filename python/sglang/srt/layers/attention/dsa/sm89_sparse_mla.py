from __future__ import annotations

import logging
import math
import os

import torch

from sglang.srt.layers.attention.dsa.sm89_debug import (
    cuda_timer,
    glm_dsa_sm89_profile_enabled,
    nvtx_range,
    profile_region,
)

logger = logging.getLogger(__name__)
_logged_kernel_impls: set[str] = set()
_logged_decode_kernels: set[str] = set()


def _log_kernel_impl_once(kernel: str) -> None:
    if kernel in _logged_kernel_impls:
        return
    logger.info(
        "GLM DSA SM89 sm89_triton prefill uses %s implementation "
        "selected by SGLANG_GLM_DSA_SM89_KERNEL.",
        kernel,
    )
    _logged_kernel_impls.add(kernel)


def _log_decode_kernel_once(selected: str) -> None:
    if selected in _logged_decode_kernels:
        return
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        SM89_DSA_DECODE_KERNEL_ENV,
    )

    logger.info(
        "GLM DSA SM89 effective decode kernel is %s selected by %s.",
        selected,
        SM89_DSA_DECODE_KERNEL_ENV,
    )
    _logged_decode_kernels.add(selected)


def _profiled_call(name: str, fn):
    profile = glm_dsa_sm89_profile_enabled()
    if not profile:
        return fn()
    with nvtx_range(name), cuda_timer(name, profile):
        return fn()


def sm89_sparse_mla_prefill_reference(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
) -> torch.Tensor:
    from sglang.srt.layers.attention.dsa.dequant_k_cache import dequantize_k_cache

    if kv_cache.dtype == torch.float8_e4m3fn:
        kv_cache = dequantize_k_cache(kv_cache)

    k_rope_cache = kv_cache[:, :, v_head_dim:]
    c_kv_cache = kv_cache[:, :, :v_head_dim]
    kv_all = torch.cat([c_kv_cache, k_rope_cache], dim=-1)
    q_all = torch.cat([q_nope, q_rope], dim=-1)

    out = torch.empty_like(q_nope)
    total_q = q_all.shape[0]
    chunk_size = 128
    arange_topk = torch.arange(page_table.shape[1], device=page_table.device)

    for start in range(0, total_q, chunk_size):
        end = min(start + chunk_size, total_q)
        row_lens = cache_seqlens[start:end].to(torch.long)
        max_len = int(row_lens.max().item()) if row_lens.numel() else 0
        if max_len == 0:
            out[start:end].zero_()
            continue

        selected = page_table[start:end, :max_len]
        valid = (arange_topk[:max_len].unsqueeze(0) < row_lens.unsqueeze(1)) & (
            selected >= 0
        )
        safe_selected = selected.clamp(min=0).to(torch.long)
        selected_kv = kv_all[safe_selected].squeeze(2)
        selected_k = selected_kv
        selected_v = selected_kv[..., :v_head_dim]

        q_chunk = q_all[start:end].unsqueeze(2)
        k_chunk = selected_k.unsqueeze(1)
        v_chunk = selected_v.unsqueeze(1)
        attn_mask = valid[:, None, None, :]
        empty_rows = ~valid.any(dim=1)

        if logit_cap and logit_cap > 0:
            scores = torch.matmul(
                q_chunk.to(torch.float32),
                k_chunk.to(torch.float32).transpose(-1, -2),
            )
            scores.mul_(sm_scale)
            scores = logit_cap * torch.tanh(scores / logit_cap)
            scores = scores.masked_fill(~attn_mask, float("-inf"))
            if empty_rows.any():
                scores[empty_rows, :, :, 0] = 0
            probs = torch.softmax(scores, dim=-1).to(v_chunk.dtype)
            chunk_out = torch.matmul(probs, v_chunk)
        else:
            chunk_out = torch.nn.functional.scaled_dot_product_attention(
                q_chunk,
                k_chunk,
                v_chunk,
                attn_mask=attn_mask,
                scale=sm_scale,
                is_causal=False,
            )

        chunk_out = chunk_out.squeeze(2)
        if empty_rows.any():
            chunk_out = torch.where(
                empty_rows[:, None, None],
                torch.zeros_like(chunk_out),
                chunk_out,
            )
        out[start:end] = chunk_out

    return out


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
    kernel = os.environ.get("SGLANG_GLM_DSA_SM89_KERNEL", "triton").strip().lower()
    if kernel == "cuda":
        if v_block not in (None, 128) or block_m not in (None, 1) or split_k not in (
            None,
            1,
        ):
            raise ValueError(
                "SGLANG_GLM_DSA_SM89_KERNEL=cuda requires "
                "v_block=128, block_m=1, and split_k=1."
            )
        _log_kernel_impl_once(kernel)
        return _profiled_call(
            "sm89_sparse_mla.cuda.total",
            lambda: sm89_sparse_mla_prefill_cuda(
                q_nope=q_nope,
                q_rope=q_rope,
                kv_cache=kv_cache,
                page_table=page_table,
                cache_seqlens=cache_seqlens,
                sm_scale=sm_scale,
                logit_cap=logit_cap,
                v_head_dim=v_head_dim,
                block_n=block_n,
            ),
        )
    if kernel != "triton":
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_KERNEL must be either 'triton' or 'cuda'; "
            f"got {kernel!r}."
        )
    _log_kernel_impl_once(kernel)

    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_triton import (
        sm89_sparse_mla_prefill_triton as _sm89_sparse_mla_prefill_triton,
    )

    return _profiled_call(
        "sm89_sparse_mla.triton.total",
        lambda: _sm89_sparse_mla_prefill_triton(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
            v_block=v_block,
            block_n=block_n,
            block_m=block_m,
            split_k=split_k,
        ),
    )


def sm89_sparse_mla_prefill_cuda(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    block_n: int | None = None,
    cuda_impl: str | None = None,
) -> torch.Tensor:
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        sm89_sparse_mla_prefill_cuda as _sm89_sparse_mla_prefill_cuda,
    )

    return _sm89_sparse_mla_prefill_cuda(
        q_nope=q_nope,
        q_rope=q_rope,
        kv_cache=kv_cache,
        page_table=page_table,
        cache_seqlens=cache_seqlens,
        sm_scale=sm_scale,
        logit_cap=logit_cap,
        v_head_dim=v_head_dim,
        block_n=block_n,
        cuda_impl=cuda_impl,
    )


def sm89_sparse_mla_decode_cuda(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    decode_kernel: str | None = None,
) -> torch.Tensor:
    with profile_region("sm89_sparse_mla.decode.cuda.total"):
        return _sm89_sparse_mla_decode_cuda(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
            decode_kernel=decode_kernel,
        )


def _sm89_sparse_mla_decode_cuda(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    decode_kernel: str | None = None,
) -> torch.Tensor:
    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        select_sm89_sparse_mla_decode_kernel,
    )

    selected = select_sm89_sparse_mla_decode_kernel(decode_kernel)
    if not math.isfinite(logit_cap) or logit_cap < 0:
        raise ValueError("sm89_cuda decode requires logit_cap to be finite and nonnegative.")
    if q_nope.ndim != 3:
        raise ValueError("sm89_cuda decode requires rank-3 q_nope.")
    batch = q_nope.shape[0]
    tensors = (q_nope, q_rope, kv_cache, page_table, cache_seqlens)
    if any(t.device != q_nope.device for t in tensors):
        raise ValueError("sm89_cuda decode tensors must share one CUDA device.")
    if not q_nope.is_cuda or q_nope.dtype != torch.bfloat16:
        raise ValueError("sm89_cuda decode requires CUDA BF16 q_nope.")
    if not (1 <= q_nope.shape[1] <= 64) or q_nope.shape[-1] != 512:
        raise ValueError("sm89_cuda decode requires q_nope=[B,H,512].")
    if q_rope.dtype != torch.bfloat16 or q_rope.shape != (*q_nope.shape[:2], 64):
        raise ValueError("sm89_cuda decode requires q_rope=[B,H,64].")
    if (
        page_table.dtype != torch.int32
        or page_table.ndim != 2
        or page_table.shape[0] != batch
        or not (1 <= page_table.shape[1] <= 4096)
    ):
        raise ValueError("sm89_cuda decode requires page_table=[B,topk].")
    if cache_seqlens.dtype != torch.int32 or cache_seqlens.shape != (batch,):
        raise ValueError("sm89_cuda decode requires cache_seqlens=[B].")
    if (
        kv_cache.ndim != 3
        or kv_cache.shape[1] != 1
        or kv_cache.dtype != torch.float8_e4m3fn
        or kv_cache.shape[-1] != 656
    ):
        raise ValueError("sm89_cuda decode requires packed FP8 E4M3 KV width 656.")
    if v_head_dim != 512:
        raise ValueError("sm89_cuda decode requires v_head_dim=512.")
    if selected == "three_stage":
        _log_decode_kernel_once(selected)
        return sm89_sparse_mla_prefill_cuda(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
            block_n=32,
            cuda_impl="tensorcore",
        )

    split_tokens = 32 if selected == "splitk32" else 64
    if batch != 1 or q_nope.shape[1] != 32 or page_table.shape[1] != 2048:
        raise ValueError(
            "sm89_cuda decode split-K requires B=1, H=32, and top-k=2048."
        )
    if logit_cap != 0:
        raise ValueError("sm89_cuda decode split-K requires logit_cap=0.")
    if any(
        stride <= 0
        for tensor in (q_nope, q_rope, page_table)
        for stride in tensor.stride()
    ):
        raise ValueError("sm89_cuda decode split-K requires positive Q and page strides.")
    if cache_seqlens.stride(0) != 1:
        raise ValueError("sm89_cuda decode split-K requires cache_seqlens.stride(0)=1.")
    if kv_cache.stride(2) != 1:
        raise ValueError("sm89_cuda decode split-K requires kv_cache.stride(2)=1.")
    if kv_cache.stride(0) < 656 or kv_cache.stride(0) % 4:
        raise ValueError(
            "sm89_cuda decode split-K requires a 4-byte divisible KV row stride of at least 656."
        )
    if kv_cache.data_ptr() % 16:
        raise ValueError("sm89_cuda decode split-K requires a 16-byte-aligned KV base.")

    from sglang.srt.layers.attention.dsa.sm89_sparse_mla_cuda import (
        sm89_sparse_mla_decode_cuda_splitk,
        splitk_workspace_schema,
    )

    workspace = splitk_workspace_schema(
        batch, q_nope.shape[1], page_table.shape[1], split_tokens
    )
    if workspace["workspace_bytes"] >= 32 * 1024 * 1024:
        raise ValueError("sm89_cuda decode split-K workspace must be smaller than 32 MiB.")
    _log_decode_kernel_once(selected)
    return sm89_sparse_mla_decode_cuda_splitk(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        sm_scale,
        logit_cap,
        v_head_dim,
        split_tokens,
    )
