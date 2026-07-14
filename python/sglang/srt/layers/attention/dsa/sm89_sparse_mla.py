from __future__ import annotations

import logging
import math
import os

import torch

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
        if (
            v_block not in (None, 128)
            or block_m not in (None, 1)
            or split_k
            not in (
                None,
                1,
            )
        ):
            raise ValueError(
                "SGLANG_GLM_DSA_SM89_KERNEL=cuda requires "
                "v_block=128, block_m=1, and split_k=1."
            )
        _log_kernel_impl_once(kernel)
        return sm89_sparse_mla_prefill_cuda(
            q_nope=q_nope,
            q_rope=q_rope,
            kv_cache=kv_cache,
            page_table=page_table,
            cache_seqlens=cache_seqlens,
            sm_scale=sm_scale,
            logit_cap=logit_cap,
            v_head_dim=v_head_dim,
            block_n=block_n,
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

    return _sm89_sparse_mla_prefill_triton(
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
        raise ValueError(
            "sm89_cuda decode requires logit_cap to be finite and nonnegative."
        )
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
        raise ValueError("sm89_cuda decode split-K requires B=1, H=32, and top-k=2048.")
    if logit_cap != 0:
        raise ValueError("sm89_cuda decode split-K requires logit_cap=0.")
    if any(
        stride <= 0
        for tensor in (q_nope, q_rope, page_table)
        for stride in tensor.stride()
    ):
        raise ValueError(
            "sm89_cuda decode split-K requires positive Q and page strides."
        )
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
        raise ValueError(
            "sm89_cuda decode split-K workspace must be smaller than 32 MiB."
        )
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
