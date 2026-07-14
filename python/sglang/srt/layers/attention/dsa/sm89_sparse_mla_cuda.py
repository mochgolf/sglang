from __future__ import annotations

import math
import os
import struct
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


SM89_DSA_DECODE_KERNEL_ENV = "SGLANG_GLM_DSA_SM89_DECODE_KERNEL"
SM89_DSA_DECODE_KERNELS = frozenset(("three_stage", "splitk32", "splitk64"))


def select_sm89_sparse_mla_decode_kernel(value: str | None = None) -> str:
    selected = value if value is not None else os.environ.get(
        SM89_DSA_DECODE_KERNEL_ENV, "three_stage"
    )
    selected = selected.strip().lower()
    if selected not in SM89_DSA_DECODE_KERNELS:
        raise ValueError(
            f"{SM89_DSA_DECODE_KERNEL_ENV} must be one of "
            "'three_stage', 'splitk32', or 'splitk64'"
        )
    return selected


def splitk_workspace_schema(
    rows: int, heads: int, topk: int, split_tokens: int, value_dim: int = 512
) -> dict[str, int]:
    if rows <= 0 or heads <= 0 or topk <= 0 or value_dim <= 0:
        raise ValueError("split-K workspace dimensions must be positive")
    if split_tokens not in (32, 64) or topk % split_tokens:
        raise ValueError("split-K workspace requires split_tokens in {32,64} dividing topk")
    splits = topk // split_tokens
    partial_o_bytes = splits * rows * heads * value_dim * 4
    partial_lse_bytes = splits * rows * heads * 4
    return {
        "splits": splits,
        "partial_o_bytes": partial_o_bytes,
        "partial_lse_bytes": partial_lse_bytes,
        "workspace_bytes": partial_o_bytes + partial_lse_bytes,
    }


def allocate_sm89_sparse_mla_decode_cuda_splitk_workspace(
    q_nope: torch.Tensor, split_tokens: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if not q_nope.is_cuda:
        raise ValueError("sm89 decode split-K workspace requires a CUDA device.")
    workspace = splitk_workspace_schema(1, 32, 2048, int(split_tokens))
    splits = workspace["splits"]
    partial_o = torch.empty(
        (splits, 1, 32, 512), device=q_nope.device, dtype=torch.float32
    )
    partial_lse = torch.empty(
        (splits, 1, 32), device=q_nope.device, dtype=torch.float32
    )
    _validate_sm89_sparse_mla_decode_cuda_splitk_workspace(
        q_nope, partial_o, partial_lse, int(split_tokens)
    )
    return partial_o, partial_lse


def select_sm89_sparse_mla_cuda_impl(cuda_impl: str | None = None) -> str:
    selected = cuda_impl or os.environ.get(
        "SGLANG_GLM_DSA_SM89_CUDA_IMPL", "simt"
    )
    selected = selected.strip().lower()
    if selected not in ("simt", "tensorcore"):
        raise ValueError(
            "SGLANG_GLM_DSA_SM89_CUDA_IMPL must be 'simt' or 'tensorcore'"
        )
    return selected


@lru_cache(maxsize=1)
def _load_sm89_sparse_mla_cuda_ext():
    csrc_dir = Path(__file__).resolve().parent / "csrc"
    verbose = os.environ.get("SGLANG_GLM_DSA_SM89_CUDA_VERBOSE", "0") == "1"
    bin_dirs = [str(Path(sys.executable).parent)]
    try:
        import ninja

        if getattr(ninja, "BIN_DIR", None):
            bin_dirs.append(str(ninja.BIN_DIR))
    except ImportError:
        pass
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    prepend = [path for path in bin_dirs if path and path not in path_entries]
    if prepend:
        os.environ["PATH"] = os.pathsep.join([*prepend, *path_entries])
    return load(
        name="sglang_sm89_sparse_mla_cuda",
        sources=[str(csrc_dir / "sm89_sparse_mla_cuda.cu")],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=[
            "-O3",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--ptxas-options=-v",
            "-gencode=arch=compute_89,code=sm_89",
        ],
        verbose=verbose,
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
    if block_n not in (None, 32):
        raise ValueError("sm89_sparse_mla_prefill_cuda currently accepts only block_n=32.")

    selected_impl = select_sm89_sparse_mla_cuda_impl(cuda_impl)
    ext = _load_sm89_sparse_mla_cuda_ext()
    op = (
        ext.sm89_sparse_mla_prefill_cuda_tensorcore
        if selected_impl == "tensorcore"
        else ext.sm89_sparse_mla_prefill_cuda
    )
    return op(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        float(sm_scale),
        float(logit_cap),
        int(v_head_dim),
    )


def sm89_sparse_mla_decode_cuda_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    split_tokens: int,
) -> torch.Tensor:
    effective_sm_scale = _validate_sm89_sparse_mla_decode_cuda_splitk(
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
    ext = _load_sm89_sparse_mla_cuda_ext()
    return ext.sm89_sparse_mla_decode_cuda_splitk(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        effective_sm_scale,
        float(logit_cap),
        int(v_head_dim),
        int(split_tokens),
    )


def sm89_sparse_mla_decode_cuda_splitk_debug(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    split_tokens: int,
):
    effective_sm_scale = _validate_sm89_sparse_mla_decode_cuda_splitk(
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
    ext = _load_sm89_sparse_mla_cuda_ext()
    return ext.sm89_sparse_mla_decode_cuda_splitk_debug(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        effective_sm_scale,
        float(logit_cap),
        int(v_head_dim),
        int(split_tokens),
    )


def sm89_sparse_mla_decode_cuda_splitk_partial(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    split_tokens: int,
) -> None:
    effective_sm_scale = _validate_sm89_sparse_mla_decode_cuda_splitk(
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
    _validate_sm89_sparse_mla_decode_cuda_splitk_workspace(
        q_nope, partial_o, partial_lse, int(split_tokens)
    )
    ext = _load_sm89_sparse_mla_cuda_ext()
    ext.sm89_sparse_mla_decode_cuda_splitk_partial(
        q_nope,
        q_rope,
        kv_cache,
        page_table,
        cache_seqlens,
        partial_o,
        partial_lse,
        effective_sm_scale,
        float(logit_cap),
        int(v_head_dim),
        int(split_tokens),
    )


def _validate_sm89_sparse_mla_decode_cuda_splitk(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    sm_scale: float,
    logit_cap: float,
    v_head_dim: int,
    split_tokens: int,
) -> float:
    tensors = (q_nope, q_rope, kv_cache, page_table, cache_seqlens)
    if not q_nope.is_cuda or any(t.device != q_nope.device for t in tensors):
        raise ValueError("sm89 decode split-K tensors must share one CUDA device.")
    if q_nope.dtype != torch.bfloat16 or q_nope.shape != (1, 32, 512):
        raise ValueError("sm89 decode split-K requires BF16 q_nope=[1,32,512].")
    if q_rope.dtype != torch.bfloat16 or q_rope.shape != (1, 32, 64):
        raise ValueError("sm89 decode split-K requires BF16 q_rope=[1,32,64].")
    if (
        kv_cache.dtype != torch.float8_e4m3fn
        or kv_cache.ndim != 3
        or not (1 <= kv_cache.shape[0] <= 2**31 - 1)
        or kv_cache.shape[1:] != (1, 656)
    ):
        raise ValueError("sm89 decode split-K requires FP8 E4M3 KV=[tokens,1,656].")
    if page_table.dtype != torch.int32 or page_table.shape != (1, 2048):
        raise ValueError("sm89 decode split-K requires int32 page_table=[1,2048].")
    if cache_seqlens.dtype != torch.int32 or cache_seqlens.shape != (1,):
        raise ValueError("sm89 decode split-K requires int32 cache_seqlens=[1].")
    input_sm_scale = float(sm_scale)
    try:
        effective_sm_scale = struct.unpack(
            "=f", struct.pack("=f", input_sm_scale)
        )[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            "sm89 decode split-K requires finite FP32 sm_scale."
        ) from error
    if not math.isfinite(input_sm_scale) or not math.isfinite(effective_sm_scale):
        raise ValueError("sm89 decode split-K requires finite FP32 sm_scale.")
    if not math.isfinite(float(logit_cap)) or float(logit_cap) != 0.0:
        raise ValueError("sm89 decode split-K requires finite logit_cap=0.")
    if int(v_head_dim) != 512:
        raise ValueError("sm89 decode split-K requires v_head_dim=512.")
    if int(split_tokens) not in (32, 64):
        raise ValueError("sm89 decode split-K requires split_tokens in {32,64}.")
    if any(
        stride <= 0
        for tensor in (q_nope, q_rope, page_table)
        for stride in tensor.stride()
    ):
        raise ValueError("sm89 decode split-K requires positive Q and page strides.")
    if cache_seqlens.stride(0) != 1:
        raise ValueError("sm89 decode split-K requires cache_seqlens.stride(0)=1.")
    if kv_cache.stride(2) != 1:
        raise ValueError("sm89 decode split-K requires kv_cache.stride(2)=1.")
    if kv_cache.stride(0) < 656 or kv_cache.stride(0) % 4:
        raise ValueError(
            "sm89 decode split-K requires a 4-byte-divisible KV row stride of at least 656."
        )
    if kv_cache.data_ptr() % 16:
        raise ValueError("sm89 decode split-K requires a 16-byte-aligned KV base.")
    workspace = splitk_workspace_schema(1, 32, 2048, int(split_tokens))
    if workspace["workspace_bytes"] >= 32 * 1024 * 1024:
        raise ValueError("sm89 decode split-K workspace must be smaller than 32 MiB.")
    return effective_sm_scale


def _validate_sm89_sparse_mla_decode_cuda_splitk_workspace(
    q_nope: torch.Tensor,
    partial_o: torch.Tensor,
    partial_lse: torch.Tensor,
    split_tokens: int,
) -> None:
    workspace = splitk_workspace_schema(1, 32, 2048, int(split_tokens))
    splits = workspace["splits"]
    if (
        not partial_o.is_cuda
        or not partial_lse.is_cuda
        or partial_o.device != q_nope.device
        or partial_lse.device != q_nope.device
    ):
        raise ValueError(
            "sm89 decode split-K workspaces must share the input CUDA device."
        )
    if partial_o.dtype != torch.float32 or partial_o.shape != (splits, 1, 32, 512):
        raise ValueError(
            f"sm89 decode split-K requires FP32 partial_o=[{splits},1,32,512]."
        )
    if partial_lse.dtype != torch.float32 or partial_lse.shape != (splits, 1, 32):
        raise ValueError(
            f"sm89 decode split-K requires FP32 partial_lse=[{splits},1,32]."
        )
    if not partial_o.is_contiguous() or not partial_lse.is_contiguous():
        raise ValueError("sm89 decode split-K workspaces must be contiguous.")
    if (
        partial_o.numel() <= 0
        or partial_lse.numel() <= 0
        or partial_o.untyped_storage().nbytes() <= 0
        or partial_lse.untyped_storage().nbytes() <= 0
    ):
        raise ValueError("sm89 decode split-K workspaces require positive storage.")
    if partial_o.data_ptr() % 32:
        raise ValueError(
            "sm89 decode split-K requires a 32-byte-aligned partial_o base."
        )
