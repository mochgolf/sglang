from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


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
