import os
from contextlib import contextmanager

import torch


def glm_dsa_sm89_profile_enabled() -> bool:
    return os.environ.get("SGLANG_GLM_DSA_SM89_PROFILE", "0") == "1"


def glm_dsa_sm89_nsys_enabled() -> bool:
    return os.environ.get("SGLANG_SM89_DECODE_NSYS_PROFILE", "0") == "1"


@contextmanager
def nvtx_range(name: str):
    enabled = torch.cuda.is_available()
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


@contextmanager
def cuda_timer(name: str, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        yield
        return

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    try:
        yield
    finally:
        end.record()
        torch.cuda.synchronize()
        print(f"[GLM-DSA-SM89] {name}: {start.elapsed_time(end):.3f} ms", flush=True)


@contextmanager
def profile_region(name: str, enabled: bool | None = None):
    if enabled is None:
        enabled = glm_dsa_sm89_profile_enabled()
    emit_nvtx = enabled or glm_dsa_sm89_nsys_enabled()
    if not emit_nvtx:
        yield
        return

    with nvtx_range(name), cuda_timer(name, enabled):
        yield
