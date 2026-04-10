from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeCapabilities:
    has_torch: bool
    has_triton: bool
    has_cuda: bool
    default_device: str


def detect_runtime_capabilities() -> RuntimeCapabilities:
    try:
        import torch

        has_torch = True
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        has_torch = False
        has_cuda = False

    try:
        import triton  # noqa: F401

        has_triton = True
    except Exception:
        has_triton = False

    default_device = "cuda" if has_cuda else "cpu"
    return RuntimeCapabilities(
        has_torch=has_torch,
        has_triton=has_triton,
        has_cuda=has_cuda,
        default_device=default_device,
    )
