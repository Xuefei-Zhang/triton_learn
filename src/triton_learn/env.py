from __future__ import annotations

# `dataclass` lets us bundle several related runtime facts into one small object.
# Here we want a single return value that can answer questions like:
# - is PyTorch importable?
# - is Triton importable?
# - is CUDA actually usable right now?
# - if we need a default device string, should it be "cpu" or "cuda"?
from dataclasses import dataclass


# `frozen=True` makes the dataclass immutable after creation.
# That is useful here because runtime capability detection should behave like a
# snapshot of the environment, not like a mutable settings object.
@dataclass(frozen=True)
class RuntimeCapabilities:
    # Whether importing `torch` succeeded.
    has_torch: bool
    # Whether importing `triton` succeeded.
    has_triton: bool
    # Whether CUDA is not only present in theory, but actually usable through PyTorch.
    has_cuda: bool
    # A convenient default device string derived from `has_cuda`.
    default_device: str


def detect_runtime_capabilities() -> RuntimeCapabilities:
    # First we try to import PyTorch, because PyTorch is the easiest place to ask
    # whether CUDA is actually available to this Python environment.
    try:
        # Importing torch may fail if the package is missing or the environment is broken.
        import torch

        # If import worked, then PyTorch exists in this environment.
        has_torch = True
        # `torch.cuda.is_available()` is the real runtime check we care about.
        # It returns True only when PyTorch can successfully initialize CUDA.
        has_cuda = bool(torch.cuda.is_available())
    except Exception:
        # If anything goes wrong during the torch import path, we treat both
        # `has_torch` and `has_cuda` as unavailable.
        has_torch = False
        has_cuda = False

    # Triton availability is checked separately from CUDA availability.
    # A machine can have Triton installed but still be unable to use CUDA,
    # and the reverse can also be true.
    try:
        # We do not use the imported module in this function body, so the variable
        # would otherwise look unused to the linter.
        import triton  # noqa: F401

        # If the import succeeds, Triton exists in this Python environment.
        has_triton = True
    except Exception:
        # If the import fails, Triton-specific paths should stay disabled.
        has_triton = False

    # This repository uses a very simple device policy:
    # - if CUDA works, default to "cuda"
    # - otherwise fall back to "cpu"
    default_device = "cuda" if has_cuda else "cpu"

    # Return a small immutable record that other parts of the codebase can use
    # for provider selection, benchmark routing, and user-facing diagnostics.
    return RuntimeCapabilities(
        has_torch=has_torch,
        has_triton=has_triton,
        has_cuda=has_cuda,
        default_device=default_device,
    )
