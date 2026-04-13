from __future__ import annotations

# `Literal` lets us say that only a few exact string values are valid providers.
from typing import Literal

# Provider selection depends on the runtime snapshot returned by `detect_runtime_capabilities`.
from triton_learn.env import detect_runtime_capabilities

# In this repository, vector add can be requested in three modes:
# - "auto": choose the best provider automatically
# - "torch": force the PyTorch/reference path
# - "triton": force the Triton/GPU path
Provider = Literal["auto", "torch", "triton"]


def choose_vector_add_provider(requested: Provider, device_type: str) -> Literal["torch", "triton"]:
    # Detect the current environment before making any provider decision.
    capabilities = detect_runtime_capabilities()

    # If the caller explicitly asked for the PyTorch path, honor that immediately.
    if requested == "torch":
        return "torch"

    # If the caller explicitly asked for Triton, we must validate that the runtime
    # actually supports that choice.
    if requested == "triton":
        # Triton vector add requires all three conditions below:
        # 1. Triton is importable
        # 2. CUDA is usable through PyTorch
        # 3. the input tensors are already on a CUDA device
        if not (capabilities.has_triton and capabilities.has_cuda and device_type == "cuda"):
            # We fail loudly instead of silently falling back, because an explicit
            # request for Triton should teach the learner when the environment is wrong.
            raise RuntimeError("Triton provider requires CUDA and Triton availability")
        return "triton"

    # Reaching this point means `requested == "auto"`.
    # In auto mode we prefer Triton only when the environment is actually ready for it.
    if capabilities.has_triton and capabilities.has_cuda and device_type == "cuda":
        return "triton"

    # Otherwise the safe fallback is always the PyTorch/reference path.
    return "torch"
