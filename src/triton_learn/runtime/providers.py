from __future__ import annotations

from typing import Literal

from triton_learn.env import detect_runtime_capabilities

Provider = Literal["auto", "torch", "triton"]


def choose_vector_add_provider(requested: Provider, device_type: str) -> Literal["torch", "triton"]:
    capabilities = detect_runtime_capabilities()

    if requested == "torch":
        return "torch"

    if requested == "triton":
        if not (capabilities.has_triton and capabilities.has_cuda and device_type == "cuda"):
            raise RuntimeError("Triton provider requires CUDA and Triton availability")
        return "triton"

    if capabilities.has_triton and capabilities.has_cuda and device_type == "cuda":
        return "triton"
    return "torch"
