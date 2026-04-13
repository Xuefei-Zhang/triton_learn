from __future__ import annotations

# The module layer exposes a learner-friendly API built on top of lower-level pieces.
import torch

# Reference implementation used when the provider resolves to the PyTorch path.
from triton_learn.baseline.reference_ops import reference_vector_add

# Provider logic decides whether we should call PyTorch or Triton.
from triton_learn.runtime.providers import Provider, choose_vector_add_provider


def vector_add(a: torch.Tensor, b: torch.Tensor, provider: Provider = "auto") -> torch.Tensor:
    # Resolve the final provider choice using both the user request and the device type.
    resolved = choose_vector_add_provider(provider, a.device.type)

    # If the provider is `torch`, stay on the simple reference path.
    if resolved == "torch":
        return reference_vector_add(a, b)

    # Import the Triton path only when needed.
    # This keeps the module lightweight on systems where Triton is unavailable.
    from triton_learn.kernels.vector_add import triton_vector_add

    # Run the GPU kernel wrapper.
    return triton_vector_add(a, b)
