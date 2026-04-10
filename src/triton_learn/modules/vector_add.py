from __future__ import annotations

import torch

from triton_learn.baseline.reference_ops import reference_vector_add
from triton_learn.runtime.providers import Provider, choose_vector_add_provider


def vector_add(a: torch.Tensor, b: torch.Tensor, provider: Provider = "auto") -> torch.Tensor:
    resolved = choose_vector_add_provider(provider, a.device.type)

    if resolved == "torch":
        return reference_vector_add(a, b)

    from triton_learn.kernels.vector_add import triton_vector_add

    return triton_vector_add(a, b)
