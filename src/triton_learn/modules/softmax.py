from __future__ import annotations

import importlib
from typing import Callable, cast

import torch

from ..baseline.reference_ops import reference_rowwise_softmax_2d
from ..runtime.providers import Provider, choose_softmax_provider


def softmax(x: torch.Tensor, provider: Provider = "auto") -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax expects a 2D tensor [rows, cols]")
    if x.shape[1] == 0:
        raise ValueError("softmax requires at least one column")

    resolved = choose_softmax_provider(provider, x.device.type)

    if resolved == "torch":
        return reference_rowwise_softmax_2d(x)

    kernel_module = importlib.import_module("triton_learn.kernels.softmax")
    triton_softmax = cast(Callable[[torch.Tensor], torch.Tensor], kernel_module.triton_softmax)

    return triton_softmax(x)
