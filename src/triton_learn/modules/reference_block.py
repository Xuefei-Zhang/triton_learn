from __future__ import annotations

import torch

from triton_learn.baseline.reference_ops import reference_attention, reference_layer_norm


def reference_attention_block(x: torch.Tensor) -> torch.Tensor:
    attended = reference_attention(x, x, x)
    return reference_layer_norm(attended + x, normalized_shape=x.shape[-1])
