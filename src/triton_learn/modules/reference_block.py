from __future__ import annotations

# This file contains a tiny compositional reference block, not a Triton kernel.
# Its job is to show how small baseline operators combine into a larger flow.
import torch

# Reuse the already-defined baseline attention and normalization helpers.
from triton_learn.baseline.reference_ops import reference_attention, reference_layer_norm


def reference_attention_block(x: torch.Tensor) -> torch.Tensor:
    # This simplified block uses the same tensor for Q, K, and V.
    # That is not a realistic trained model setup, but it is enough to demonstrate
    # the shape flow of attention plus a residual connection.
    attended = reference_attention(x, x, x)

    # Add the residual connection (`attended + x`) and normalize across the last dimension.
    return reference_layer_norm(attended + x, normalized_shape=x.shape[-1])
