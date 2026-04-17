from __future__ import annotations

# `math` is used for the square-root factor in scaled dot-product attention.
import math

# PyTorch gives us a correctness-oriented reference implementation for every operator here.
import torch

# `torch.nn.functional` exposes functional versions of common neural-network operations.
import torch.nn.functional as F


def reference_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # A reference implementation should fail clearly on obviously invalid input.
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    # For vector add, the PyTorch baseline is just regular tensor addition.
    return a + b


def reference_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # We delegate directly to PyTorch so later Triton implementations can compare
    # against a well-tested baseline.
    return torch.softmax(x, dim=dim)


def reference_rowwise_softmax_2d(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax expects a 2D tensor [rows, cols]")
    if x.shape[1] == 0:
        raise ValueError("softmax requires at least one column")

    return torch.softmax(x, dim=-1)


def reference_layer_norm(x: torch.Tensor, normalized_shape: int) -> torch.Tensor:
    # PyTorch expects `normalized_shape` as a tuple, even if we are only normalizing
    # over one trailing dimension.
    return F.layer_norm(x, (normalized_shape,))


def reference_linear(
    # `x` is the input activation tensor.
    x: torch.Tensor,
    # `weight` follows PyTorch's `F.linear` layout: [out_features, in_features].
    weight: torch.Tensor,
    # Bias is optional because some projections in transformer-style code omit it.
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    # `F.linear` computes `x @ weight.T + bias` under the hood.
    return F.linear(x, weight, bias)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # `d` is the hidden size per token per head for this simplified attention path.
    d = q.shape[-1]

    # Compute attention scores by multiplying Q with K^T.
    # The transpose swaps the last two dimensions so every query position can score
    # against every key position.
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)

    # Normalize scores into probabilities along the last dimension.
    probs = torch.softmax(scores, dim=-1)

    # Use those probabilities to mix the value tensor.
    return torch.matmul(probs, v)
