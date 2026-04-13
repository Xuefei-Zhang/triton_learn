from __future__ import annotations

# We still use PyTorch modules here because this file is about composition and flow,
# not about replacing every piece with Triton yet.
import torch
from torch import nn

# These reference operations let us build a tiny transformer-like path while keeping
# correctness and readability high.
from triton_learn.baseline.reference_ops import (
    reference_attention,
    reference_layer_norm,
    reference_linear,
)


# `nn.Module` is the standard PyTorch base class for model components.
class ToyTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        # Always initialize the parent class first so PyTorch can register parameters.
        super().__init__()

        # Store the embedding width so we can reuse it later in normalization.
        self.embed_dim = embed_dim

        # These three linear layers act like Q, K, and V projections.
        # `bias=False` keeps the example simpler and closer to many optimized attention paths.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Final output projection after attention mixing.
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project the input into query, key, and value spaces.
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Run the simplified reference attention implementation.
        attended = reference_attention(q, k, v)

        # Apply the output projection with the functional linear helper.
        mixed = reference_linear(attended, self.out_proj.weight)

        # Add the residual connection and normalize the final result.
        return reference_layer_norm(mixed + x, normalized_shape=self.embed_dim)
