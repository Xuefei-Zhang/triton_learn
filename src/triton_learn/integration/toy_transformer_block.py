from __future__ import annotations

import torch
from torch import nn

from triton_learn.baseline.reference_ops import (
    reference_attention,
    reference_layer_norm,
    reference_linear,
)


class ToyTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attended = reference_attention(q, k, v)
        mixed = reference_linear(attended, self.out_proj.weight)
        return reference_layer_norm(mixed + x, normalized_shape=self.embed_dim)
