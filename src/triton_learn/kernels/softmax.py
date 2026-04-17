from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import torch

if TYPE_CHECKING:
    ConstExpr: TypeAlias = int

try:
    import triton
    import triton.language as tl
    from triton.language import constexpr as triton_constexpr
except Exception:  # pragma: no cover - exercised through runtime checks
    triton = None
    tl = None
    triton_constexpr = None
else:
    ConstExpr = triton_constexpr


if triton is not None and tl is not None:
    tl_lang = tl

    @triton.jit
    def _rowwise_softmax_kernel(
        x_ptr,
        out_ptr,
        row_stride,
        n_cols,
        BLOCK_SIZE: ConstExpr,
    ):
        row_idx = tl_lang.program_id(axis=0)
        row_start = row_idx * row_stride
        col_offsets = tl_lang.arange(0, BLOCK_SIZE)
        mask = col_offsets < n_cols

        row = tl_lang.load(x_ptr + row_start + col_offsets, mask=mask, other=-float("inf"))
        row_max = tl_lang.max(row, axis=0)
        numerator = tl_lang.exp(row - row_max)
        denominator = tl_lang.sum(numerator, axis=0)
        softmax_row = numerator / denominator

        tl_lang.store(out_ptr + row_start + col_offsets, softmax_row, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2:
        raise ValueError("softmax expects a 2D tensor [rows, cols]")
    if x.shape[1] == 0:
        raise ValueError("softmax requires at least one column")
    if x.dtype != torch.float32:
        raise RuntimeError("Triton softmax currently supports float32 inputs only")

    if triton is None:
        raise RuntimeError("Triton is not available")

    if x.device.type != "cuda":
        raise RuntimeError("Triton softmax requires CUDA tensors")

    x_contiguous = x.contiguous()
    rows, cols = x_contiguous.shape
    out = torch.empty_like(x_contiguous)
    block_size = triton.next_power_of_2(cols)

    _rowwise_softmax_kernel[(rows,)](
        x_contiguous,
        out,
        x_contiguous.stride(0),
        cols,
        BLOCK_SIZE=block_size,
    )

    return out
