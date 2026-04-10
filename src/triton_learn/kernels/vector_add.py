from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised via capability routing
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _vector_add_kernel(
        a_ptr,
        b_ptr,
        out_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, a + b, mask=mask)


def triton_vector_add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if triton is None:
        raise RuntimeError("Triton is not available")
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise RuntimeError("Triton vector add requires CUDA tensors")

    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)
    out = torch.empty_like(a_flat)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    _vector_add_kernel[grid](a_flat, b_flat, out, n_elements, BLOCK_SIZE=block_size)
    return out.reshape(a.shape)
