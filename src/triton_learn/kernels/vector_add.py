from __future__ import annotations

# PyTorch is still used around the Triton kernel for input validation,
# output allocation, reshaping, and user-facing tensor handling.
import torch

# Triton import is optional at import time, because this repository is designed
# to remain readable and partially runnable even on machines without Triton.
try:
    # `triton` is the top-level package that gives us launch helpers like `cdiv`.
    import triton

    # `triton.language` is usually imported as `tl`; this is where the kernel-side
    # operations such as `tl.load`, `tl.store`, and `tl.arange` live.
    import triton.language as tl
except Exception:  # pragma: no cover - exercised via capability routing
    # If Triton import fails, we record that by setting both names to `None`.
    # Later runtime checks turn that into a clear user-facing error.
    triton = None
    tl = None


# We only define the actual Triton kernel if Triton imported successfully.
if triton is not None:
    # `@triton.jit` tells Triton to compile this Python function into a GPU kernel.
    # Conceptually, you can think of it as: "this function is no longer ordinary
    # Python code; Triton will lower it into GPU code that runs in parallel."
    @triton.jit
    def _vector_add_kernel(
        # Pointer to the first element of input tensor `a` in device memory.
        a_ptr,
        # Pointer to the first element of input tensor `b` in device memory.
        b_ptr,
        # Pointer to the first element of the output tensor in device memory.
        out_ptr,
        # Total number of scalar elements we need to process.
        n_elements,
        # `tl.constexpr` means Triton treats this as a compile-time constant.
        # That matters because block size influences code generation and loop shapes.
        BLOCK_SIZE: tl.constexpr,
    ):
        # Every Triton program instance gets an integer id inside the launch grid.
        # In this simple 1D kernel, we only use `axis=0`.
        pid = tl.program_id(axis=0)

        # Each program instance owns one contiguous block of `BLOCK_SIZE` elements.
        # So the start of this block is just `pid * BLOCK_SIZE`.
        block_start = pid * BLOCK_SIZE

        # `tl.arange(0, BLOCK_SIZE)` creates vector offsets inside the block:
        # [0, 1, 2, ..., BLOCK_SIZE-1].
        # Adding `block_start` shifts those local offsets into global tensor positions.
        offsets = block_start + tl.arange(0, BLOCK_SIZE)

        # Near the end of the tensor, the last block may extend past `n_elements`.
        # The mask marks which offsets are valid and which would be out of bounds.
        mask = offsets < n_elements

        # `tl.load` reads a block of values from GPU memory.
        # Pointer arithmetic in Triton works by adding offsets to the base pointer.
        # The mask prevents invalid reads on the tail block.
        a = tl.load(a_ptr + offsets, mask=mask)
        b = tl.load(b_ptr + offsets, mask=mask)

        # `tl.store` writes the elementwise sum back to the output pointer.
        # The same mask prevents invalid writes beyond the true tensor length.
        tl.store(out_ptr + offsets, a + b, mask=mask)


def triton_vector_add(a: torch.Tensor, b: torch.Tensor, block_size: int = 1024) -> torch.Tensor:
    # Triton kernels here assume elementwise inputs with identical shape.
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    # If Triton is not importable, there is no way to launch the kernel.
    if triton is None:
        raise RuntimeError("Triton is not available")

    # This starter kernel is written only for CUDA tensors.
    if a.device.type != "cuda" or b.device.type != "cuda":
        raise RuntimeError("Triton vector add requires CUDA tensors")

    # We flatten the inputs into 1D contiguous buffers because this kernel is written
    # as a simple 1D elementwise kernel over scalar elements.
    a_flat = a.contiguous().reshape(-1)
    b_flat = b.contiguous().reshape(-1)

    # Allocate the output buffer on the same device with the same dtype and shape.
    out = torch.empty_like(a_flat)

    # Count the number of scalar elements that the kernel must cover.
    n_elements = out.numel()

    def grid(meta):
        # Triton launches kernels over a grid of program instances.
        # `triton.cdiv` computes ceiling division, so this returns enough programs
        # to cover all elements even when `n_elements` is not divisible by block size.
        return (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    # `_vector_add_kernel[grid](...)` is Triton's launch syntax.
    # Read it as: "launch this kernel over the given grid with these runtime args
    # and these compile-time meta-parameters."
    _vector_add_kernel[grid](a_flat, b_flat, out, n_elements, BLOCK_SIZE=block_size)

    # Restore the original tensor shape before returning to the caller.
    return out.reshape(a.shape)
