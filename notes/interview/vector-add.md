# Interview Note: Triton Vector Add

## What problem does it solve?

It is the simplest useful Triton kernel for learning how one launch maps many program instances to many elements.

## Key ideas

- each program instance owns one block of elements
- `tl.arange` builds offsets inside that block
- `mask` handles the tail
- `tl.load` and `tl.store` operate on pointer-plus-offset expressions

## Why does block size matter?

It changes how much data each program instance handles and can affect launch shape and hardware utilization.

## Why keep a PyTorch baseline?

Because correctness must have a trusted oracle. The baseline path lets you compare values before you start trusting performance claims.
