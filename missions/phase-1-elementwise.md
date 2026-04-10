# Phase 1 — Elementwise Kernels

## Why This Phase Exists

Elementwise kernels are the cleanest place to learn program mapping, pointer arithmetic, masks, and block size tradeoffs.

## Before You Code

- explain how one program instance maps to one chunk of data
- explain what happens when the data length is not divisible by the block size

## Mission

1. Read `docs/concepts/grid-program-id-offset-mask.md`.
2. Study `src/triton_learn/kernels/vector_add.py`.
3. Modify block size locally and re-run the benchmark.
4. Implement one new unary or binary elementwise exercise in `playground/`.

## Test and Benchmark Checklist

- compare the reference and Triton paths
- write at least one edge-case test for a non-round size

## Project Integration

Elementwise kernels usually belong first in `kernels/`, then receive a user-facing wrapper under `modules/`.

## After You Finish

You should be able to explain the relationship between `grid`, `program_id`, `offsets`, `mask`, and `tl.load`.
