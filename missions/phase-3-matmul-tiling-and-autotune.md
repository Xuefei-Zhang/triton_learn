# Phase 3 — Matmul, Tiling, and Autotune

## Why This Phase Exists

Matmul is where Triton starts to feel like real performance engineering rather than syntax practice.

## Before You Code

- explain the roles of M, N, and K
- explain why tiling improves reuse
- explain what autotune is trying to optimize

## Mission

1. Read `docs/concepts/tiling-blocking-and-data-reuse.md`.
2. Benchmark the reference linear path on a few shapes.
3. Sketch the future matmul kernel interface.
4. Decide which meta-parameters will eventually be tunable.

## Test and Benchmark Checklist

- keep correctness separate from performance claims
- document which shapes are interesting and why

## Project Integration

Phase 3 work should eventually extend `baseline/`, `kernels/`, `modules/`, and `benchmarks/` together.

## After You Finish

You should be able to explain why a correct matmul can still be a bad GPU kernel.
