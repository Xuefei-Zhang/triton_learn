# Phase 0 — Bootstrap and GPU Foundations

## Why This Phase Exists

You cannot reason about Triton if you do not yet have a usable mental model for kernels, program ids, launch grids, and why GPUs behave differently from CPUs.

## Before You Code

You should be able to explain:

- what a GPU kernel is
- why masks exist
- what `tl.program_id` roughly means
- why correctness must be checked against a reference implementation

## Mission

1. Read `docs/concepts/glossary-gpu-basics.md`.
2. Read `docs/concepts/glossary-triton-basics.md`.
3. Run `pytest -q`.
4. Run `python benchmarks/vector_add_bench.py`.
5. Open the working vector-add implementation and trace the data flow.

## Test and Benchmark Checklist

- runtime tests pass
- baseline tests pass
- Triton vector-add test either passes or skips cleanly
- benchmark script runs without crashing

## Project Integration

At the end of this phase you should know where new kernels, modules, and integration paths belong in the repo.

## After You Finish

You should be able to explain why this starter repo has both a baseline implementation and a Triton implementation of vector add.
