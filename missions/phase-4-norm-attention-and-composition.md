# Phase 4 — Norm, Attention, and Composition

## Why This Phase Exists

This is where separate ideas start forming something that looks like a real model path rather than isolated demos.

## Before You Code

- explain the data flow of attention at a high level
- explain why norm and attention are good optimization targets

## Mission

1. Read `docs/architecture/final-project-overview.md`.
2. Study `src/triton_learn/integration/toy_transformer_block.py`.
3. Trace how reference attention and norm are composed.
4. Write down which subpaths would be best Triton candidates next.

## Test and Benchmark Checklist

- verify output shape and dtype first
- benchmark modules separately before claiming end-to-end wins

## Project Integration

This phase is where reusable modules and composed integration paths must stay clearly separated.

## After You Finish

You should be able to argue for or against optimizing a specific subpath rather than saying “optimize everything.”
