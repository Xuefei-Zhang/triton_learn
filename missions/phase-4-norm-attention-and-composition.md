# Phase 4 — Norm, Attention, and Composition

## Why This Phase Exists

This is where separate ideas start forming something that looks like a real model path rather than isolated demos.

## Before You Code

- explain the data flow of attention at a high level
- explain why norm and attention are good optimization targets

## Mission

1. Read `docs/architecture/final-project-overview.md`.
2. Run `python playground/phase4_attention_flow_demo.py`.
3. Study `src/triton_learn/integration/toy_transformer_block.py`.
4. Fill out `playground/phase4_norm_attention_exercise.py`.
5. Fill out `playground/phase4_target_selection_exercise.py`.
6. Write down which subpaths would be best Triton candidates next.

## Test and Benchmark Checklist

- verify output shape and dtype first
- benchmark modules separately before claiming end-to-end wins

## Project Integration

This phase is where reusable modules and composed integration paths must stay clearly separated.

## After You Finish

You should be able to argue for or against optimizing a specific subpath rather than saying “optimize everything.”
