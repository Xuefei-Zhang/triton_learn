# Repository Layout

This repository uses three top-level mental buckets:

## 1. Code: `src/triton_learn/`

This is the only importable code root.

- `baseline/` is the correctness oracle.
- `kernels/` contains Triton kernels.
- `runtime/` contains capability checks and provider routing.
- `modules/` contains learner-facing APIs.
- `integration/` contains composed examples.

## 2. Learning path: `missions/`

This is the main curriculum. If you want to know what to read and do next, start here.

## 3. Reference material: `docs/`

This contains stable lookup material. The missions point into this directory when you need deeper explanation.

## Why not duplicate everything?

Curriculum repos rot when the same explanation exists in multiple places. This repo keeps one main learning path and a separate reference layer so updates remain coherent.
