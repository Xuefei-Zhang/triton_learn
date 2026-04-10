# Phase 3 — Matmul, Tiling, and Autotune

## 0. Why This Phase Exists

Matmul is where memory layout, blocking, and reuse start dominating the quality of the kernel.

## 1. Learning Targets

- explain M/N/K clearly
- explain why tiling changes performance
- prepare a future autotune search space

## 2. Big Picture

This phase shifts your focus from “can I launch a kernel?” to “can I organize work well for the hardware?”

## 3. Core Concepts Broken Down

- tiling
- blocking
- data reuse
- autotune configs

## 4. Syntax and Keywords

- meta-parameters for tile sizes
- grouped scheduling ideas

## 5. Mental Execution Model

One program instance handles a tile, not a single scalar output.

## 6. Minimal Worked Example

Write out the data flow for a tiled matmul before touching code.

## 7. Common Bugs and Pitfalls

- mixing up math dimensions and memory layout
- choosing tile sizes without measurement

## 8. Before You Code

Explain why a correct matmul can still be a poor GPU kernel.

## 9. Mission

Follow `missions/phase-3-matmul-tiling-and-autotune.md`.

Starter assets for this phase:

- `playground/phase3_tiling_demo.py`
- `playground/phase3_matmul_exercise.py`
- `playground/phase3_autotune_exercise.py`
- `src/triton_learn/runtime/phase3_support.py`

## 10. Test and Benchmark Checklist

- benchmark multiple shapes
- keep correctness separate from performance claims

## 11. Project Integration

This phase prepares future `linear` and attention subpaths.

## 12. After You Finish

You should be able to justify at least one autotune dimension instead of saying “I tried random values.”

## 13. Optional Deep Dives

- `docs/concepts/tiling-blocking-and-data-reuse.md`
