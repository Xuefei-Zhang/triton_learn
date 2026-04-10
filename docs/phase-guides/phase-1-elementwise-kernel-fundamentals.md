# Phase 1 — Elementwise Kernel Fundamentals

## 0. Why This Phase Exists

Elementwise kernels are the cleanest place to internalize the Triton execution model.

## 1. Learning Targets

- map launch grid to tensor slices
- explain block size tradeoffs
- write and test a simple elementwise kernel

## 2. Big Picture

This is where Triton stops being abstract. You now use the vector-add pattern as a template for future unary and binary operations.

## 3. Core Concepts Broken Down

- offsets
- contiguous flattening
- masks
- launch meta-parameters

## 4. Syntax and Keywords

- `triton.cdiv`
- `BLOCK_SIZE`
- pointer-plus-offset memory expressions

## 5. Mental Execution Model

Every program instance computes exactly one block of elements.

## 6. Minimal Worked Example

Study the existing vector-add kernel, then write a variation in `playground/`.

## 7. Common Bugs and Pitfalls

- wrong offset math
- shape mismatch assumptions
- silently relying on contiguity

## 8. Before You Code

Explain how the final partial block is handled.

## 9. Mission

Follow `missions/phase-1-elementwise.md`.

Starter assets for this phase:

- `playground/phase1_block_mapping_demo.py`
- `playground/phase1_vector_add_exercise.py`
- `playground/phase1_unary_op_exercise.py`
- `src/triton_learn/runtime/phase1_support.py`

## 10. Test and Benchmark Checklist

- compare against the reference path
- test at least one non-round input size

## 11. Project Integration

The elementwise pattern should become a reusable module boundary, not just a one-off kernel.

## 12. After You Finish

You should be able to derive a new elementwise kernel from the vector-add pattern without copy-pasting blindly.

## 13. Optional Deep Dives

- `docs/concepts/grid-program-id-offset-mask.md`
