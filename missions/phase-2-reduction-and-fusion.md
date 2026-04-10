# Phase 2 — Reduction and Fusion

## Why This Phase Exists

Reduction is the first time you need to think beyond simple elementwise mapping. It is also where numerics become impossible to ignore.

## Before You Code

- explain why softmax needs numerical stabilization
- explain why fusion can reduce memory traffic

## Mission

1. Read `docs/concepts/numerical-stability-in-softmax.md`.
2. Run `python playground/phase2_softmax_steps_demo.py`.
3. Implement the reasoning steps in `playground/phase2_reduction_exercise.py`.
4. Fill out `playground/phase2_softmax_exercise.py` before attempting any Triton softmax code.
5. Design the future `kernels/softmax.py` API before writing code.

## Test and Benchmark Checklist

- define tolerance expectations before benchmarking
- compare fused and unfused thinking even if you do not fully implement the fused kernel yet

## Project Integration

This phase sets up future `modules/softmax.py` work and later attention integration.

## After You Finish

You should be able to explain why “fewer kernels” can matter for speed even when the math is unchanged.
