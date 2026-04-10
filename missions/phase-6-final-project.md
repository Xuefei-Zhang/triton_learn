# Phase 6 — Final Project Integration

## Why This Phase Exists

The end state is not “I solved some tutorials.” The end state is “I built a small optimization lab and can explain its tradeoffs.”

## Before You Code

- explain the boundaries between baseline, kernels, runtime, modules, and integration
- explain what evidence you need before claiming an optimization is worthwhile

## Mission

1. Choose a small Transformer-style optimization target.
2. Keep a reference path and one optimized path.
3. Add correctness tests, benchmarks, and a write-up.
4. Document when the optimized path should fall back to the reference path.

## Test and Benchmark Checklist

- correctness comes first
- benchmarks specify shapes and dtypes
- tradeoffs are written down, not just observed privately

## Project Integration

The final project should live primarily under `integration/` with supporting modules and runtime logic below it.

## After You Finish

You should be able to give an engineering walkthrough of the project without opening the code.
