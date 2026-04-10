# Phase 5 — Debugging, Profiling, and Performance Engineering

## Why This Phase Exists

Past this point the question is not “can I write a kernel?” but “can I explain why it is right, why it is fast, and why a different configuration changes the result?”

## Before You Code

- explain the difference between correctness bugs and performance bugs
- explain why an untrustworthy benchmark is dangerous

## Mission

1. Read `docs/profiling/how-to-read-a-triton-benchmark.md`.
2. Read `docs/concepts/when-to-look-at-ir-ptx-asm.md`.
3. Use the benchmark and profiling helpers on vector add.
4. Write one profiling note in `notes/`.

## Test and Benchmark Checklist

- write down the exact hardware/software context
- confirm you are comparing equal work

## Project Integration

Phase 5 adds evidence and interpretation, not just more code.

## After You Finish

You should be able to say not only which provider is faster, but also what you think the limiting factor is.
