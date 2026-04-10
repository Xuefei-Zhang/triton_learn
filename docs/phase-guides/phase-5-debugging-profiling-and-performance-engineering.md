# Phase 5 — Debugging, Profiling, and Performance Engineering

## 0. Why This Phase Exists

Performance work becomes real only when you can explain your measurements and debug incorrect kernels systematically.

## 1. Learning Targets

- distinguish correctness bugs from performance bugs
- read and write meaningful benchmark notes
- know when lower-level generated code inspection is worthwhile

## 2. Big Picture

This phase upgrades you from “I ran a benchmark” to “I know what the benchmark is telling me.”

## 3. Core Concepts Broken Down

- trusted benchmarks
- profiling workflows
- likely sources of Triton performance regressions

## 4. Syntax and Keywords

- no single syntax focus; this phase is more about method than new language features

## 5. Mental Execution Model

Always ask what you measured, what changed, and what stayed fixed.

## 6. Minimal Worked Example

Run the benchmark and profiling helper for vector add, then write a short note interpreting the results.

## 7. Common Bugs and Pitfalls

- profiling the wrong thing
- trusting tiny or noisy benchmark differences
- skipping correctness validation

## 8. Before You Code

Explain why “faster once” is not enough evidence.

## 9. Mission

Follow `missions/phase-5-debugging-profiling-and-performance-engineering.md`.

## 10. Test and Benchmark Checklist

- record shapes, dtype, hardware, provider
- write at least one interpretation note

## 11. Project Integration

This phase strengthens `benchmarks/`, `profiling/`, and `notes/`.

## 12. After You Finish

You should be able to defend your measurement methodology.

## 13. Optional Deep Dives

- `docs/profiling/how-to-read-a-triton-benchmark.md`
- `docs/concepts/when-to-look-at-ir-ptx-asm.md`
