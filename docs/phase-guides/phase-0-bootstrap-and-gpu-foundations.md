# Phase 0 — Bootstrap and GPU Foundations

## 0. Why This Phase Exists

You need a mental model before you need more code. If you do not understand what a kernel, grid, program id, and mask are doing at a high level, later Triton code will feel memorized instead of reasoned about.

## 1. Learning Targets

- understand the role of PyTorch vs Triton in this repository
- understand the minimal Triton execution model
- run the starter tests and benchmark without guessing what they mean
- explain why the repo keeps both a baseline path and a Triton path

## 2. Big Picture

This repository is structured like a small engineering lab, not a pile of disconnected notebooks. The baseline path tells you what correct behavior looks like. The Triton path tells you how custom GPU kernels can replace part of that baseline. The missions tell you what to learn next.

## 3. Core Concepts Broken Down

- CPU vs GPU execution
- what a kernel is
- what one Triton program instance owns
- why tails require masks
- why correctness must come before benchmarking

## 4. Syntax and Keywords

Read these in context of the working vector-add kernel:

- `@triton.jit`
- `tl.program_id`
- `tl.arange`
- `tl.load`
- `tl.store`
- `tl.constexpr`

## 5. Mental Execution Model

One launch creates many program instances. Each program instance calculates a block start, builds offsets, loads values, adds them, and stores the result. The mask prevents invalid memory accesses for the last partial block.

## 6. Minimal Worked Example

Use `src/triton_learn/kernels/vector_add.py` as the example. Read it line by line and connect each line to the execution model above.

## 7. Common Bugs and Pitfalls

- confusing grid size with tensor shape
- forgetting that masks are required for tails
- benchmarking before validating against the baseline

## 8. Before You Code

Make sure you can explain:

- why the baseline implementation exists
- what `tl.program_id` is doing in vector add
- why the output tensor is allocated before the launch

## 9. Mission

Start with `missions/phase-0-bootstrap.md` and complete every checklist item.

## 10. Test and Benchmark Checklist

- `pytest -q`
- `python benchmarks/vector_add_bench.py`
- `python profiling/profile_vector_add.py`

## 11. Project Integration

At this point, the integration task is understanding the repo layout: baseline → runtime → modules → integration.

## 12. After You Finish

You should be able to explain this repo to another learner without opening the code first.

## 13. Optional Deep Dives

- `docs/concepts/glossary-gpu-basics.md`
- `docs/architecture/repository-layout.md`
