# Phase 2 — Reduction and Fusion

## 0. Why This Phase Exists

Reductions force you to reason about cooperation across values, and fusion introduces the first serious performance argument for Triton.

## 1. Learning Targets

- explain stable softmax
- understand why fused operations reduce memory traffic
- prepare for a future softmax kernel implementation

## 2. Big Picture

You move from “one output depends on one input position” to “one output depends on many input positions.”

## 3. Core Concepts Broken Down

- reduction
- broadcast
- stable softmax
- fusion

## 4. Syntax and Keywords

- `tl.max`
- `tl.sum`
- reduction axes

## 5. Mental Execution Model

Rows become the natural unit of work for early reduction examples.

## 6. Minimal Worked Example

Prototype row-wise max and row-wise sum in `playground/` before attempting fused softmax.

## 7. Common Bugs and Pitfalls

- unstable exponentials
- wrong reduction axis
- thinking fewer kernels always means better code

## 8. Before You Code

Explain why subtracting the maximum is required in practice.

## 9. Mission

Follow `missions/phase-2-reduction-and-fusion.md`.

## 10. Test and Benchmark Checklist

- define numeric tolerances
- validate against PyTorch first

## 11. Project Integration

This phase prepares the future softmax and attention paths.

## 12. After You Finish

You should be able to explain the difference between an algebraically correct softmax and a numerically stable implementation.

## 13. Optional Deep Dives

- `docs/concepts/numerical-stability-in-softmax.md`
