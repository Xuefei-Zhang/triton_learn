# Phase 4 — Norm, Attention, and Composition

## 0. Why This Phase Exists

The repo should start feeling like a model optimization project rather than a collection of kernels.

## 1. Learning Targets

- understand the flow of a small attention-style block
- identify good optimization candidates
- compose modules cleanly

## 2. Big Picture

This phase turns isolated techniques into a miniature system.

## 3. Core Concepts Broken Down

- norm
- attention flow
- module composition
- boundary between reusable module and experiment code

## 4. Syntax and Keywords

- no new mandatory Triton syntax here; the emphasis is on architectural composition

## 5. Mental Execution Model

Think in terms of dataflow across reference and optimized paths.

## 6. Minimal Worked Example

Trace the provided `ToyTransformerBlock` and annotate where future Triton replacements could live.

## 7. Common Bugs and Pitfalls

- over-optimizing before identifying the hot path
- mixing reusable modules with integration glue

## 8. Before You Code

Explain why not every part of a model should be optimized first.

## 9. Mission

Follow `missions/phase-4-norm-attention-and-composition.md`.

## 10. Test and Benchmark Checklist

- validate shapes and outputs first
- then benchmark subpaths

## 11. Project Integration

This phase lives mainly in `integration/` and future `modules/` growth.

## 12. After You Finish

You should be able to argue for a specific next optimization target inside the block.

## 13. Optional Deep Dives

- `docs/architecture/final-project-overview.md`
