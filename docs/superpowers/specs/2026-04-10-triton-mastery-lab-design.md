# Triton Mastery Lab Design

## Goal

Create a self-contained Triton learning repository that takes a learner from minimal Triton/GPU intuition to the point where they can explain, test, benchmark, and integrate custom Triton kernels into a small engineering project.

## Core Design Decisions

1. **One code root**: all importable code lives under `src/triton_learn/`.
2. **One learning-path root**: `missions/` is the phase-by-phase curriculum.
3. **Docs are reference, not a duplicate curriculum**: `docs/` supports missions.
4. **Starter repo over solved repo**: phase structure is complete, but advanced phases remain guided work rather than fully pre-solved code.
5. **Before/After discipline**: each phase explicitly defines what you must understand before coding and what you must be able to explain after finishing.

## Repository Architecture

### Code Layers

- `baseline/`: the correctness oracle implemented with PyTorch
- `kernels/`: pure Triton kernel definitions and thin kernel-level wrappers
- `runtime/`: capability detection, provider routing, launch decisions
- `modules/`: user-facing module-style APIs that choose between providers
- `integration/`: composed examples such as a toy transformer-like block

### Learning Layers

- `missions/`: the main sequence, phases 0–6
- `docs/concepts/`: targeted explanations of key ideas
- `docs/architecture/`: why the repo is organized the way it is
- `docs/profiling/`: how to interpret measurements
- `notes/`: personal synthesis and interview-ready explanations

## Phase Model

Every phase contains these sections:

1. Why This Phase Exists
2. Learning Targets
3. Big Picture
4. Core Concepts Broken Down
5. Syntax and Keywords
6. Mental Execution Model
7. Common Bugs and Pitfalls
8. Before You Code
9. Mission
10. Test and Benchmark Checklist
11. Project Integration
12. After You Finish
13. Optional Deep Dives

## Starter Implementation Scope

The repository initially ships with:

- working environment/runtime detection
- working PyTorch reference operators
- working Triton vector-add path
- provider-dispatch module boundary
- toy integration module
- tests and benchmarks
- complete mission scaffolding for phases 0–6

The repository intentionally does **not** pre-solve all advanced phase kernels. Those phases are designed to be implemented by the learner under the mission structure.

## Final Project Shape

The long-term target is a mini Transformer-style optimization lab that compares:

- PyTorch reference implementations
- Triton-optimized paths
- benchmark and profiling evidence
- correctness and tradeoff analysis

## Success Criteria

The starter repo is successful if a returning learner can:

- set it up quickly
- read a phase mission and start immediately
- run tests and benchmarks
- compare a baseline implementation with a Triton implementation
- understand where later phase work should be added
