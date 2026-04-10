# Triton Mastery Lab Starter Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a usable Triton Mastery Lab starter repository with missions, reference operators, one working Triton kernel path, tests, benchmarks, and git initialization.

**Architecture:** Keep one importable package root under `src/triton_learn/`. Use `missions/` as the primary learning path and keep `docs/` as reference/supporting material. Implement only the code that should already be solved in a starter repo; leave advanced phases as guided work.

**Tech Stack:** Python, pytest, ruff, optional torch/triton runtime, Triton benchmarking helpers when available.

---

## Chunk 1: Repository skeleton and design artifacts

**Files:**
- Create: `README.md`, `.gitignore`, `pyproject.toml`, `Makefile`
- Create: `docs/superpowers/specs/2026-04-10-triton-mastery-lab-design.md`
- Create: `docs/superpowers/plans/2026-04-10-triton-mastery-lab-starter.md`
- Move: `docs/reference/2026-04-07-language-mastery-lab-generic-design.md`

- [ ] Create the root files.
- [ ] Preserve the original generic design doc under `docs/reference/`.
- [ ] Describe the intended starter scope clearly so the learner knows what is already implemented and what is intentionally left for later phases.

## Chunk 2: Learning path and reference docs

**Files:**
- Create: `missions/phase-0-bootstrap.md` … `missions/phase-6-final-project.md`
- Create: concept, architecture, and profiling docs under `docs/`
- Create: `notes/README.md`, `notes/interview/vector-add.md`

- [ ] Write mission files that include before/after sections.
- [ ] Add glossary and concept docs for common Triton/GPU stumbling blocks.
- [ ] Keep docs reference-oriented; do not duplicate full mission walkthroughs in `docs/`.

## Chunk 3: Tests first for code layers

**Files:**
- Create tests under `tests/runtime/`, `tests/baseline/`, `tests/modules/`, `tests/integration/`, `tests/kernels/`

- [ ] Write tests for runtime capability detection.
- [ ] Write tests for PyTorch reference operators.
- [ ] Write tests for module/provider dispatch.
- [ ] Write an integration test for a toy transformer-style block.
- [ ] Write a CUDA/Triton-gated test for vector add.

## Chunk 4: Minimal code to satisfy tests

**Files:**
- Create `src/triton_learn/...`

- [ ] Implement capability detection and provider dispatch.
- [ ] Implement reference baseline ops with PyTorch.
- [ ] Implement one working Triton vector-add kernel path.
- [ ] Implement one composed integration module that uses the reference path.

## Chunk 5: Benchmarks, profiling helpers, and verification

**Files:**
- Create `benchmarks/vector_add_bench.py`
- Create `profiling/profile_vector_add.py`

- [ ] Add repeatable vector-add benchmark script.
- [ ] Add simple profiling helper.
- [ ] Run test and lint verification.

## Chunk 6: Git repository initialization

**Files:**
- Initialize repository in place.

- [ ] Run `git init`.
- [ ] Set remote to `https://github.com/Xuefei-Zhang/triton_learn.git`.
- [ ] Do not push unless explicitly requested.
