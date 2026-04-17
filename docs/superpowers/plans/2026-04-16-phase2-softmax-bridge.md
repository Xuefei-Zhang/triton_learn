# Phase 2 Softmax Bridge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the approved Phase 2 row-wise 2D softmax bridge with baseline, provider dispatch, Triton float32 kernel wrapper, tests, docs, and benchmark support.

**Architecture:** Mirror the existing vector-add vertical slice so each layer has one clear responsibility: baseline correctness wrapper, runtime provider chooser, optional Triton kernel wrapper, and module dispatch entrypoint. Keep all behavior limited to 2D last-dimension softmax, reject empty-column inputs, constrain the Triton path to float32, and make CUDA/Triton support optional at runtime.

**Tech Stack:** Python, PyTorch, Triton, pytest, optional CUDA runtime.

---

## Chunk 1: Baseline and runtime provider path

**Files:**
- Modify: `src/triton_learn/baseline/reference_ops.py`
- Modify: `src/triton_learn/baseline/__init__.py`
- Modify: `src/triton_learn/runtime/providers.py`
- Modify: `src/triton_learn/runtime/__init__.py`
- Modify: `tests/baseline/test_reference_ops.py`
- Create: `tests/runtime/test_providers.py`

- [ ] Write failing tests for `reference_rowwise_softmax_2d` success and 2D validation.
- [ ] Run `pytest tests/baseline/test_reference_ops.py -q` and confirm the new tests fail because the function does not exist yet.
- [ ] Write failing tests for `choose_softmax_provider` torch, explicit Triton failure, CPU auto fallback, and CUDA auto routing.
- [ ] Run `pytest tests/runtime/test_providers.py -q` and confirm the new tests fail because the chooser does not exist yet.
- [ ] Implement the minimal baseline wrapper and provider chooser.
- [ ] Export the new public APIs from the baseline and runtime packages.
- [ ] Re-run both targeted test files until they pass.

## Chunk 2: Module dispatch path

**Files:**
- Create: `src/triton_learn/modules/softmax.py`
- Modify: `src/triton_learn/modules/__init__.py`
- Modify: `src/triton_learn/__init__.py`
- Create: `tests/modules/test_softmax_module.py`

- [ ] Write failing tests for CPU auto fallback, explicit torch, explicit Triton failure, shared 2D validation, and lazy Triton import.
- [ ] Run `pytest tests/modules/test_softmax_module.py -q` and confirm failure because the module entrypoint is missing.
- [ ] Implement the minimal dispatch module and package exports.
- [ ] Re-run the module tests until they pass.

## Chunk 3: Triton kernel path

**Files:**
- Create: `src/triton_learn/kernels/softmax.py`
- Modify: `src/triton_learn/kernels/__init__.py`
- Create: `tests/kernels/test_softmax.py`

- [ ] Write a failing CUDA/Triton-gated correctness test using a non-power-of-two column count.
- [ ] Write failing CPU-side validation tests for 2D input, Triton availability, and CUDA device requirements.
- [ ] Run `pytest tests/kernels/test_softmax.py -q` and confirm the new failures occur because the kernel module does not exist yet.
- [ ] Implement the minimal stable row-wise Triton softmax wrapper using masked loads/stores and `triton.next_power_of_2`.
- [ ] Re-run the kernel tests until CPU validations pass and CUDA correctness passes or skips cleanly.

## Chunk 4: Benchmark and docs

**Files:**
- Create: `benchmarks/softmax_bench.py`
- Create: `docs/superpowers/specs/2026-04-16-phase2-softmax-bridge-design.md`
- Create: `docs/superpowers/plans/2026-04-16-phase2-softmax-bridge.md`

- [ ] Add the standalone benchmark script mirroring the vector-add benchmark style with CSV output and CPU fallback.
- [ ] Write the feature spec describing the approved scope and non-goals.
- [ ] Write the implementation plan describing the TDD execution order for the softmax bridge.

## Chunk 5: Final verification

**Files:**
- Verify only modified softmax-related files.

- [ ] Run `lsp_diagnostics` on all changed production files and resolve any real errors.
- [ ] Run `pytest tests/baseline/test_reference_ops.py tests/runtime/test_providers.py tests/modules/test_softmax_module.py tests/kernels/test_softmax.py -q`.
- [ ] Run `pytest -q` to confirm the worktree remains green, with CUDA/Triton tests skipped when unavailable.
- [ ] Run `python benchmarks/softmax_bench.py` and confirm it produces CSV-shaped output.
