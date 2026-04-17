# Phase 2 Softmax Bridge Design

## Goal

Add one runnable row-wise 2D softmax path that extends the repository's existing vector-add layering into Phase 2 without broadening scope beyond last-dimension softmax over `[rows, cols]` tensors.

## Approved Scope

- Add `reference_rowwise_softmax_2d(x)` as the validated baseline wrapper for 2D inputs.
- Add `choose_softmax_provider(requested, device_type)` with the same provider policy shape as `choose_vector_add_provider`.
- Add `triton_softmax(x)` as an optional Triton kernel wrapper for CUDA 2D float32 tensors.
- Add `softmax(x, provider="auto")` as the module-level dispatch API.
- Export the new softmax APIs from the relevant package `__init__.py` files.
- Add baseline, runtime, module, and kernel tests, with CUDA/Triton tests skipped when unavailable.
- Add a benchmark script in `benchmarks/softmax_bench.py` with CSV output and CPU fallback.

## Non-Goals

- No higher-rank softmax support.
- No reshaping tricks to simulate unsupported dimensions.
- No empty-column inputs.
- No profiling work in this phase.
- No provider expansion beyond `auto`, `torch`, and `triton`.

## Design

The bridge mirrors the current vector-add architecture so learners can compare two complete vertical slices. The baseline layer owns user-facing correctness validation for the simple PyTorch path. The runtime layer owns provider resolution only. The module layer validates the public API, resolves the provider, and lazy-imports the Triton kernel so CPU-only environments stay usable. The kernel layer owns Triton-specific validation and the row-wise implementation details. To keep the first executable Phase 2 step narrow and predictable, the Triton path is explicitly limited to float32 inputs and rejects empty-column tensors.

For the Triton kernel, each program instance handles one row. The wrapper computes `BLOCK_SIZE` with `triton.next_power_of_2(cols)` so non-power-of-two widths are still covered by a single masked row load/store pattern. The kernel uses masked loads with `other=-inf`, subtracts the row max for numerical stability, exponentiates the shifted values, sums across the row, and stores normalized probabilities back with the same mask.

## Testing Strategy

- Baseline tests verify the new reference wrapper matches `torch.softmax(..., dim=-1)` and rejects non-2D inputs with the shared error message.
- Runtime tests verify explicit torch selection, explicit Triton failure when unavailable, CPU auto fallback, and CUDA/Triton auto routing when capabilities are present.
- Module tests verify baseline dispatch, explicit provider behavior, shared validation, and lazy Triton kernel import behavior.
- Kernel tests verify CPU-side validation and CUDA correctness for a non-power-of-two column count when CUDA/Triton are available.

## Success Criteria

- The softmax path is runnable from the module layer and mirrors the vector-add layering.
- CPU-only environments pass all relevant non-CUDA tests and skip the CUDA/Triton correctness test cleanly.
- The benchmark script produces stable CSV-shaped output on both CPU fallback and CUDA-capable environments.
