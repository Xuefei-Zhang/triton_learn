from __future__ import annotations

# These top-level imports are all standard-library helpers used by this standalone script.
import sys

# `time.perf_counter()` gives us a high-resolution wall-clock timer.
import time

# `Path` helps us compute the repository root relative to this file.
from pathlib import Path

# `ROOT` is the repository directory one level above `benchmarks/`.
ROOT = Path(__file__).resolve().parents[1]
# `SRC` is the source-package directory we want Python to import from.
SRC = ROOT / "src"
# Because this benchmark is run as a standalone script, we manually insert `src/`
# into `sys.path` so `import triton_learn...` works without extra shell setup.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _mean_ms(fn, iterations: int = 100) -> float:
    # Capture the starting timestamp before repeated execution.
    start = time.perf_counter()
    # Run the callable multiple times to smooth out one-off noise.
    for _ in range(iterations):
        fn()
    # Capture the ending timestamp after the repeated loop.
    end = time.perf_counter()

    # Convert average seconds-per-run into milliseconds-per-run.
    return (end - start) * 1000 / iterations


def main() -> None:
    # Try to import torch lazily so this script can fail gracefully on environments
    # that do not yet have runtime dependencies installed.
    try:
        import torch
    except Exception:
        print("torch is not installed; benchmark skipped")
        return

    # Reference implementation used as the baseline timing target.
    from triton_learn.baseline.reference_ops import reference_vector_add

    # Runtime capability detection tells us whether CUDA/Triton are usable.
    from triton_learn.env import detect_runtime_capabilities

    # Snapshot the environment once at the start of the script.
    capabilities = detect_runtime_capabilities()
    # A few simple tensor sizes to benchmark.
    sizes = [1024, 4096, 16384, 65536]

    # If CUDA is not available, we still run the benchmark on CPU for the reference path.
    if not capabilities.has_cuda:
        print("CUDA not available; running CPU reference timings only")
        device = "cpu"
        provider = "reference"
    else:
        # If CUDA works, we benchmark both the reference and Triton paths on GPU tensors.
        device = "cuda"
        provider = "reference+triton"

    # Print a tiny header so the output is easy to interpret or paste into notes.
    print(f"device={device} provider={provider}")
    print("size,torch_ms,triton_ms")

    # Start with no Triton implementation, then enable it only if the runtime supports it.
    triton_impl = None
    if capabilities.has_cuda and capabilities.has_triton:
        from triton_learn.kernels.vector_add import triton_vector_add

        triton_impl = triton_vector_add

    # Benchmark each tensor size separately.
    for size in sizes:
        # Create two random input vectors on the selected device.
        a = torch.randn(size, device=device, dtype=torch.float32)
        b = torch.randn(size, device=device, dtype=torch.float32)

        # Bind the current loop tensors into a small callable so `_mean_ms` can repeatedly call it.
        def run_reference(a=a, b=b):
            return reference_vector_add(a, b)

        # Measure the reference implementation first.
        torch_ms = _mean_ms(run_reference)

        # Use `nan` when Triton is unavailable so the output table still has a stable shape.
        triton_ms = float("nan")
        if triton_impl is not None:
            # Synchronize before timing so previous GPU work does not leak into this measurement.
            torch.cuda.synchronize()

            # Same idea as `run_reference`, but for the Triton implementation.
            def run_triton(a=a, b=b, triton_impl=triton_impl):
                return triton_impl(a, b)

            # Measure the Triton implementation.
            triton_ms = _mean_ms(run_triton)

            # Synchronize again so the measured kernel work is actually complete.
            torch.cuda.synchronize()

        # Print one CSV-style line per tensor size.
        print(f"{size},{torch_ms:.6f},{triton_ms:.6f}")


if __name__ == "__main__":
    # Standard Python script entrypoint.
    main()
