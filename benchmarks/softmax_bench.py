from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _mean_ms(fn, iterations: int = 100) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000 / iterations


def main() -> None:
    try:
        import torch
    except Exception:
        print("torch is not installed; benchmark skipped")
        return

    from triton_learn.baseline.reference_ops import reference_rowwise_softmax_2d
    from triton_learn.env import detect_runtime_capabilities

    capabilities = detect_runtime_capabilities()
    shapes = [(128, 127), (256, 257), (512, 511), (1024, 1023)]

    if not capabilities.has_cuda:
        print("CUDA not available; running CPU reference timings only")
        device = "cpu"
        provider = "reference"
    else:
        device = "cuda"
        provider = "reference+triton"

    print(f"device={device} provider={provider}")
    print("rows,cols,torch_ms,triton_ms")

    triton_impl = None
    if capabilities.has_cuda and capabilities.has_triton:
        from triton_learn.kernels.softmax import triton_softmax

        triton_impl = triton_softmax

    for rows, cols in shapes:
        x = torch.randn(rows, cols, device=device, dtype=torch.float32)

        def run_reference(x=x):
            return reference_rowwise_softmax_2d(x)

        torch_ms = _mean_ms(run_reference)

        triton_ms = float("nan")
        if triton_impl is not None:
            torch.cuda.synchronize()

            def run_triton(x=x, triton_impl=triton_impl):
                return triton_impl(x)

            triton_ms = _mean_ms(run_triton)
            torch.cuda.synchronize()

        print(f"{rows},{cols},{torch_ms:.6f},{triton_ms:.6f}")


if __name__ == "__main__":
    main()
