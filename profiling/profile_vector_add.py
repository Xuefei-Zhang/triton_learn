from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    try:
        import torch
    except Exception:
        print("torch is not installed; profiling helper skipped")
        return

    from triton_learn.baseline.reference_ops import reference_vector_add
    from triton_learn.env import detect_runtime_capabilities

    capabilities = detect_runtime_capabilities()
    device = capabilities.default_device
    a = torch.randn(4096, device=device, dtype=torch.float32)
    b = torch.randn(4096, device=device, dtype=torch.float32)

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        reference_vector_add(a.cpu(), b.cpu())

    print("Vector add profiling helper")
    print(f"default_device={capabilities.default_device}")
    print(f"has_triton={capabilities.has_triton}")
    print("Top CPU profiler events:")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))


if __name__ == "__main__":
    main()
