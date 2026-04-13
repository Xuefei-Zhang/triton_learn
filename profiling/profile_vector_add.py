from __future__ import annotations

# This script is a tiny profiling helper, so its imports are all standard-library only.
import sys
from pathlib import Path

# Compute the repository root and source directory from the current file location.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
# Insert `src/` into `sys.path` so the package can be imported when this script is run directly.
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    # Import torch lazily so the script can exit cleanly on incomplete environments.
    try:
        import torch
    except Exception:
        print("torch is not installed; profiling helper skipped")
        return

    # Import the simple baseline op we want to profile.
    from triton_learn.baseline.reference_ops import reference_vector_add

    # Import runtime detection so the profiling output includes useful context.
    from triton_learn.env import detect_runtime_capabilities

    # Snapshot capability information once at the start.
    capabilities = detect_runtime_capabilities()
    # Use the runtime's default device string when constructing example tensors.
    device = capabilities.default_device

    # Create two example input vectors.
    a = torch.randn(4096, device=device, dtype=torch.float32)
    b = torch.randn(4096, device=device, dtype=torch.float32)

    # Open a CPU profiler context and run the reference operation inside it.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        # We move to CPU here because this helper is intentionally minimal and meant to
        # demonstrate profiler usage rather than deep GPU profiling methodology.
        reference_vector_add(a.cpu(), b.cpu())

    # Print a small human-readable summary.
    print("Vector add profiling helper")
    print(f"default_device={capabilities.default_device}")
    print(f"has_triton={capabilities.has_triton}")
    print("Top CPU profiler events:")
    # `key_averages().table(...)` renders the most expensive profiler events as text.
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=5))


if __name__ == "__main__":
    # Standard Python script entrypoint.
    main()
