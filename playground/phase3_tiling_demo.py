from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def main() -> None:
    from triton_learn.runtime.phase3_support import (
        describe_matmul_problem,
        phase3_autotune_configs,
        phase3_problem_cases,
    )

    print("Phase 3 tiling demo")
    print("=" * 19)

    configs = phase3_autotune_configs()
    base_config = configs[0]

    for case in phase3_problem_cases():
        summary = describe_matmul_problem(
            m=case.m,
            n=case.n,
            k=case.k,
            block_m=base_config.block_m,
            block_n=base_config.block_n,
            block_k=base_config.block_k,
        )
        print(f"case={case.name}")
        print(f"  m={case.m}, n={case.n}, k={case.k}")
        print(f"  output_shape={summary.output_shape}")
        print(f"  tile_grid={summary.tile_grid}")
        print(f"  k_tiles={summary.k_tiles}")
        print(f"  note={case.note}")
        print()

    print("Starter autotune configs")
    for index, config in enumerate(configs, start=1):
        print(
            f"  config_{index}: block_m={config.block_m}, block_n={config.block_n}, "
            f"block_k={config.block_k}, warps={config.num_warps}, stages={config.num_stages}"
        )


if __name__ == "__main__":
    main()
