from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triton_learn.runtime.phase1_support import phase1_exercise_cases, summarize_block_mapping


def main() -> None:
    print("Phase 1 block mapping demo")
    print("=" * 28)

    for case in phase1_exercise_cases():
        num_elements = 1
        for dim in case.shape:
            num_elements *= dim
        summary = summarize_block_mapping(num_elements=num_elements, block_size=case.block_size)
        print(f"case={case.name}")
        print(f"  shape={case.shape}")
        print(f"  block_size={case.block_size}")
        print(f"  num_programs={summary.num_programs}")
        print(f"  last_program_start={summary.last_program_start}")
        print(f"  tail_elements={summary.tail_elements}")
        print(f"  note={case.note}")
        print()


if __name__ == "__main__":
    main()
