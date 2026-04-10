from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triton_learn.runtime.phase2_support import phase2_exercise_rows, stable_softmax_steps


def main() -> None:
    print("Phase 2 softmax steps demo")
    print("=" * 27)

    for row in phase2_exercise_rows():
        summary = stable_softmax_steps(row.values)
        print(f"row={row.name}")
        print(f"  values={summary.values}")
        print(f"  row_max={summary.row_max}")
        print(f"  shifted={summary.shifted}")
        print(f"  exponentials={summary.exponentials}")
        print(f"  denominator={summary.denominator}")
        print(f"  probabilities={summary.probabilities}")
        print(f"  note={row.note}")
        print()


if __name__ == "__main__":
    main()
