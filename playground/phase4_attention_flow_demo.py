from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from triton_learn.runtime.phase4_support import (
    describe_attention_flow,
    phase4_composition_questions,
    phase4_optimization_targets,
)


def main() -> None:
    print("Phase 4 attention flow demo")
    print("=" * 27)

    print("Flow steps")
    for step in describe_attention_flow():
        print(f"  - {step.name}: {step.description} (candidate={step.optimization_candidate})")

    print()
    print("Optimization targets")
    for target in phase4_optimization_targets():
        print(f"  - {target.name}: benefit={target.expected_benefit} | {target.reason}")

    print()
    print("Composition questions")
    for question in phase4_composition_questions():
        print(f"  - {question}")


if __name__ == "__main__":
    main()
