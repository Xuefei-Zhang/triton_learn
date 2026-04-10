from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class StableSoftmaxSummary:
    values: list[float]
    row_max: float
    shifted: list[float]
    exponentials: list[float]
    denominator: float
    probabilities: list[float]


@dataclass(frozen=True)
class Phase2ExerciseRow:
    name: str
    values: list[float]
    note: str


def stable_softmax_steps(values: list[float]) -> StableSoftmaxSummary:
    if not values:
        raise ValueError("values must not be empty")

    row_max = max(values)
    shifted = [value - row_max for value in values]
    exponentials = [math.exp(value) for value in shifted]
    denominator = sum(exponentials)
    probabilities = [value / denominator for value in exponentials]

    return StableSoftmaxSummary(
        values=list(values),
        row_max=row_max,
        shifted=shifted,
        exponentials=exponentials,
        denominator=denominator,
        probabilities=probabilities,
    )


def phase2_exercise_rows() -> list[Phase2ExerciseRow]:
    return [
        Phase2ExerciseRow(
            name="balanced",
            values=[1.0, 2.0, 3.0],
            note="A small row where every stable-softmax step can be checked by hand.",
        ),
        Phase2ExerciseRow(
            name="large-magnitude",
            values=[1000.0, 1001.0, 1002.0],
            note="Use this to see why subtracting the row max matters.",
        ),
        Phase2ExerciseRow(
            name="mixed-sign",
            values=[-3.0, 0.0, 5.0],
            note="A row with both negative and positive values.",
        ),
    ]
