from __future__ import annotations

# `math.exp` is used here so the learner can see stable softmax one scalar step at a time.
import math
from dataclasses import dataclass


# This summary object keeps every intermediate quantity of stable softmax visible.
@dataclass(frozen=True)
class StableSoftmaxSummary:
    # Original row values.
    values: list[float]
    # Maximum value in the row, used for numerical stabilization.
    row_max: float
    # Values after subtracting the row maximum.
    shifted: list[float]
    # Exponentials of the shifted values.
    exponentials: list[float]
    # Sum of exponentials, used to normalize into probabilities.
    denominator: float
    # Final softmax probabilities.
    probabilities: list[float]


# A small named example row used in the learning exercises.
@dataclass(frozen=True)
class Phase2ExerciseRow:
    name: str
    values: list[float]
    note: str


def stable_softmax_steps(values: list[float]) -> StableSoftmaxSummary:
    # Stable softmax on an empty row does not make sense.
    if not values:
        raise ValueError("values must not be empty")

    # First find the row maximum.
    row_max = max(values)

    # Then shift every value down by that maximum.
    # This is the classic trick that keeps large positive inputs from exploding in `exp`.
    shifted = [value - row_max for value in values]

    # Now compute exponentials in the safer shifted space.
    exponentials = [math.exp(value) for value in shifted]

    # Sum the exponentials to get the softmax denominator.
    denominator = sum(exponentials)

    # Divide each exponential by the denominator to get probabilities.
    probabilities = [value / denominator for value in exponentials]

    # Return all intermediate values so the learner can inspect each step.
    return StableSoftmaxSummary(
        values=list(values),
        row_max=row_max,
        shifted=shifted,
        exponentials=exponentials,
        denominator=denominator,
        probabilities=probabilities,
    )


def phase2_exercise_rows() -> list[Phase2ExerciseRow]:
    # These rows are deliberately chosen to show:
    # - a small hand-computable example
    # - a large-magnitude numerical-stability example
    # - a mixed-sign example
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
