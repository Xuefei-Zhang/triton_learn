import math

import pytest


def test_stable_softmax_steps_reports_row_max_and_shifted_values():
    from triton_learn.runtime.phase2_support import stable_softmax_steps

    summary = stable_softmax_steps([1.0, 2.0, 3.0])

    assert summary.row_max == 3.0
    assert summary.shifted == [-2.0, -1.0, 0.0]
    assert len(summary.probabilities) == 3
    assert pytest.approx(sum(summary.probabilities), rel=1e-6, abs=1e-6) == 1.0


def test_stable_softmax_steps_handles_large_values_without_overflowing():
    from triton_learn.runtime.phase2_support import stable_softmax_steps

    summary = stable_softmax_steps([1000.0, 1001.0, 1002.0])

    assert math.isfinite(summary.denominator)
    assert all(math.isfinite(value) for value in summary.exponentials)
    assert pytest.approx(sum(summary.probabilities), rel=1e-6, abs=1e-6) == 1.0


def test_stable_softmax_steps_rejects_empty_rows():
    from triton_learn.runtime.phase2_support import stable_softmax_steps

    with pytest.raises(ValueError):
        stable_softmax_steps([])


def test_phase2_exercise_rows_include_balanced_and_large_magnitude_examples():
    from triton_learn.runtime.phase2_support import phase2_exercise_rows

    rows = phase2_exercise_rows()

    assert len(rows) >= 3
    assert any(row.name == "large-magnitude" for row in rows)
    assert any(row.name == "mixed-sign" for row in rows)
