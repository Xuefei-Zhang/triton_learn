import pytest


def test_summarize_block_mapping_for_exact_division():
    from triton_learn.runtime.phase1_support import summarize_block_mapping

    summary = summarize_block_mapping(num_elements=1024, block_size=256)

    assert summary.num_programs == 4
    assert summary.tail_elements == 0
    assert summary.covers_all_elements is True


def test_summarize_block_mapping_for_partial_tail():
    from triton_learn.runtime.phase1_support import summarize_block_mapping

    summary = summarize_block_mapping(num_elements=1000, block_size=256)

    assert summary.num_programs == 4
    assert summary.tail_elements == 232
    assert summary.last_program_start == 768


def test_summarize_block_mapping_rejects_non_positive_values():
    from triton_learn.runtime.phase1_support import summarize_block_mapping

    with pytest.raises(ValueError):
        summarize_block_mapping(num_elements=0, block_size=256)

    with pytest.raises(ValueError):
        summarize_block_mapping(num_elements=128, block_size=0)


def test_phase1_exercise_cases_include_tail_and_multidimensional_examples():
    from triton_learn.runtime.phase1_support import phase1_exercise_cases

    cases = phase1_exercise_cases()

    assert len(cases) >= 3
    assert any(case.name == "tail-case" for case in cases)
    assert any(len(case.shape) == 2 for case in cases)
