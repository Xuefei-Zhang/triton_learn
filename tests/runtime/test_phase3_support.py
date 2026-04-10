import pytest


def test_describe_matmul_problem_reports_output_shape_and_tile_grid():
    from triton_learn.runtime.phase3_support import describe_matmul_problem

    summary = describe_matmul_problem(m=128, n=256, k=64, block_m=64, block_n=128, block_k=32)

    assert summary.output_shape == (128, 256)
    assert summary.tile_grid == (2, 2)
    assert summary.k_tiles == 2


def test_describe_matmul_problem_rejects_non_positive_values():
    from triton_learn.runtime.phase3_support import describe_matmul_problem

    with pytest.raises(ValueError):
        describe_matmul_problem(m=0, n=256, k=64, block_m=64, block_n=128, block_k=32)

    with pytest.raises(ValueError):
        describe_matmul_problem(m=128, n=256, k=64, block_m=0, block_n=128, block_k=32)


def test_phase3_autotune_configs_include_multiple_tradeoff_points():
    from triton_learn.runtime.phase3_support import phase3_autotune_configs

    configs = phase3_autotune_configs()

    assert len(configs) >= 3
    assert any(config.block_m == 64 for config in configs)
    assert any(config.num_warps >= 4 for config in configs)


def test_phase3_problem_cases_include_square_and_rectangular_examples():
    from triton_learn.runtime.phase3_support import phase3_problem_cases

    cases = phase3_problem_cases()

    assert len(cases) >= 3
    assert any(case.name == "square" for case in cases)
    assert any(case.m != case.n for case in cases)
