from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MatmulProblemSummary:
    m: int
    n: int
    k: int
    block_m: int
    block_n: int
    block_k: int
    output_shape: tuple[int, int]
    tile_grid: tuple[int, int]
    k_tiles: int


@dataclass(frozen=True)
class Phase3AutotuneConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    note: str


@dataclass(frozen=True)
class Phase3ProblemCase:
    name: str
    m: int
    n: int
    k: int
    note: str


def describe_matmul_problem(
    m: int,
    n: int,
    k: int,
    block_m: int,
    block_n: int,
    block_k: int,
) -> MatmulProblemSummary:
    values = (m, n, k, block_m, block_n, block_k)
    if any(value <= 0 for value in values):
        raise ValueError("all dimensions and block sizes must be positive")

    tile_grid = (
        (m + block_m - 1) // block_m,
        (n + block_n - 1) // block_n,
    )
    k_tiles = (k + block_k - 1) // block_k

    return MatmulProblemSummary(
        m=m,
        n=n,
        k=k,
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        output_shape=(m, n),
        tile_grid=tile_grid,
        k_tiles=k_tiles,
    )


def phase3_autotune_configs() -> list[Phase3AutotuneConfig]:
    return [
        Phase3AutotuneConfig(
            block_m=64,
            block_n=64,
            block_k=32,
            num_warps=4,
            num_stages=2,
            note="Balanced starter config for medium square-ish problems.",
        ),
        Phase3AutotuneConfig(
            block_m=128,
            block_n=64,
            block_k=32,
            num_warps=4,
            num_stages=3,
            note="Bigger M tile to favor taller output regions.",
        ),
        Phase3AutotuneConfig(
            block_m=64,
            block_n=128,
            block_k=32,
            num_warps=8,
            num_stages=3,
            note="Bigger N tile to favor wider output regions.",
        ),
    ]


def phase3_problem_cases() -> list[Phase3ProblemCase]:
    return [
        Phase3ProblemCase(
            name="square",
            m=128,
            n=128,
            k=128,
            note="The cleanest starting point for understanding tiled matmul.",
        ),
        Phase3ProblemCase(
            name="tall-output",
            m=256,
            n=64,
            k=128,
            note="Useful for thinking about asymmetric tile choices.",
        ),
        Phase3ProblemCase(
            name="wide-output",
            m=64,
            n=256,
            k=128,
            note="Helps build intuition for N-heavy output grids.",
        ),
    ]
