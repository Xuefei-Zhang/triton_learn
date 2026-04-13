from __future__ import annotations

# These helpers are not matmul kernels; they are teaching tools for reasoning about
# tiled matmul and autotune choices before writing a Triton kernel.
from dataclasses import dataclass


# Summarizes one matmul problem together with one candidate tile configuration.
@dataclass(frozen=True)
class MatmulProblemSummary:
    # Output height.
    m: int
    # Output width.
    n: int
    # Reduction dimension shared by the two inputs.
    k: int
    # Tile height processed by one program instance.
    block_m: int
    # Tile width processed by one program instance.
    block_n: int
    # Chunk size used while stepping through the K dimension.
    block_k: int
    # Shape of the final output matrix.
    output_shape: tuple[int, int]
    # Number of tiles needed across M and N.
    tile_grid: tuple[int, int]
    # Number of chunks needed to traverse the K dimension.
    k_tiles: int


# One possible autotune configuration.
@dataclass(frozen=True)
class Phase3AutotuneConfig:
    block_m: int
    block_n: int
    block_k: int
    num_warps: int
    num_stages: int
    note: str


# One named matrix-shape example for reasoning exercises.
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
    # Every dimension and tile size must be positive, or the problem definition is invalid.
    values = (m, n, k, block_m, block_n, block_k)
    if any(value <= 0 for value in values):
        raise ValueError("all dimensions and block sizes must be positive")

    # Ceiling division tells us how many tiles we need along M and N.
    tile_grid = (
        (m + block_m - 1) // block_m,
        (n + block_n - 1) // block_n,
    )

    # K is also processed in chunks, so we count how many such chunks are needed.
    k_tiles = (k + block_k - 1) // block_k

    # Return a summary object that the learner can inspect or print.
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
    # These are not magic values; they are starter points that help the learner think
    # about how tile shapes and warp counts might trade off against each other.
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
    # Provide a square case and two asymmetric cases so the learner can see that
    # "good tile shape" depends on the output geometry.
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
