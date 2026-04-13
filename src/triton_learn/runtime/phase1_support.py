from __future__ import annotations

# Dataclasses are a convenient way to bundle several related teaching values together.
from dataclasses import dataclass


# This object summarizes how a 1D elementwise kernel launch covers a tensor.
@dataclass(frozen=True)
class BlockMappingSummary:
    # Total scalar elements that need processing.
    num_elements: int
    # Number of elements one program instance is asked to handle.
    block_size: int
    # Number of program instances needed to cover the whole input.
    num_programs: int
    # Global offset where the final program instance begins.
    last_program_start: int
    # How many valid elements are actually present in the last partial block.
    tail_elements: int
    # Whether the computed launch covers at least the real data range.
    covers_all_elements: bool


# Each exercise case gives the learner one concrete shape/block configuration to think about.
@dataclass(frozen=True)
class Phase1ExerciseCase:
    name: str
    shape: tuple[int, ...]
    block_size: int
    note: str


def summarize_block_mapping(num_elements: int, block_size: int) -> BlockMappingSummary:
    # These helpers are educational, so invalid inputs should fail explicitly.
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    # Ceiling division computes how many blocks/programs we need to cover all elements.
    num_programs = (num_elements + block_size - 1) // block_size

    # The final program starts at the beginning of the last logical block.
    last_program_start = (num_programs - 1) * block_size

    # `remaining` tells us how many real elements are left in the final block.
    remaining = num_elements - last_program_start

    # If the final block is partial, that partial size becomes `tail_elements`.
    # If the final block is exactly full, there is no tail.
    tail_elements = remaining if remaining < block_size else 0

    # Package the computed teaching values into one immutable summary object.
    return BlockMappingSummary(
        num_elements=num_elements,
        block_size=block_size,
        num_programs=num_programs,
        last_program_start=last_program_start,
        tail_elements=tail_elements,
        covers_all_elements=last_program_start < num_elements,
    )


def phase1_exercise_cases() -> list[Phase1ExerciseCase]:
    # Return a few representative cases:
    # - exact division
    # - a tail/mask case
    # - a 2D tensor that still becomes a 1D launch in this starter repo
    return [
        Phase1ExerciseCase(
            name="exact-division",
            shape=(1024,),
            block_size=256,
            note="Every program owns a full block.",
        ),
        Phase1ExerciseCase(
            name="tail-case",
            shape=(1000,),
            block_size=256,
            note="The final program needs a mask for the tail.",
        ),
        Phase1ExerciseCase(
            name="matrix-flattened",
            shape=(17, 33),
            block_size=128,
            note="A 2D tensor is flattened before the kernel launch in the starter path.",
        ),
    ]
