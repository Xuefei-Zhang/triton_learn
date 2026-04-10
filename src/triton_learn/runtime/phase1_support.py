from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BlockMappingSummary:
    num_elements: int
    block_size: int
    num_programs: int
    last_program_start: int
    tail_elements: int
    covers_all_elements: bool


@dataclass(frozen=True)
class Phase1ExerciseCase:
    name: str
    shape: tuple[int, ...]
    block_size: int
    note: str


def summarize_block_mapping(num_elements: int, block_size: int) -> BlockMappingSummary:
    if num_elements <= 0:
        raise ValueError("num_elements must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    num_programs = (num_elements + block_size - 1) // block_size
    last_program_start = (num_programs - 1) * block_size
    remaining = num_elements - last_program_start
    tail_elements = remaining if remaining < block_size else 0

    return BlockMappingSummary(
        num_elements=num_elements,
        block_size=block_size,
        num_programs=num_programs,
        last_program_start=last_program_start,
        tail_elements=tail_elements,
        covers_all_elements=last_program_start < num_elements,
    )


def phase1_exercise_cases() -> list[Phase1ExerciseCase]:
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
