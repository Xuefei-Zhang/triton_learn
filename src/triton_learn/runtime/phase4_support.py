from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttentionFlowStep:
    name: str
    description: str
    optimization_candidate: bool


@dataclass(frozen=True)
class OptimizationTarget:
    name: str
    expected_benefit: str
    reason: str


def describe_attention_flow() -> list[AttentionFlowStep]:
    return [
        AttentionFlowStep(
            name="qkv-projection",
            description="Project the input into query, key, and value tensors.",
            optimization_candidate=False,
        ),
        AttentionFlowStep(
            name="score-computation",
            description="Compute attention scores from query and key.",
            optimization_candidate=True,
        ),
        AttentionFlowStep(
            name="softmax-path",
            description="Normalize scores into probabilities.",
            optimization_candidate=True,
        ),
        AttentionFlowStep(
            name="value-mixing",
            description="Apply the probabilities to the value tensor.",
            optimization_candidate=True,
        ),
        AttentionFlowStep(
            name="output-projection",
            description="Project the mixed result back to the model space.",
            optimization_candidate=False,
        ),
    ]


def phase4_optimization_targets() -> list[OptimizationTarget]:
    return [
        OptimizationTarget(
            name="softmax-path",
            expected_benefit="memory-traffic",
            reason="Softmax often benefits from fusion and fewer global-memory round trips.",
        ),
        OptimizationTarget(
            name="score-computation",
            expected_benefit="matmul-tiling",
            reason="Score computation is closely tied to matmul-like blocking and reuse.",
        ),
        OptimizationTarget(
            name="norm-path",
            expected_benefit="reduction-fusion",
            reason="Normalization combines reduction and elementwise work in one local region.",
        ),
    ]


def phase4_composition_questions() -> list[str]:
    return [
        "Which part of the block is currently the clearest optimization target?",
        "Which intermediate tensors are created only to be consumed immediately?",
        "What part of the flow should stay reference-only until correctness is trusted?",
        "If you optimize one subpath, how will you compare it against the baseline block?",
    ]
