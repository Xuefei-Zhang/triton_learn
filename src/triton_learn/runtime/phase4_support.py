from __future__ import annotations

# Phase 4 is about system composition and choosing worthwhile optimization targets,
# so these helpers describe flow and priorities rather than running Triton kernels.
from dataclasses import dataclass


# One step in a simplified attention-style dataflow.
@dataclass(frozen=True)
class AttentionFlowStep:
    name: str
    description: str
    # Whether this step looks like a promising optimization candidate.
    optimization_candidate: bool


# One possible subpath that a learner might choose to optimize next.
@dataclass(frozen=True)
class OptimizationTarget:
    name: str
    # A short label for the kind of benefit we expect.
    expected_benefit: str
    # Human-readable explanation of why this target is interesting.
    reason: str


def describe_attention_flow() -> list[AttentionFlowStep]:
    # Return a simplified end-to-end attention flow in execution order.
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
    # These targets are framed in beginner-friendly language so the learner can justify
    # optimization choices instead of saying "optimize everything".
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
    # These questions are intentionally phrased as prompts for design thinking.
    return [
        "Which part of the block is currently the clearest optimization target?",
        "Which intermediate tensors are created only to be consumed immediately?",
        "What part of the flow should stay reference-only until correctness is trusted?",
        "If you optimize one subpath, how will you compare it against the baseline block?",
    ]
