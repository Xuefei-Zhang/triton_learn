def test_describe_attention_flow_returns_expected_stage_names():
    from triton_learn.runtime.phase4_support import describe_attention_flow

    flow = describe_attention_flow()

    assert flow[0].name == "qkv-projection"
    assert flow[-1].name == "output-projection"
    assert any(step.optimization_candidate for step in flow)


def test_phase4_optimization_targets_include_attention_and_norm():
    from triton_learn.runtime.phase4_support import phase4_optimization_targets

    targets = phase4_optimization_targets()

    names = {target.name for target in targets}
    assert "softmax-path" in names
    assert "norm-path" in names
    assert any(target.expected_benefit == "memory-traffic" for target in targets)


def test_phase4_composition_questions_are_non_empty_and_actionable():
    from triton_learn.runtime.phase4_support import phase4_composition_questions

    questions = phase4_composition_questions()

    assert len(questions) >= 3
    assert all(question.endswith("?") for question in questions)
