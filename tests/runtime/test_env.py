def test_runtime_capability_detection_exposes_expected_fields():
    from triton_learn.env import detect_runtime_capabilities

    capabilities = detect_runtime_capabilities()

    assert hasattr(capabilities, "has_torch")
    assert hasattr(capabilities, "has_triton")
    assert hasattr(capabilities, "has_cuda")
    assert hasattr(capabilities, "default_device")
