import pytest


def test_choose_softmax_provider_returns_torch_when_requested():
    from triton_learn.runtime.providers import choose_softmax_provider

    assert choose_softmax_provider("torch", "cpu") == "torch"


def test_choose_softmax_provider_raises_for_explicit_triton_without_cuda_or_triton():
    from triton_learn.runtime.providers import choose_softmax_provider

    with pytest.raises(RuntimeError, match="Triton provider requires CUDA and Triton availability"):
        choose_softmax_provider("triton", "cpu")


def test_choose_softmax_provider_returns_torch_for_auto_on_cpu():
    from triton_learn.runtime.providers import choose_softmax_provider

    assert choose_softmax_provider("auto", "cpu") == "torch"


def test_choose_softmax_provider_returns_triton_for_auto_on_cuda_when_available(monkeypatch):
    from triton_learn.env import RuntimeCapabilities
    from triton_learn.runtime.providers import choose_softmax_provider

    monkeypatch.setattr(
        "triton_learn.runtime.providers.detect_runtime_capabilities",
        lambda: RuntimeCapabilities(
            has_torch=True,
            has_triton=True,
            has_cuda=True,
            default_device="cuda",
        ),
    )

    assert choose_softmax_provider("auto", "cuda") == "triton"
