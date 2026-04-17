import importlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")


def test_softmax_auto_provider_uses_reference_path_on_cpu():
    from triton_learn.modules.softmax import softmax

    x = torch.randn(4, 7, dtype=torch.float32)

    out = softmax(x, provider="auto")

    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_softmax_torch_provider_matches_torch_softmax():
    from triton_learn.modules.softmax import softmax

    x = torch.randn(3, 5, dtype=torch.float32)

    out = softmax(x, provider="torch")

    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_softmax_requires_2d_input():
    from triton_learn.modules.softmax import softmax

    x = torch.randn(2, 3, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"softmax expects a 2D tensor \[rows, cols\]"):
        softmax(x)


def test_softmax_requires_at_least_one_column():
    from triton_learn.modules.softmax import softmax

    x = torch.empty(3, 0, dtype=torch.float32)

    with pytest.raises(ValueError, match="softmax requires at least one column"):
        softmax(x)


def test_softmax_triton_provider_requires_cuda_and_triton():
    from triton_learn.modules.softmax import softmax

    x = torch.randn(2, 8, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="Triton provider requires CUDA and Triton availability"):
        softmax(x, provider="triton")


def test_softmax_triton_path_lazy_imports_kernel(monkeypatch):
    module = importlib.import_module("triton_learn.modules.softmax")

    x = torch.randn(2, 3, dtype=torch.float32)
    sentinel = torch.full_like(x, 0.5)

    fake_module = types.ModuleType("triton_learn.kernels.softmax")

    def fake_triton_softmax(tensor):
        assert tensor is x
        return sentinel

    fake_module.triton_softmax = fake_triton_softmax
    monkeypatch.setitem(sys.modules, "triton_learn.kernels.softmax", fake_module)
    monkeypatch.setattr(module, "choose_softmax_provider", lambda provider, device_type: "triton")

    out = module.softmax(x, provider="auto")

    assert out is sentinel


def test_softmax_auto_provider_uses_triton_path_when_runtime_selects_it(monkeypatch):
    module = importlib.import_module("triton_learn.modules.softmax")

    x = torch.randn(2, 4, dtype=torch.float32)
    sentinel = torch.full_like(x, 0.25)

    fake_module = types.ModuleType("triton_learn.kernels.softmax")
    fake_module.triton_softmax = lambda tensor: sentinel
    monkeypatch.setitem(sys.modules, "triton_learn.kernels.softmax", fake_module)
    monkeypatch.setattr(module, "choose_softmax_provider", lambda provider, device_type: "triton")

    out = module.softmax(x, provider="auto")

    assert out is sentinel
