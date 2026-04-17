import pytest

torch = pytest.importorskip("torch")


def _has_cuda() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for Triton kernel tests")
def test_triton_softmax_matches_torch_on_cuda_for_non_power_of_two_columns():
    pytest.importorskip("triton")
    from triton_learn.kernels.softmax import triton_softmax

    x = torch.randn(11, 37, device="cuda", dtype=torch.float32)

    out = triton_softmax(x)

    torch.testing.assert_close(out, torch.softmax(x, dim=-1), rtol=1e-4, atol=1e-4)


def test_triton_softmax_requires_2d_input():
    from triton_learn.kernels.softmax import triton_softmax

    x = torch.randn(2, 3, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"softmax expects a 2D tensor \[rows, cols\]"):
        triton_softmax(x)


def test_triton_softmax_requires_at_least_one_column():
    from triton_learn.kernels.softmax import triton_softmax

    x = torch.empty(3, 0, dtype=torch.float32)

    with pytest.raises(ValueError, match="softmax requires at least one column"):
        triton_softmax(x)


def test_triton_softmax_requires_float32_inputs():
    from triton_learn.kernels.softmax import triton_softmax

    x = torch.randn(2, 8, dtype=torch.float16)

    with pytest.raises(RuntimeError, match="Triton softmax currently supports float32 inputs only"):
        triton_softmax(x)


def test_triton_softmax_requires_triton_when_module_imported_without_it(monkeypatch):
    import importlib

    module = importlib.import_module("triton_learn.kernels.softmax")
    x = torch.randn(2, 8, dtype=torch.float32)

    monkeypatch.setattr(module, "triton", None)

    with pytest.raises(RuntimeError, match="Triton is not available"):
        module.triton_softmax(x)


def test_triton_softmax_requires_cuda_tensors_when_triton_is_available(monkeypatch):
    import importlib

    module = importlib.import_module("triton_learn.kernels.softmax")
    x = torch.randn(2, 8, dtype=torch.float32)

    monkeypatch.setattr(module, "triton", object())

    with pytest.raises(RuntimeError, match="Triton softmax requires CUDA tensors"):
        module.triton_softmax(x)
