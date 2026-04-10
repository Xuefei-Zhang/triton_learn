import pytest


torch = pytest.importorskip("torch")


def test_auto_provider_uses_reference_path_on_cpu():
    from triton_learn.modules.vector_add import vector_add

    a = torch.randn(32, dtype=torch.float32)
    b = torch.randn(32, dtype=torch.float32)

    out = vector_add(a, b, provider="auto")

    torch.testing.assert_close(out, a + b)


def test_torch_provider_matches_torch_add():
    from triton_learn.modules.vector_add import vector_add

    a = torch.randn(32, dtype=torch.float32)
    b = torch.randn(32, dtype=torch.float32)

    out = vector_add(a, b, provider="torch")

    torch.testing.assert_close(out, a + b)


def test_explicit_triton_provider_requires_cuda_and_triton():
    from triton_learn.modules.vector_add import vector_add

    a = torch.randn(8)
    b = torch.randn(8)

    with pytest.raises(RuntimeError):
        vector_add(a, b, provider="triton")
