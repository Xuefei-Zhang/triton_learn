import pytest

torch = pytest.importorskip("torch")


def _has_cuda() -> bool:
    return torch.cuda.is_available()


@pytest.mark.skipif(not _has_cuda(), reason="CUDA is required for Triton kernel tests")
def test_triton_vector_add_matches_torch_on_cuda():
    pytest.importorskip("triton")
    from triton_learn.kernels.vector_add import triton_vector_add

    a = torch.randn(1024, device="cuda", dtype=torch.float32)
    b = torch.randn(1024, device="cuda", dtype=torch.float32)

    out = triton_vector_add(a, b)

    torch.testing.assert_close(out, a + b, rtol=1e-4, atol=1e-4)
