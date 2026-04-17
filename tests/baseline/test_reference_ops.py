import pytest

torch = pytest.importorskip("torch")


def test_reference_vector_add_matches_torch_add():
    from triton_learn.baseline.reference_ops import reference_vector_add

    a = torch.randn(16, dtype=torch.float32)
    b = torch.randn(16, dtype=torch.float32)

    out = reference_vector_add(a, b)

    torch.testing.assert_close(out, a + b)


def test_reference_softmax_matches_torch_softmax():
    from triton_learn.baseline.reference_ops import reference_softmax

    x = torch.randn(8, 16, dtype=torch.float32)

    out = reference_softmax(x, dim=-1)

    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_reference_rowwise_softmax_2d_matches_torch_softmax_last_dim():
    from triton_learn.baseline.reference_ops import reference_rowwise_softmax_2d

    x = torch.randn(8, 16, dtype=torch.float32)

    out = reference_rowwise_softmax_2d(x)

    torch.testing.assert_close(out, torch.softmax(x, dim=-1))


def test_reference_rowwise_softmax_2d_requires_2d_input():
    from triton_learn.baseline.reference_ops import reference_rowwise_softmax_2d

    x = torch.randn(2, 3, 4, dtype=torch.float32)

    with pytest.raises(ValueError, match=r"softmax expects a 2D tensor \[rows, cols\]"):
        reference_rowwise_softmax_2d(x)


def test_reference_rowwise_softmax_2d_requires_at_least_one_column():
    from triton_learn.baseline.reference_ops import reference_rowwise_softmax_2d

    x = torch.empty(3, 0, dtype=torch.float32)

    with pytest.raises(ValueError, match="softmax requires at least one column"):
        reference_rowwise_softmax_2d(x)


def test_reference_layer_norm_preserves_shape():
    from triton_learn.baseline.reference_ops import reference_layer_norm

    x = torch.randn(4, 8, 16, dtype=torch.float32)

    out = reference_layer_norm(x, normalized_shape=16)

    assert out.shape == x.shape


def test_reference_linear_matches_manual_formula():
    from triton_learn.baseline.reference_ops import reference_linear

    x = torch.randn(2, 4, dtype=torch.float32)
    weight = torch.randn(3, 4, dtype=torch.float32)
    bias = torch.randn(3, dtype=torch.float32)

    out = reference_linear(x, weight, bias)

    torch.testing.assert_close(out, x @ weight.T + bias)


def test_reference_attention_returns_expected_shape():
    from triton_learn.baseline.reference_ops import reference_attention

    q = torch.randn(2, 4, 8, dtype=torch.float32)
    k = torch.randn(2, 4, 8, dtype=torch.float32)
    v = torch.randn(2, 4, 8, dtype=torch.float32)

    out = reference_attention(q, k, v)

    assert out.shape == q.shape


def test_reference_vector_add_rejects_shape_mismatch():
    from triton_learn.baseline.reference_ops import reference_vector_add

    a = torch.randn(4)
    b = torch.randn(5)

    with pytest.raises(ValueError):
        reference_vector_add(a, b)
