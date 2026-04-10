import pytest

torch = pytest.importorskip("torch")


def test_toy_transformer_block_preserves_batch_and_sequence_dimensions():
    from triton_learn.integration.toy_transformer_block import ToyTransformerBlock

    block = ToyTransformerBlock(embed_dim=16)
    x = torch.randn(2, 5, 16, dtype=torch.float32)

    out = block(x)

    assert out.shape == x.shape
