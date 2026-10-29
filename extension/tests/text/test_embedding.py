import numpy as np
import pytest
import torch

from causal_policy.text import embedding


def test_last_token_pool_left_padding():
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    assert torch.equal(embedding.last_token_pool(hidden, mask), hidden[:, -1])


def test_last_token_pool_right_padding():
    hidden = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    assert torch.equal(embedding.last_token_pool(hidden, mask), torch.stack([hidden[0, 1], hidden[1, 2]]))


def test_truncate_normalize_keeps_leading_coordinates():
    pooled = torch.tensor([[3.0, 4.0, 100.0], [0.0, 2.0, -7.0]])
    emb = embedding.truncate_normalize(pooled, 2)
    assert emb.dtype == np.float64
    assert np.allclose(emb, [[0.6, 0.8], [0.0, 1.0]])


def test_check_embeddings_rejects_bad_rows():
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    embedding.check_embeddings([1, 2], [1, 2], emb, 2)
    with pytest.raises(AssertionError):
        embedding.check_embeddings([1, 2], [1, 2], np.array([[1.0, 0.0], [1.0, 0.0]]), 2)
    with pytest.raises(AssertionError):
        embedding.check_embeddings([1, 2], [1, 2], np.array([[2.0, 0.0], [0.0, 1.0]]), 2)
