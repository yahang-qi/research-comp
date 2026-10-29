import numpy as np
import pytest

from causal_policy.policy import generator


def test_groups_and_sampling():
    groups = generator.build_groups([5, 3, 5, 9, 2], [1, -1, 1, 1, -1])
    assert groups == {1: [5, 9], -1: [2, 3]}
    choices = np.array([1, -1, 1, 1, -1] * 200)
    texts = generator.sample_texts(groups, choices, np.random.default_rng(0))
    assert all(t in groups[int(c)] for t, c in zip(texts, choices))
    assert set(texts) == {2, 3, 5, 9}


def test_empty_group_cannot_be_recommended():
    groups = generator.build_groups([5, 9], [1, 1])
    with pytest.raises(AssertionError):
        generator.sample_texts(groups, np.array([1, -1]), np.random.default_rng(0))
