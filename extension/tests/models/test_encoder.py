import numpy as np
import pytest

from causal_policy.models import encoder


def bank(n=200, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, dim))
    return U / np.linalg.norm(U, axis=1, keepdims=True), rng.permutation(n) + 1


def test_balanced_labels_and_cut():
    U, ids = bank()
    s = encoder.scores(np.ones(U.shape[1]), U)
    labels, a = encoder.balanced_labels(s, ids)
    assert (labels == 1).sum() == (labels == -1).sum() == 100
    assert s[labels == 1].min() > a > s[labels == -1].max()
    srt = np.sort(s)
    assert a == pytest.approx((srt[99] + srt[100]) / 2)


def test_ties_broken_by_text_id():
    s = np.array([0.0, 1.0, 1.0, 1.0, 2.0, -1.0])
    ids = np.array([10, 30, 20, 40, 50, 60])
    labels, a = encoder.balanced_labels(s, ids)
    # Increasing order: -1 (60), 0 (10), 1 (20), 1 (30), 1 (40), 2 (50); the top three are 30, 40, 50.
    assert labels.tolist() == [-1, 1, -1, 1, 1, -1] and a == 1.0


def test_labels_invariant_to_positive_scaling():
    U, ids = bank()
    w = np.random.default_rng(1).normal(size=U.shape[1])
    l1, _ = encoder.balanced_labels(encoder.scores(w, U), ids)
    l2, _ = encoder.balanced_labels(encoder.scores(3.7 * w, U), ids)
    l3, _ = encoder.balanced_labels(encoder.scores(-w, U), ids)
    assert np.array_equal(l1, l2) and np.array_equal(l1, -l3)


def test_surrogate_matches_hard_labels_away_from_cut():
    U, ids = bank()
    s = encoder.scores(np.random.default_rng(2).normal(size=U.shape[1]), U)
    labels, a = encoder.balanced_labels(s, ids)
    phi = encoder.surrogate(s, a)
    far = np.abs(s - a) > 1e-9
    assert np.array_equal(np.sign(phi[far]), labels[far])
    eps = 1e-6
    assert np.allclose(encoder.surrogate_grad(s, a), (encoder.surrogate(s + eps, a) - encoder.surrogate(s - eps, a)) / (2 * eps), atol=1e-8)


def test_initial_weights_are_unit_and_label_free():
    U, _ = bank()
    starts = encoder.initial_weights(U, n_random=3, seed=5)
    assert len(starts) == 4 and all(np.isclose(np.linalg.norm(v), 1.0) for v in starts)
    centred = U - U.mean(axis=0)
    top = np.linalg.eigh(centred.T @ centred)[1][:, -1]
    assert abs(starts[0] @ top) == pytest.approx(1.0)
    again = encoder.initial_weights(U, n_random=3, seed=5)
    assert all(np.array_equal(a, b) for a, b in zip(starts, again))
