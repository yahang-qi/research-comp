"""Balanced text encoder phi_theta.

Scores s(t) = w'u(t) for all bank texts are ranked in increasing order, ties
broken by text ID; the top half is labelled +1 and the rest -1. The cut point
a is the midpoint of the two middle ordered scores. For the backward pass the
hard labels are replaced by the surrogate 2 sigma(s - a) - 1 with a held fixed
(straight-through approximation).
"""

import numpy as np
from scipy.special import expit


def scores(w, U_bank):
    return U_bank @ w


def balanced_labels(s, text_ids):
    """Return the balanced +-1 labels of the bank texts and the cut point a."""
    s = np.asarray(s)
    n = len(s)
    assert n % 2 == 0, "the bank must have an even number of texts"
    half = n // 2
    order = np.lexsort((np.asarray(text_ids), s))  # increasing score, then increasing text ID
    labels = np.full(n, -1, dtype=int)
    labels[order[half:]] = 1
    s_sorted = s[order]
    return labels, float((s_sorted[half - 1] + s_sorted[half]) / 2.0)


def surrogate(s, a):
    return 2.0 * expit(np.asarray(s) - a) - 1.0


def surrogate_grad(s, a):
    p = expit(np.asarray(s) - a)
    return 2.0 * p * (1.0 - p)


def initial_weights(U_bank, n_random=0, seed=0):
    """Unit first principal component of the centred bank embeddings (sign: largest entry positive),
    followed by n_random seeded random unit vectors."""
    Uc = U_bank - U_bank.mean(axis=0)
    v = np.linalg.svd(Uc, full_matrices=False)[2][0]
    v = v * np.sign(v[np.argmax(np.abs(v))])
    starts = [v / np.linalg.norm(v)]
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    for _ in range(n_random):
        r = rng.normal(size=U_bank.shape[1])
        starts.append(r / np.linalg.norm(r))
    return starts
