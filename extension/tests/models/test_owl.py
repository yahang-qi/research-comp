import numpy as np
import pytest
from sklearn.metrics.pairwise import pairwise_kernels

from causal_policy.models import owl


def toy(n=200, seed=0):
    # Optimal rule: A = +1 when x1 > 0. Reward 1 when the assigned A matches it.
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    A = rng.choice([-1, 1], n)
    reward = (A == np.where(X[:, 0] > 0, 1, -1)).astype(float)
    folds = rng.permutation(np.arange(n) % 5 + 1)
    return X, A, reward, folds


def gram(model, X):
    params = {"rbf": {"gamma": model.gamma}, "poly": {"gamma": model.gamma, "degree": model.degree, "coef0": model.coef0}, "linear": {}}[model.kernel]
    return pairwise_kernels(X, X, metric=model.kernel, **params)


def test_kernels_match_their_formulas():
    X = np.random.default_rng(1).normal(size=(6, 3))
    sq = ((X[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    assert np.allclose(gram(owl.make_svc("gaussian", 1.0, 0.7), X), np.exp(-sq / (2 * 0.7 ** 2)))
    assert np.allclose(gram(owl.make_svc("poly2", 1.0), X), (1 + X @ X.T) ** 2)
    assert np.allclose(gram(owl.make_svc("linear", 1.0), X), X @ X.T)


@pytest.mark.parametrize("kernel,kparam", [("linear", None), ("gaussian", 1.0), ("poly2", None)])
def test_learns_separable_rule(kernel, kparam):
    X, A, reward, _ = toy()
    fit = owl.fit_owl(X, A, reward, kernel, C=1.0, kparam=kparam)
    Xt = np.random.default_rng(2).uniform(-1, 1, size=(500, 2))
    assert np.mean(fit.predict(Xt) == np.where(Xt[:, 0] > 0, 1, -1)) > 0.9


def test_zero_weight_rows_do_not_change_fit():
    X, A, reward, _ = toy()
    fit = owl.fit_owl(X, A, reward, "gaussian", C=1.0, kparam=1.0)
    keep = reward > 0
    fit_kept = owl.fit_owl(X[keep], A[keep], reward[keep], "gaussian", C=1.0, kparam=1.0)
    Xt = np.random.default_rng(3).uniform(-1, 1, size=(300, 2))
    assert np.allclose(fit.model.decision_function(Xt), fit_kept.model.decision_function(Xt))


def test_swapping_treatments_flips_rule():
    X, A, reward, _ = toy()
    fit = owl.fit_owl(X, A, reward, "poly2", C=1.0)
    fit_flip = owl.fit_owl(X, -A, reward, "poly2", C=1.0)
    Xt = np.random.default_rng(4).uniform(-1, 1, size=(300, 2))
    # Equal up to the solver tolerance (libsvm tol = 1e-3).
    f, f_flip = fit.model.decision_function(Xt), fit_flip.model.decision_function(Xt)
    assert np.allclose(f, -f_flip, atol=1e-2)
    assert np.array_equal(fit.predict(Xt), -fit_flip.predict(Xt))


def test_one_treatment_among_positive_weights_gives_constant_rule():
    X = np.zeros((4, 1))
    fit = owl.fit_owl(X, np.array([1, -1, 1, -1]), np.array([1.0, 0.0, 2.0, 0.0]), "linear", C=1.0)
    assert fit.constant == 1 and fit.predict(np.ones((3, 1))).tolist() == [1, 1, 1]
    with pytest.raises(AssertionError):
        owl.fit_owl(X, np.array([1, -1, 1, -1]), np.array([1.0, -1.0, 2.0, 0.0]), "linear", C=1.0)


@pytest.mark.parametrize("kernel,kparam", [("linear", None), ("gaussian", 0.8), ("poly2", None)])
def test_rkhs_norm_matches_direct_computation(kernel, kparam):
    X, A, reward, _ = toy(n=80)
    fit = owl.fit_owl(X, A, reward, kernel, C=1.0, kparam=kparam)
    sv, a = fit.model.support_vectors_, fit.model.dual_coef_[0]
    if kernel == "linear":
        K = sv @ sv.T
    elif kernel == "gaussian":
        K = np.exp(-((sv[:, None, :] - sv[None, :, :]) ** 2).sum(-1) / (2 * kparam ** 2))
    else:
        K = (1 + sv @ sv.T) ** 2
    assert owl.rkhs_norm_sq(fit) == pytest.approx(a @ K @ a)
    if kernel == "linear":
        assert owl.rkhs_norm_sq(fit) == pytest.approx(float(fit.model.coef_[0] @ fit.model.coef_[0]))
    const = owl.fit_owl(np.zeros((2, 1)), np.array([1, 1]), np.array([1.0, 1.0]), kernel, C=1.0, kparam=kparam)
    assert owl.rkhs_norm_sq(const) == 0.0


def test_owl_value_is_horvitz_thompson():
    reward = np.array([1.0, 0.0, 1.0, 1.0])
    A = np.array([1, 1, -1, -1])
    d = np.array([1, -1, -1, 1])
    assert owl.owl_value(reward, A, d, 0.5) == pytest.approx((1.0 + 1.0) / 0.5 / 4)


def test_tune_selects_best_cv_value_and_refits():
    X, A, reward, folds = toy()
    res = owl.tune_owl(X, A, reward, folds, "gaussian")
    values = [g["cv_value"] for g in res["grid"]]
    best = res["grid"][int(np.argmax(values))]
    assert (res["selected_c"], res["selected_kparam"]) == (best["c"], best["kparam"])
    assert len(res["grid"]) == len(owl.C_VALUES) * len(owl.BANDWIDTH_MULTIPLIERS)
    assert res["fit"].C == pytest.approx(res["selected_c"] / len(A))
    kparams = [g["kparam"] for g in res["grid"][: len(owl.BANDWIDTH_MULTIPLIERS)]]
    assert kparams == sorted(kparams, reverse=True)
