import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from causal_policy.models import outcome_model as om
from causal_policy.models.features import design_matrix, quadratic_covariates


def toy(n=150, dim=4, seed=0):
    # Text embedding with one frame direction; the frame effect depends on x1.
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n), rng.uniform(-1, 1, n)])
    z = rng.choice([-1, 1], n)
    U = rng.normal(scale=0.1, size=(n, dim))
    U[:, 0] += z
    p = 0.45 + 0.35 * np.tanh(2 * X[:, 0]) * z
    Y = (rng.uniform(size=n) <= p).astype(float)
    folds = rng.permutation(np.arange(n) % 5 + 1)
    return X, U, Y, folds


def test_refit_matches_sklearn_on_original_scale():
    X, U, Y, _ = toy()
    m = om._fit(quadratic_covariates(X), U, Y, 0.1)
    F = design_matrix(quadratic_covariates(X), U)
    scaler = StandardScaler().fit(F)
    clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=10000).fit(scaler.transform(F), Y)
    assert np.allclose(om.predict_outcome(m, X, U), clf.predict_proba(scaler.transform(F))[:, 1], atol=1e-8)


def test_matrix_prediction_matches_pairs():
    X, U, Y, _ = toy()
    m = om._fit(quadratic_covariates(X), U, Y, 0.1)
    texts = U[:7]
    H = om.predict_outcome_matrix(m, X[:5], texts)
    for i in range(5):
        assert np.allclose(H[i], om.predict_outcome(m, np.repeat(X[i:i + 1], 7, axis=0), texts))


def test_tuning_selects_minimum_cv_log_loss():
    X, U, Y, folds = toy()
    res = om.fit_outcome_model(X, U, Y, folds)
    losses = res["cv_log_loss"]
    assert res["selected_C"] == min(losses, key=lambda c: (losses[c], c))
    assert res["model"].C == res["selected_C"]


def test_cross_fit_ignores_own_fold_outcomes():
    X, U, Y, folds = toy()
    grid = om.C_GRID[::3]
    h = om.cross_fit(X, U, Y, folds, grid)
    Y2 = Y.copy()
    Y2[folds == 1] = 1 - Y2[folds == 1]
    h2 = om.cross_fit(X, U, Y2, folds, grid)
    assert np.isfinite(h).all() and ((h > 0) & (h < 1)).all()
    assert np.array_equal(h[folds == 1], h2[folds == 1])
    assert not np.array_equal(h[folds == 2], h2[folds == 2])


def test_best_texts_is_brute_force_argmax_with_id_ties():
    X, U, Y, _ = toy()
    m = om._fit(quadratic_covariates(X), U, Y, 0.1)
    ids = np.array([40, 7, 19, 3, 25])
    texts = U[:5]
    chosen = om.best_texts(m, X[:20], ids, texts)
    H = om.predict_outcome_matrix(m, X[:20], texts)
    assert np.array_equal(chosen, ids[np.argmax(H, axis=1)])
    dup = np.vstack([texts[1], texts[1]])
    assert (om.best_texts(m, X[:3], np.array([9, 4]), dup) == 4).all()


def test_constant_outcomes_predict_their_mean():
    X, U, _, folds = toy(n=20)
    res = om.fit_outcome_model(X, U, np.zeros(20), folds)
    assert res["model"].constant == 0.0
    assert (om.predict_outcome(res["model"], X, U) == 0.0).all()
    assert om.log_loss(np.array([0.0, 1.0]), np.array([0.0, 0.0])) == pytest.approx(-np.log(om.EPS) / 2)
