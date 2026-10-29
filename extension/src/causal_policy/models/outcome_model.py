"""Outcome model h(x, t) = P(Y = 1 | X = x, T = t).

    h(x, t) = sigma(alpha + beta_x' r(x) + beta_t' u(t) + r(x)' B u(t)),

fitted by L2-penalized logistic regression on standardized features with an
unpenalized intercept. The inverse penalty is chosen by the lowest mean
held-out log loss over the given folds (ties: smaller value) and the model is
refitted on all data. If the training outcomes are all equal, the model
predicts their mean.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from causal_policy.models.features import design_matrix, quadratic_covariates

C_GRID = [float(10.0 ** e) for e in np.arange(-4.0, 2.0 + 0.25, 0.5)]
EPS = 1e-15


@dataclass
class OutcomeModel:
    C: float | None
    intercept: float  # logit = intercept + R beta_r + U beta_u + rowwise(R B U')
    beta_r: np.ndarray
    beta_u: np.ndarray
    B: np.ndarray
    constant: float | None = None  # set when the training outcomes are all equal


def log_loss(y, p):
    p = np.clip(p, EPS, 1.0 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1.0 - p)))


def _fit(R, U, Y, C):
    if Y.min() == Y.max():
        return OutcomeModel(C, 0.0, np.zeros(R.shape[1]), np.zeros(U.shape[1]), np.zeros((R.shape[1], U.shape[1])), float(Y.mean()))
    F = design_matrix(R, U)
    scaler = StandardScaler().fit(F)
    clf = LogisticRegression(C=C, solver="lbfgs", max_iter=10000).fit(scaler.transform(F), Y)
    # Undo the standardization: logit = a + sum_j w_j (f_j - m_j) / s_j.
    w = clf.coef_[0] / scaler.scale_
    intercept = float(clf.intercept_[0] - w @ scaler.mean_)
    p, q = R.shape[1], U.shape[1]
    return OutcomeModel(C, intercept, w[:p], w[p:p + q], w[p + q:].reshape(p, q))


def logit_pairs(model, R, U):
    """Logits for row-aligned pairs (r_i, u_i)."""
    return model.intercept + R @ model.beta_r + U @ model.beta_u + np.einsum("ij,jk,ik->i", R, model.B, U)


def logit_matrix(model, R, U_texts):
    """Logits for every person (rows of R) and every text (rows of U_texts)."""
    return model.intercept + (R @ model.beta_r)[:, None] + (U_texts @ model.beta_u)[None, :] + R @ model.B @ U_texts.T


def _proba(model, logits):
    if model.constant is not None:
        return np.full(np.shape(logits), model.constant)
    return 1.0 / (1.0 + np.exp(-logits))


def predict_outcome(model, X, U):
    """h(x_i, t_i) for row-aligned covariates X and text embeddings U."""
    R = quadratic_covariates(X)
    return _proba(model, logit_pairs(model, R, U))


def predict_outcome_matrix(model, X, U_texts):
    """h(x_i, t_k) for every person i and text k."""
    R = quadratic_covariates(X)
    return _proba(model, logit_matrix(model, R, U_texts))


def _tune(R, U, Y, folds, c_grid):
    cv_log_loss = []
    for C in c_grid:
        losses = []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            m = _fit(R[tr], U[tr], Y[tr], C)
            losses.append(log_loss(Y[te], _proba(m, logit_pairs(m, R[te], U[te]))))
        cv_log_loss.append(float(np.mean(losses)))
    best = int(np.argmin(cv_log_loss))
    return c_grid[best], cv_log_loss


def fit_outcome_model(X, U, Y, folds, c_grid=C_GRID):
    """Tune C by fold-wise held-out log loss and refit on all data."""
    R = quadratic_covariates(X)
    Y = np.asarray(Y, dtype=float)
    folds = np.asarray(folds)
    C, cv_log_loss = _tune(R, U, Y, folds, c_grid)
    return {"model": _fit(R, U, Y, C), "selected_C": C, "cv_log_loss": dict(zip(c_grid, cv_log_loss))}


def cross_fit(X, U, Y, folds, c_grid=C_GRID):
    """Cross-fitted h^(-i)(X_i, T_i): for each fold k, tune on the other folds (leaving one of them out at a time), refit on them, and predict fold k."""
    R = quadratic_covariates(X)
    Y = np.asarray(Y, dtype=float)
    folds = np.asarray(folds)
    out = np.full(len(Y), np.nan)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        C, _ = _tune(R[tr], U[tr], Y[tr], folds[tr], c_grid)
        m = _fit(R[tr], U[tr], Y[tr], C)
        out[te] = _proba(m, logit_pairs(m, R[te], U[te]))
    return out


def best_texts(model, X, text_ids, U_texts):
    """For each person, the text with the highest h(x, t) among the given texts (ties: smaller text ID)."""
    order = np.argsort(text_ids, kind="stable")
    ids = np.asarray(text_ids)[order]
    if model.constant is not None:
        return np.full(len(X), ids[0])
    # Rank on the logit scale, where probabilities near 0 or 1 do not tie.
    L = logit_matrix(model, quadratic_covariates(X), U_texts[order])
    return ids[np.argmax(L, axis=1)]
