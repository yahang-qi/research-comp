"""Outcome weighted learning (Zhao et al., 2012) with the hinge surrogate.

The rule d(x) = sign f(x) solves the weighted SVM

    min_f  (1/2) ||f||^2 + C sum_i w_i max(0, 1 - A_i f(X_i)),   w_i = R_i / pi(A_i | X_i),

whose dual has the box 0 <= alpha_i <= C w_i, as in DynTxRegime::owl with
lambdas = C. Rewards must be non-negative. Observations with zero weight have
alpha_i = 0 and are dropped before fitting.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.svm import SVC

KERNELS = ("linear", "gaussian", "poly2")
C_VALUES = (0.5, 4.0, 32.0, 256.0, 2048.0)
BANDWIDTH_MULTIPLIERS = (0.25, 0.5, 1.0, 2.0)


def make_svc(kernel, C, kparam=None):
    """Gaussian: K(x, x') = exp(-||x - x'||^2 / (2 kparam^2)); poly2: K(x, x') = (1 + <x, x'>)^2."""
    if kernel == "linear":
        return SVC(kernel="linear", C=C)
    if kernel == "gaussian":
        return SVC(kernel="rbf", gamma=1.0 / (2.0 * kparam ** 2), C=C)
    if kernel == "poly2":
        return SVC(kernel="poly", degree=2, gamma=1.0, coef0=1.0, C=C)
    raise ValueError(f"unknown kernel {kernel!r}")


@dataclass
class OWLFit:
    kernel: str
    C: float
    kparam: float | None
    model: SVC | None
    constant: int | None  # set when the positive-weight observations all share one treatment

    def predict(self, X):
        if self.constant is not None:
            return np.full(len(X), self.constant, dtype=int)
        return self.model.predict(X).astype(int)


def fit_owl(X, A, reward, kernel, C, kparam=None, propensity=0.5):
    """Fit the OWL rule; `propensity` is P(A_i | X_i) for each observation (or a constant)."""
    A = np.asarray(A)
    w = np.asarray(reward, dtype=float) / np.broadcast_to(np.asarray(propensity, dtype=float), A.shape)
    assert (w >= 0).all(), "OWL rewards must be non-negative"
    keep = w > 0
    classes = np.unique(A[keep])
    if len(classes) == 0:
        return OWLFit(kernel, C, kparam, None, 1)
    if len(classes) == 1:
        return OWLFit(kernel, C, kparam, None, int(classes[0]))
    model = make_svc(kernel, C, kparam)
    model.fit(X[keep], A[keep], sample_weight=w[keep])
    return OWLFit(kernel, C, kparam, model, None)


def rkhs_norm_sq(fit):
    """||f||^2 of the fitted decision function in its kernel's RKHS (0 for a constant rule)."""
    if fit.model is None:
        return 0.0
    m = fit.model
    params = {"rbf": {"gamma": m._gamma}, "poly": {"gamma": m._gamma, "degree": m.degree, "coef0": m.coef0}, "linear": {}}[m.kernel]
    K = pairwise_kernels(m.support_vectors_, metric=m.kernel, **params)
    a = m.dual_coef_[0]
    return float(a @ K @ a)


def owl_value(reward, A, d, propensity=0.5):
    """Horvitz-Thompson value estimate mean(R 1{A = d} / pi(A | X))."""
    A = np.asarray(A)
    return float(np.mean(np.asarray(reward, dtype=float) * (A == d) / np.broadcast_to(np.asarray(propensity, dtype=float), A.shape)))


def bandwidth_grid(X, multipliers=BANDWIDTH_MULTIPLIERS):
    """Gaussian kparam values: multiples of the median pairwise distance of the covariates."""
    med = float(np.median(pdist(X)))
    return [m * med for m in multipliers]


def tune_owl(X, A, reward, folds, kernel, c_values=C_VALUES, kparams=None, propensity=0.5):
    """Select (c, kparam) by the mean held-out value over the folds, then refit on all data.

    C = c / n with n the full training size, for the fold fits and the refit alike.
    Grid order is increasing c, then decreasing kparam; ties go to the earlier point.
    """
    A = np.asarray(A)
    reward = np.asarray(reward, dtype=float)
    folds = np.asarray(folds)
    prop = np.broadcast_to(np.asarray(propensity, dtype=float), A.shape)
    n = len(A)
    if kernel == "gaussian":
        kparams = sorted(bandwidth_grid(X) if kparams is None else kparams, reverse=True)
    else:
        kparams = [None]
    grid = [(c, k) for c in c_values for k in kparams]
    cv_values = []
    for c, k in grid:
        values = []
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            fit = fit_owl(X[tr], A[tr], reward[tr], kernel, c / n, k, prop[tr])
            values.append(owl_value(reward[te], A[te], fit.predict(X[te]), prop[te]))
        cv_values.append(float(np.mean(values)))
    best = int(np.argmax(cv_values))
    c, k = grid[best]
    return {
        "selected_c": c,
        "selected_kparam": k,
        "grid": [{"c": gc, "kparam": gk, "cv_value": v} for (gc, gk), v in zip(grid, cv_values)],
        "fit": fit_owl(X, A, reward, kernel, c / n, k, prop),
    }
