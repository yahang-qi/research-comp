"""Covariate and covariate-text features."""

import numpy as np

QUADRATIC_NAMES = ("x1", "x2", "x3", "x1^2", "x2^2", "x3^2", "x1*x2", "x1*x3", "x2*x3")


def quadratic_covariates(X):
    """r(x) = (x1, x2, x3, x1^2, x2^2, x3^2, x1 x2, x1 x3, x2 x3) for rows of X = (x1, x2, x3)."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return np.column_stack([x1, x2, x3, x1 ** 2, x2 ** 2, x3 ** 2, x1 * x2, x1 * x3, x2 * x3])


def design_matrix(R, U):
    """[r, u, r (x) u] row by row; the interaction r_j u_k is column len(r) + len(u) + j * len(u) + k."""
    n = R.shape[0]
    return np.hstack([R, U, (R[:, :, None] * U[:, None, :]).reshape(n, -1)])
