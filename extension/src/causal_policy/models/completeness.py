"""Completeness loss.

For attribute values c (hard +-1 labels or surrogate values) the reduced model
m_xi(x, c) = xi_0 + xi_x'r(x) + xi_z c + xi_xz'r(x) c is fitted to the
cross-fitted outcome predictions h^(-i) by ridge regression,

    xi_hat = argmin (1/n) sum_i (h_i - m_xi(x_i, c_i))^2 + kappa (||xi_x||^2 + xi_z^2 + ||xi_xz||^2),

with kappa = lambda_reg / lambda_comp, which is the minimizer over xi of the
joint objective. The completeness loss is the first term at xi_hat. Its
profiled value adds kappa times the penalty; because xi_hat is optimal, the
gradient of the profiled value in c_i is -(2/n)(h_i - m_i)(xi_z + xi_xz'r_i).
"""

import numpy as np


def m_design(R, c):
    c = np.asarray(c, dtype=float)[:, None]
    return np.hstack([np.ones_like(c), R, c, R * c])


def fit_xi(h, R, c, kappa):
    """Ridge solution with an unpenalized intercept (minimum-norm when the design is rank deficient)."""
    Z = m_design(R, c)
    n, k = Z.shape
    D = np.eye(k)
    D[0, 0] = 0.0
    A = np.vstack([Z / np.sqrt(n), np.sqrt(kappa) * D])
    rhs = np.concatenate([np.asarray(h, dtype=float) / np.sqrt(n), np.zeros(k)])
    return np.linalg.lstsq(A, rhs, rcond=None)[0]


def completeness_loss(h, R, c, kappa):
    """Return the loss, its profiled value (loss + kappa * penalty), the gradient of the profiled value in c, and xi_hat."""
    h = np.asarray(h, dtype=float)
    xi = fit_xi(h, R, c, kappa)
    resid = h - m_design(R, c) @ xi
    n, p = R.shape
    xi_z, xi_xz = xi[1 + p], xi[2 + p:]
    loss = float(np.mean(resid ** 2))
    profiled = loss + kappa * float(xi[1:] @ xi[1:])
    grad_c = -(2.0 / n) * resid * (xi_z + R @ xi_xz)
    return loss, profiled, grad_c, xi
