import numpy as np
import pytest

from causal_policy.models import completeness
from causal_policy.models.features import quadratic_covariates


def data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n), rng.uniform(-1, 1, n)])
    R = quadratic_covariates(X)
    c = rng.choice([-1.0, 1.0], n)
    h = rng.uniform(0.1, 0.9, n)
    return R, c, h


def test_fit_xi_matches_direct_ridge():
    R, c, h = data()
    Z = completeness.m_design(R, c)
    n, k = Z.shape
    D = np.eye(k)
    D[0, 0] = 0
    kappa = 0.03
    direct = np.linalg.solve(Z.T @ Z / n + kappa * D, Z.T @ h / n)
    assert np.allclose(completeness.fit_xi(h, R, c, kappa), direct)


def test_loss_is_zero_for_exactly_linear_target():
    R, c, _ = data()
    xi = np.random.default_rng(3).normal(size=2 * R.shape[1] + 2)
    h = completeness.m_design(R, c) @ xi
    loss, profiled, grad, xi_hat = completeness.completeness_loss(h, R, c, 0.0)
    assert loss == pytest.approx(0.0, abs=1e-20) and np.allclose(xi_hat, xi)


@pytest.mark.parametrize("kappa", [0.0, 0.05])
def test_gradient_in_c_matches_finite_differences(kappa):
    R, _, h = data()
    c = np.random.default_rng(4).uniform(-0.9, 0.9, len(h))  # surrogate values
    _, _, grad, _ = completeness.completeness_loss(h, R, c, kappa)
    eps = 1e-6
    for i in (0, 7, 33):
        cp, cm = c.copy(), c.copy()
        cp[i] += eps
        cm[i] -= eps
        fd = (completeness.completeness_loss(h, R, cp, kappa)[1] - completeness.completeness_loss(h, R, cm, kappa)[1]) / (2 * eps)
        assert grad[i] == pytest.approx(fd, rel=1e-4, abs=1e-9)
