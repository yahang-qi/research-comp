import numpy as np

from causal_policy.models.features import QUADRATIC_NAMES, design_matrix, quadratic_covariates


def test_quadratic_covariates_order():
    X = np.array([[2.0, 3.0, 5.0]])
    assert quadratic_covariates(X).tolist() == [[2, 3, 5, 4, 9, 25, 6, 10, 15]]
    assert len(QUADRATIC_NAMES) == 9


def test_design_matrix_layout():
    rng = np.random.default_rng(0)
    R, U = rng.normal(size=(4, 9)), rng.normal(size=(4, 128))
    F = design_matrix(R, U)
    assert F.shape == (4, 9 + 128 + 9 * 128)
    assert np.array_equal(F[:, :9], R) and np.array_equal(F[:, 9:137], U)
    j, k = 7, 41
    assert np.allclose(F[:, 137 + j * 128 + k], R[:, j] * U[:, k])
