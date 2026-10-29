import numpy as np
import pytest

from causal_policy.data import simulation
from causal_policy.policy import evaluation


def population_sample(n=200000, seed=0):
    x1, x2, x3 = simulation.draw_covariates(np.random.default_rng(seed), n)
    return simulation.b_fn(x2, x3), simulation.g_fn(x1, x2)


def test_optimal_and_constant_policies():
    b, g = population_sample()
    z_opt = np.where(g > 0, 1, -1)
    opt = evaluation.policy_metrics(b, g, z_opt, z_opt)
    assert opt["treatment_error"] == 0.0
    assert opt["value"] == pytest.approx(evaluation.V_OPT, abs=3e-3)
    assert opt["regret_norm"] == pytest.approx(0.0, abs=5e-3) and opt["gain_fraction"] == pytest.approx(1.0, abs=0.02)
    const = evaluation.policy_metrics(b, g, np.ones_like(z_opt), z_opt)
    assert const["gain_fraction"] == pytest.approx(0.0, abs=0.02) and const["share_gain"] == 1.0


def test_perfect_outcome_model_and_contrast():
    b, g = population_sample(n=1000)
    mu = b + g
    fit = evaluation.outcome_fit_metrics(mu, mu)
    assert fit["rmse_mu"] == 0.0 and fit["expected_log_loss"] == pytest.approx(fit["oracle_log_loss"])
    worse = evaluation.outcome_fit_metrics(np.full_like(mu, mu.mean()), mu)
    assert worse["expected_log_loss"] > fit["oracle_log_loss"]
    c = evaluation.contrast_metrics(b + g, b - g, g)
    assert c["contrast_rmse"] == pytest.approx(0.0) and c["contrast_sign_agreement"] == 1.0
    assert evaluation.contrast_metrics(b - g, b + g, g)["contrast_sign_agreement"] == 0.0


def test_attribute_recovery_allows_sign_reversal():
    z = np.array([1, 1, -1, -1])
    assert evaluation.attribute_recovery(z, z) == 1.0
    assert evaluation.attribute_recovery(-z, z) == 1.0
    assert evaluation.attribute_recovery(np.array([1, -1, -1, -1]), z) == 0.75
    assert evaluation.attribute_recovery(z, z, mask=np.array([True, False, True, False])) == 1.0
    assert np.isnan(evaluation.attribute_recovery(z, z, mask=np.zeros(4, dtype=bool)))


def test_metric_identities():
    b = np.array([0.4, 0.5]); g = np.array([0.2, -0.1]); z = np.array([1, 1])
    m = evaluation.policy_metrics(b, g, z, np.array([1, -1]))
    assert m["value"] == pytest.approx(0.5)
    assert m["treatment_error"] == 0.5
    assert m["gain"] == pytest.approx(0.5 - evaluation.V_CONST)
    assert m["regret_norm"] == pytest.approx((evaluation.V_OPT - 0.5) / evaluation.V_OPT)
    assert evaluation.constant_frame_metrics()["gain_fraction"] == 0.0
