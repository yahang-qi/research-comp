"""Evaluate a fitted text policy on the test sample with the known outcome model.

The value of delivering text t to a person with covariates x is
mu(x, t) = b(x) + z(t) g(x); values use these probabilities, not simulated
outcomes, and the population constants V* and V_const from the plan.
"""

import numpy as np
from scipy.special import xlogy

from causal_policy.data.simulation import PLAN_VALUES

V_OPT = PLAN_VALUES["V_opt"]
V_CONST = PLAN_VALUES["V_gain"]


def policy_metrics(b, g, z_delivered, z_opt):
    """Value, normalized regret, gain, gain fraction, treatment error and frame shares."""
    b, g, z_delivered, z_opt = (np.asarray(v) for v in (b, g, z_delivered, z_opt))
    value = float(np.mean(b + z_delivered * g))
    return {
        "value": value,
        "regret_norm": (V_OPT - value) / V_OPT,
        "gain": value - V_CONST,
        "gain_fraction": (value - V_CONST) / (V_OPT - V_CONST),
        "treatment_error": float(np.mean(z_delivered != z_opt)),
        "share_gain": float(np.mean(z_delivered == 1)),
        "share_loss": float(np.mean(z_delivered == -1)),
    }


def outcome_fit_metrics(h, mu, eps=1e-15):
    """RMSE of h to the true probability mu, and the expected log loss of h vs the oracle (h = mu) under Y ~ Bernoulli(mu)."""
    h = np.clip(np.asarray(h, dtype=float), eps, 1 - eps)
    mu = np.asarray(mu, dtype=float)
    return {
        "rmse_mu": float(np.sqrt(np.mean((h - mu) ** 2))),
        "expected_log_loss": float(np.mean(-(xlogy(mu, h) + xlogy(1 - mu, 1 - h)))),
        "oracle_log_loss": float(np.mean(-(xlogy(mu, mu) + xlogy(1 - mu, 1 - mu)))),
    }


def contrast_metrics(h_gain_mean, h_loss_mean, g):
    """Estimated frame contrast vs the true contrast 2 g(x), and the share of people for whom it prefers the optimal frame."""
    diff = np.asarray(h_gain_mean) - np.asarray(h_loss_mean)
    g = np.asarray(g)
    return {
        "contrast_rmse": float(np.sqrt(np.mean((diff - 2 * g) ** 2))),
        "contrast_sign_agreement": float(np.mean(np.sign(diff) == np.sign(g))),
    }


def attribute_recovery(labels, z_true, mask=None):
    """Agreement of learned +-1 labels with the true frames, allowing one global sign reversal (0.5 = chance)."""
    labels, z_true = np.asarray(labels), np.asarray(z_true)
    if mask is not None:
        labels, z_true = labels[mask], z_true[mask]
    if len(labels) == 0:
        return float("nan")
    a = float(np.mean(labels == z_true))
    return max(a, 1.0 - a)


def constant_frame_metrics():
    """Reference metrics of the best constant frame (gain framing) from the plan's constants."""
    return {"value": V_CONST, "regret_norm": (V_OPT - V_CONST) / V_OPT, "gain": 0.0, "gain_fraction": 0.0}
