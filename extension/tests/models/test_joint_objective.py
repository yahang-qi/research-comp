import numpy as np
import pytest

from causal_policy.models import encoder
from causal_policy.models.features import quadratic_covariates
from causal_policy.models.joint_objective import joint_objective


def setup(n=80, n_bank=40, dim=6, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_bank, dim))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    ids = rng.permutation(n_bank) + 1
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n), rng.uniform(-1, 1, n)])
    return {
        "U_bank": U, "bank_ids": ids, "text_index": rng.integers(0, n_bank, n), "Y": rng.integers(0, 2, n),
        "d": rng.choice([-1, 1], n), "h_cf": rng.uniform(0.1, 0.9, n), "R": quadratic_covariates(X),
    }, rng.normal(size=dim)


@pytest.mark.parametrize("lam_comp,lam_reg", [(0.0, 0.01), (2.0, 0.0), (2.0, 0.05)])
def test_gradient_matches_finite_differences_of_surrogate_objective(lam_comp, lam_reg):
    args, w = setup()
    cut = joint_objective(w, **args, lam_comp=lam_comp, lam_reg=lam_reg)["cut"]
    res = joint_objective(w, **args, lam_comp=lam_comp, lam_reg=lam_reg, forward="surrogate", cut=cut)
    eps = 1e-6
    fd = np.zeros_like(w)
    for k in range(len(w)):
        wp, wm = w.copy(), w.copy()
        wp[k] += eps
        wm[k] -= eps
        fd[k] = (joint_objective(wp, **args, lam_comp=lam_comp, lam_reg=lam_reg, forward="surrogate", cut=cut)["objective"]
                 - joint_objective(wm, **args, lam_comp=lam_comp, lam_reg=lam_reg, forward="surrogate", cut=cut)["objective"]) / (2 * eps)
    assert np.allclose(res["grad"], fd, rtol=1e-4, atol=1e-8)


def test_hard_forward_uses_plan_owl_value():
    args, w = setup()
    res = joint_objective(w, **args, lam_comp=0.0, lam_reg=0.0)
    labels, _ = encoder.balanced_labels(encoder.scores(w, args["U_bank"]), args["bank_ids"])
    c = labels[args["text_index"]]
    v_plan = 2.0 / len(c) * np.sum(args["Y"] * (c == args["d"]))
    assert res["value_owl"] == pytest.approx(v_plan) and res["objective"] == pytest.approx(-v_plan)
    assert np.array_equal(res["labels"], labels)
