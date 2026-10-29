"""Joint text OWL: alternating fit of the encoder phi_theta and the attribute policy d_psi.

For fixed (lambda_comp, lambda_reg), starting from each initial w:
  repeat up to MAX_OUTER times
    psi step:   fit d_psi by hinge-loss OWL on the current learned attributes,
                with C = 1 / (2 lambda_reg n) (penalty lambda_reg ||psi||^2);
    theta step: up to MAX_INNER normalized gradient steps on w (kept at unit
                length) for -V_OWL + lambda_comp (L_comp + kappa ||xi||^2),
                keeping the best w seen; the balanced labels and cut point are
                recomputed after each step;
  until the labels no longer change. The policy is refitted on the final labels
and the fit with the smallest
    J = -V_OWL + lambda_comp (L_comp + kappa ||xi||^2) + lambda_reg ||psi||^2
over the starts is kept. (lambda_comp, lambda_reg) are tuned by the held-out
value over the stored folds, refitting everything within each training split.
"""

from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import pdist

from causal_policy.models.encoder import balanced_labels, scores
from causal_policy.models.features import quadratic_covariates
from causal_policy.models.joint_objective import joint_objective
from causal_policy.models.outcome_model import cross_fit
from causal_policy.models.owl import OWLFit, fit_owl, rkhs_norm_sq

LAM_COMP_GRID = (0.0, 1.0, 10.0, 100.0)
LAM_REG_GRID = (1.0, 1 / 8, 1 / 64, 1 / 512, 1 / 4096)
MAX_OUTER = 20
MAX_INNER = 10
STEP_SIZE = 0.05


@dataclass
class JointFit:
    w: np.ndarray
    labels: np.ndarray  # learned attribute of every bank text
    policy: OWLFit
    objective: float
    value_owl: float
    completeness: float
    start: int  # index into the starts; 0 = first principal component
    outer_iterations: int
    lam_comp: float
    lam_reg: float


def gaussian_bandwidth(X):
    return float(np.median(pdist(X)))


def _fit_policy(X, c, Y, kernel, lam_reg, kparam):
    return fit_owl(X, c, Y, kernel, 1.0 / (2.0 * lam_reg * len(Y)), kparam)


def fit_joint(X, Y, text_index, h_cf, U_bank, bank_ids, kernel, lam_comp, lam_reg, starts):
    """Fit joint text OWL on one training set; text_index[i] is the bank row of observation i's text."""
    Y = np.asarray(Y, dtype=float)
    R = quadratic_covariates(X)
    kparam = gaussian_bandwidth(X) if kernel == "gaussian" else None

    def objective(w, d):
        return joint_objective(w, U_bank, bank_ids, text_index, Y, d, h_cf, R, lam_comp, lam_reg)

    best = None
    for si, w0 in enumerate(starts):
        w = w0 / np.linalg.norm(w0)
        labels = balanced_labels(scores(w, U_bank), bank_ids)[0]
        outer = 0
        for outer in range(1, MAX_OUTER + 1):
            d = _fit_policy(X, labels[text_index], Y, kernel, lam_reg, kparam).predict(X)
            cur, best_w, best_obj = w, w, None
            for step in range(MAX_INNER + 1):
                res = objective(cur, d)
                if best_obj is None or res["objective"] < best_obj:
                    best_w, best_obj = cur, res["objective"]
                g = res["grad"]
                if step == MAX_INNER or not np.any(g):
                    break
                cur = cur - STEP_SIZE * g / np.linalg.norm(g)
                cur = cur / np.linalg.norm(cur)
            w = best_w
            new_labels = balanced_labels(scores(w, U_bank), bank_ids)[0]
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
        policy = _fit_policy(X, labels[text_index], Y, kernel, lam_reg, kparam)
        res = objective(w, policy.predict(X))
        J = res["objective"] + lam_reg * rkhs_norm_sq(policy)
        if best is None or J < best.objective:
            best = JointFit(w, labels, policy, J, res["value_owl"], res["completeness"], si, outer, lam_comp, lam_reg)
    return best


def held_out_value(fit, X, Y, text_index):
    """Plan's value estimate (2/n) sum Y 1{phi(T) = d(X)} on held-out observations."""
    c = fit.labels[text_index]
    return float(2.0 / len(Y) * np.sum(np.asarray(Y) * (c == fit.policy.predict(X))))


def tune_joint(X, Y, folds, text_index, U_bank, bank_ids, kernels, starts, lam_comp_grid=LAM_COMP_GRID, lam_reg_grid=LAM_REG_GRID):
    """Tune (lambda_comp, lambda_reg) for each kernel by the mean held-out value over the folds, then refit on all data.

    Ties go to the larger lambda_reg, then the smaller lambda_comp. The cross-fitted outcome
    predictions are computed once per training split and shared across kernels and grid points.
    """
    Y = np.asarray(Y)
    folds = np.asarray(folds)
    U_obs = U_bank[text_index]
    grid = [(lc, lr) for lr in sorted(lam_reg_grid, reverse=True) for lc in sorted(lam_comp_grid)]
    cv = {kernel: np.zeros((len(np.unique(folds)), len(grid))) for kernel in kernels}
    for fi, f in enumerate(np.unique(folds)):
        tr, te = folds != f, folds == f
        h_tr = cross_fit(X[tr], U_obs[tr], Y[tr], folds[tr])
        for kernel in kernels:
            for gi, (lc, lr) in enumerate(grid):
                fit = fit_joint(X[tr], Y[tr], text_index[tr], h_tr, U_bank, bank_ids, kernel, lc, lr, starts)
                cv[kernel][fi, gi] = held_out_value(fit, X[te], Y[te], text_index[te])
    h_all = cross_fit(X, U_obs, Y, folds)
    out = {}
    for kernel in kernels:
        means = cv[kernel].mean(axis=0)
        best = int(np.argmax(means))
        lc, lr = grid[best]
        out[kernel] = {
            "fit": fit_joint(X, Y, text_index, h_all, U_bank, bank_ids, kernel, lc, lr, starts),
            "selected_lam_comp": lc,
            "selected_lam_reg": lr,
            "cv_value": float(means[best]),
            "grid": [{"lam_comp": g[0], "lam_reg": g[1], "cv_value": float(v)} for g, v in zip(grid, means)],
        }
    return out
