"""Joint text-OWL objective in the encoder weights w, for a fixed attribute policy d.

    J(w) = -V_OWL + lambda_comp * [L_comp + kappa * Omega_xi],
    V_OWL = (2/n) sum_i Y_i 1{c_i = d_i} = (2/n) sum_i Y_i (1 + c_i d_i) / 2,

with c_i the attribute of observation i's text, xi profiled out by ridge
(kappa = lambda_reg / lambda_comp), and the policy penalty lambda_reg ||psi||^2
omitted because it does not depend on w. There is no penalty on w: the balanced
labels do not change when w is rescaled, so w is kept at unit length instead.
With forward="hard" the balanced hard labels are used and the gradient is the
straight-through approximation through the surrogate; with forward="surrogate"
the surrogate replaces the labels in the forward pass at a fixed cut point, and
the gradient is exact.
"""

import numpy as np

from causal_policy.models.completeness import completeness_loss
from causal_policy.models.encoder import balanced_labels, scores, surrogate, surrogate_grad


def joint_objective(w, U_bank, bank_ids, text_index, Y, d, h_cf, R, lam_comp, lam_reg, forward="hard", cut=None):
    """Return the objective, its gradient in w, and the components.

    text_index[i] is the row of U_bank holding observation i's text; d is the policy's choice d(X_i).
    """
    Y = np.asarray(Y, dtype=float)
    d = np.asarray(d, dtype=float)
    n = len(Y)
    s_bank = scores(w, U_bank)
    labels, a = balanced_labels(s_bank, bank_ids)
    if forward == "hard":
        c = labels[text_index].astype(float)
    else:
        a = a if cut is None else cut
        c = surrogate(s_bank[text_index], a)

    value = (2.0 / n) * float(np.sum(Y * (1.0 + c * d) / 2.0))
    grad_c = -(1.0 / n) * Y * d
    comp = {"loss": 0.0, "profiled": 0.0, "xi": None}
    if lam_comp > 0:
        loss, profiled, g_comp, xi = completeness_loss(h_cf, R, c, lam_reg / lam_comp)
        comp = {"loss": loss, "profiled": profiled, "xi": xi}
        grad_c = grad_c + lam_comp * g_comp

    objective = -value + lam_comp * comp["profiled"]
    dc_ds = surrogate_grad(s_bank[text_index], a)
    grad_w = U_bank[text_index].T @ (grad_c * dc_ds)
    return {"objective": objective, "grad": grad_w, "value_owl": value, "completeness": comp["loss"],
            "completeness_profiled": comp["profiled"], "xi": comp["xi"], "labels": labels, "cut": a}
