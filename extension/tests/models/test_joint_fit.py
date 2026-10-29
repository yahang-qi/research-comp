import numpy as np
import pytest

from causal_policy.models import joint_fit
from causal_policy.models.encoder import initial_weights
from causal_policy.models.outcome_model import cross_fit
from causal_policy.policy.evaluation import attribute_recovery


def planted(n=300, n_bank=40, dim=6, seed=0):
    # Bank with a frame direction in coordinate 0; the frame effect depends on x1.
    rng = np.random.default_rng(seed)
    z_bank = np.repeat([1, -1], n_bank // 2)
    U = rng.normal(scale=0.3, size=(n_bank, dim))
    U[:, 0] += z_bank
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    bank_ids = rng.permutation(n_bank) + 1
    X = np.column_stack([rng.uniform(-1, 1, n), rng.uniform(0, 1, n), rng.uniform(-1, 1, n)])
    text_index = rng.integers(0, n_bank, n)
    mu = 0.45 + 0.35 * np.tanh(3 * X[:, 0]) * z_bank[text_index]
    Y = (rng.uniform(size=n) <= mu).astype(int)
    folds = rng.permutation(np.arange(n) % 5 + 1)
    return X, Y, folds, text_index, U, bank_ids, z_bank


def test_recovers_planted_frames_from_random_starts():
    X, Y, folds, text_index, U, bank_ids, z_bank = planted()
    h_cf = cross_fit(X, U[text_index], Y, folds)
    starts = initial_weights(U, n_random=2, seed=1)[1:]
    fit = joint_fit.fit_joint(X, Y, text_index, h_cf, U, bank_ids, "linear", 10.0, 1 / 64, starts)
    assert (fit.labels == 1).sum() == (fit.labels == -1).sum() == len(z_bank) // 2
    assert attribute_recovery(fit.labels, z_bank) >= 0.9
    assert np.isclose(np.linalg.norm(fit.w), 1.0)


def test_keeps_start_with_smallest_objective():
    X, Y, folds, text_index, U, bank_ids, _ = planted(n=120)
    h_cf = cross_fit(X, U[text_index], Y, folds)
    starts = initial_weights(U, n_random=2, seed=3)
    single = [joint_fit.fit_joint(X, Y, text_index, h_cf, U, bank_ids, "poly2", 1.0, 1 / 8, [w]) for w in starts]
    combined = joint_fit.fit_joint(X, Y, text_index, h_cf, U, bank_ids, "poly2", 1.0, 1 / 8, starts)
    assert combined.objective == pytest.approx(min(f.objective for f in single))
    assert combined.start == int(np.argmin([f.objective for f in single]))


def test_tuning_fits_never_see_the_held_out_fold(monkeypatch):
    X, Y, folds, text_index, U, bank_ids, _ = planted(n=100)
    seen = []
    original = joint_fit.fit_joint

    def recording(Xf, *args, **kwargs):
        seen.append(Xf.copy())
        return original(Xf, *args, **kwargs)

    monkeypatch.setattr(joint_fit, "fit_joint", recording)
    res = joint_fit.tune_joint(X, Y, folds, text_index, U, bank_ids, ["linear"], initial_weights(U)[:1], lam_comp_grid=(0.0, 1.0), lam_reg_grid=(1 / 8,))
    fold_rows = {f: {tuple(r) for r in X[folds == f]} for f in np.unique(folds)}
    for Xf in seen[:-1]:  # all but the final refit on the full sample
        rows = {tuple(r) for r in Xf}
        held_out = [f for f, fr in fold_rows.items() if not (rows & fr)]
        assert len(held_out) == 1 and len(rows) == len(X) - len(fold_rows[held_out[0]])
    assert len(seen[-1]) == len(X)
    best = max(res["linear"]["grid"], key=lambda g: g["cv_value"])
    assert res["linear"]["cv_value"] == best["cv_value"]
