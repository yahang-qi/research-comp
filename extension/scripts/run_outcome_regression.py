"""Outcome regression on the synthetic text experiment, for one training sample size.

For each replication, fit the outcome model h(x, t) on the covariates, the
embedding of the assigned text and the outcome, and deliver to each test
person the observed training text with the highest h(x, t). The policy is
evaluated with the known outcome probabilities, and the fit of h is compared
with the true probability mu(x, t) on the test sample (one design-drawn text
per person), through the frame contrast, and through the cross-fitted
predictions on the training sample. Simulator truth is used only for evaluation.
"""

import argparse
import os
import sys

import numpy as np

from causal_policy.data.samples import load_bank_frames, load_test, load_training_reps
from causal_policy.data.text_bank import read_embeddings
from causal_policy.io import fmt, write_csv
from causal_policy.models.outcome_model import best_texts, cross_fit, fit_outcome_model, predict_outcome, predict_outcome_matrix
from causal_policy.policy.evaluation import contrast_metrics, outcome_fit_metrics, policy_metrics

POLICY_KEYS = ("value", "regret_norm", "gain", "gain_fraction", "treatment_error", "share_gain", "share_loss")
FIT_KEYS = ("cv_log_loss", "rmse_mu", "expected_log_loss", "oracle_log_loss", "contrast_rmse", "contrast_sign_agreement", "crossfit_rmse_mu")
HEADER = ["method", "n", "rep", "n_texts_observed", "selected_C"] + list(POLICY_KEYS) + list(FIT_KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-end", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ids, E, _ = read_embeddings(args.embeddings)
    row_of = {t: i for i, t in enumerate(ids)}
    frame_of = load_bank_frames(args.sim_dir)
    gain_ids = np.array(sorted(t for t, z in frame_of.items() if z == 1))
    loss_ids = np.array(sorted(t for t, z in frame_of.items() if z == -1))

    reps = load_training_reps(args.sim_dir, args.n)
    X_test, test_truth = load_test(args.sim_dir)
    b, g, z_opt = test_truth.b, test_truth.g, test_truth.z_opt

    # One design-drawn text per test person for the fit check: Z uniform, then a uniform text of that frame.
    rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(3,)))
    z_fit = rng.choice(np.array([-1, 1]), len(X_test))
    j_fit = rng.integers(0, len(gain_ids), len(X_test))
    t_fit = np.where(z_fit == 1, gain_ids[j_fit], loss_ids[j_fit])
    mu_fit = b + z_fit * g
    U_fit = E[[row_of[t] for t in t_fit]]
    U_gain, U_loss = E[[row_of[t] for t in gain_ids]], E[[row_of[t] for t in loss_ids]]

    rows = []
    for rep in range(args.rep_start, args.rep_end + 1):
        observed, truth = reps[rep]
        X, Y, folds, text_ids = observed.X, observed.Y, observed.folds, observed.text_ids
        U = E[[row_of[t] for t in text_ids]]
        mu_train = truth.mu

        res = fit_outcome_model(X, U, Y, folds)
        model = res["model"]
        observed = np.unique(text_ids)
        delivered = best_texts(model, X_test, observed, E[[row_of[t] for t in observed]])
        pm = policy_metrics(b, g, np.array([frame_of[t] for t in delivered]), z_opt)

        fm = outcome_fit_metrics(predict_outcome(model, X_test, U_fit), mu_fit)
        cm = contrast_metrics(predict_outcome_matrix(model, X_test, U_gain).mean(axis=1), predict_outcome_matrix(model, X_test, U_loss).mean(axis=1), g)
        crossfit_rmse = float(np.sqrt(np.mean((cross_fit(X, U, Y, folds) - mu_train) ** 2)))
        fit = {"cv_log_loss": res["cv_log_loss"][res["selected_C"]], **fm, **cm, "crossfit_rmse_mu": crossfit_rmse}

        rows.append(["Outcome-regression", args.n, rep, len(observed), res["selected_C"]] + [fmt(pm[k]) for k in POLICY_KEYS] + [fmt(fit[k]) for k in FIT_KEYS])
        print(f"n={args.n} rep={rep} done", file=sys.stderr, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, f"per_rep_n{args.n:04d}_reps{args.rep_start:03d}-{args.rep_end:03d}.csv"), HEADER, rows)


if __name__ == "__main__":
    main()
