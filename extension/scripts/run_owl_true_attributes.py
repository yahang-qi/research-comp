"""OWL with true attributes on the synthetic text experiment, for one training sample size.

For each replication, OWL learns d(x) in {-1, +1} from the covariates, the true
frame Z of the assigned text, and the outcome Y (reward Y, propensity 1/2),
with linear, Gaussian and degree-two polynomial kernels tuned on the stored
folds. G delivers to each test person a text drawn uniformly from the training
texts whose true frame is d(x), and the policy is evaluated with the known
outcome probabilities on the test sample.
"""

import argparse
import os
import sys

import numpy as np

from causal_policy.data.samples import load_bank_frames, load_test, load_training_reps
from causal_policy.io import fmt, write_csv
from causal_policy.models.owl import KERNELS, tune_owl
from causal_policy.policy.evaluation import policy_metrics
from causal_policy.policy.generator import build_groups, sample_texts

METHOD_NAMES = {"linear": "OWL-true-Linear", "gaussian": "OWL-true-Gaussian", "poly2": "OWL-true-Poly2"}
HEADER = ["method", "kernel", "n", "rep", "n_y1", "selected_c", "selected_kparam", "cv_value", "constant_rule",
          "value", "regret_norm", "gain", "gain_fraction", "treatment_error", "share_gain", "share_loss"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-reps", type=int, default=None, help="run only the first replications (for timing)")
    args = parser.parse_args()

    reps = load_training_reps(args.sim_dir, args.n)
    frame_of = load_bank_frames(args.sim_dir)
    X_test, test_truth = load_test(args.sim_dir)
    b, g, z_opt = test_truth.b, test_truth.g, test_truth.z_opt

    rows = []
    for rep in list(reps)[:args.max_reps]:
        observed, truth = reps[rep]
        X, Y, folds, text_ids = observed.X, observed.Y, observed.folds, observed.text_ids
        A = truth.z
        assert all(frame_of[t] == a for t, a in zip(text_ids, A))
        groups = build_groups(text_ids, A)
        for k, kernel in enumerate(KERNELS):
            res = tune_owl(X, A, Y.astype(float), folds, kernel, propensity=0.5)
            d = res["fit"].predict(X_test)
            rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(2, args.n, int(rep), k)))
            delivered = sample_texts(groups, d, rng)
            z_delivered = np.array([frame_of[t] for t in delivered])
            m = policy_metrics(b, g, z_delivered, z_opt)
            best_cv = max(gp["cv_value"] for gp in res["grid"])
            rows.append([METHOD_NAMES[kernel], kernel, args.n, int(rep), int(Y.sum()), res["selected_c"],
                         "" if res["selected_kparam"] is None else fmt(res["selected_kparam"]), fmt(best_cv),
                         int(res["fit"].constant is not None)] + [fmt(m[key]) for key in HEADER[9:]])
        print(f"n={args.n} rep={rep} done", file=sys.stderr, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, f"per_rep_n{args.n:04d}.csv"), HEADER, rows)


if __name__ == "__main__":
    main()
