"""Joint text OWL on the synthetic text experiment, for one training sample size.

For each replication and policy kernel, tune (lambda_comp, lambda_reg) by the
held-out value over the stored folds, refit the encoder and the attribute
policy on the full training sample, and let G deliver to each test person a
text drawn uniformly from the observed training texts in the learned group
the policy chooses (or the other group if the chosen one has no observed
text). The policy is evaluated with the known outcome probabilities, and the
learned attributes are compared with the true frames, which are used only for
evaluation.
"""

import argparse
import json
import os
import sys

import numpy as np

from causal_policy.data.samples import load_bank_frames, load_test, load_training_reps
from causal_policy.data.text_bank import read_embeddings
from causal_policy.io import fmt, write_csv
from causal_policy.models.encoder import balanced_labels, initial_weights, scores
from causal_policy.models.joint_fit import tune_joint
from causal_policy.models.owl import KERNELS
from causal_policy.policy.evaluation import attribute_recovery, policy_metrics
from causal_policy.policy.generator import build_groups, sample_texts

N_RANDOM_STARTS = 4
METHOD_NAMES = {"linear": "Joint-text-OWL-Linear", "gaussian": "Joint-text-OWL-Gaussian", "poly2": "Joint-text-OWL-Poly2"}
POLICY_KEYS = ("value", "regret_norm", "gain", "gain_fraction", "treatment_error", "share_gain", "share_loss")
HEADER = ["method", "kernel", "n", "rep", "selected_lam_comp", "selected_lam_reg", "cv_value", "start", "outer_iterations",
          "group_size_plus", "group_size_minus", "overridden", "objective", "value_owl_train", "completeness_train",
          "recovery_all", "recovery_unobserved", "recovery_start"] + list(POLICY_KEYS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--kernels", nargs="+", default=list(KERNELS), choices=KERNELS)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-end", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    ids, E, _ = read_embeddings(args.embeddings)
    bank_ids = np.array(ids)
    row_of = {t: i for i, t in enumerate(ids)}
    frame_of = load_bank_frames(args.sim_dir)
    z_bank = np.array([frame_of[t] for t in ids])
    reps = load_training_reps(args.sim_dir, args.n)
    X_test, test_truth = load_test(args.sim_dir)
    starts = initial_weights(E, n_random=N_RANDOM_STARTS, seed=args.seed)

    rows, grids = [], {}
    for rep in range(args.rep_start, args.rep_end + 1):
        obs, _ = reps[rep]
        text_index = np.array([row_of[t] for t in obs.text_ids])
        unobserved = np.ones(len(ids), dtype=bool)
        unobserved[np.unique(text_index)] = False
        tuned = tune_joint(obs.X, obs.Y, obs.folds, text_index, E, bank_ids, args.kernels, starts)
        for kernel in args.kernels:
            res = tuned[kernel]
            fit = res["fit"]
            groups = build_groups(obs.text_ids, fit.labels[text_index])
            d = fit.policy.predict(X_test)
            missing = [c for c in (1, -1) if c not in groups]
            overridden = int(np.sum(d == missing[0])) if missing else 0
            if missing:
                d = np.where(d == missing[0], -missing[0], d)
            k = KERNELS.index(kernel)
            rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(5, args.n, rep, k)))
            delivered = sample_texts(groups, d, rng)
            pm = policy_metrics(test_truth.b, test_truth.g, np.array([frame_of[t] for t in delivered]), test_truth.z_opt)
            start_labels = balanced_labels(scores(starts[fit.start], E), bank_ids)[0]
            rows.append([METHOD_NAMES[kernel], kernel, args.n, rep, res["selected_lam_comp"], fmt(res["selected_lam_reg"]), fmt(res["cv_value"]),
                         "pca" if fit.start == 0 else f"random{fit.start}", fit.outer_iterations,
                         len(groups.get(1, [])), len(groups.get(-1, [])), overridden, fmt(fit.objective), fmt(fit.value_owl), fmt(fit.completeness),
                         fmt(attribute_recovery(fit.labels, z_bank)), fmt(attribute_recovery(fit.labels, z_bank, unobserved)),
                         fmt(attribute_recovery(start_labels, z_bank))] + [fmt(pm[key]) for key in POLICY_KEYS])
            grids[f"{kernel}_rep{rep}"] = res["grid"]
        print(f"n={args.n} rep={rep} done", file=sys.stderr, flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    tag = f"n{args.n:04d}_{'-'.join(args.kernels)}_reps{args.rep_start:03d}-{args.rep_end:03d}"
    write_csv(os.path.join(args.out_dir, f"per_rep_{tag}.csv"), HEADER, rows)
    with open(os.path.join(args.out_dir, f"cv_grid_{tag}.json"), "w", encoding="utf-8") as f:
        json.dump(grids, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    main()
