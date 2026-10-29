"""Diagnostic: does the completeness loss single out the true frames?

For each replication, compute the cross-fitted outcome predictions h^(-i) and,
with kappa = 0, the completeness loss of
  - the true-frame partition of the bank,
  - random balanced partitions of the bank,
  - the partition reached by straight-through descent on w using the
    completeness term only (no OWL term), started at the bank's first
    principal component, and started at seeded random unit vectors.
The frame labels are used only for evaluation.

  run        one n and a range of replications -> per_rep_nXXXX_repsAAA-BBB.csv
  summarize  combine per_rep files -> summary.csv and report.md
"""

import argparse
import csv
import glob
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

from causal_policy.data.samples import load_bank_frames, load_training_reps
from causal_policy.data.text_bank import read_embeddings
from causal_policy.io import fmt, write_csv
from causal_policy.models.completeness import completeness_loss
from causal_policy.models.encoder import balanced_labels, initial_weights, scores
from causal_policy.models.features import quadratic_covariates
from causal_policy.models.joint_objective import joint_objective
from causal_policy.models.outcome_model import cross_fit

HEADER = ["n", "rep", "loss_true", "loss_random_mean", "loss_random_sd", "share_random_below_true",
          "loss_start", "agreement_start", "loss_descent", "agreement_descent", "steps_run",
          "agreement_random_start_mean", "agreement_random_descent_mean", "loss_random_descent_best", "agreement_random_descent_best"]
METRICS = HEADER[2:]


def agreement(labels, z_bank):
    """Agreement with the true frames, allowing one global sign reversal."""
    a = float(np.mean(labels == z_bank))
    return max(a, 1.0 - a)


def descend(w0, U_bank, bank_ids, text_index, h_cf, R, steps, step_size):
    """Normalized gradient steps on the completeness-only objective, with w kept on the unit sphere.
    Returns the labels with the smallest completeness loss seen, that loss, and the number of steps run."""
    w = w0.copy()
    Y0, d0 = np.zeros(len(h_cf)), np.zeros(len(h_cf))
    best = None
    unchanged, prev, steps_run = 0, None, 0
    for step in range(steps + 1):
        res = joint_objective(w, U_bank, bank_ids, text_index, Y0, d0, h_cf, R, lam_comp=1.0, lam_reg=0.0)
        if best is None or res["completeness"] < best[1]:
            best = (res["labels"].copy(), res["completeness"])
        unchanged = unchanged + 1 if prev is not None and np.array_equal(prev, res["labels"]) else 0
        prev = res["labels"]
        steps_run = step
        g = res["grad"]
        if step == steps or unchanged >= 20 or not np.any(g):
            break
        w = w - step_size * g / np.linalg.norm(g)
        w /= np.linalg.norm(w)
    return best[0], best[1], steps_run


def run(args):
    ids, E, _ = read_embeddings(args.embeddings)
    bank_ids = np.array(ids)
    row_of = {t: i for i, t in enumerate(ids)}
    frame_of = load_bank_frames(args.sim_dir)
    z_bank = np.array([frame_of[t] for t in ids])
    reps = load_training_reps(args.sim_dir, args.n)
    starts = initial_weights(E, n_random=args.n_starts, seed=args.seed)
    w_start, random_starts = starts[0], starts[1:]
    rows = []
    for rep in range(args.rep_start, args.rep_end + 1):
        observed, _ = reps[rep]
        text_index = np.array([row_of[t] for t in observed.text_ids])
        R = quadratic_covariates(observed.X)
        h_cf = cross_fit(observed.X, E[text_index], observed.Y, observed.folds)

        def loss_of(bank_labels):
            return completeness_loss(h_cf, R, bank_labels[text_index].astype(float), 0.0)[0]

        loss_true = loss_of(z_bank)
        rng = np.random.default_rng(np.random.SeedSequence(args.seed, spawn_key=(4, args.n, rep)))
        random_losses = np.array([loss_of(np.where(rng.permutation(len(ids)) < len(ids) // 2, 1, -1)) for _ in range(args.n_random)])
        start_labels, _ = balanced_labels(scores(w_start, E), bank_ids)
        labels, loss_desc, steps_run = descend(w_start, E, bank_ids, text_index, h_cf, R, args.steps, args.step_size)
        rand_start_agree, rand_desc = [], []
        for w0 in random_starts:
            rand_start_agree.append(agreement(balanced_labels(scores(w0, E), bank_ids)[0], z_bank))
            lab, loss_r, _ = descend(w0, E, bank_ids, text_index, h_cf, R, args.steps, args.step_size)
            rand_desc.append((loss_r, agreement(lab, z_bank)))
        best_rand = min(rand_desc)
        rows.append([args.n, rep, fmt(loss_true), fmt(random_losses.mean()), fmt(random_losses.std(ddof=1)),
                     fmt(np.mean(random_losses <= loss_true)), fmt(loss_of(start_labels)), fmt(agreement(start_labels, z_bank)),
                     fmt(loss_desc), fmt(agreement(labels, z_bank)), steps_run,
                     fmt(np.mean(rand_start_agree)), fmt(np.mean([a for _, a in rand_desc])), fmt(best_rand[0]), fmt(best_rand[1])])
        print(f"n={args.n} rep={rep} done", file=sys.stderr, flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, f"per_rep_n{args.n:04d}_reps{args.rep_start:03d}-{args.rep_end:03d}.csv"), HEADER, rows)


def summarize(args):
    rows = []
    for path in sorted(glob.glob(os.path.join(args.out_dir, "per_rep_n*.csv"))):
        with open(path, encoding="utf-8") as f:
            rows += list(csv.DictReader(f))
    ns = sorted({int(r["n"]) for r in rows})
    summary = []
    for n in ns:
        sel = [r for r in rows if int(r["n"]) == n]
        cell = {"n": n, "reps": len(sel)}
        for key in METRICS:
            v = np.array([float(r[key]) for r in sel])
            cell[f"{key}_mean"], cell[f"{key}_se"] = float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))
        cell["reps_true_below_all_random"] = sum(float(r["share_random_below_true"]) == 0.0 for r in sel)
        summary.append(cell)
    header = list(summary[0])
    write_csv(os.path.join(args.out_dir, "summary.csv"), header, [[fmt(c[k]) if isinstance(c[k], float) else c[k] for k in header] for c in summary])

    def ms(c, key, digits=4):
        return f"{c[key + '_mean']:.{digits}f} ({c[key + '_se']:.{digits}f})"

    lines = [
        "# Completeness-loss diagnostic",
        "",
        "Does the completeness loss single out the true gain/loss partition of the bank? For each replication, the cross-fitted outcome predictions h^(-i) are computed from the observed training data, and the completeness loss (ridge penalty kappa = 0) is evaluated for the true-frame partition, for random balanced partitions of the 200 bank texts, for the partition given by the bank's first principal component (the encoder's starting point), and for the partition reached by straight-through descent on the encoder weights with the completeness term only (no OWL term), started at the first principal component and at seeded random unit vectors. Descent uses normalized gradient steps of size "
        f"{args.step_size} with the weights kept at unit norm, at most {args.steps} steps, stopping after 20 steps without a change in the partition, and keeps the partition with the smallest loss. True frames are used only for evaluation. Agreement is with the true frames after allowing one global sign reversal (0.5 = chance).",
        "",
        f"Random partitions per replication: {args.n_random}. Random starts per replication: {args.n_starts}. Entries are means over replications with Monte Carlo standard errors in parentheses.",
        "",
        "| n | Reps | Loss, true partition | Loss, random partitions | Share of random partitions with loss <= true | Replications where true beats all random | Loss, PCA start | Agreement, PCA start | Loss, after descent | Agreement, after descent |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in summary:
        lines.append(f"| {c['n']} | {c['reps']} | {ms(c, 'loss_true', 5)} | {ms(c, 'loss_random_mean', 5)} | {ms(c, 'share_random_below_true', 3)} | {c['reps_true_below_all_random']}/{c['reps']} | {ms(c, 'loss_start', 5)} | {ms(c, 'agreement_start', 3)} | {ms(c, 'loss_descent', 5)} | {ms(c, 'agreement_descent', 3)} |")
    lines += [
        "",
        "Descent from random starts (no principal-component head start):",
        "",
        "| n | Agreement, random starts | Agreement after descent, mean over starts | Agreement after descent, start with smallest loss | Loss after descent, smallest |",
        "|---|---|---|---|---|",
    ]
    for c in summary:
        lines.append(f"| {c['n']} | {ms(c, 'agreement_random_start_mean', 3)} | {ms(c, 'agreement_random_descent_mean', 3)} | {ms(c, 'agreement_random_descent_best', 3)} | {ms(c, 'loss_random_descent_best', 5)} |")
    lines += ["", "## Files", "", f"- Per-replicate results: `{args.out_dir}/per_rep_n*.csv`", f"- Summary: `{args.out_dir}/summary.csv`", ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(), "n_random": args.n_random, "n_starts": args.n_starts, "steps": args.steps, "step_size": args.step_size, "seed": args.seed}, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("run", "summarize"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sim-dir", default="dataset/simulation")
    parser.add_argument("--embeddings", default="dataset/text_bank/embeddings.csv")
    parser.add_argument("--n", type=int)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--rep-end", type=int, default=20)
    parser.add_argument("--n-random", type=int, default=200)
    parser.add_argument("--n-starts", type=int, default=4, help="random starts besides the principal component")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--step-size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    run(args) if args.command == "run" else summarize(args)


if __name__ == "__main__":
    main()
