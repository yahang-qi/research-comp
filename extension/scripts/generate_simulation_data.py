"""Generate the synthetic training and test samples for the text treatment experiment.

Covariates X1 ~ U[-1,1], X2 ~ U[0,1], X3 ~ U[-1,1]; frame Z ~ Uniform{-1,+1}
independent of X; text T = t_{Z,J} with J ~ Uniform{1,...,K}; outcome
Y = 1{U <= mu(X,T)} with mu = b(x) + z(t) g(x). Observed data (what methods may
see) and simulator truth are written to separate folders.
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy

from causal_policy.data.simulation import (K, N_REPS, N_TEST, N_VALUES, N_FOLDS, PLAN_TOLERANCE, PLAN_VALUES, b_fn, check_population_values, check_training_sample,
                               draw_covariates, draw_training_sample, g_fn, population_values, rng_test_sample, rng_training_sample)
from causal_policy.data.text_bank import read_bank
from causal_policy.io import fmt, sha256_of, write_csv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    pop = population_values()
    check_population_values(pop)
    print(f"population values: {json.dumps(pop)}", file=sys.stderr)

    bank = read_bank(args.bank_dir, K)
    assert not os.path.exists(args.out_dir), f"{args.out_dir} already exists"
    obs_dir = os.path.join(args.out_dir, "observed")
    truth_dir = os.path.join(args.out_dir, "truth")
    os.makedirs(obs_dir)
    os.makedirs(truth_dir)

    write_csv(os.path.join(truth_dir, "bank_index.csv"), ["z", "j", "text_id"], [[z, j + 1, t] for z in (1, -1) for j, t in enumerate(bank[z])])

    sums = {"rows": 0, "x1": 0.0, "x2": 0.0, "x3": 0.0, "z_gain": 0, "y": 0, "j_counts": {1: np.zeros(K, int), -1: np.zeros(K, int)}}
    obs_train_header = ["rep", "person_id", "fold", "x1", "x2", "x3", "text_id", "y"]
    truth_train_header = ["rep", "person_id", "z", "j", "u", "b", "g", "mu"]
    for n in N_VALUES:
        obs_rows, truth_rows = [], []
        for rep in range(1, N_REPS + 1):
            rng = rng_training_sample(args.seed, n, rep)
            s = draw_training_sample(rng, n, bank)
            check_training_sample(s, n, bank)
            for i in range(n):
                obs_rows.append([rep, i + 1, int(s["fold"][i]), fmt(s["x1"][i]), fmt(s["x2"][i]), fmt(s["x3"][i]), s["text_id"][i], int(s["y"][i])])
                truth_rows.append([rep, i + 1, int(s["z"][i]), int(s["j"][i]), fmt(s["u"][i]), fmt(s["b"][i]), fmt(s["g"][i]), fmt(s["mu"][i])])
            sums["rows"] += n
            for key in ("x1", "x2", "x3", "y"):
                sums[key] += float(s[key].sum())
            sums["z_gain"] += int((s["z"] == 1).sum())
            for z in (1, -1):
                sums["j_counts"][z] += np.bincount(s["j"][s["z"] == z] - 1, minlength=K)
        assert len(obs_rows) == len(truth_rows) == n * N_REPS
        write_csv(os.path.join(obs_dir, f"train_n{n:04d}.csv"), obs_train_header, obs_rows)
        write_csv(os.path.join(truth_dir, f"train_n{n:04d}.csv"), truth_train_header, truth_rows)

    rng = rng_test_sample(args.seed)
    x1, x2, x3 = draw_covariates(rng, N_TEST)
    b, g = b_fn(x2, x3), g_fn(x1, x2)
    assert (b - np.abs(g) >= 0.0).all() and (b + np.abs(g) <= 0.90 + 1e-12).all()
    write_csv(os.path.join(obs_dir, "test.csv"), ["person_id", "x1", "x2", "x3"], [[i + 1, fmt(x1[i]), fmt(x2[i]), fmt(x3[i])] for i in range(N_TEST)])
    write_csv(os.path.join(truth_dir, "test.csv"), ["person_id", "b", "g", "mu_gain", "mu_loss", "z_opt"],
              [[i + 1, fmt(b[i]), fmt(g[i]), fmt(b[i] + g[i]), fmt(b[i] - g[i]), 1 if g[i] > 0 else -1] for i in range(N_TEST)])

    rows = sums["rows"]
    j_freq = {z: sums["j_counts"][z] / sums["j_counts"][z].sum() for z in (1, -1)}
    sanity = {
        "train_rows": rows,
        "train_mean_x1": sums["x1"] / rows,
        "train_mean_x2": sums["x2"] / rows,
        "train_mean_x3": sums["x3"] / rows,
        "train_share_z_gain": sums["z_gain"] / rows,
        "train_mean_y": sums["y"] / rows,
        "train_text_index_share_range_gain": [float(j_freq[1].min()), float(j_freq[1].max())],
        "train_text_index_share_range_loss": [float(j_freq[-1].min()), float(j_freq[-1].max())],
        "test_mean_b_plus_abs_g": float(np.mean(b + np.abs(g))),
        "test_mean_b_plus_g": float(np.mean(b + g)),
        "test_mean_b_minus_g": float(np.mean(b - g)),
        "test_share_gain_optimal": float(np.mean(g > 0)),
    }
    print(f"sanity: {json.dumps(sanity)}", file=sys.stderr)

    files = sorted(os.path.relpath(os.path.join(d, f), args.out_dir) for d, _, fs in os.walk(args.out_dir) for f in fs)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Synthetic training and test samples for the text treatment experiment. Methods read only observed/; truth/ is for data generation and evaluation.",
        "design": {
            "covariates": "X1 ~ U[-1,1], X2 ~ U[0,1], X3 ~ U[-1,1], independent; p = 3",
            "frame": "Z ~ Uniform{-1,+1}, independent of X; +1 = gain, -1 = loss",
            "text": "J ~ Uniform{1,...,K}; T = t_{Z,J}; within each frame texts are indexed j = 1,...,K in increasing text_id order (truth/bank_index.csv)",
            "K": K,
            "baseline": "b(x) = 0.40 + 0.10 x2 + 0.05 x3",
            "frame_effect": "g(x) = 0.35 tanh(2 [x1 + 0.80 (1 - x2)^2])",
            "outcome_probability": "mu(x, t) = b(x) + z(t) g(x)",
            "outcome": "U ~ U[0,1], Y = 1{U <= mu}",
            "optimal_frame": "z_opt(x) = +1 if g(x) > 0, -1 otherwise",
            "n_values": list(N_VALUES),
            "replications": N_REPS,
            "test_size": N_TEST,
            "test_sample": "covariates only, shared across replications and methods; values and regret use the known b and g",
            "folds": f"{N_FOLDS} folds per training sample, a random permutation of (i mod {N_FOLDS}) + 1, sizes as equal as possible",
        },
        "randomness": {
            "root_seed": args.seed,
            "generator": "numpy.random.default_rng (PCG64)",
            "training_sample_seed": "SeedSequence(root_seed, spawn_key=(0, n, rep))",
            "test_sample_seed": "SeedSequence(root_seed, spawn_key=(1,))",
            "draw_order_training": "x1, x2, x3, z, j, u, fold permutation",
            "draw_order_test": "x1, x2, x3",
        },
        "population_values": {"computed": pop, "plan": PLAN_VALUES, "tolerance": PLAN_TOLERANCE, "method": "closed-form integral over x1, adaptive quadrature over x2"},
        "sanity": sanity,
        "columns": {
            "observed/train_nXXXX.csv": {"rep": "replication 1-100", "person_id": "1-n within replication", "fold": "tuning fold 1-5", "x1": "motivation", "x2": "confidence", "x3": "baseline walking activity", "text_id": "assigned text", "y": "walk completed (0/1)"},
            "truth/train_nXXXX.csv": {"rep": "replication", "person_id": "person", "z": "true frame of the assigned text", "j": "index of the text within its frame", "u": "uniform draw for y", "b": "b(x)", "g": "g(x)", "mu": "outcome probability"},
            "observed/test.csv": {"person_id": "1-10000", "x1": "motivation", "x2": "confidence", "x3": "baseline walking activity"},
            "truth/test.csv": {"person_id": "person", "b": "b(x)", "g": "g(x)", "mu_gain": "b + g", "mu_loss": "b - g", "z_opt": "optimal frame"},
            "truth/bank_index.csv": {"z": "frame", "j": "index within frame", "text_id": "text"},
        },
        "float_format": "Python repr of float64 (round-trip exact)",
        "inputs": {
            "texts": os.path.join(args.bank_dir, "texts.jsonl"),
            "texts_sha256": sha256_of(os.path.join(args.bank_dir, "texts.jsonl")),
            "simulator_labels": os.path.join(args.bank_dir, "simulator_labels.jsonl"),
            "simulator_labels_sha256": sha256_of(os.path.join(args.bank_dir, "simulator_labels.jsonl")),
        },
        "files_sha256": {f: sha256_of(os.path.join(args.out_dir, f)) for f in files},
        "log_path": args.log_path,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"wrote {len(files)} data files and metadata.json to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
