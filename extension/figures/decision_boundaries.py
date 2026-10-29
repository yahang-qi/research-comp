"""Figure: fitted decision boundaries of representative runs against the true boundary.

Rows are methods (OWL with true attributes, joint text OWL) at n = 200 and
800, and columns the policy kernels (linear, Gaussian, poly2). For each method,
kernel and n the run shown is the median-regret replication: the 100 replications are sorted by
their recorded normalized regret in increasing order (ties by the smaller
replication number) and the 50th is taken. That replication is refitted from
its seed with the same library calls as the run scripts, and the script stops
unless the refitted regret equals the recorded one. Each panel shades, over
(x1, x2) with x3 = 0, the probability that the delivered text is gain-framed;
the solid line is its 0.5 contour and the dashed line the true boundary
x1 = -0.80 (1 - x2)^2.
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from causal_policy.data.samples import load_bank_frames, load_test, load_training_reps  # noqa: E402
from causal_policy.data.text_bank import read_embeddings  # noqa: E402
from causal_policy.io import fmt, write_csv  # noqa: E402
from causal_policy.models.encoder import initial_weights  # noqa: E402
from causal_policy.models.joint_fit import tune_joint  # noqa: E402
from causal_policy.models.owl import KERNELS, tune_owl  # noqa: E402
from causal_policy.policy.evaluation import policy_metrics  # noqa: E402
from causal_policy.policy.generator import build_groups, sample_texts  # noqa: E402

N_VALUES = (200, 800)
# (method prefix in the per-replicate files, row label)
METHODS = (("OWL-true", "OWL with true attributes"), ("Joint-text-OWL", "Joint text OWL"))
# (kernel, suffix in the method name, column title)
KERNEL_COLUMNS = (("linear", "Linear", "Linear kernel"), ("gaussian", "Gaussian", "Gaussian kernel"), ("poly2", "Poly2", "Polynomial (degree 2) kernel"))
ROWS = [(prefix, label, n) for prefix, label in METHODS for n in N_VALUES]
N_RANDOM_STARTS = 4  # as in scripts/run_joint_text_owl.py
GRID = 201
SEED = 0
INK, MUTED = "#0b0b0b", "#52514e"
# Diverging pair of the reference palette with a neutral gray midpoint, as light tints for filled regions.
REGION_CMAP = LinearSegmentedColormap.from_list("loss_gain", ["#f3b8b7", "#f0efec", "#a9c9ef"])


def median_regret_rep(rows):
    """The 50th replication when sorted by recorded regret (increasing), ties by replication number."""
    assert len(rows) == 100 and len({int(r["rep"]) for r in rows}) == 100
    ordered = sorted(rows, key=lambda r: (float(r["regret_norm"]), int(r["rep"])))
    return ordered[49]


def read_rows(run_dirs):
    rows = []
    for d in run_dirs:
        for path in sorted(glob.glob(os.path.join(d, "per_rep_n*.csv"))):
            with open(path, encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
    return rows


def deliver(groups, d, seed_key):
    rng = np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=seed_key))
    return sample_texts(groups, d, rng)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--sim-dir", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    rows = read_rows(args.run_dirs)
    ids, E, _ = read_embeddings(args.embeddings)
    bank_ids = np.array(ids)
    row_of = {t: i for i, t in enumerate(ids)}
    frame_of = load_bank_frames(args.sim_dir)
    X_test, truth = load_test(args.sim_dir)
    starts = initial_weights(E, n_random=N_RANDOM_STARTS, seed=SEED)

    x1g, x2g = np.meshgrid(np.linspace(-1, 1, GRID), np.linspace(0, 1, GRID))
    X_grid = np.column_stack([x1g.ravel(), x2g.ravel(), np.zeros(x1g.size)])

    def regret(z_delivered):
        return policy_metrics(truth.b, truth.g, z_delivered, truth.z_opt)["regret_norm"]

    panels, records = {}, []
    for n in N_VALUES:
        reps = load_training_reps(args.sim_dir, n)
        for prefix, _ in METHODS:
            for kernel, suffix, _ in KERNEL_COLUMNS:
                method = f"{prefix}-{suffix}"
                k = KERNELS.index(kernel)
                chosen = median_regret_rep([r for r in rows if r["method"] == method and int(r["n"]) == n])
                rep = int(chosen["rep"])
                obs, tr = reps[rep]
                if prefix == "OWL-true":
                    fit = tune_owl(obs.X, tr.z, obs.Y.astype(float), obs.folds, kernel, propensity=0.5)["fit"]
                    groups = build_groups(obs.text_ids, tr.z)
                    delivered = deliver(groups, fit.predict(X_test), (2, n, rep, k))
                    p_gain = (fit.predict(X_grid) == 1).astype(float)
                else:
                    text_index = np.array([row_of[t] for t in obs.text_ids])
                    fit = tune_joint(obs.X, obs.Y, obs.folds, text_index, E, bank_ids, [kernel], starts)[kernel]["fit"]
                    groups = build_groups(obs.text_ids, fit.labels[text_index])
                    missing = [c for c in (1, -1) if c not in groups]

                    def chosen_groups(X, fit=fit, missing=missing):
                        d = fit.policy.predict(X)
                        return np.where(d == missing[0], -missing[0], d) if missing else d

                    delivered = deliver(groups, chosen_groups(X_test), (5, n, rep, k))
                    share_gain = {c: np.mean([frame_of[t] == 1 for t in texts]) for c, texts in groups.items()}
                    p_gain = np.array([share_gain[c] for c in chosen_groups(X_grid)])
                refit = regret(np.array([frame_of[t] for t in delivered]))
                assert fmt(refit) == chosen["regret_norm"], f"{method}, n = {n}, rep {rep}: refitted regret {refit!r} != recorded {chosen['regret_norm']}"
                panels[(prefix, n, kernel)] = (rep, refit, p_gain.reshape(x1g.shape))
                records.append([method, n, rep, chosen["regret_norm"], fmt(refit)])
                print(f"{method} n={n}: replication {rep}, regret {refit:.4f}", flush=True)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(len(ROWS), len(KERNEL_COLUMNS), figsize=(8.6, 10.2), sharex=True, sharey=True)
    x2_line = np.linspace(0, 1, 400)
    for ri, (prefix, row_label, n) in enumerate(ROWS):
        for ci, (kernel, _, col_title) in enumerate(KERNEL_COLUMNS):
            ax = axes[ri, ci]
            rep, reg, P = panels[(prefix, n, kernel)]
            im = ax.imshow(P, origin="lower", extent=(-1, 1, 0, 1), aspect="auto", cmap=REGION_CMAP, vmin=0, vmax=1, interpolation="nearest")
            if 0 < P.mean() < 1:
                ax.contour(x1g, x2g, P, levels=[0.5], colors=INK, linewidths=1.2)
            ax.plot(-0.80 * (1 - x2_line) ** 2, x2_line, color=INK, linestyle=(0, (4, 3)), linewidth=1.0)
            ax.set_xlim(-1, 1)
            ax.set_ylim(0, 1)
            ax.text(0.97, 0.04, f"replication {rep}\nregret = {reg:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
                    color=INK, bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
            if ri == 0:
                ax.set_title(col_title, fontsize=9, color=INK)
            if ri == len(ROWS) - 1:
                ax.set_xlabel(r"Motivation $x_1$")
            if ci == 0:
                ax.set_ylabel(f"{row_label}, n = {n}\n" + r"Confidence $x_2$", fontsize=8.5)
    handles = [plt.Line2D([], [], color=INK, linewidth=1.2, label="fitted boundary (0.5 contour)"),
               plt.Line2D([], [], color=INK, linestyle=(0, (4, 3)), linewidth=1.0, label=r"true boundary $x_1 = -0.80(1-x_2)^2$")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.45, 1.0))
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    cax = fig.add_axes((0.915, 0.25, 0.015, 0.5))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Probability the delivered text is gain-framed", fontsize=8)
    cb.outline.set_linewidth(0.5)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "decision_boundaries.pdf")
    fig.savefig(out)
    write_csv(os.path.join(args.out_dir, "decision_boundaries_runs.csv"), ["method", "n", "rep", "regret_recorded", "regret_refitted"], records)
    print(out)


if __name__ == "__main__":
    main()
