"""Figure: value, normalized regret and mis-treatment rate against the training sample size.

Rows are the three measures and columns the policy kernels. Each panel shows
joint text OWL and OWL with true attributes for that kernel, as means over
replications with +-1.96 Monte Carlo SE bars. Outcome regression is not shown:
it chooses among all observed texts, which is not feasible when the text space
is too large to enumerate. The mis-treatment rate is the percentage of test
people whose delivered text has a frame different from their optimal frame.
Dashed lines mark the best constant policy (yellow) and the optimal policy
(green); both are identified in the shared legend.
"""

import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from causal_policy.data.simulation import population_values  # noqa: E402
from causal_policy.policy.evaluation import V_CONST, V_OPT  # noqa: E402

N_VALUES = (30, 100, 200, 400, 800)
REPS = 100
KERNELS = (("Linear", "Linear"), ("Gaussian", "Gaussian"), ("Poly2", "Polynomial (degree 2)"))
# Navy and muted red, with distinct markers and line styles so identity does not rest on colour alone.
# The dashed series uses longer dashes than the thin gray reference lines.
SERIES = (
    ("Joint text OWL", "Joint-text-OWL-{k}", "#1f4e79", "o", "-"),
    ("OWL with true attributes", "OWL-true-{k}", "#c0392b", "s", (0, (6, 2.5))),
)
# The best constant frame (gain for everyone) mis-treats everyone for whom loss framing is optimal.
MIS_CONST = 100.0 * population_values()["share_loss_optimal"]
# (summary key, y label, scale, reference lines)
ROWS = (
    ("value", "Value", 1.0, ((V_CONST, "best constant policy"), (V_OPT, "optimal policy"))),
    ("regret_norm", "Normalized regret", 1.0, (((V_OPT - V_CONST) / V_OPT, "best constant policy"), (0.0, "optimal policy"))),
    ("treatment_error", "Mis-treatment rate (%)", 100.0, ((MIS_CONST, "best constant policy"), (0.0, "optimal policy"))),
)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
# Reference lines: a dark yellow (readable on white) for the best constant policy, green for the optimum.
REFERENCE_COLORS = {"best constant policy": "#d4a017", "optimal policy": "#2e7d32"}
REFERENCE_STYLE = {"linestyle": (0, (4, 3)), "linewidth": 1.0}


def read_summary(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {(r["method"], int(r["n"])): r for r in rows}


def series_values(summary, method, key):
    mean, se = [], []
    for n in N_VALUES:
        r = summary[(method, n)]
        assert int(r["reps"]) == REPS and int(r[f"{key}_reps"]) == REPS, f"{method}, n = {n}: expected {REPS} replications"
        mean.append(float(r[f"{key}_mean"]))
        se.append(float(r[f"{key}_se"]))
    return np.array(mean), np.array(se)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    summary = read_summary(args.summary)

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.6})
    fig, axes = plt.subplots(len(ROWS), len(KERNELS), figsize=(8.0, 7.1), sharex=True, sharey="row")
    offsets = np.array([-0.02, 0.02])  # small horizontal dodge in log10(n) so error bars do not overlap
    for ci, (kernel, kernel_label) in enumerate(KERNELS):
        for ri, (key, ylabel, scale, refs) in enumerate(ROWS):
            ax = axes[ri, ci]
            for value, label in refs:
                ax.axhline(value, color=REFERENCE_COLORS[label], zorder=2.5, **REFERENCE_STYLE)  # above grid lines, below data
            for si, (label, method, color, marker, linestyle) in enumerate(SERIES):
                mean, se = series_values(summary, method.format(k=kernel), key)
                mean, se = scale * mean, scale * se
                x = np.array(N_VALUES) * 10 ** offsets[si]
                ax.errorbar(x, mean, yerr=1.96 * se, color=color, marker=marker, markersize=4.5, linewidth=1.5, linestyle=linestyle,
                            elinewidth=1.0, capsize=0, markeredgecolor="white", markeredgewidth=0.6, label=label, zorder=3)
            ax.set_xscale("log")
            ax.set_xticks(N_VALUES)
            ax.set_xticklabels([str(n) for n in N_VALUES], fontsize=7.5)
            ax.minorticks_off()
            ax.grid(axis="y", color=GRID, linewidth=0.6, zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if ri == 0:
                ax.set_title(kernel_label + " kernel", fontsize=9, color=INK)
            if ci == 0:
                ax.set_ylabel(ylabel)
            if ri == len(ROWS) - 1:
                ax.set_xlabel("Training sample size n")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    for label in ("optimal policy", "best constant policy"):
        handles.append(plt.Line2D([], [], color=REFERENCE_COLORS[label], **REFERENCE_STYLE))
        labels.append(label.capitalize())
    fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, 1.0), handlelength=2.6)
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=1.2)
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "value_regret_gain_vs_n.pdf")
    fig.savefig(out)
    print(out)


if __name__ == "__main__":
    main()
