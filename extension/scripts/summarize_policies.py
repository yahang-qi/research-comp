"""Summarize policy runs on the text experiment: means and Monte Carlo SEs by method and n.

Reads every per_rep_n*.csv in the given run directories (one row per method,
n and replication) and writes summary.csv and report.md to the output
directory. Outcome-model fit metrics and learned-attribute checks are summarized
for methods that report them.
"""

import argparse
import csv
import glob
import json
import os
from datetime import datetime, timezone

import numpy as np

from causal_policy.io import fmt, write_csv
from causal_policy.policy.evaluation import V_CONST, V_OPT, constant_frame_metrics

POLICY_METRICS = ("value", "regret_norm", "gain", "gain_fraction", "treatment_error", "share_gain")
FIT_METRICS = ("cv_log_loss", "rmse_mu", "expected_log_loss", "oracle_log_loss", "contrast_rmse", "contrast_sign_agreement", "crossfit_rmse_mu")
ATTRIBUTE_METRICS = ("recovery_all", "recovery_unobserved", "recovery_start", "completeness_train", "overridden")


def cell_text(c, key, digits):
    return f"{c[key + '_mean']:.{digits}f} ({c[key + '_se']:.{digits}f})"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", required=True, help="repeat for several runs")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--title", default="Policy comparison")
    args = parser.parse_args()

    rows = []
    for d in args.run_dir:
        for path in sorted(glob.glob(os.path.join(d, "per_rep_n*.csv"))):
            with open(path, encoding="utf-8") as f:
                rows += list(csv.DictReader(f))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    ns = sorted({int(r["n"]) for r in rows})

    summary = []
    for method in methods:
        for n in ns:
            sel = [r for r in rows if r["method"] == method and int(r["n"]) == n]
            if not sel:
                continue
            cell = {"method": method, "n": n, "reps": len(sel)}
            assert len({int(r["rep"]) for r in sel}) == len(sel), f"duplicate replications for {method}, n = {n}"
            if "constant_rule" in sel[0]:
                cell["constant_rules"] = sum(int(r["constant_rule"]) for r in sel)
            if "start" in sel[0]:
                cell["share_pca_start"] = float(np.mean([r["start"] == "pca" for r in sel]))
                cell["share_lam_comp_positive"] = float(np.mean([float(r["selected_lam_comp"]) > 0 for r in sel]))
            for key in POLICY_METRICS + FIT_METRICS + ATTRIBUTE_METRICS:
                if key in sel[0]:
                    # Average over replications where the metric is defined (e.g. no unobserved texts).
                    v = np.array([float(r[key]) for r in sel])
                    v = v[np.isfinite(v)]
                    cell[f"{key}_reps"] = len(v)
                    cell[f"{key}_mean"] = float(v.mean()) if len(v) else float("nan")
                    cell[f"{key}_se"] = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
            summary.append(cell)
    header = list(dict.fromkeys(k for c in summary for k in c))
    os.makedirs(args.out_dir, exist_ok=True)
    write_csv(os.path.join(args.out_dir, "summary.csv"), header, [[fmt(c[k]) if isinstance(c.get(k), float) else c.get(k, "") for k in header] for c in summary])

    const = constant_frame_metrics()
    lines = [
        f"# {args.title}",
        "",
        "Means over replications with Monte Carlo standard errors in parentheses. Value is the expected walking-completion probability on the 10,000-person test sample under the delivered texts, computed from the known outcome model. Regret is normalized by V*; gain fraction is (V - V_const)/(V* - V_const). Treatment error is the share of test people whose delivered text has a frame different from their optimal frame.",
        "",
        f"References: best constant frame (gain) V_const = {V_CONST:.4f} (regret {const['regret_norm']:.4f}, gain fraction 0); optimal policy V* = {V_OPT:.4f}; gain framing is optimal for 63.3% of the population.",
        "",
        "## Policy",
        "",
        "| Method | n | Reps | Value | Regret | Gain fraction | Treatment error | Share gain |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in summary:
        lines.append(f"| {c['method']} | {c['n']} | {c['reps']} | {cell_text(c, 'value', 4)} | {cell_text(c, 'regret_norm', 4)} | {cell_text(c, 'gain_fraction', 3)} | {cell_text(c, 'treatment_error', 3)} | {cell_text(c, 'share_gain', 3)} |")
    fit_cells = [c for c in summary if "rmse_mu_mean" in c]
    if fit_cells:
        lines += [
            "",
            "## Outcome model fit",
            "",
            "CV log loss is the held-out log loss at the selected penalty on the stored folds (tuning criterion, no truth used). The remaining columns compare the fitted h with the true probability mu: RMSE and expected log loss on the test sample with one design-drawn text per person (oracle = expected log loss of mu itself); the frame contrast (mean of h over gain texts minus over loss texts) against 2g(x), and the share of test people for whom it prefers the optimal frame; and the RMSE of the cross-fitted predictions on the training sample.",
            "",
            "| Method | n | CV log loss | Test RMSE to mu | Expected log loss | Oracle log loss | Contrast RMSE | Contrast sign agreement | Cross-fit RMSE to mu |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in fit_cells:
            lines.append(f"| {c['method']} | {c['n']} | {cell_text(c, 'cv_log_loss', 3)} | {cell_text(c, 'rmse_mu', 3)} | {cell_text(c, 'expected_log_loss', 3)} | {c['oracle_log_loss_mean']:.3f} | {cell_text(c, 'contrast_rmse', 3)} | {cell_text(c, 'contrast_sign_agreement', 3)} | {cell_text(c, 'crossfit_rmse_mu', 3)} |")
    attr_cells = [c for c in summary if "recovery_all_mean" in c]
    if attr_cells:
        lines += [
            "",
            "## Learned attributes",
            "",
            "Recovery is the agreement of the learned bank labels with the true frames after allowing one global sign reversal (0.5 = chance), over all 200 bank texts and over the bank texts not observed in the training sample (averaged over the replications that have such texts; their number is in brackets). The starting point's recovery is that of the start that won (the bank's first principal component, or a random start). Completeness loss is at the final fit on the training sample. Overridden is the number of test people whose chosen group had no observed training text.",
            "",
            "| Method | n | Recovery, all texts | Recovery, unobserved texts | Recovery, winning start | Completeness loss | Won by PC start | lambda_comp > 0 | Overridden |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in attr_cells:
            lines.append(f"| {c['method']} | {c['n']} | {cell_text(c, 'recovery_all', 3)} | {cell_text(c, 'recovery_unobserved', 3)} [{c['recovery_unobserved_reps']}] | {cell_text(c, 'recovery_start', 3)} | {cell_text(c, 'completeness_train', 4)} | {c['share_pca_start']:.2f} | {c['share_lam_comp_positive']:.2f} | {c['overridden_mean']:.1f} |")
    lines += ["", "## Files", ""] + [f"- Per-replicate results: `{d}/per_rep_n*.csv`" for d in args.run_dir] + [f"- Summary: `{args.out_dir}/summary.csv`", ""]
    with open(os.path.join(args.out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({"created_at_utc": datetime.now(timezone.utc).isoformat(), "run_dirs": args.run_dir, "methods": methods, "n_values": ns, "V_const": V_CONST, "V_opt": V_OPT}, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
