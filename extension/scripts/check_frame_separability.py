"""Check whether the saved text embeddings preserve the gain/loss distinction.

Fits an L2-regularized logistic regression predicting the true frame from the
cached embeddings u(t), using nested cross-validation on the text bank: five
stratified outer folds; within each outer training set, five-fold CV selects
the penalty by log loss; the refitted model predicts the outer test fold.
The frame labels are used only for this check.
"""

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import scipy
import sklearn

from causal_policy.data.text_bank import read_embeddings, read_labels
from causal_policy.io import fmt, sha256_of, write_csv
from causal_policy.text.separability import C_GRID, N_INNER, N_OUTER, nested_cv, summarize, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--embedding-metadata", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    ids, X, feature_names = read_embeddings(args.embeddings)
    labels = read_labels(args.labels)
    assert len(ids) == len(set(ids)) == 200, "expected 200 unique texts"
    assert set(ids) == set(labels), "text_id sets of embeddings and labels differ"
    assert X.shape == (200, 128) and np.isfinite(X).all()
    y = np.array([1 if labels[t] == 1 else 0 for t in ids])
    n_gain, n_loss = int(y.sum()), int(len(y) - y.sum())
    assert n_gain == n_loss == 100

    with open(args.embedding_metadata, encoding="utf-8") as f:
        emb_meta = json.load(f)
    assert emb_meta["output_sha256"] == sha256_of(args.embeddings), "embeddings do not match their metadata"

    prob, fold_of, folds = nested_cv(X, y, args.seed, test_size_per_frame=20)
    pred, pooled, fold_summary = summarize(y, prob, folds)

    os.makedirs(args.out_dir, exist_ok=True)
    pred_path = os.path.join(args.out_dir, "predictions.csv")
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    report_path = os.path.join(args.out_dir, "report.md")
    order = np.argsort(ids)
    write_csv(pred_path, ["text_id", "outer_fold", "y_true", "prob_gain", "y_pred"],
              [[ids[i], int(fold_of[i]), int(y[i]), fmt(prob[i]), int(pred[i])] for i in order])

    metrics = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "n_texts": len(ids),
        "n_features": X.shape[1],
        "n_gain": n_gain,
        "n_loss": n_loss,
        "label_coding": "y = 1 if Z = +1 (gain), y = 0 if Z = -1 (loss)",
        "n_outer_folds": N_OUTER,
        "n_inner_folds": N_INNER,
        "outer_test_size_per_frame": 20,
        "outer_train_size": len(ids) - len(ids) // N_OUTER,
        "model": "StandardScaler + L2-penalized LogisticRegression (lbfgs)",
        "selection_criterion": "lowest mean inner-CV log loss",
        "classification_threshold": 0.5,
        "C_grid": C_GRID,
        "embedding_model": {k: emb_meta[k] for k in ("model_id", "model_revision", "embedding_dim")},
        "inputs": {
            "embeddings": args.embeddings,
            "embeddings_sha256": sha256_of(args.embeddings),
            "labels": args.labels,
            "labels_sha256": sha256_of(args.labels),
        },
        "folds": folds,
        "pooled": pooled,
        "fold_summary": fold_summary,
        "outputs": {"metrics": metrics_path, "predictions": pred_path, "report": report_path},
        "log_path": args.log_path,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")
    write_report(report_path, metrics)

    print(f"pooled accuracy {pooled['accuracy']:.3f} ({pooled['n_correct']}/{len(y)}), AUC {pooled['auc']:.3f}, log loss {pooled['log_loss']:.3f}", file=sys.stderr)
    print(f"selected C per fold: {[f['selected_C'] for f in folds]}", file=sys.stderr)


if __name__ == "__main__":
    main()
