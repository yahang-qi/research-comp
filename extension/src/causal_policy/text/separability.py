"""Check whether the text embeddings preserve the gain/loss distinction.

An L2-regularized logistic regression predicts the true frame from u(t) under
nested cross-validation on the text bank: stratified outer folds; within each
outer training set, inner CV selects the penalty by log loss; the refitted
model predicts the outer test fold.
"""

import numpy as np
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_OUTER = 5
N_INNER = 5
C_GRID = [float(10.0 ** e) for e in np.arange(-4.0, 4.0 + 0.25, 0.5)]


def fold_metrics(y, p):
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, (p >= 0.5).astype(int))),
        "auc": float(roc_auc_score(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
    }


def nested_cv(X, y, seed, test_size_per_frame=None):
    """Return held-out probabilities, each text's outer fold, and per-fold results."""
    outer = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    inner = StratifiedKFold(n_splits=N_INNER, shuffle=True, random_state=seed)
    prob = np.full(len(y), np.nan)
    fold_of = np.full(len(y), -1)
    folds = []
    for k, (train, test) in enumerate(outer.split(X, y), start=1):
        assert not set(train) & set(test)
        if test_size_per_frame is not None:
            assert int(y[test].sum()) == test_size_per_frame and int((1 - y[test]).sum()) == test_size_per_frame
        pipe = make_pipeline(StandardScaler(), LogisticRegression(solver="lbfgs", max_iter=100000))
        search = GridSearchCV(pipe, {"logisticregression__C": C_GRID}, scoring="neg_log_loss", cv=inner, refit=True)
        search.fit(X[train], y[train])
        assert search.best_estimator_.named_steps["standardscaler"].n_samples_seen_ == len(train)
        p = search.predict_proba(X[test])[:, 1]
        prob[test] = p
        fold_of[test] = k
        folds.append({
            "fold": k,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "inner_cv_mean_log_loss": {repr(c): float(-s) for c, s in zip(C_GRID, search.cv_results_["mean_test_score"])},
            "selected_C": float(search.best_params_["logisticregression__C"]),
            "test": fold_metrics(y[test], p),
        })
    assert np.isfinite(prob).all() and (fold_of > 0).all(), "some text lacks a held-out prediction"
    return prob, fold_of, folds


def summarize(y, prob, folds):
    """Pooled held-out metrics with an exact binomial CI and p-value, and fold mean/SD."""
    pred = (prob >= 0.5).astype(int)
    n_correct = int((pred == y).sum())
    ci = binomtest(n_correct, len(y), 0.5).proportion_ci(confidence_level=0.95, method="exact")
    pooled = fold_metrics(y, prob)
    pooled.update({
        "n_correct": n_correct,
        "accuracy_ci95": [float(ci.low), float(ci.high)],
        "accuracy_pvalue_vs_half": float(binomtest(n_correct, len(y), 0.5, alternative="greater").pvalue),
    })
    fold_summary = {
        key: {"mean": float(np.mean([f["test"][key] for f in folds])), "sd": float(np.std([f["test"][key] for f in folds], ddof=1))}
        for key in ("accuracy", "auc", "log_loss")
    }
    return pred, pooled, fold_summary


def write_report(path, m):
    pooled = m["pooled"]
    acc = pooled["accuracy"]
    lines = [
        "# Frame separability of the saved text embeddings",
        "",
        "## Purpose",
        "",
        "Check whether the cached text embeddings u(t) preserve the gain/loss distinction, by fitting an L2-regularized logistic regression that predicts the true frame from u(t) and measuring its held-out performance. The frame labels are used only for this check, never to train the encoder.",
        "",
        "## Data",
        "",
        f"- Embeddings: `{m['inputs']['embeddings']}` ({m['n_texts']} texts x {m['n_features']} features), SHA-256 `{m['inputs']['embeddings_sha256']}`.",
        f"- Embedding model: `{m['embedding_model']['model_id']}` at revision `{m['embedding_model']['model_revision']}`, {m['embedding_model']['embedding_dim']}-dim Matryoshka truncation, L2-normalized.",
        f"- Labels: `Z` from `{m['inputs']['labels']}`, SHA-256 `{m['inputs']['labels_sha256']}`; y = 1 for gain (Z = +1), y = 0 for loss (Z = -1). Counts: {m['n_gain']} gain, {m['n_loss']} loss.",
        "",
        "## Method",
        "",
        f"- Nested cross-validation with seed {m['seed']}. The texts are split into {m['n_outer_folds']} stratified outer folds of {m['outer_test_size_per_frame']} gain and {m['outer_test_size_per_frame']} loss texts each.",
        f"- For each outer fold, the other {m['outer_train_size']} texts form the training set. Within it, stratified {m['n_inner_folds']}-fold CV selects the inverse penalty C from 10^-4, 10^-3.5, ..., 10^4 ({len(m['C_grid'])} values) by the lowest mean CV log loss.",
        "- Model: standardization fitted on the training texts, followed by L2-penalized logistic regression (lbfgs). It is refitted on all training texts with the selected C and predicts the outer test texts.",
        "- Every text receives exactly one held-out prediction. Metrics are computed on the pooled held-out predictions and per outer fold; a predicted probability of at least 0.5 is classified as gain.",
        "",
        "## Results",
        "",
        "Pooled held-out predictions (chance level 0.5 for accuracy and AUC):",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Accuracy | {acc:.3f} ({pooled['n_correct']}/{pooled['n']}) |",
        f"| Accuracy exact 95% CI | [{pooled['accuracy_ci95'][0]:.3f}, {pooled['accuracy_ci95'][1]:.3f}] |",
        f"| One-sided binomial p-value vs 0.5 | {pooled['accuracy_pvalue_vs_half']:.3g} |",
        f"| ROC AUC | {pooled['auc']:.3f} |",
        f"| Log loss | {pooled['log_loss']:.3f} |",
        "",
        "Per outer fold:",
        "",
        "| Outer fold | Selected C | Accuracy | AUC | Log loss |",
        "|---|---|---|---|---|",
    ]
    for f in m["folds"]:
        t = f["test"]
        lines.append(f"| {f['fold']} | {f['selected_C']:.3g} | {t['accuracy']:.3f} | {t['auc']:.3f} | {t['log_loss']:.3f} |")
    s = m["fold_summary"]
    lines += [
        f"| Mean (SD) | | {s['accuracy']['mean']:.3f} ({s['accuracy']['sd']:.3f}) | {s['auc']['mean']:.3f} ({s['auc']['sd']:.3f}) | {s['log_loss']['mean']:.3f} ({s['log_loss']['sd']:.3f}) |",
        "",
        "SD is the sample standard deviation across the outer folds.",
        "",
        "## Notes",
        "",
        "- The binomial CI and p-value treat the pooled held-out predictions as independent. The outer-fold models share training texts, so this holds only approximately.",
        "",
        "## Files",
        "",
        f"- Metrics: `{m['outputs']['metrics']}`",
        f"- Held-out predictions: `{m['outputs']['predictions']}`",
        f"- Log: `{m['log_path']}`",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
