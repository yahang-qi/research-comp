import numpy as np

from causal_policy.text import separability


def toy_data(n_per_frame=40, dim=6, shift=1.5, seed=0):
    rng = np.random.default_rng(seed)
    y = np.repeat([1, 0], n_per_frame)
    X = rng.normal(size=(2 * n_per_frame, dim))
    X[:, 0] += shift * (2 * y - 1)
    return X, y


def test_nested_cv_folds_cover_each_text_once_and_are_stratified():
    X, y = toy_data()
    prob, fold_of, folds = separability.nested_cv(X, y, seed=0, test_size_per_frame=8)
    assert np.isfinite(prob).all() and ((prob >= 0) & (prob <= 1)).all()
    assert sorted(np.bincount(fold_of)[1:].tolist()) == [16] * separability.N_OUTER
    for k in range(1, separability.N_OUTER + 1):
        assert y[fold_of == k].sum() == 8
    assert all(f["n_train"] + f["n_test"] == len(y) for f in folds)
    assert all(f["selected_C"] in separability.C_GRID for f in folds)


def test_separable_data_is_detected_and_summary_is_consistent():
    X, y = toy_data(shift=3.0)
    prob, _, folds = separability.nested_cv(X, y, seed=0)
    pred, pooled, fold_summary = separability.summarize(y, prob, folds)
    assert pooled["accuracy"] > 0.9 and pooled["auc"] > 0.95
    assert pooled["n_correct"] == int((pred == y).sum())
    assert pooled["accuracy_ci95"][0] <= pooled["accuracy"] <= pooled["accuracy_ci95"][1]
    assert np.isclose(fold_summary["accuracy"]["mean"], np.mean([f["test"]["accuracy"] for f in folds]))

