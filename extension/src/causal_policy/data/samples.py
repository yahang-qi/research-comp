"""Loading the generated dataset (dataset/simulation).

Observed data (what methods may use) and simulator truth (for data generation,
evaluation and true-attribute methods only) are returned as separate objects.
"""

import os
from dataclasses import dataclass

import numpy as np

from causal_policy.io import read_csv

COVARIATES = ("x1", "x2", "x3")


@dataclass
class ObservedSample:
    X: np.ndarray
    Y: np.ndarray
    folds: np.ndarray
    text_ids: np.ndarray


@dataclass
class TrainingTruth:
    z: np.ndarray
    mu: np.ndarray


@dataclass
class TestTruth:
    b: np.ndarray
    g: np.ndarray
    z_opt: np.ndarray


def _column(rows, key, dtype=float):
    return np.array([dtype(r[key]) for r in rows])


def _covariates(rows):
    return np.column_stack([_column(rows, k) for k in COVARIATES])


def load_bank_frames(sim_dir):
    """{text_id: true frame} from truth/bank_index.csv."""
    return {int(r["text_id"]): int(r["z"]) for r in read_csv(os.path.join(sim_dir, "truth", "bank_index.csv"))}


def load_training_reps(sim_dir, n):
    """{rep: (ObservedSample, TrainingTruth)} for training sample size n, in replication order."""
    obs = read_csv(os.path.join(sim_dir, "observed", f"train_n{n:04d}.csv"))
    truth = read_csv(os.path.join(sim_dir, "truth", f"train_n{n:04d}.csv"))
    assert [(r["rep"], r["person_id"]) for r in obs] == [(r["rep"], r["person_id"]) for r in truth], "observed and truth rows differ"
    rep_of = _column(obs, "rep", int)
    out = {}
    for rep in np.unique(rep_of):
        idx = np.flatnonzero(rep_of == rep)
        o = [obs[i] for i in idx]
        t = [truth[i] for i in idx]
        out[int(rep)] = (
            ObservedSample(_covariates(o), _column(o, "y", int), _column(o, "fold", int), _column(o, "text_id", int)),
            TrainingTruth(_column(t, "z", int), _column(t, "mu")),
        )
    return out


def load_test(sim_dir):
    """Observed test covariates and the test truth."""
    obs = read_csv(os.path.join(sim_dir, "observed", "test.csv"))
    truth = read_csv(os.path.join(sim_dir, "truth", "test.csv"))
    assert [r["person_id"] for r in obs] == [r["person_id"] for r in truth], "observed and truth test rows differ"
    return _covariates(obs), TestTruth(_column(truth, "b"), _column(truth, "g"), _column(truth, "z_opt", int))
