import numpy as np
import pytest

from causal_policy import io
from causal_policy.data import samples


def write_dataset(root, truth_rows=None):
    (root / "observed").mkdir()
    (root / "truth").mkdir()
    obs = [[1, 1, 1, 0.1, 0.2, 0.3, 7, 1], [1, 2, 2, -0.1, 0.5, 0.0, 9, 0], [2, 1, 1, 0.4, 0.6, -0.2, 9, 1]]
    io.write_csv(root / "observed" / "train_n0002.csv", ["rep", "person_id", "fold", "x1", "x2", "x3", "text_id", "y"], obs)
    truth = truth_rows or [[1, 1, 1, 3, 0.5, 0.4, 0.1, 0.5], [1, 2, -1, 2, 0.9, 0.45, -0.2, 0.65], [2, 1, -1, 2, 0.1, 0.5, 0.1, 0.4]]
    io.write_csv(root / "truth" / "train_n0002.csv", ["rep", "person_id", "z", "j", "u", "b", "g", "mu"], truth)
    io.write_csv(root / "observed" / "test.csv", ["person_id", "x1", "x2", "x3"], [[1, 0.0, 0.5, 0.0], [2, 0.5, 0.1, -0.5]])
    io.write_csv(root / "truth" / "test.csv", ["person_id", "b", "g", "mu_gain", "mu_loss", "z_opt"], [[1, 0.45, 0.2, 0.65, 0.25, 1], [2, 0.4, -0.1, 0.3, 0.5, -1]])
    io.write_csv(root / "truth" / "bank_index.csv", ["z", "j", "text_id"], [[1, 3, 7], [-1, 2, 9]])


def test_training_reps_keep_observed_and_truth_apart(tmp_path):
    write_dataset(tmp_path)
    reps = samples.load_training_reps(tmp_path, 2)
    assert list(reps) == [1, 2]
    observed, truth = reps[1]
    assert observed.X.tolist() == [[0.1, 0.2, 0.3], [-0.1, 0.5, 0.0]]
    assert observed.Y.tolist() == [1, 0] and observed.folds.tolist() == [1, 2] and observed.text_ids.tolist() == [7, 9]
    assert truth.z.tolist() == [1, -1] and np.allclose(truth.mu, [0.5, 0.65])
    assert not hasattr(observed, "z") and not hasattr(observed, "mu")
    assert reps[2][0].text_ids.tolist() == [9] and reps[2][1].z.tolist() == [-1]


def test_test_sample_and_bank_frames(tmp_path):
    write_dataset(tmp_path)
    X, truth = samples.load_test(tmp_path)
    assert X.shape == (2, 3) and truth.z_opt.tolist() == [1, -1] and np.allclose(truth.g, [0.2, -0.1])
    assert samples.load_bank_frames(tmp_path) == {7: 1, 9: -1}


def test_mismatched_observed_and_truth_rows_are_rejected(tmp_path):
    write_dataset(tmp_path, truth_rows=[[1, 1, 1, 3, 0.5, 0.4, 0.1, 0.5], [1, 3, -1, 2, 0.9, 0.45, -0.2, 0.65], [2, 1, -1, 2, 0.1, 0.5, 0.1, 0.4]])
    with pytest.raises(AssertionError):
        samples.load_training_reps(tmp_path, 2)
