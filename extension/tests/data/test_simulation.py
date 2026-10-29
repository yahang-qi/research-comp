import numpy as np
import pytest

from causal_policy.data import simulation


def toy_bank():
    return {1: list(range(1, simulation.K + 1)), -1: list(range(simulation.K + 1, 2 * simulation.K + 1))}


def test_population_values_match_plan():
    pop = simulation.population_values()
    simulation.check_population_values(pop)
    assert pop["share_gain_optimal"] == pytest.approx(19 / 30, abs=1e-12)
    assert pop["V_const"] == pop["V_gain"]


def test_mu_range_on_grid():
    x1, x2, x3 = np.meshgrid(np.linspace(-1, 1, 41), np.linspace(0, 1, 21), np.linspace(-1, 1, 41))
    b, g = simulation.b_fn(x2, x3), simulation.g_fn(x1, x2)
    assert (b - np.abs(g) >= 0).all() and (b + np.abs(g) <= 0.90 + 1e-12).all()


@pytest.mark.parametrize("n", simulation.N_VALUES)
def test_training_sample(n):
    bank = toy_bank()
    s = simulation.draw_training_sample(simulation.rng_training_sample(0, n, 1), n, bank)
    simulation.check_training_sample(s, n, bank)
    for t, z, j in zip(s["text_id"], s["z"], s["j"]):
        assert t == bank[int(z)][int(j) - 1]
    assert np.array_equal(s["y"], (s["u"] <= s["mu"]).astype(int))
    assert np.bincount(s["fold"])[1:].tolist() == [n // simulation.N_FOLDS] * simulation.N_FOLDS


def test_samples_reproducible_and_distinct():
    bank = toy_bank()
    a = simulation.draw_training_sample(simulation.rng_training_sample(0, 30, 1), 30, bank)
    b = simulation.draw_training_sample(simulation.rng_training_sample(0, 30, 1), 30, bank)
    c = simulation.draw_training_sample(simulation.rng_training_sample(0, 30, 2), 30, bank)
    assert np.array_equal(a["x1"], b["x1"]) and a["text_id"] == b["text_id"]
    assert not np.array_equal(a["x1"], c["x1"])
    t1 = simulation.draw_covariates(simulation.rng_test_sample(0), 5)
    t2 = simulation.draw_covariates(simulation.rng_test_sample(0), 5)
    assert all(np.array_equal(u, v) for u, v in zip(t1, t2))
