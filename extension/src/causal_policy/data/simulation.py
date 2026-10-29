"""Data-generating process of the text treatment experiment.

Covariates X1 ~ U[-1,1], X2 ~ U[0,1], X3 ~ U[-1,1]; frame Z ~ Uniform{-1,+1}
independent of X; text T = t_{Z,J} with J ~ Uniform{1,...,K}; outcome
Y = 1{U <= mu(X,T)} with mu = b(x) + z(t) g(x).
"""

import numpy as np
from scipy.integrate import quad

K = 100
N_VALUES = (30, 100, 200, 400, 800)
N_REPS = 100
N_TEST = 10000
N_FOLDS = 5

# Population values stated in the experiment plan.
PLAN_VALUES = {
    "V_gain": 0.5370794093,
    "V_loss": 0.3629205907,
    "V_opt": 0.6865127935,
    "Delta": 0.1494333842,
}
PLAN_TOLERANCE = 1e-8


def b_fn(x2, x3):
    return 0.40 + 0.10 * x2 + 0.05 * x3


def g_fn(x1, x2):
    return 0.35 * np.tanh(2.0 * (x1 + 0.80 * (1.0 - x2) ** 2))


def log_cosh(a):
    return np.logaddexp(a, -a) - np.log(2.0)


def population_values():
    # Integrate over x1 in closed form: the antiderivative of tanh(2(x1 + c)) is
    # log cosh(2(x1 + c)) / 2, and |.| splits at x1 = -c, which lies in [-1, 0].
    # The remaining integral over x2 ~ U[0,1] is done numerically.
    def c_of(x2):
        return 0.80 * (1.0 - x2) ** 2

    def mean_g_given_x2(x2):
        c = c_of(x2)
        return 0.35 * 0.5 * 0.5 * (log_cosh(2.0 * (1.0 + c)) - log_cosh(2.0 * (c - 1.0)))

    def mean_abs_g_given_x2(x2):
        c = c_of(x2)
        return 0.35 * 0.5 * 0.5 * (log_cosh(2.0 * (1.0 + c)) + log_cosh(2.0 * (1.0 - c)))

    opts = {"epsabs": 1e-14, "epsrel": 1e-13, "limit": 200}
    e_b = 0.40 + 0.10 * 0.5 + 0.05 * 0.0
    e_g = quad(mean_g_given_x2, 0.0, 1.0, **opts)[0]
    e_abs_g = quad(mean_abs_g_given_x2, 0.0, 1.0, **opts)[0]
    # P(g > 0) = E[(1 + c(X2)) / 2] = 1/2 + 0.4 E[(1 - X2)^2].
    p_gain_opt = quad(lambda x2: (1.0 + c_of(x2)) / 2.0, 0.0, 1.0, **opts)[0]
    v_gain = e_b + e_g
    v_loss = e_b - e_g
    v_opt = e_b + e_abs_g
    v_const = max(v_gain, v_loss)
    return {
        "E_b": e_b,
        "E_g": e_g,
        "E_abs_g": e_abs_g,
        "V_gain": v_gain,
        "V_loss": v_loss,
        "V_const": v_const,
        "V_opt": v_opt,
        "Delta": v_opt - v_const,
        "share_gain_optimal": p_gain_opt,
        "share_loss_optimal": 1.0 - p_gain_opt,
    }


def check_population_values(pop):
    for key, value in PLAN_VALUES.items():
        assert abs(pop[key] - value) < PLAN_TOLERANCE, f"{key}: computed {pop[key]!r}, plan {value!r}"


def rng_training_sample(seed, n, rep):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(0, n, rep)))


def rng_test_sample(seed):
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(1,)))


def draw_covariates(rng, n):
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    x3 = rng.uniform(-1.0, 1.0, n)
    return x1, x2, x3


def draw_training_sample(rng, n, bank):
    x1, x2, x3 = draw_covariates(rng, n)
    z = rng.choice(np.array([-1, 1]), n)
    j = rng.integers(1, K + 1, n)
    u = rng.uniform(0.0, 1.0, n)
    fold = rng.permutation(np.arange(n) % N_FOLDS + 1)
    b = b_fn(x2, x3)
    g = g_fn(x1, x2)
    mu = b + z * g
    y = (u <= mu).astype(int)
    text_id = [bank[int(zi)][int(ji) - 1] for zi, ji in zip(z, j)]
    return {"x1": x1, "x2": x2, "x3": x3, "z": z, "j": j, "u": u, "fold": fold, "b": b, "g": g, "mu": mu, "y": y, "text_id": text_id}


def check_training_sample(s, n, bank):
    assert len(s["y"]) == n
    assert (s["mu"] >= 0.0).all() and (s["mu"] <= 0.90 + 1e-12).all(), "mu outside [0, 0.90]"
    assert np.array_equal(s["y"], (s["u"] <= s["mu"]).astype(int))
    assert set(np.unique(s["z"])) <= {-1, 1} and s["j"].min() >= 1 and s["j"].max() <= K
    for t, zi in zip(s["text_id"], s["z"]):
        assert t in bank[int(zi)], "text frame does not match z"
    sizes = np.bincount(s["fold"], minlength=N_FOLDS + 1)[1:]
    assert sizes.max() - sizes.min() <= 1, "fold sizes not as equal as possible"
