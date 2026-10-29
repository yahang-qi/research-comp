import numpy as np

from causal_policy import io


def test_float_round_trip(tmp_path):
    values = np.random.default_rng(0).normal(size=50)
    io.write_csv(tmp_path / "v.csv", ["id", "v"], [[i, io.fmt(v)] for i, v in enumerate(values)])
    lines = (tmp_path / "v.csv").read_text().splitlines()
    assert lines[0] == "id,v"
    assert np.array_equal(np.array([float(line.split(",")[1]) for line in lines[1:]]), values)
    rows = io.read_csv(tmp_path / "v.csv")
    assert np.array_equal(np.array([float(r["v"]) for r in rows]), values)
