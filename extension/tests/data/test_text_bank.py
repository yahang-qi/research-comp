import json

import pytest

from causal_policy import io
from causal_policy.data import text_bank


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")


def test_read_bank_orders_ids_within_frame(tmp_path):
    write_jsonl(tmp_path / "texts.jsonl", [{"text_id": t, "text": f"text {t}"} for t in (4, 1, 3, 2)])
    write_jsonl(tmp_path / "simulator_labels.jsonl", [{"text_id": 4, "Z": 1}, {"text_id": 1, "Z": -1}, {"text_id": 3, "Z": -1}, {"text_id": 2, "Z": 1}])
    assert text_bank.read_bank(tmp_path, 2) == {1: [2, 4], -1: [1, 3]}
    assert text_bank.read_labels(tmp_path / "simulator_labels.jsonl") == {4: 1, 1: -1, 3: -1, 2: 1}
    with pytest.raises(AssertionError):
        text_bank.read_bank(tmp_path, 3)


def test_read_bank_rejects_mismatched_ids(tmp_path):
    write_jsonl(tmp_path / "texts.jsonl", [{"text_id": 1, "text": "a"}, {"text_id": 2, "text": "b"}])
    write_jsonl(tmp_path / "simulator_labels.jsonl", [{"text_id": 1, "Z": 1}, {"text_id": 3, "Z": -1}])
    with pytest.raises(AssertionError):
        text_bank.read_bank(tmp_path, 1)


def test_read_texts_sorted_and_embeddings_ids_are_int(tmp_path):
    write_jsonl(tmp_path / "texts.jsonl", [{"text_id": 10, "text": "b"}, {"text_id": 2, "text": "a"}])
    assert text_bank.read_texts(tmp_path / "texts.jsonl") == [(2, "a"), (10, "b")]
    io.write_csv(tmp_path / "e.csv", ["text_id", "u_001"], [[2, io.fmt(0.5)], [10, io.fmt(-0.25)]])
    ids, X, names = text_bank.read_embeddings(tmp_path / "e.csv")
    assert ids == [2, 10] and names == ["u_001"] and X.tolist() == [[0.5], [-0.25]]
