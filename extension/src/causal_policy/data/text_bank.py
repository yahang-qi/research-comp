"""Reading the text bank: texts, simulator labels, frame banks and embeddings."""

import csv
import os

import numpy as np

from causal_policy.io import read_jsonl


def read_texts(path):
    """Return (text_id, text) pairs sorted by text_id."""
    records = [(rec["text_id"], rec["text"]) for rec in read_jsonl(path)]
    ids = [r[0] for r in records]
    assert len(set(ids)) == len(ids), "duplicate text_id"
    assert all(isinstance(t, str) and t.strip() for _, t in records), "empty text"
    return sorted(records)


def read_labels(path):
    """Return {text_id: Z} from simulator_labels.jsonl."""
    labels = {}
    for rec in read_jsonl(path):
        assert rec["text_id"] not in labels, "duplicate text_id in labels"
        assert rec["Z"] in (-1, 1)
        labels[rec["text_id"]] = rec["Z"]
    return labels


def read_bank(bank_dir, k):
    """Return {+1: [...], -1: [...]}, each frame's text IDs in increasing order."""
    text_ids = [r["text_id"] for r in read_jsonl(os.path.join(bank_dir, "texts.jsonl"))]
    labels = {r["text_id"]: r["Z"] for r in read_jsonl(os.path.join(bank_dir, "simulator_labels.jsonl"))}
    assert len(text_ids) == len(set(text_ids)) == len(labels), "text and label files differ in size"
    assert set(text_ids) == set(labels), "text and label files differ in text_id"
    bank = {z: sorted(t for t, lab in labels.items() if lab == z) for z in (1, -1)}
    assert len(bank[1]) == len(bank[-1]) == k, "expected K texts per frame"
    return bank


def read_embeddings(path):
    """Return integer text IDs, the embedding matrix, and the feature names."""
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header = rows[0]
    assert header[0] == "text_id"
    ids = [int(r[0]) for r in rows[1:]]
    X = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    return ids, X, header[1:]
