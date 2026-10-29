"""Generic file helpers: hashing, JSONL reading, and round-trip CSV writing."""

import csv
import hashlib
import json


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def fmt(v):
    """Round-trip exact text for a float."""
    return repr(float(v))


def write_csv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path):
    """Rows of a CSV file as dicts of strings."""
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))
