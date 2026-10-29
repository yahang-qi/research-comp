"""Compute fixed text embeddings u(t) for the text bank.

Reads only texts.jsonl (no frame labels), encodes each text as a plain
document with Qwen3-Embedding, takes the last-token pooled vector, keeps the
first `dim` coordinates (Matryoshka truncation), and L2-normalizes.
"""

import argparse
import csv
import json
import os
import platform
import sys
from datetime import datetime, timezone

import numpy as np
import torch
import transformers

from causal_policy.data.text_bank import read_texts
from causal_policy.io import fmt, sha256_of, write_csv
from causal_policy.text.embedding import MAX_LENGTH, MODEL_ID, SEED, check_embeddings, embed, load_model, set_determinism, truncate_normalize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--texts", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--n-texts", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-metadata", required=True)
    parser.add_argument("--log-path", default=None)
    args = parser.parse_args()

    set_determinism(args.num_threads)

    records = read_texts(args.texts)
    assert len(records) == args.n_texts, f"expected {args.n_texts} texts, found {len(records)}"
    ids = [r[0] for r in records]
    texts = [r[1] for r in records]

    tokenizer, model = load_model(args.model_revision)
    source_dim = model.config.hidden_size
    assert args.dim <= source_dim

    emb = truncate_normalize(embed(texts, tokenizer, model, args.batch_size), args.dim)
    check_embeddings(ids, ids, emb, args.dim)

    columns = ["text_id"] + [f"u_{j:03d}" for j in range(1, args.dim + 1)]
    write_csv(args.out_csv, columns, [[text_id] + [fmt(v) for v in row] for text_id, row in zip(ids, emb)])

    # Re-read the written file and recheck it.
    with open(args.out_csv, encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == columns
    reread_ids = [int(r[0]) for r in rows[1:]]
    reread = np.array([[float(v) for v in r[1:]] for r in rows[1:]])
    assert reread_ids == ids and np.array_equal(reread, emb)
    check_embeddings(reread_ids, ids, reread, args.dim)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "model_revision": args.model_revision,
        "embedding_source_dim": source_dim,
        "embedding_dim": args.dim,
        "dimension_reduction": f"Matryoshka truncation to the first {args.dim} coordinates of the pooled vector",
        "pooling": "last token (tokenizer appends <|endoftext|>)",
        "normalization": "L2 after truncation",
        "instruction": None,
        "max_length": MAX_LENGTH,
        "padding_side": "left",
        "dtype": "float32",
        "device": "cpu",
        "batch_size": args.batch_size,
        "num_threads": args.num_threads,
        "seed": SEED,
        "deterministic_algorithms": True,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "input_file": os.path.basename(args.texts),
        "input_sha256": sha256_of(args.texts),
        "n_texts": len(ids),
        "row_order": "sorted by text_id",
        "columns": columns,
        "float_format": "Python repr of float64 (round-trip exact)",
        "output_file": os.path.basename(args.out_csv),
        "output_sha256": sha256_of(args.out_csv),
        "log_path": args.log_path,
    }
    with open(args.out_metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")
    print(f"wrote {len(ids)} x {args.dim} embeddings to {args.out_csv}", file=sys.stderr)
    print(f"output_sha256 {metadata['output_sha256']}", file=sys.stderr)


if __name__ == "__main__":
    main()
