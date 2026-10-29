"""Fixed text embeddings u(t) from Qwen3-Embedding.

Each text is encoded as a plain document (no instruction), the last-token
pooled vector is kept to its first `dim` coordinates (Matryoshka truncation),
and the result is L2-normalized.
"""

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
MAX_LENGTH = 8192
SEED = 0


def set_determinism(num_threads):
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)


def load_model(revision):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision, padding_side="left")
    model = AutoModel.from_pretrained(MODEL_ID, revision=revision, dtype=torch.float32)
    model.eval()
    probe = tokenizer(["probe"])["input_ids"][0]
    assert probe[-1] == tokenizer.convert_tokens_to_ids("<|endoftext|>"), "tokenizer does not append <|endoftext|>"
    return tokenizer, model


def last_token_pool(last_hidden_states, attention_mask):
    # As in the model card; with left padding the last position is the last token.
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[torch.arange(batch_size), sequence_lengths]


@torch.no_grad()
def embed(texts, tokenizer, model, batch_size):
    pooled = []
    for start in range(0, len(texts), batch_size):
        batch = tokenizer(texts[start:start + batch_size], padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        lengths = batch["attention_mask"].sum(dim=1)
        assert int(lengths.max()) < MAX_LENGTH, "text reached max_length and may be truncated"
        out = model(**batch)
        pooled.append(last_token_pool(out.last_hidden_state, batch["attention_mask"]))
    return torch.cat(pooled)


def truncate_normalize(pooled, dim):
    return F.normalize(pooled[:, :dim], p=2, dim=1).numpy().astype(np.float64)


def check_embeddings(ids, expected_ids, emb, dim):
    assert emb.shape == (len(expected_ids), dim), emb.shape
    assert np.isfinite(emb).all(), "non-finite embedding"
    norms = np.linalg.norm(emb, axis=1)
    assert np.abs(norms - 1).max() < 1e-6, norms
    assert set(ids) == set(expected_ids) and len(ids) == len(expected_ids), "text_id mismatch"
    assert len(np.unique(emb, axis=0)) == len(ids), "identical embedding rows"
