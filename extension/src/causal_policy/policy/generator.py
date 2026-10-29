"""Generator G: deliver a text for a chosen attribute value.

G groups the observed training texts by their attribute (true frame or learned
group label) and draws a text uniformly from the chosen group. Only groups that
contain observed texts can be recommended.
"""

import numpy as np


def build_groups(text_ids, labels):
    """Return {label: sorted distinct observed text IDs with that label}."""
    groups = {}
    for t, lab in zip(text_ids, labels):
        groups.setdefault(int(lab), set()).add(t)
    return {lab: sorted(ts) for lab, ts in groups.items()}


def sample_texts(groups, choices, rng):
    """Draw one text uniformly from groups[c] for each chosen label c."""
    choices = np.asarray(choices)
    missing = set(np.unique(choices).tolist()) - set(groups)
    assert not missing, f"no observed training text in group(s) {sorted(missing)}"
    out = np.empty(len(choices), dtype=object)
    for lab, texts in groups.items():
        idx = np.flatnonzero(choices == lab)
        out[idx] = [texts[i] for i in rng.integers(0, len(texts), len(idx))]
    return out.tolist()
