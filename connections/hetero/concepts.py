"""Per-puzzle concept nodes from neighbor caches (build_phrase_cache / build_neighbor_cache).

For one puzzle, every neighbor (phrase partner, ConceptNet concept, Wiktionary
category) shared by MIN..MAX of its 16 words becomes a concept node. Its features
are its source (one-hot), how many puzzle words it links (one-hot), and how rare
it is across the whole puzzle vocabulary (log inverse frequency).
"""

import json
from pathlib import Path

import numpy as np
from scipy import sparse
import torch


class ConceptSource:
    def __init__(self, directory):
        directory = Path(directory)
        self.matrix = (sparse.load_npz(directory / "embeddings.npz") > 0).astype(np.float32).tocsr()
        self.rows = json.loads((directory / "word_to_row.json").read_text())
        frequency = np.asarray(self.matrix.sum(axis=0)).ravel()
        self.idf = np.log(self.matrix.shape[0] / np.maximum(frequency, 1)).astype(np.float32)
        self.directory = str(directory)


def parse_sources(items):
    sources = {}
    for item in items:
        name, sep, directory = item.partition("=")
        if not sep or name in sources:
            raise ValueError(f"Expected unique NAME=DIR, got {item!r}")
        sources[name] = ConceptSource(directory)
    return sources


def feature_dim(n_sources, min_count, max_count):
    return n_sources + (max_count - min_count + 1) + 1


def puzzle_concepts(words, sources, min_count=2, max_count=8):
    """Incidence (C, 16), features (C, F) and source index (C,) for one puzzle."""
    blocks, features, kinds = [], [], []
    n = len(sources)
    for k, source in enumerate(sources.values()):
        block = source.matrix[[source.rows[w] for w in words]].toarray().T  # (P, 16)
        counts = block.sum(axis=1)
        keep = np.flatnonzero((counts >= min_count) & (counts <= max_count))
        if len(keep) == 0:
            continue
        block = block[keep]
        one_hot_source = np.zeros((len(keep), n), np.float32)
        one_hot_source[:, k] = 1
        one_hot_count = np.zeros((len(keep), max_count - min_count + 1), np.float32)
        one_hot_count[np.arange(len(keep)), counts[keep].astype(int) - min_count] = 1
        idf = source.idf[keep][:, None] / 10.0
        blocks.append(block)
        features.append(np.concatenate([one_hot_source, one_hot_count, idf], axis=1))
        kinds.append(np.full(len(keep), k))
    if not blocks:
        return {"incidence": torch.zeros(0, 16), "features": torch.zeros(0, feature_dim(n, min_count, max_count)),
                "kind": torch.zeros(0, dtype=torch.long)}
    return {"incidence": torch.tensor(np.concatenate(blocks)),
            "features": torch.tensor(np.concatenate(features)),
            "kind": torch.tensor(np.concatenate(kinds), dtype=torch.long)}
