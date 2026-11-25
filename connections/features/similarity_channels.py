"""Extra word-similarity sources, each giving a 16x16 matrix per puzzle.

A spec is KIND:DIR:
  cos:DIR  cosine between single vectors (embed_data.py / embed_static.py cache)
  max:DIR  best cosine over all sense pairs (build_sense_cache.py cache), so two
           words match if any of their meanings do

Used as extra residual terms and edge features by train_gnn.py (--similarity)
and compared without training by evaluation/combine_similarities.py.
"""

import json
from pathlib import Path

import numpy as np


class SimilarityChannel:
    def __init__(self, spec):
        kind, _, directory = spec.partition(":")
        directory = Path(directory)
        if kind == "cos":
            vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
            rows = json.loads((directory / "word_to_row.json").read_text())
            self.groups = {word: vectors[[row]] for word, row in rows.items()}
        elif kind == "max":
            vectors = np.load(directory / "senses.npy", allow_pickle=False)
            rows = json.loads((directory / "word_to_rows.json").read_text())
            self.groups = {word: vectors[indices] for word, indices in rows.items()}
        else:
            raise ValueError(f"Unknown similarity kind {kind!r} in {spec!r}; use cos:DIR or max:DIR")
        if not all(np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4) for v in self.groups.values()):
            raise ValueError(f"{spec}: vectors must be unit length")
        self.spec = spec

    def matrix(self, words):
        """16x16 similarities (diagonal 1): the max within each word-by-word block."""
        missing = [word for word in words if word not in self.groups]
        if missing:
            raise KeyError(f"{self.spec}: no vectors for {missing}")
        blocks = [self.groups[word] for word in words]
        starts = np.cumsum([0] + [len(block) for block in blocks[:-1]])
        stacked = np.concatenate(blocks)
        full = stacked @ stacked.T
        sims = np.maximum.reduceat(np.maximum.reduceat(full, starts, axis=0), starts, axis=1)
        np.fill_diagonal(sims, 1.0)
        return sims.astype(np.float32)


def parse(items):
    """NAME=KIND:DIR strings -> {name: SimilarityChannel}, in order."""
    channels = {}
    for item in items:
        name, sep, spec = item.partition("=")
        if not sep or not name or name in channels:
            raise ValueError(f"Expected a unique NAME=KIND:DIR, got {item!r}")
        channels[name] = SimilarityChannel(spec)
    return channels
