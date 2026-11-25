"""Extra word-similarity sources, each giving a 16x16 matrix per puzzle.

A spec is KIND:DIR:
  cos:DIR  cosine between single vectors (embed_data.py / embed_static.py cache)
  max:DIR  best cosine over all sense pairs (build_sense_cache.py cache), so two
           words match if any of their meanings do
  sparse:DIR  cosine between sparse vectors (build_phrase_cache.py), e.g. shared
           phrase partners; words with an all-zero row have similarity 0
  phrase:DIR  puzzle-specific phrase completion (build_phrase_cache.py): two
           words score 1 if they share a phrase partner that 3-5 of the
           puzzle's 16 words share (a "___ ROLL" group plus at most one red
           herring), else 0. Partners shared by many words are too generic.
  puzzle:DIR  precomputed 16x16 matrices, one per puzzle (matrices.npy and
           index.json keyed by the puzzle's word list in order), e.g. the
           out-of-fold global-graph similarities of connections/hetero/global_graph.py

Used as extra residual terms and edge features by train_gnn.py (--similarity)
and compared without training by evaluation/combine_similarities.py.
"""

import json
from pathlib import Path

import numpy as np


PHRASE_MIN, PHRASE_MAX = 3, 5  # puzzle words sharing a partner, for phrase:DIR


class SimilarityChannel:
    def __init__(self, spec):
        kind, _, directory = spec.partition(":")
        directory = Path(directory)
        self.sparse = None
        self.puzzle = None
        self.phrase = kind == "phrase"
        if kind == "puzzle":
            self.puzzle = np.load(directory / "matrices.npy", allow_pickle=False)
            self.index = json.loads((directory / "index.json").read_text())
            self.spec = spec
            return
        if kind in ("sparse", "phrase"):
            from scipy import sparse

            self.sparse = sparse.load_npz(directory / "embeddings.npz").tocsr()
            if self.phrase:
                self.sparse = (self.sparse > 0).astype(np.float32).tocsr()
            self.rows = json.loads((directory / "word_to_row.json").read_text())
            self.spec = spec
            return
        if kind == "cos":
            vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
            rows = json.loads((directory / "word_to_row.json").read_text())
            self.groups = {word: vectors[[row]] for word, row in rows.items()}
        elif kind == "max":
            vectors = np.load(directory / "senses.npy", allow_pickle=False)
            rows = json.loads((directory / "word_to_rows.json").read_text())
            self.groups = {word: vectors[indices] for word, indices in rows.items()}
        else:
            raise ValueError(f"Unknown similarity kind {kind!r} in {spec!r}; use cos:, max:, sparse:, phrase: or puzzle:DIR")
        if not all(np.allclose(np.linalg.norm(v, axis=1), 1, atol=1e-4) for v in self.groups.values()):
            raise ValueError(f"{spec}: vectors must be unit length")
        self.spec = spec

    def matrix(self, words):
        """16x16 similarities (diagonal 1): the max within each word-by-word block."""
        if self.puzzle is not None:
            key = "|".join(words)
            if key not in self.index:
                raise KeyError(f"{self.spec}: no matrix for this puzzle ({words[:3]}...)")
            return self.puzzle[self.index[key]].astype(np.float32)
        if self.sparse is not None:
            missing = [word for word in words if word not in self.rows]
            if missing:
                raise KeyError(f"{self.spec}: no row for {missing}")
            block = self.sparse[[self.rows[word] for word in words]]
            if self.phrase:
                counts = np.asarray(block.sum(axis=0)).ravel()
                block = block[:, np.flatnonzero((counts >= PHRASE_MIN) & (counts <= PHRASE_MAX))]
                sims = ((block @ block.T).toarray() > 0).astype(np.float32)
            else:
                sims = (block @ block.T).toarray()
            np.fill_diagonal(sims, 1.0)
            return sims.astype(np.float32)
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
