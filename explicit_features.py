"""Deterministic word relationships; no answer labels or fitted vocabularies."""

import json
import hashlib
from pathlib import Path
import re

import numpy as np
import torch


NODE_FEATURES = ('length', 'palindrome', 'vowel_fraction', 'repetition_fraction',
                 'has_apostrophe', 'has_space')
EDGE_FEATURES = ('anagram', 'reversal', 'shared_prefix', 'shared_suffix',
                 'one_edit_apart', 'substring', 'same_pronunciation')


def letters(word):
    return ''.join(c for c in word.lower() if c.isalpha())


def one_edit_apart(a, b):
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b):
        a, b = b, a
    i = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(a))
    return a[i:] == b[i + 1:]


def shared_prefix(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def orthographic_inputs(words, pronunciations):
    normalized = [letters(re.sub(r'\([^()]*\)', '', w)) for w in words]
    nodes = []
    for word, text in zip(words, normalized):
        size = max(1, len(text))
        nodes.append([len(text) / 32, float(len(text) > 1 and text == text[::-1]),
                      sum(c in 'aeiou' for c in text) / size,
                      (len(text) - len(set(text))) / size,
                      float("'" in word or '’' in word), float(' ' in word)])
    edges = torch.zeros(len(words), len(words), len(EDGE_FEATURES))
    phone_sets = [{tuple(v) for v in variants} for variants in pronunciations]
    for i, a in enumerate(normalized):
        for j in range(i + 1, len(words)):
            b = normalized[j]
            if not a or not b:
                continue
            denominator = max(1, min(len(a), len(b)))
            values = [a != b and sorted(a) == sorted(b), a != b and a == b[::-1],
                      shared_prefix(a, b) / denominator, shared_prefix(a[::-1], b[::-1]) / denominator,
                      one_edit_apart(a, b), a != b and (a in b or b in a),
                      bool(phone_sets[i] & phone_sets[j])]
            edges[i, j] = edges[j, i] = torch.tensor(values, dtype=torch.float32)
    return torch.tensor(nodes, dtype=torch.float32), edges


class HomophoneCache:
    """Candidate semantics: maximum pair similarity plus explicit availability."""

    def __init__(self, folder):
        folder = Path(folder)
        self.metadata = json.loads((folder / 'metadata.json').read_text())
        if self.metadata.get('tokenization') != 'strip-parenthetical-stressless-v2':
            raise ValueError('Rebuild homophone cache with current pronunciation tokenization')
        self.words = json.loads((folder / 'words.json').read_text())
        self.vectors = np.load(folder / 'vectors.npy', allow_pickle=False)
        if self.vectors.ndim != 2 or self.vectors.shape[1] != 768 or not np.isfinite(self.vectors).all():
            raise ValueError('Invalid homophone candidate vectors')
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise ValueError('Homophone candidate vectors must be normalized')
        for entry in self.words.values():
            if any(not isinstance(i, int) or not 0 <= i < len(self.vectors) for i in entry['rows']):
                raise ValueError('Homophone cache contains invalid candidate indices')

    def validate_inputs(self, data_dir):
        for split, expected in self.metadata.get('input_sha256', {}).items():
            actual = hashlib.sha256((Path(data_dir) / f'{split}.json').read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'Homophone cache is stale for {split}; rebuild it')

    def edges(self, words):
        indices = []
        for word in words:
            if word not in self.words:
                raise ValueError(f'Word missing from homophone cache: {word!r}; rebuild cache for these inputs')
            indices.append(self.words[word]['rows'])
        result = np.zeros((len(words), len(words), 2), dtype=np.float32)
        for i, a in enumerate(indices):
            for j in range(i + 1, len(indices)):
                b = indices[j]
                if a and b:
                    result[i, j] = result[j, i] = [float((self.vectors[a] @ self.vectors[b].T).max()), 1]
        return torch.from_numpy(result)
