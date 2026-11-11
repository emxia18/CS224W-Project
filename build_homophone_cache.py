"""Retrieve CMUdict homophones and encode their meanings with the cached MPNet.

Reads only input words from train/validation, never answers or category labels.
"""

import argparse
from collections import defaultdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import numpy as np
import torch

from embed_data import MODEL, ROOT
from word_channels import WordTokenizer


def retrieve_candidates(tokenizer, words):
    reverse = defaultdict(set)
    for word in tokenizer.dictionary:
        if re.fullmatch(r"[a-z]+(?:'[a-z]+)?", word):
            for variant in tokenizer.variants(word):
                reverse[tuple(variant)].add(word)
    result = {}
    for word in words:
        candidates = set()
        for variant in tokenizer.variants(word):
            candidates.update(reverse.get(tuple(variant), ()))
        candidates.discard(tokenizer.pronunciation_word(word))
        result[word] = sorted(candidates)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/embeddings/homophones')
    parser.add_argument('--embedding-dir', type=Path, default=ROOT / 'data/embeddings')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    torch.set_num_threads(2)
    tokenizer = WordTokenizer(pronunciation=True)
    words, hashes = set(), {}
    for split in ('train', 'val'):
        raw = (args.input_dir / f'{split}.json').read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        words.update(w for puzzle in json.loads(raw) for w in puzzle['words'])
    candidates = retrieve_candidates(tokenizer, sorted(words))
    vocabulary = sorted({w for values in candidates.values() for w in values})
    print(f'{sum(bool(v) for v in candidates.values())}/{len(words)} input words have alternative '
          f'homophones; encoding {len(vocabulary)} distinct candidates.', flush=True)
    if vocabulary:
        from sentence_transformers import SentenceTransformer
        cached = ROOT / 'data/model_cache/models--sentence-transformers--all-mpnet-base-v2'
        embedding_metadata = json.loads((args.embedding_dir / 'metadata.json').read_text())
        revision = embedding_metadata.get('revision') or (cached / 'refs/main').read_text().strip()
        model = SentenceTransformer(str(cached / 'snapshots' / revision), device=args.device)
        vectors = model.encode(vocabulary, batch_size=args.batch_size, normalize_embeddings=True,
                               convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
    else:
        revision, vectors = None, np.empty((0, 768), dtype=np.float32)
    if vectors.shape != (len(vocabulary), 768) or not np.isfinite(vectors).all():
        raise ValueError('Expected finite 768-dimensional candidate vectors')
    rows = {w: i for i, w in enumerate(vocabulary)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'vectors.npy', vectors, allow_pickle=False)
    (args.output_dir / 'words.json').write_text(json.dumps({
        w: {'candidates': values, 'rows': [rows[c] for c in values]}
        for w, values in candidates.items()}, indent=2) + '\n')
    (args.output_dir / 'metadata.json').write_text(json.dumps(dict(
        model=MODEL, model_revision=revision, tokenization='strip-parenthetical-stressless-v2',
        cmudict_version=version('cmudict'), input_sha256=hashes,
        candidate_count=len(vocabulary), candidates_sha256=hashlib.sha256(
            json.dumps(vocabulary).encode()).hexdigest()), indent=2) + '\n')


if __name__ == '__main__':
    main()
