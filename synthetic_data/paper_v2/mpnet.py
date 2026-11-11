"""Score every quartet with cached MPNet; select color-spaced representatives.

Usage: .venv/bin/python 'data generation/paper_v2/mpnet.py' pools.json scores.json
Input: list of {id, words: [eight strings], required: [optional anchors]}.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REVISION = 'e8c3b32edf5434bc2275fc9bab85f82640a19130'
MODEL = ROOT / 'data/model_cache/models--sentence-transformers--all-mpnet-base-v2/snapshots' / REVISION
COLORS = ('yellow', 'green', 'blue', 'purple')


def rank_quartets(words, embeddings, required=()):
    """Enumerate all 70; impose optional construction constraints after scoring."""
    if len(words) != 8 or len(set(words)) != 8:
        raise ValueError('Every pool must have eight distinct words')
    if not set(required) <= set(words):
        raise ValueError('Required anchors missing from pool')
    rows = []
    for indices in itertools.combinations(range(8), 4):
        selected = [words[i] for i in indices]
        score = float(np.mean([np.dot(embeddings[i], embeddings[j])
                               for i, j in itertools.combinations(indices, 2)]))
        rows.append({'words': selected, 'similarity': score,
                     'eligible': set(required) <= set(selected)})
    eligible = [r for r in rows if r['eligible']]
    if not eligible:
        raise ValueError('No eligible quartet')
    lo = min(r['similarity'] for r in eligible)
    hi = max(r['similarity'] for r in eligible)
    # The printed paper omits the minimum offset and reverses the intermediate
    # color order. Use equally spaced points between its stated extrema.
    targets = dict(zip(COLORS, (hi, lo + 2 * (hi-lo)/3, lo + (hi-lo)/3, lo)))
    selected = {}
    for color, target in targets.items():
        row = min(eligible, key=lambda r: (abs(r['similarity']-target), r['words']))
        selected[color] = {**row, 'target': target}
    return {'quartets': rows, 'eligible_count': len(eligible), 'required': list(required),
            'min_similarity': lo, 'max_similarity': hi, 'selections': selected}


def score_pools(pools):
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    words = sorted({w for p in pools for w in p['words']})
    if not words:
        raise ValueError('No pools')
    model = SentenceTransformer(str(MODEL), device='cpu', local_files_only=True)
    vectors = model.encode([w.lower() for w in words], normalize_embeddings=True,
                           convert_to_numpy=True, batch_size=32, show_progress_bar=False)
    if not np.isfinite(vectors).all():
        raise ValueError('Nonfinite embeddings')
    mapping = dict(zip(words, vectors))
    results = {}
    for pool in pools:
        if pool['id'] in results:
            raise ValueError('Duplicate pool id')
        results[pool['id']] = rank_quartets(pool['words'],
                                          [mapping[w] for w in pool['words']],
                                          pool.get('required', []))
    return {'model': 'sentence-transformers/all-mpnet-base-v2', 'revision': REVISION,
            'input_case': 'lowercase', 'normalized': True,
            'input_sha256': hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest(),
            'pools': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    pools = json.loads(args.input.read_text())
    scores = score_pools(pools)
    args.output.write_text(json.dumps(scores, indent=2) + '\n')
    print(f"Scored {len(pools)} pools / {70 * len(pools)} quartets -> {args.output}")
