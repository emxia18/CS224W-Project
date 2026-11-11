"""Frozen per-word MPNet representations conditioned on all puzzle words."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from connections.paths import ROOT


def puzzle_key(words):
    return hashlib.sha256(json.dumps(sorted(words), ensure_ascii=False).encode()).hexdigest()


def puzzle_text(words):
    """Canonical order avoids dependence on the shuffled dataset word order."""
    ordered = sorted(words)
    text, spans = '', []
    for word in ordered:
        if text:
            text += ' ; '
        start = len(text)
        # MPNet drops emoji presentation selectors during normalization.
        text += word.replace('\ufe0f', '').replace('\ufe0e', '').strip()
        spans.append((start, len(text)))
    return ordered, text, spans


def pool_words(hidden, offsets, spans):
    pooled = []
    for start, end in spans:
        mask = (offsets[:, 0] >= start) & (offsets[:, 1] <= end) & (offsets[:, 1] > offsets[:, 0])
        if not mask.any() or int(offsets[mask, 0].min()) != start or int(offsets[mask, 1].max()) != end:
            raise ValueError('Word missing or truncated during contextual tokenization')
        pooled.append(hidden[mask].mean(0))
    return F.normalize(torch.stack(pooled), dim=-1)


class ContextualCache:
    def __init__(self, folder):
        folder = Path(folder)
        self.metadata = json.loads((folder / 'metadata.json').read_text())
        self.index = json.loads((folder / 'index.json').read_text())
        self.vectors = np.load(folder / 'embeddings.npy', allow_pickle=False)
        if (self.vectors.shape != (len(self.index), 16, 768)
                or not np.isfinite(self.vectors).all()
                or not np.allclose(np.linalg.norm(self.vectors, axis=-1), 1, atol=1e-5)):
            raise ValueError('Expected normalized contextual vectors of shape (puzzles,16,768)')

    def validate_split(self, path):
        expected = self.metadata['input_sha256'].get(path.stem)
        if expected != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f'Contextual cache missing or stale for {path}; regenerate it')

    def lookup(self, words):
        entry = self.index[puzzle_key(words)]
        positions = {word: i for i, word in enumerate(entry['words'])}
        return self.vectors[entry['row'], [positions[w] for w in words]]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'data/processed')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'data/embeddings/contextual')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.batch_size < 1 or args.threads < 1:
        parser.error('Batch size and threads must be positive')
    torch.set_num_threads(args.threads)
    from transformers import AutoModel, AutoTokenizer
    base = ROOT / 'data/model_cache/models--sentence-transformers--all-mpnet-base-v2'
    revision = json.loads((ROOT / 'data/embeddings/metadata.json').read_text()).get('revision')
    revision = revision or (base / 'refs/main').read_text().strip()
    snapshot = base / 'snapshots' / revision
    tokenizer = AutoTokenizer.from_pretrained(snapshot, use_fast=True, local_files_only=True)
    model = AutoModel.from_pretrained(snapshot, local_files_only=True).to(args.device).eval()
    hashes, puzzles = {}, {}
    for split in ('train', 'val'):
        raw = (args.input_dir / f'{split}.json').read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        for record in json.loads(raw):
            words = record['words']
            if len(words) != 16 or len(set(words)) != 16:
                raise ValueError('Expected 16 distinct puzzle words')
            puzzles[puzzle_key(words)] = puzzle_text(words)
    items, vectors, index = list(puzzles.items()), [], {}
    with torch.inference_mode():
        for start in range(0, len(items), args.batch_size):
            batch = items[start:start + args.batch_size]
            encoded = tokenizer([value[1] for _, value in batch], padding=True,
                                truncation=False, return_offsets_mapping=True, return_tensors='pt')
            offsets = encoded.pop('offset_mapping')
            if encoded['input_ids'].shape[1] > 384:
                raise ValueError('Puzzle exceeds MPNet sentence encoder length budget')
            hidden = model(**{k: v.to(args.device) for k, v in encoded.items()}).last_hidden_state
            for i, (key, (words, _, spans)) in enumerate(batch):
                index[key] = {'row': len(vectors), 'words': words}
                vectors.append(pool_words(hidden[i], offsets[i].to(args.device), spans).cpu().numpy())
            print(f'Encoded {len(vectors)}/{len(items)} puzzles', flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / 'embeddings.npy', np.stack(vectors).astype(np.float32), allow_pickle=False)
    (args.output_dir / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
    (args.output_dir / 'metadata.json').write_text(json.dumps({
        'mode': 'joint-word-token-mean-v1', 'model': 'sentence-transformers/all-mpnet-base-v2',
        'revision': revision, 'order': 'alphabetical', 'separator': ' ; ',
        'normalized': True, 'input_sha256': hashes,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
