"""One heterogeneous graph over every puzzle word, trained by link prediction (Option 2).

Nodes
  word      all puzzle words (train/val/test inputs), features = fastText and
            Numberbatch vectors
  <source>  one node type per neighbor cache (phrase partners, ConceptNet
            concepts, Wiktionary categories), kept if linked to 2..MAX words;
            feature = log degree

Edges
  word <-> concept, one relation per source
  word <-> word "same_group", from training answers only

Training (link prediction): each epoch splits the given training puzzles into
--chunks parts; each step takes one part as targets and the same-group edges of
the other parts as message-passing edges. All 120 word pairs of each target
puzzle are scored (24 same-group pairs,
96 different-group pairs, so negatives are the puzzle's own red herrings).
Pairs are scored by the cosine of the final word embeddings. Each embedding is
[normalized input vector, GNN output] with the GNN part starting near zero, so
the untrained model scores pairs by input cosine and the graph learns a
correction (without this, averaging over hundreds of concept neighbors washes
out each word's own meaning).

No leakage into the per-puzzle model (out-of-fold stacking): training puzzles
are split into K folds; fold k's 16x16 similarity matrices come from a graph
model trained on the other folds, whose same-group edges never include fold k.
Validation and test matrices come from a model trained on all training puzzles.
Validation and test answers are never edges.

Output (--output-dir): matrices.npy (puzzles, 16, 16) and index.json (puzzle
key -> row), for `--similarity NAME=puzzle:DIR` in train_gnn.py and
combine_similarities.py; plus metadata.json with link-prediction metrics.
"""

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
from scipy import sparse
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv

from connections.paths import ROOT


def puzzle_key(words):
    return "|".join(words)


def load_graph_inputs(args):
    rows = json.loads((args.word_dir / "word_to_row.json").read_text())
    words = sorted(rows, key=rows.get)
    features = [np.load(args.word_dir / "embeddings.npy")]
    for extra in args.extra_word_dir:
        extra_rows = json.loads((extra / "word_to_row.json").read_text())
        vectors = np.load(extra / "embeddings.npy")
        features.append(vectors[[extra_rows[w] for w in words]])
    x = torch.tensor(np.concatenate(features, axis=1), dtype=torch.float32)
    sources = {}
    for item in args.source:
        name, _, directory = item.partition("=")
        directory = Path(directory)
        matrix = (sparse.load_npz(directory / "embeddings.npz") > 0).tocsc()
        source_rows = json.loads((directory / "word_to_row.json").read_text())
        matrix = matrix[[source_rows[w] for w in words]]
        degree = np.asarray(matrix.sum(axis=0)).ravel()
        keep = np.flatnonzero((degree >= 2) & (degree <= args.max_degree))
        matrix = matrix[:, keep].tocoo()
        sources[name] = {"edges": torch.tensor(np.vstack([matrix.row, matrix.col]), dtype=torch.long),
                         "count": len(keep),
                         "x": torch.tensor(np.log(degree[keep])[:, None], dtype=torch.float32)}
    return words, {w: i for i, w in enumerate(words)}, x, sources


def same_group_edges(puzzles, index):
    pairs = []
    for record in puzzles:
        for answer in record["answers"]:
            ids = [index[w] for w in answer["words"]]
            pairs += [(a, b) for a in ids for b in ids if a != b]
    if not pairs:
        return torch.zeros(2, 0, dtype=torch.long)
    return torch.tensor(pairs, dtype=torch.long).t()


def pair_targets(puzzles, index):
    """All 120 within-puzzle pairs: (2, P) word ids and (P,) 0/1 same-group labels."""
    left, right, labels = [], [], []
    for record in puzzles:
        group = {w: j for j, a in enumerate(record["answers"]) for w in a["words"]}
        ws = record["words"]
        for i in range(16):
            for j in range(i + 1, 16):
                left.append(index[ws[i]])
                right.append(index[ws[j]])
                labels.append(float(group[ws[i]] == group[ws[j]]))
    return torch.tensor([left, right]), torch.tensor(labels)


class GlobalGNN(nn.Module):
    def __init__(self, sources, word_dim, hidden=128, layers=2, dropout=0.2, conv="sage"):
        super().__init__()
        self.word_in = nn.Linear(word_dim, hidden)
        self.concept_in = nn.ModuleDict({name: nn.Linear(1, hidden) for name in sources})
        relations = [("word", "same_group", "word")]
        for name in sources:
            relations += [("word", f"in_{name}", name), (name, f"has_{name}", "word")]
        self.convs = nn.ModuleList(
            HeteroConv({rel: (SAGEConv((hidden, hidden), hidden) if conv == "sage" else
                              GATConv((hidden, hidden), hidden, add_self_loops=False))
                        for rel in relations}, aggr="mean")
            for _ in range(layers))
        self.dropout = dropout
        self.out = nn.Linear(hidden, hidden)
        nn.init.normal_(self.out.weight, std=1e-3)
        nn.init.zeros_(self.out.bias)
        self.temperature = nn.Parameter(torch.tensor(10.0))

    def forward(self, data):
        h = {"word": F.relu(self.word_in(data["word"].x))}
        for name, layer in self.concept_in.items():
            h[name] = F.relu(layer(data[name].x))
        for conv in self.convs:
            update = conv(h, data.edge_index_dict)
            h = {k: F.dropout(F.relu(update[k] + h[k]), self.dropout, self.training) if k in update else h[k]
                 for k in h}
        skip = F.normalize(data["word"].x, dim=-1)
        return F.normalize(torch.cat([skip, self.out(h["word"])], dim=-1), dim=-1)


def build_data(x, sources, group_edges):
    data = HeteroData()
    data["word"].x = x
    data["word", "same_group", "word"].edge_index = group_edges
    for name, source in sources.items():
        data[name].x = source["x"]
        data["word", f"in_{name}", name].edge_index = source["edges"]
        data[name, f"has_{name}", "word"].edge_index = source["edges"].flip(0)
    return data


def train_embeddings(train_puzzles, x, sources, index, args, seed, eval_puzzles=None):
    """Train on the given puzzles; return word embeddings using all their same-group edges."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    model = GlobalGNN(sources, x.shape[1], args.hidden, args.layers, args.dropout, args.conv)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epochs):
        model.train()
        order = rng.sample(train_puzzles, len(train_puzzles))
        parts = [order[i::args.chunks] for i in range(args.chunks)]
        for k, target in enumerate(parts):
            message = [p for j, part in enumerate(parts) if j != k for p in part]
            data = build_data(x, sources, same_group_edges(message, index))
            pairs, labels = pair_targets(target, index)
            emb = model(data)
            logits = model.temperature * (emb[pairs[0]] * emb[pairs[1]]).sum(-1) - model.temperature / 2
            weight = torch.where(labels > 0, 0.5 / labels.mean(), 0.5 / (1 - labels.mean()))
            loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        emb = model(build_data(x, sources, same_group_edges(train_puzzles, index)))
    metrics = {}
    if eval_puzzles:
        pairs, labels = pair_targets(eval_puzzles, index)
        sims = (emb[pairs[0]] * emb[pairs[1]]).sum(-1).numpy()
        order = np.argsort(-sims)
        positives = labels.numpy().astype(bool)
        ranks = np.empty(len(sims))
        ranks[order] = np.arange(len(sims))
        n_pos, n_neg = positives.sum(), (~positives).sum()
        metrics["pair_auc"] = float(1 - (ranks[positives].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg))
    return emb, metrics


def matrices(puzzles, emb, index):
    out = []
    for record in puzzles:
        ids = [index[w] for w in record["words"]]
        m = (emb[ids] @ emb[ids].t()).numpy()
        np.fill_diagonal(m, 1.0)
        out.append(m.astype(np.float32))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--word-dir", type=Path, default=ROOT / "data/embeddings/fasttext-crawl")
    parser.add_argument("--extra-word-dir", type=Path, action="append", default=[],
                        help="More single-vector caches to concatenate as word features")
    parser.add_argument("--source", action="append", default=[], metavar="NAME=DIR",
                        help="Neighbor cache (build_phrase_cache / build_neighbor_cache)")
    parser.add_argument("--max-degree", type=int, default=50)
    parser.add_argument("--folds", type=int, default=5,
                        help="Out-of-fold models for training puzzles; 0 = tuning mode (held-out training AUC)")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--conv", choices=("sage", "gat"), default="sage",
                        help="sage: mean of neighbors; gat: attention over neighbors")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--chunks", type=int, default=2,
                        help="Optimizer steps per epoch (target chunks)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    splits = {s: json.loads((args.data_dir / f"{s}.json").read_text()) for s in ("train", "val", "test")}
    words, index, x, sources = load_graph_inputs(args)
    print(f"Graph: {len(words)} words; " + ", ".join(
        f"{n}: {s['count']} nodes / {s['edges'].shape[1]} links" for n, s in sources.items()), flush=True)

    train = splits["train"]
    if args.folds == 0:  # tuning mode: train on 80% of training puzzles, score the other 20%
        start = time.time()
        split = np.random.default_rng(args.seed).permutation(len(train)) % 5
        inside = [train[i] for i in range(len(train)) if split[i] != 0]
        outside = [train[i] for i in range(len(train)) if split[i] == 0]
        _, m = train_embeddings(inside, x, sources, index, args, args.seed, eval_puzzles=outside)
        print(f"tuning: held-out training pair AUC {m['pair_auc']:.3f} ({time.time() - start:.0f}s)", flush=True)
        return
    folds = np.random.default_rng(args.seed).permutation(len(train)) % args.folds
    keys, rows, metrics = [], [], {"folds": []}
    start = time.time()
    for k in range(args.folds):
        inside = [train[i] for i in range(len(train)) if folds[i] != k]
        outside = [train[i] for i in range(len(train)) if folds[i] == k]
        emb, m = train_embeddings(inside, x, sources, index, args, args.seed + k, eval_puzzles=outside)
        rows += matrices(outside, emb, index)
        keys += [puzzle_key(r["words"]) for r in outside]
        metrics["folds"].append(m)
        print(f"fold {k}: out-of-fold pair AUC {m['pair_auc']:.3f} ({time.time() - start:.0f}s)", flush=True)
    emb, m = train_embeddings(train, x, sources, index, args, args.seed + args.folds, eval_puzzles=splits["val"])
    metrics["val_pair_auc"] = m["pair_auc"]
    print(f"all training puzzles: validation pair AUC {m['pair_auc']:.3f}", flush=True)
    for split in ("val", "test"):
        rows += matrices(splits[split], emb, index)
        keys += [puzzle_key(r["words"]) for r in splits[split]]
    if len(set(keys)) != len(keys):
        raise ValueError("Two puzzles share the same word list; keys must be unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "matrices.npy", np.stack(rows), allow_pickle=False)
    (args.output_dir / "index.json").write_text(json.dumps({k: i for i, k in enumerate(keys)}) + "\n")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config["extra_word_dir"] = [str(p) for p in args.extra_word_dir]
    config.update(metrics=metrics, input_sha256={
        s: hashlib.sha256((args.data_dir / f"{s}.json").read_bytes()).hexdigest() for s in splits})
    (args.output_dir / "metadata.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Saved {len(keys)} puzzle matrices -> {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
