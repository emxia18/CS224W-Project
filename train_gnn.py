"""Train a GATv2 quartet scorer using cached, frozen word embeddings."""

import argparse
from functools import lru_cache
from itertools import combinations
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv


ROOT = Path(__file__).resolve().parent
QUARTETS = list(combinations(range(16), 4))
QUARTET_IDS = {group: index for index, group in enumerate(QUARTETS)}
FULL_MASK = (1 << 16) - 1
EDGE_INDEX = torch.tensor(
    [(i, j) for i in range(16) for j in range(16)], dtype=torch.long
).t().contiguous()


@lru_cache(maxsize=None)
def partition_options(mask):
    """Anchor each group on the first remaining word to avoid duplicate partitions."""
    remaining = [i for i in range(16) if mask & (1 << i)]
    options = []
    for rest in combinations(remaining[1:], 3):
        group = (remaining[0], *rest)
        group_mask = sum(1 << i for i in group)
        options.append((QUARTET_IDS[group], mask ^ group_mask))
    return options


def decode(scores):
    """Return quartet indices for the maximum-total-logit exact partition."""
    scores = np.asarray(scores)
    if scores.shape != (1820,) or not np.isfinite(scores).all():
        raise ValueError("Decoder requires 1,820 finite scores")
    choices = {}

    @lru_cache(maxsize=None)
    def solve(mask):
        if mask == 0:
            return 0.0
        best, choice = max(
            (float(scores[index]) + solve(rest), (index, rest))
            for index, rest in partition_options(mask)
        )
        choices[mask] = choice
        return best

    solve(FULL_MASK)
    groups, mask = [], FULL_MASK
    while mask:
        index, mask = choices[mask]
        groups.append(index)
    return groups


class QuartetGNN(nn.Module):
    def __init__(self, hidden=128, dropout=0.1):
        super().__init__()
        self.project = nn.Linear(768, hidden)
        self.gat = GATv2Conv(
            hidden, hidden, heads=1, edge_dim=1, dropout=dropout,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.phi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.register_buffer("edge_index", EDGE_INDEX.clone())
        self.register_buffer("quartets", torch.tensor(QUARTETS, dtype=torch.long))

    def forward(self, x, quartet_ids=None):
        # Cached embeddings are unit length, so dot products are cosine similarities.
        source, target = self.edge_index
        edge_attr = (x[source] * x[target]).sum(dim=-1, keepdim=True)
        h = F.relu(self.project(x))
        h = self.norm(h + self.dropout(F.elu(self.gat(h, self.edge_index, edge_attr))))
        groups = self.quartets if quartet_ids is None else self.quartets[quartet_ids]
        pooled = self.phi(h)[groups].sum(dim=1)
        return self.rho(pooled).squeeze(-1)


def load_puzzles(path, vectors, word_to_row, device):
    records = json.loads(path.read_text(encoding="utf-8"))
    puzzles = []
    for record in records:
        words = record["words"]
        positions = {word: index for index, word in enumerate(words)}
        positives = {
            QUARTET_IDS[tuple(sorted(positions[word] for word in answer["words"]))]
            for answer in record["answers"]
        }
        puzzles.append({
            "record": record,
            "x": torch.tensor(vectors[[word_to_row[word] for word in words]], device=device),
            "positives": positives,
            "negatives": sorted(set(range(1820)) - positives),
        })
    return puzzles


@torch.no_grad()
def evaluate(model, puzzles):
    model.eval()
    correct_counts, predictions = [], []
    start = time.perf_counter()
    for puzzle in puzzles:
        predicted = decode(model(puzzle["x"]).cpu().numpy())
        correct_counts.append(len(set(predicted) & puzzle["positives"]))
        predictions.append(predicted)
    return {
        "mean_correct_groups": float(np.mean(correct_counts)),
        "exact_accuracy": float(np.mean(np.array(correct_counts) == 4)),
        "seconds_per_puzzle": (time.perf_counter() - start) / len(puzzles),
    }, predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/gnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; small graphs favor few threads")
    parser.add_argument("--train-limit", type=int, help="Use only the first N training puzzles")
    parser.add_argument("--val-limit", type=int, help="Use only the first N validation puzzles")
    parser.add_argument("--overfit", type=int, metavar="N", help="Train and evaluate on N training puzzles")
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.hidden, args.threads) < 1
            or not 1 <= args.negatives <= 1816 or not 0 <= args.dropout < 1
            or args.lr <= 0 or args.weight_decay < 0):
        parser.error("Invalid training hyperparameters")
    for limit in (args.train_limit, args.val_limit, args.overfit):
        if limit is not None and limit < 1:
            parser.error("Puzzle limits must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)

    vectors = np.load(args.embedding_dir / "embeddings.npy", allow_pickle=False)
    if vectors.ndim != 2 or vectors.shape[1] != 768 or not np.isfinite(vectors).all():
        parser.error("Expected finite 768-dimensional cached embeddings")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        parser.error("Cached embeddings must be normalized")
    mapping = json.loads((args.embedding_dir / "word_to_row.json").read_text())
    train = load_puzzles(args.data_dir / "train.json", vectors, mapping, args.device)
    if args.overfit:
        train = train[:args.overfit]
        validation = train
        print("OVERFIT CHECK: evaluation uses training puzzles, not held-out validation.")
    else:
        train = train[:args.train_limit]
        validation = load_puzzles(args.data_dir / "val.json", vectors, mapping, args.device)[:args.val_limit]
    if not train or not validation:
        parser.error("Training and evaluation sets must be nonempty")

    model = QuartetGNN(args.hidden, args.dropout).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Training on {len(train)} puzzles; evaluating on {len(validation)}; device={args.device}", flush=True)
    best, stale, history = (-1.0, -1.0), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for puzzle in rng.sample(train, len(train)):
            ids = sorted(puzzle["positives"]) + rng.sample(puzzle["negatives"], args.negatives)
            logits = model(puzzle["x"], torch.tensor(ids, device=args.device))
            # Give the four positives and the sampled negatives equal total weight.
            loss = 0.5 * (F.softplus(-logits[:4]).mean() + F.softplus(logits[4:]).mean())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, validation)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics})
        print(f"Epoch {epoch:03d} loss={np.mean(losses):.4f} groups={metrics['mean_correct_groups']:.3f}/4 "
              f"exact={metrics['exact_accuracy']:.1%}", flush=True)
        key = (metrics["mean_correct_groups"], metrics["exact_accuracy"])
        if key > best:
            best, stale = key, 0
            torch.save({"model_state": model.state_dict(), "config": config,
                        "epoch": epoch, "metrics": metrics}, args.output_dir / "best.pt")
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        if stale >= args.patience:
            print("Early stopping.")
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics, predictions = evaluate(model, validation)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    example = validation[0]["record"]
    print(f"\nBest epoch: {checkpoint['epoch']}; metrics: {metrics}")
    print(f"Example: {example['contest']}")
    for index in predictions[0]:
        print("Predicted:", [example["words"][i] for i in QUARTETS[index]])
    for answer in example["answers"]:
        print("Official:", answer["words"])


if __name__ == "__main__":
    main()
