"""Train the GATv2 quartet scorer on cached, frozen word embeddings.

Run from the repository root:
    python -m connections.model.train_gnn --all-quartets --pairwise-features

Loads data/processed/{train,val}.json (never test), trains with balanced BCE over
quartets, selects the best epoch on validation, and writes config.json,
history.json, best.pt and metrics.json to --output-dir.
"""

import argparse
from importlib.metadata import version
from itertools import permutations
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from connections.features.explicit_features import EDGE_FEATURES, NODE_FEATURES, HomophoneCache
from connections.features.word_channels import WordTokenizer
from connections.model.curriculum import TASKS, compatible_quartets, sample_hints, task_probabilities
from connections.model.decoding import DECODERS, QUARTET_IDS, QUARTETS
from connections.model.gnn import GRAPHS, QuartetGNN
from connections.paths import ROOT


def load_puzzles(path, vectors, word_to_row, device, tokenizer=None, contextual=None):
    if contextual is not None:
        contextual.validate_split(path)
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
            "word_inputs": tokenizer.encode(words, device) if tokenizer is not None else None,
            "x": torch.tensor(contextual.lookup(words) if contextual is not None else
                              vectors[[word_to_row[word] for word in words]], device=device),
            "positives": positives,
            "negatives": sorted(set(range(1820)) - positives),
            "hard_negatives": [
                index for index, group in enumerate(QUARTETS)
                if index not in positives and any(
                    len(set(group) & set(QUARTETS[pos])) == 3 for pos in positives
                )
            ],
        })
    return puzzles


def matched_word_overlap(predicted, official):
    """Maximum word overlap under a one-to-one matching of four groups (0–16)."""
    predicted = [set(QUARTETS[index]) for index in predicted]
    official = [set(QUARTETS[index]) for index in sorted(official)]
    return max(
        sum(len(predicted[i] & official[j]) for i, j in enumerate(order))
        for order in permutations(range(4))
    )


@torch.no_grad()
def evaluate(model, puzzles, hint_task="full", hint_seed=12345, decoder="exact"):
    """Evaluate fixed hints when requested; default is always unassisted."""
    model.eval()
    decode_fn = DECODERS[decoder]
    hint_rng = random.Random(hint_seed)
    correct_counts, overlaps, predictions = [], [], []
    start = time.perf_counter()
    for puzzle in puzzles:
        hints, allowed = None, None
        if hint_task != "full":
            official = [QUARTETS[index] for index in sorted(puzzle["positives"])]
            hints, revealed = sample_hints(official, hint_task, hint_rng, puzzle["x"].device)
            allowed = compatible_quartets(QUARTETS, revealed)
        predicted = decode_fn(model(puzzle["x"], hints=hints, word_inputs=puzzle.get("word_inputs")).cpu().numpy(), allowed)
        correct_counts.append(len(set(predicted) & puzzle["positives"]))
        overlaps.append(matched_word_overlap(predicted, puzzle["positives"]))
        predictions.append(predicted)
    return {
        "mean_correct_groups": float(np.mean(correct_counts)),
        "exact_accuracy": float(np.mean(np.array(correct_counts) == 4)),
        "mean_matched_words": float(np.mean(overlaps)),
        "word_overlap_accuracy": float(np.mean(overlaps) / 16),
        "seconds_per_puzzle": (time.perf_counter() - start) / len(puzzles),
    }, predictions


def training_diagnostics(model, puzzles, seed):
    """Measure completion and partition recovery on the same training subset."""
    return {
        task: evaluate(model, puzzles, hint_task=task, hint_seed=seed)[0]
        for task in ("full", "triplets", "pairs")
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--contextual-dir", type=Path,
                        help="Use cached puzzle-contextual node embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/gnn")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--selection-metric", choices=("groups", "overlap"), default="groups",
                        help="Validation metric for checkpoint selection and early stopping")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pairwise-features", action="store_true",
                        help="Append mean/min/max/std of quartet cosine similarities")
    parser.add_argument("--learned-relations", type=int, default=0,
                        help="Number of trainable semantic similarity projections (try 4)")
    parser.add_argument("--spelling-features", action="store_true", help="Train a character CNN")
    parser.add_argument("--pronunciation-features", action="store_true", help="Train a CMUdict phoneme CNN")
    parser.add_argument("--graph", choices=tuple(GRAPHS), default="complete",
                        help="complete: message passing over all word pairs; "
                             "none: self-loops only (no-message-passing control, same parameters)")
    parser.add_argument("--orthographic-features", action="store_true", help="Explicit spelling and phonetic relationships")
    parser.add_argument("--homophone-features", action="store_true", help="Semantic similarity of retrieved homophone candidates")
    parser.add_argument("--homophone-cache", type=Path, default=ROOT / "data/embeddings/homophones")
    parser.add_argument("--keep-pronunciation-qualifiers", action="store_true", help="Legacy pronunciation preprocessing for comparisons")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--negatives", type=int, default=32)
    parser.add_argument("--all-quartets", action="store_true",
                        help="Train on all 4 positives and 1,816 negatives per puzzle")
    parser.add_argument("--curriculum", choices=("none", "fixed", "progressive"), default="none",
                        help="Training-only membership hints; validation is always unassisted")
    parser.add_argument("--hint-features", action="store_true",
                        help="Include hint edge channel even without curriculum, for matched comparisons")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1, help="CPU threads; small graphs favor few threads")
    parser.add_argument("--train-limit", type=int, help="Use only the first N training puzzles")
    parser.add_argument("--val-limit", type=int, help="Use only the first N validation puzzles")
    parser.add_argument("--diagnostic-puzzles", type=int, default=16,
                        help="Fixed training subset for full/triplet/pair diagnostics; 0 disables")
    parser.add_argument("--overfit", type=int, metavar="N", help="Train and evaluate on N training puzzles")
    args = parser.parse_args()
    if args.curriculum != "none":
        if not args.all_quartets:
            parser.error("Curriculum requires --all-quartets (all hint-compatible candidates)")
        if args.overfit:
            parser.error("Run curriculum on held-out validation; omit --overfit")
        args.hint_features = True
    if (min(args.epochs, args.patience, args.hidden, args.threads) < 1
            or not 1 <= args.negatives <= 1816 or not 0 <= args.dropout < 1
            or args.lr <= 0 or args.weight_decay < 0):
        parser.error("Invalid training hyperparameters")
    if args.learned_relations < 0:
        parser.error("--learned-relations must be nonnegative")
    if args.diagnostic_puzzles < 0:
        parser.error("--diagnostic-puzzles must be nonnegative")
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
    contextual = None
    if args.contextual_dir is not None:
        from connections.embeddings.contextual_embeddings import ContextualCache
        contextual = ContextualCache(args.contextual_dir)
    if args.homophone_features and args.keep_pronunciation_qualifiers:
        parser.error("Homophone cache requires qualifier-stripping pronunciation preprocessing")
    homophones = HomophoneCache(args.homophone_cache) if args.homophone_features else None
    if homophones is not None:
        homophones.validate_inputs(args.data_dir)
    tokenizer = (WordTokenizer(pronunciation=args.pronunciation_features,
                              strip_qualifiers=not args.keep_pronunciation_qualifiers,
                              orthographic_features=args.orthographic_features, homophone_cache=homophones)
                 if args.spelling_features or args.pronunciation_features or args.orthographic_features
                 or args.homophone_features else None)
    train = load_puzzles(args.data_dir / "train.json", vectors, mapping, args.device, tokenizer, contextual)
    if args.overfit:
        train = train[:args.overfit]
        validation = train
        print("OVERFIT CHECK: evaluation uses training puzzles, not held-out validation.")
    else:
        train = train[:args.train_limit]
        validation = load_puzzles(args.data_dir / "val.json", vectors, mapping, args.device, tokenizer, contextual)[:args.val_limit]
    if not train or not validation:
        parser.error("Training and evaluation sets must be nonempty")
    diagnostic_subset = random.Random(args.seed + 2).sample(
        train, min(args.diagnostic_puzzles, len(train))
    )

    model = QuartetGNN(args.hidden, args.dropout, args.pairwise_features, args.hint_features,
                       args.learned_relations, args.spelling_features, args.pronunciation_features,
                       args.orthographic_features, args.homophone_features, args.graph).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["parameters"] = sum(p.numel() for p in model.parameters())
    print(f"Graph={args.graph}; parameters={config['parameters']:,}", flush=True)
    if tokenizer is not None:
        config["word_channel_tokenization"] = ("lowercase-utf8-cmudict-stressless-v1"
            if args.keep_pronunciation_qualifiers else "strip-parenthetical-stressless-v2")
    if args.pronunciation_features or args.orthographic_features or args.homophone_features:
        config["cmudict_version"] = version("cmudict")
    if homophones is not None:
        config["homophone_cache_metadata"] = homophones.metadata
    if args.orthographic_features:
        config["orthographic_node_features"] = NODE_FEATURES
        config["orthographic_edge_features"] = EDGE_FEATURES
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (args.output_dir / "diagnostic_subset.json").write_text(json.dumps({
        "dates": [puzzle["record"]["date"] for puzzle in diagnostic_subset],
        "subset_seed": args.seed + 2, "hint_seed": args.seed + 3,
    }, indent=2) + "\n")
    print(f"Training on {len(train)} puzzles; evaluating on {len(validation)}; device={args.device}", flush=True)
    if args.pronunciation_features:
        for name, puzzles in (("train", train), ("evaluation", validation)):
            known = sum(int(p["word_inputs"]["phonemes"].ne(0).any(-1).any(-1).sum()) for p in puzzles)
            print(f"CMUdict coverage [{name}]: {known}/{16 * len(puzzles)} word occurrences", flush=True)
    print("Training quartets per puzzle: " + (
        "all hint-compatible quartets" if args.curriculum != "none" else
        "1820 (4 positives + 1816 negatives)" if args.all_quartets or args.overfit
        else f"{4 + args.negatives} (4 positives + {args.negatives} sampled negatives)"
    ), flush=True)
    best, stale, history = (-1.0, -1.0), 0, []
    # Use a separate RNG so full-puzzle comparisons retain their original sampling.
    hint_rng = random.Random(args.seed + 1)
    print(f"Curriculum={args.curriculum}; validation always has zero membership hints.", flush=True)
    if diagnostic_subset:
        print(f"Training diagnostics: {len(diagnostic_subset)} fixed puzzles, fixed hints; "
              f"hint edge channel={'enabled' if args.hint_features else 'disabled (decoder constraints only)' }.",
              flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        task_counts = dict.fromkeys(TASKS, 0)
        task_losses = {task: [] for task in TASKS}
        probabilities = task_probabilities(args.curriculum, epoch, args.epochs)
        for puzzle in rng.sample(train, len(train)):
            task = hint_rng.choices(TASKS, weights=probabilities)[0]
            hints = None
            if task != "full":
                official = [QUARTETS[index] for index in sorted(puzzle["positives"])]
                hints, revealed = sample_hints(official, task, hint_rng, args.device)
                eligible = compatible_quartets(QUARTETS, revealed)
                negatives = [index for index in eligible if index not in puzzle["positives"]]
            elif args.all_quartets or args.overfit:
                negatives = puzzle["negatives"]
            else:
                hard_count = min(args.negatives // 2, len(puzzle["hard_negatives"]))
                hard = rng.sample(puzzle["hard_negatives"], hard_count)
                remaining = sorted(set(puzzle["negatives"]) - set(hard))
                negatives = hard + rng.sample(remaining, args.negatives - hard_count)
            ids = sorted(puzzle["positives"]) + negatives
            logits = model(puzzle["x"], torch.tensor(ids, device=args.device), hints=hints,
                           word_inputs=puzzle.get("word_inputs"))
            # Average each class separately so negatives cannot dominate the loss.
            loss = 0.5 * (F.softplus(-logits[:4]).mean() + F.softplus(logits[4:]).mean())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
            task_counts[task] += 1
            task_losses[task].append(loss.item())
        metrics, _ = evaluate(model, validation)
        diagnostics = (training_diagnostics(model, diagnostic_subset, args.seed + 3)
                       if diagnostic_subset else {})
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), **metrics,
                        "training_diagnostics": diagnostics,
                        "task_counts": task_counts, "task_probabilities": probabilities,
                        "task_losses": {task: float(np.mean(values))
                                        for task, values in task_losses.items() if values}})
        print(f"Epoch {epoch:03d} loss={np.mean(losses):.4f} groups={metrics['mean_correct_groups']:.3f}/4 "
              f"overlap={metrics['mean_matched_words']:.2f}/16 "
              f"exact={metrics['exact_accuracy']:.1%}", flush=True)
        if args.curriculum != "none":
            print(f"  Training tasks: {task_counts}", flush=True)
        for task, diagnostic in diagnostics.items():
            print(f"  Train diagnostic [{task}]: groups={diagnostic['mean_correct_groups']:.3f}/4 "
                  f"overlap={diagnostic['mean_matched_words']:.2f}/16 "
                  f"exact={diagnostic['exact_accuracy']:.1%}", flush=True)
        primary = "mean_matched_words" if args.selection_metric == "overlap" else "mean_correct_groups"
        key = (metrics[primary], metrics["exact_accuracy"])
        if key > best:
            best, stale = key, 0
            torch.save({"model_state": model.state_dict(), "config": config,
                        "epoch": epoch, "metrics": metrics}, args.output_dir / "best.pt")
        else:
            stale += 1
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        # Let every progressive stage run before counting stopping patience.
        if args.curriculum == "progressive" and probabilities[-1] < 1.0:
            stale = 0
        if stale >= args.patience:
            print("Early stopping.")
            break

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics, predictions = evaluate(model, validation)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    # Same checkpoint, greedy decoding: isolates the benefit of joint partitioning.
    greedy_metrics, _ = evaluate(model, validation, decoder="greedy")
    (args.output_dir / "metrics_greedy.json").write_text(json.dumps(greedy_metrics, indent=2) + "\n")
    print(f"Greedy decoding: groups={greedy_metrics['mean_correct_groups']:.3f}/4 "
          f"exact={greedy_metrics['exact_accuracy']:.1%}", flush=True)
    if diagnostic_subset:
        diagnostics = training_diagnostics(model, diagnostic_subset, args.seed + 3)
        (args.output_dir / "training_diagnostics.json").write_text(json.dumps({
            "checkpoint_epoch": checkpoint["epoch"], "metrics": diagnostics,
        }, indent=2) + "\n")
    example = validation[0]["record"]
    print(f"\nBest epoch: {checkpoint['epoch']}; metrics: {metrics}")
    print(f"Example: {example['contest']}")
    for index in predictions[0]:
        print("Predicted:", [example["words"][i] for i in QUARTETS[index]])
    for answer in example["answers"]:
        print("Official:", answer["words"])


if __name__ == "__main__":
    main()
