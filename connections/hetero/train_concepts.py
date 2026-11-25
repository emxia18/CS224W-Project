"""Train QuartetGNN + concept-node branch (per-puzzle heterogeneous graph).

Uses the same inputs, baseline-weight fit, optimizer, loop (one step per puzzle,
gradient clipping 1.0, seeded shuffle), evaluation and outputs as
connections/model/train_gnn.py with --residual --loss crf, so runs are directly
comparable. The difference is the concept branch (connections/hetero/model.py),
fed by --concept NAME=DIR neighbor caches. Validation only.

Example (final model + ConceptNet, plus concept nodes from four sources):
  python -m connections.hetero.train_concepts --seed 0 --output-dir runs/hetero/concepts_seed0 \\
    --orthographic-features --homophone-features \\
    --similarity wn=max:data/embeddings/senses/wordnet-mpnet ... \\
    --concept pc=data/embeddings/phrases/numberbatch-open --concept cn=data/embeddings/neighbors/conceptnet
"""

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from connections.features.explicit_features import HomophoneCache
from connections.features.similarity_channels import parse as parse_similarities
from connections.features.word_channels import WordTokenizer
from connections.hetero.concepts import feature_dim, parse_sources, puzzle_concepts
from connections.hetero.model import ConceptQuartetGNN
from connections.model.crf import crf_loss
from connections.model.gnn import QuartetGNN
from connections.model.train_gnn import evaluate, fit_baseline_weights, load_puzzles, save_predictions
from connections.paths import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "data/embeddings/fasttext-crawl")
    parser.add_argument("--similarity", action="append", default=[], metavar="NAME=KIND:DIR")
    parser.add_argument("--concept", action="append", default=[], metavar="NAME=DIR",
                        help="Neighbor cache whose shared neighbors become concept nodes")
    parser.add_argument("--concept-min", type=int, default=2)
    parser.add_argument("--concept-max", type=int, default=8)
    parser.add_argument("--concept-hidden", type=int, default=64)
    parser.add_argument("--orthographic-features", action="store_true")
    parser.add_argument("--homophone-features", action="store_true")
    parser.add_argument("--homophone-cache", type=Path, default=ROOT / "data/embeddings/homophones")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--graph", default="complete")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--val-limit", type=int)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    rng = random.Random(args.seed)

    vectors = np.load(args.embedding_dir / "embeddings.npy", allow_pickle=False)
    mapping = json.loads((args.embedding_dir / "word_to_row.json").read_text())
    homophones = HomophoneCache(args.homophone_cache) if args.homophone_features else None
    if homophones is not None:
        homophones.validate_inputs(args.data_dir)
    tokenizer = (WordTokenizer(pronunciation=False, strip_qualifiers=True,
                               orthographic_features=args.orthographic_features, homophone_cache=homophones)
                 if args.orthographic_features or args.homophone_features else None)
    train = load_puzzles(args.data_dir / "train.json", vectors, mapping, "cpu", tokenizer)[:args.train_limit]
    validation = load_puzzles(args.data_dir / "val.json", vectors, mapping, "cpu", tokenizer)[:args.val_limit]

    channels = parse_similarities(args.similarity)
    sources = parse_sources(args.concept)
    sizes = []
    for puzzle in train + validation:
        words = puzzle["record"]["words"]
        if channels:
            puzzle["similarities"] = torch.tensor(np.stack([c.matrix(words) for c in channels.values()]))
        graph = puzzle_concepts(words, sources, args.concept_min, args.concept_max)
        puzzle["word_inputs"] = dict(puzzle.get("word_inputs") or {}, concept_graph=graph)
        sizes.append(graph["incidence"].shape[0])
    print(f"Concept nodes per puzzle: median {int(np.median(sizes))}, max {max(sizes)} "
          f"(sources: {', '.join(sources) or 'none'})", flush=True)

    base = QuartetGNN(args.hidden, args.dropout, pairwise_features=True,
                      orthographic_features=args.orthographic_features,
                      homophone_features=args.homophone_features, graph=args.graph,
                      input_dim=vectors.shape[1], residual=True, extra_similarities=len(channels))
    weights, loss0 = fit_baseline_weights(train)
    base.residual_scale.data.fill_(weights[0])
    if channels:
        base.similarity_scales.data.copy_(torch.tensor(weights[1:]))
    print(f"Baseline weights fit on {len(train)} training puzzles: a={weights[0]:.3f} "
          + " ".join(f"{n}={w:.3f}" for n, w in zip(channels, weights[1:])) + f" (train crf loss {loss0:.4f})",
          flush=True)
    # Build the concept branch without consuming the global RNG, so that with no
    # concept sources training is identical to train_gnn.py for the same seed.
    rng_state = torch.get_rng_state()
    model = ConceptQuartetGNN(base, vectors.shape[1], max(len(sources), 1),
                              feature_dim(max(len(sources), 1), args.concept_min, args.concept_max),
                              args.concept_hidden, args.dropout)
    torch.set_rng_state(rng_state)
    calibration = ("base.residual_scale", "base.similarity_scales")
    optimizer = torch.optim.AdamW(
        [{"params": [p for n, p in model.named_parameters() if n not in calibration]},
         {"params": [p for n, p in model.named_parameters() if n in calibration], "weight_decay": 0.0}],
        lr=args.lr, weight_decay=args.weight_decay)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(parameters=sum(p.numel() for p in model.parameters()), initial_weights=weights,
                  concept_nodes_median=float(np.median(sizes)))
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    print(f"Parameters: {config['parameters']:,}", flush=True)

    initial, _ = evaluate(model, validation)
    (args.output_dir / "initial_metrics.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(f"Epoch 000 (untrained) groups={initial['mean_correct_groups']:.3f}/4 "
          f"({sum(initial['correct_per_puzzle'])} groups)", flush=True)
    best, history = (-1.0, -1.0), []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for puzzle in rng.sample(train, len(train)):
            scores = model(puzzle["x"], word_inputs=puzzle["word_inputs"], similarities=puzzle.get("similarities"))
            loss = crf_loss(scores, torch.tensor(sorted(puzzle["positives"])))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())
        metrics, _ = evaluate(model, validation)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        **{k: v for k, v in metrics.items() if k != "correct_per_puzzle"}})
        print(f"Epoch {epoch:03d} loss={np.mean(losses):.4f} groups={metrics['mean_correct_groups']:.3f}/4 "
              f"({sum(metrics['correct_per_puzzle'])} groups) exact={metrics['exact_accuracy']:.1%}", flush=True)
        key = (metrics["mean_correct_groups"], metrics["exact_accuracy"])
        if key > best:
            best = key
            torch.save({"model_state": model.state_dict(), "epoch": epoch, "metrics": metrics},
                       args.output_dir / "best.pt")
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n")

    checkpoint = torch.load(args.output_dir / "best.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics, predictions = evaluate(model, validation)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    greedy, _ = evaluate(model, validation, decoder="greedy")
    (args.output_dir / "metrics_greedy.json").write_text(json.dumps(greedy, indent=2) + "\n")
    save_predictions(args.output_dir / "predictions.json", validation, predictions, metrics)
    print(f"Best epoch {checkpoint['epoch']}: {sum(metrics['correct_per_puzzle'])} groups, "
          f"exact {metrics['exact_accuracy']:.1%}", flush=True)


if __name__ == "__main__":
    main()
