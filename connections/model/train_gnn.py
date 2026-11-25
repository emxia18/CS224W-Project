"""Train the GATv2 quartet scorer on cached, frozen word embeddings.

Run from the repository root:
    python -m connections.model.train_gnn --all-quartets --pairwise-features

Loads data/processed/{train,val}.json (never test), trains with balanced BCE over
quartets (or, with --loss crf, a partition-level CRF loss), selects the best epoch
on validation, and writes config.json, history.json, best.pt, metrics.json and
predictions.json to --output-dir.

Best configuration so far (fastText, see results/residual_crf_embeddings.md):
    python -m connections.model.train_gnn --embedding-dir data/embeddings/fasttext-crawl \
        --pairwise-features --residual --loss crf --epochs 8 --patience 8 --seed 0
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
from connections.model.crf import crf_loss
from connections.model.decoding import DECODERS, QUARTET_IDS, QUARTETS
from connections.features.similarity_channels import parse as parse_similarities
from connections.model.gnn import GRAPHS, QuartetGNN, quartet_cosine_sum, quartet_pair_sum
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
        predicted = decode_fn(model(puzzle["x"], hints=hints, word_inputs=puzzle.get("word_inputs"),
                                     similarities=puzzle.get("similarities")).cpu().numpy(), allowed)
        correct_counts.append(len(set(predicted) & puzzle["positives"]))
        overlaps.append(matched_word_overlap(predicted, puzzle["positives"]))
        predictions.append(predicted)
    return {
        "mean_correct_groups": float(np.mean(correct_counts)),
        "exact_accuracy": float(np.mean(np.array(correct_counts) == 4)),
        "mean_matched_words": float(np.mean(overlaps)),
        "word_overlap_accuracy": float(np.mean(overlaps) / 16),
        "seconds_per_puzzle": (time.perf_counter() - start) / len(puzzles),
        "correct_per_puzzle": [int(c) for c in correct_counts],
    }, predictions


def balanced_bce(logits):
    """Positives first. Average each class separately so negatives cannot dominate."""
    return 0.5 * (F.softplus(-logits[..., :4]).mean(-1) + F.softplus(logits[..., 4:]).mean(-1))


@torch.no_grad()
def fit_residual_scale(puzzles, loss_name, high=100.0, iterations=60, batch=32):
    """Initial cosine scale a for the residual scorer, fit on training puzzles only.

    Minimizes the mean training loss of score = a * cosine_sum + b (the residual
    correction is zero at initialization apart from its bias b). Returns
    (a, b, loss). CRF is invariant to b (every partition has four quartets), so
    b = 0 and a golden-section search finds a; the CRF loss is convex in a.
    BCE needs the bias to put logits on the right side of zero (without it, the
    many positive-cosine negatives drive a toward 0), so (a, b) are fit jointly
    with L-BFGS; balanced BCE is convex in (a, b).
    """
    cosine = torch.stack([quartet_cosine_sum(p["x"], torch.tensor(QUARTETS, device=p["x"].device))
                          for p in puzzles])
    positives = torch.tensor([sorted(p["positives"]) for p in puzzles], device=cosine.device)
    if loss_name == "bce":
        order = torch.stack([torch.cat((pos, torch.tensor(p["negatives"], device=cosine.device)))
                             for pos, p in zip(positives, puzzles)])
        cosine = cosine.gather(1, order)

    if loss_name == "bce":
        with torch.enable_grad():
            params = torch.zeros(2, device=cosine.device, requires_grad=True)
            optimizer = torch.optim.LBFGS([params], max_iter=200, line_search_fn="strong_wolfe")

            def closure():
                optimizer.zero_grad()
                value = balanced_bce(params[0] * cosine + params[1]).mean()
                value.backward()
                return value

            optimizer.step(closure)
        scale, bias = params.tolist()
        return scale, bias, float(balanced_bce(scale * cosine + bias).mean())

    def loss(a):
        return float(torch.cat([crf_loss(a * cosine[i:i + batch], positives[i:i + batch])
                                for i in range(0, len(puzzles), batch)]).mean())

    ratio = (5 ** 0.5 - 1) / 2
    low, top = 0.0, high
    left, right = top - ratio * (top - low), low + ratio * (top - low)
    f_left, f_right = loss(left), loss(right)
    for _ in range(iterations):
        if f_left < f_right:
            top, right, f_right = right, left, f_left
            left = top - ratio * (top - low)
            f_left = loss(left)
        else:
            low, left, f_left = left, right, f_right
            right = low + ratio * (top - low)
            f_right = loss(right)
    scale = (low + top) / 2
    if scale > 0.95 * high:
        raise RuntimeError(f"Residual scale search hit its upper bound ({high}); raise it")
    return scale, 0.0, loss(scale)


def fit_baseline_weights(puzzles, iterations=200, batch=32):
    """Initial (a, b_1..b_K) for score = a * cosine_sum + sum_k b_k * similarity_sum_k.

    Fit on training puzzles only by minimizing the CRF loss, which is convex in the
    weights (log-sum-exp of linear functions minus a linear function). Returns
    (weights, loss). Accumulated over batches: the CRF tables are large.
    """
    quartets = torch.tensor(QUARTETS, device=puzzles[0]["x"].device)
    with torch.no_grad():
        channels = torch.stack([torch.stack([quartet_cosine_sum(p["x"], quartets)]
                                            + [quartet_pair_sum(m, quartets) for m in p["similarities"]])
                                for p in puzzles]).double()
    positives = torch.tensor([sorted(p["positives"]) for p in puzzles], device=channels.device)
    weights = torch.full((channels.shape[1],), 5.0 / channels.shape[1], dtype=torch.float64,
                         device=channels.device, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=iterations, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        total = 0.0
        for i in range(0, len(puzzles), batch):
            scores = torch.einsum("pkq,k->pq", channels[i:i + batch], weights)
            loss = crf_loss(scores, positives[i:i + batch]).sum() / len(puzzles)
            loss.backward()
            total += loss.item()
        return torch.tensor(total, dtype=torch.float64)

    optimizer.step(closure)
    return weights.detach().cpu().tolist(), float(closure())


def save_predictions(path, puzzles, predictions, metrics):
    """Per-puzzle predictions, for paired comparisons between runs."""
    path.write_text(json.dumps([{
        "date": puzzle["record"].get("date"),
        "correct_groups": correct,
        "predicted": [[puzzle["record"]["words"][i] for i in QUARTETS[index]] for index in predicted],
    } for puzzle, predicted, correct in zip(puzzles, predictions, metrics["correct_per_puzzle"])],
        indent=1) + "\n")


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
    parser.add_argument("--residual", action="store_true",
                        help="Score = a * cosine_sum + learned correction (starts at the cosine baseline)")
    parser.add_argument("--similarity", action="append", default=[], metavar="NAME=KIND:DIR",
                        help="Extra word-similarity source (cos:DIR single-vector cache, max:DIR "
                             "sense cache); repeatable. Edge feature, readout statistics and, with "
                             "--residual, an extra baseline term")
    parser.add_argument("--loss", choices=("bce", "crf"), default="bce",
                        help="bce: balanced BCE per quartet; crf: log-likelihood of the gold partition "
                             "over all 2,627,625 partitions (uses all 1,820 quartets)")
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
        if args.loss == "crf":
            parser.error("--loss crf does not support hint curricula")
    if args.similarity and args.residual and args.loss != "crf":
        parser.error("--similarity with --residual needs --loss crf (weights are fit with the CRF)")
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
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        parser.error("Expected a finite 2-D embedding matrix")
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
    similarity_channels = parse_similarities(args.similarity)
    if similarity_channels:
        for puzzle in train + ([] if args.overfit else validation):
            words = puzzle["record"]["words"]
            puzzle["similarities"] = torch.tensor(
                np.stack([channel.matrix(words) for channel in similarity_channels.values()]),
                device=args.device)
        print(f"Extra similarities: {', '.join(similarity_channels)}", flush=True)
    diagnostic_subset = random.Random(args.seed + 2).sample(
        train, min(args.diagnostic_puzzles, len(train))
    )

    input_dim = train[0]["x"].shape[1]
    model = QuartetGNN(args.hidden, args.dropout, args.pairwise_features, args.hint_features,
                       args.learned_relations, args.spelling_features, args.pronunciation_features,
                       args.orthographic_features, args.homophone_features, args.graph,
                       input_dim=input_dim, residual=args.residual,
                       extra_similarities=len(similarity_channels)).to(args.device)
    initial_scale, initial_similarity_scales = None, None
    if args.residual and similarity_channels:
        (initial_scale, *initial_similarity_scales), initial_loss = fit_baseline_weights(train)
        initial_bias = 0.0
        model.residual_scale.data.fill_(initial_scale)
        model.similarity_scales.data.copy_(torch.tensor(initial_similarity_scales))
        print(f"Baseline weights fit on {len(train)} training puzzles: a={initial_scale:.3f} "
              + " ".join(f"{name}={w:.3f}" for name, w in zip(similarity_channels, initial_similarity_scales))
              + f" (train crf loss {initial_loss:.4f})", flush=True)
    elif args.residual:
        initial_scale, initial_bias, initial_loss = fit_residual_scale(train, args.loss)
        model.residual_scale.data.fill_(initial_scale)
        model.rho[-1].bias.data.fill_(initial_bias)
        print(f"Residual scale fit on {len(train)} training puzzles: a={initial_scale:.3f} "
              f"b={initial_bias:.3f} "
              f"(train {args.loss} loss {initial_loss:.4f})", flush=True)
    # The baseline weights are calibration constants, so they are not decayed toward 0.
    calibration = ("residual_scale", "similarity_scales")
    decayed = [p for name, p in model.named_parameters() if name not in calibration]
    undecayed = [p for name, p in model.named_parameters() if name in calibration]
    optimizer = torch.optim.AdamW([{"params": decayed}, {"params": undecayed, "weight_decay": 0.0}],
                                  lr=args.lr, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["parameters"] = sum(p.numel() for p in model.parameters())
    config["input_dim"] = input_dim
    config["initial_residual_scale"] = initial_scale
    config["initial_residual_bias"] = initial_bias if args.residual else None
    config["initial_similarity_scales"] = (dict(zip(similarity_channels, initial_similarity_scales))
                                           if initial_similarity_scales else None)
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
        "all 1820, scored as whole partitions (CRF)" if args.loss == "crf" else
        "all hint-compatible quartets" if args.curriculum != "none" else
        "1820 (4 positives + 1816 negatives)" if args.all_quartets or args.overfit
        else f"{4 + args.negatives} (4 positives + {args.negatives} sampled negatives)"
    ), flush=True)
    # Epoch 0: the untrained model. With --residual this equals the cosine baseline.
    initial, _ = evaluate(model, validation)
    (args.output_dir / "initial_metrics.json").write_text(json.dumps(initial, indent=2) + "\n")
    print(f"Epoch 000 (untrained) groups={initial['mean_correct_groups']:.3f}/4 "
          f"({sum(initial['correct_per_puzzle'])} groups) exact={initial['exact_accuracy']:.1%}", flush=True)
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
            elif args.loss == "crf" or args.all_quartets or args.overfit:
                negatives = puzzle["negatives"]
            else:
                hard_count = min(args.negatives // 2, len(puzzle["hard_negatives"]))
                hard = rng.sample(puzzle["hard_negatives"], hard_count)
                remaining = sorted(set(puzzle["negatives"]) - set(hard))
                negatives = hard + rng.sample(remaining, args.negatives - hard_count)
            if args.loss == "crf":
                scores = model(puzzle["x"], hints=hints, word_inputs=puzzle.get("word_inputs"),
                               similarities=puzzle.get("similarities"))
                loss = crf_loss(scores, torch.tensor(sorted(puzzle["positives"]), device=args.device))
            else:
                ids = sorted(puzzle["positives"]) + negatives
                logits = model(puzzle["x"], torch.tensor(ids, device=args.device), hints=hints,
                               word_inputs=puzzle.get("word_inputs"),
                               similarities=puzzle.get("similarities"))
                loss = balanced_bce(logits)
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
                        "residual_scale": model.residual_scale.item() if args.residual else None,
                        "similarity_scales": (dict(zip(similarity_channels, model.similarity_scales.tolist()))
                                              if args.residual and similarity_channels else None),
                        "training_diagnostics": diagnostics,
                        "task_counts": task_counts, "task_probabilities": probabilities,
                        "task_losses": {task: float(np.mean(values))
                                        for task, values in task_losses.items() if values}})
        print(f"Epoch {epoch:03d} loss={np.mean(losses):.4f} groups={metrics['mean_correct_groups']:.3f}/4 "
              f"overlap={metrics['mean_matched_words']:.2f}/16 "
              f"exact={metrics['exact_accuracy']:.1%}"
              + (f" a={model.residual_scale.item():.3f}" if args.residual else "")
              + ("".join(f" {name}={w:.3f}" for name, w in zip(similarity_channels, model.similarity_scales.tolist()))
                 if args.residual and similarity_channels else ""), flush=True)
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
    save_predictions(args.output_dir / "predictions.json", validation, predictions, metrics)
    print(f"Greedy decoding: groups={greedy_metrics['mean_correct_groups']:.3f}/4 "
          f"exact={greedy_metrics['exact_accuracy']:.1%}", flush=True)
    if diagnostic_subset:
        diagnostics = training_diagnostics(model, diagnostic_subset, args.seed + 3)
        (args.output_dir / "training_diagnostics.json").write_text(json.dumps({
            "checkpoint_epoch": checkpoint["epoch"], "metrics": diagnostics,
        }, indent=2) + "\n")
    example = validation[0]["record"]
    summary = {key: value for key, value in metrics.items() if key != "correct_per_puzzle"}
    print(f"\nBest epoch: {checkpoint['epoch']}; metrics: {summary}")
    print(f"Example: {example['contest']}")
    for index in predictions[0]:
        print("Predicted:", [example["words"][i] for i in QUARTETS[index]])
    for answer in example["answers"]:
        print("Official:", answer["words"])


if __name__ == "__main__":
    main()
