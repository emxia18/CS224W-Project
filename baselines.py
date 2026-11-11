"""Zero-parameter baselines: random partitions and summed cosine similarity.

Nothing here is trained or selected on validation data, so these numbers are
directly comparable across splits. Test is excluded unless requested explicitly.
"""

import argparse
from itertools import combinations
import json
from pathlib import Path
import random
import time

import numpy as np

from train_gnn import DECODERS, QUARTETS, QUARTET_IDS, load_puzzles, matched_word_overlap


ROOT = Path(__file__).resolve().parent
PAIRS = np.array(list(combinations(range(4), 2)))
QUARTET_ARRAY = np.array(QUARTETS)
# A fixed 4-set is a block in 5,775 of the 2,627,625 partitions (those of the
# other 12 words), so each official group is recovered with probability 1/455.
RANDOM_GROUPS_EXPECTED = 4 / 455


def cosine_sum_scores(x):
    """Sum of the six pairwise cosines for every quartet (embeddings are unit length)."""
    similarity = x @ x.T
    return similarity[QUARTET_ARRAY[:, PAIRS[:, 0]], QUARTET_ARRAY[:, PAIRS[:, 1]]].sum(axis=1)


def summarize(correct, overlaps, seconds):
    correct, overlaps = np.asarray(correct), np.asarray(overlaps)
    return {
        "puzzles": len(correct),
        "mean_correct_groups": float(correct.mean()),
        "total_correct_groups": int(correct.sum()),
        "exact_accuracy": float((correct == 4).mean()),
        "exact_puzzles": int((correct == 4).sum()),
        "mean_matched_words": float(overlaps.mean()),
        "seconds_per_puzzle": seconds / len(correct),
    }


def evaluate_scores(puzzles, decoder):
    correct, overlaps = [], []
    start = time.perf_counter()
    for puzzle in puzzles:
        predicted = DECODERS[decoder](cosine_sum_scores(puzzle["x"]))
        correct.append(len(set(predicted) & puzzle["positives"]))
        overlaps.append(matched_word_overlap(predicted, puzzle["positives"]))
    return summarize(correct, overlaps, time.perf_counter() - start)


def evaluate_random(puzzles, samples, rng):
    """Monte Carlo over uniform random partitions (no decoder involved)."""
    correct, overlaps = [], []
    start = time.perf_counter()
    for puzzle in puzzles:
        for _ in range(samples):
            order = rng.sample(range(16), 16)
            predicted = [QUARTET_IDS[tuple(sorted(order[k:k + 4]))] for k in range(0, 16, 4)]
            correct.append(len(set(predicted) & puzzle["positives"]))
            overlaps.append(matched_word_overlap(predicted, puzzle["positives"]))
    result = summarize(correct, overlaps, time.perf_counter() - start)
    result["puzzles"] = len(puzzles)
    result["samples_per_puzzle"] = samples
    result["analytic_mean_correct_groups"] = RANDOM_GROUPS_EXPECTED
    result["analytic_exact_accuracy"] = 1 / 2_627_625
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--embedding-dir", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "runs/baselines")
    parser.add_argument("--splits", nargs="+", default=["train", "val"],
                        choices=("train", "val", "test"))
    parser.add_argument("--random-samples", type=int, default=200,
                        help="Random partitions per puzzle for the chance baseline")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.random_samples < 1:
        parser.error("--random-samples must be positive")

    vectors = np.load(args.embedding_dir / "embeddings.npy", allow_pickle=False)
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5):
        parser.error("Cached embeddings must be normalized")
    mapping = json.loads((args.embedding_dir / "word_to_row.json").read_text())

    results = {}
    for split in args.splits:
        puzzles = load_puzzles(args.data_dir / f"{split}.json", vectors, mapping, "cpu")
        for puzzle in puzzles:
            puzzle["x"] = puzzle["x"].numpy().astype(np.float64)
        results[split] = {
            "random": evaluate_random(puzzles, args.random_samples, random.Random(args.seed)),
            "cosine_exact": evaluate_scores(puzzles, "exact"),
            "cosine_greedy": evaluate_scores(puzzles, "greedy"),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((args.embedding_dir / "metadata.json").read_text())
    (args.output_dir / "metrics.json").write_text(json.dumps({
        "embedding_model": metadata.get("model"), "embedding_revision": metadata.get("revision"),
        "random_seed": args.seed, "results": results,
    }, indent=2) + "\n")

    print("| Split | Baseline | Groups / puzzle | Total groups | Exact puzzles | Matched words / 16 |")
    print("|---|---|---:|---:|---:|---:|")
    for split, rows in results.items():
        for name, m in rows.items():
            total = (f"{m['total_correct_groups'] / m['samples_per_puzzle']:.1f}"
                     if name == "random" else str(m["total_correct_groups"]))
            exact = (f"{m['exact_accuracy'] * m['puzzles']:.2f}" if name == "random"
                     else str(m["exact_puzzles"]))
            print(f"| {split} | {name} | {m['mean_correct_groups']:.3f} | {total} / {4 * m['puzzles']} "
                  f"| {exact} / {m['puzzles']} | {m['mean_matched_words']:.3f} |")
    print(f"\nSaved {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
