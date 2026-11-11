"""Rank embedding caches with the zero-parameter cosine-sum baseline, by category.

Each --embedding-dir is a cache written by embed_data.py. Every quartet is scored by
the sum of its six pairwise cosines and decoded with the exact partition decoder,
exactly as in baselines.py. Nothing is trained or tuned, so the train split is as
valid an evaluation as validation and gives about 4.7x more puzzles. The test split
is never loaded.

Validation groups use the category tags from analyze_failures.py (manually reviewed
synonyms). Train has no synonym annotations, so there synonyms and themes are merged
into one "Synonym or thematic" row; the rule-based tags (phrase completion,
spelling, sound) are the same on both splits. Descriptions are used only after
scoring, never as input.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from connections.evaluation.analyze_failures import TYPES, category_type
from connections.evaluation.baselines import cosine_sum_scores
from connections.model.decoding import QUARTETS, decode
from connections.model.train_gnn import load_puzzles, matched_word_overlap


from connections.paths import ROOT
MERGED = "Synonym or thematic"
SHORT = {TYPES[0]: "Synonym", TYPES[1]: "Thematic", TYPES[2]: "Phrase compl.",
         TYPES[3]: "Spelling", TYPES[4]: "Sound", MERGED: "Syn.+thematic"}


def evaluate(puzzles, split):
    """Per-group recovery, rank, and category; per-puzzle correct and overlap."""
    groups, per_puzzle = [], []
    for i, puzzle in enumerate(puzzles):
        scores = cosine_sum_scores(puzzle["x"])
        predicted = decode(scores)
        record = puzzle["record"]
        positions = {word: k for k, word in enumerate(record["words"])}
        for j, answer in enumerate(record["answers"]):
            group = tuple(sorted(positions[word] for word in answer["words"]))
            official = QUARTETS.index(group)
            kind = category_type(answer["answerDescription"], i if split == "val" else -1, j)
            if split != "val" and kind in TYPES[:2]:
                kind = MERGED
            groups.append({"puzzle": i, "type": kind, "recovered": official in predicted,
                           "rank": int(1 + (scores > scores[official]).sum())})
        correct = len(set(predicted) & puzzle["positives"])
        per_puzzle.append((correct, matched_word_overlap(predicted, puzzle["positives"])))
    return groups, np.array(per_puzzle)


def bootstrap_diff(a, b, samples, rng):
    """95% CI of the difference in total groups, resampling puzzles (paired)."""
    diff = a - b
    idx = rng.integers(0, len(diff), size=(samples, len(diff)))
    totals = diff[idx].sum(axis=1)
    return float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_dirs", nargs="+", type=Path,
                        help="Embedding caches; the first is the reference for paired CIs")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--splits", nargs="+", default=["val", "train"], choices=("train", "val"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/embeddings/compare.json")
    args = parser.parse_args()

    results = {}
    for directory in args.embedding_dirs:
        vectors = np.load(directory / "embeddings.npy", allow_pickle=False)
        if not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-4):
            parser.error(f"{directory}: cached embeddings must be normalized")
        mapping = json.loads((directory / "word_to_row.json").read_text())
        metadata = json.loads((directory / "metadata.json").read_text())
        label = directory.name
        results[label] = {"model": metadata.get("model"), "dim": int(vectors.shape[1]),
                          "lowercase": metadata.get("lowercase", False),
                          "prefix": metadata.get("prefix", ""), "splits": {}}
        for split in args.splits:
            puzzles = load_puzzles(args.data_dir / f"{split}.json", vectors, mapping, "cpu")
            for puzzle in puzzles:
                puzzle["x"] = puzzle["x"].numpy().astype(np.float64)
            groups, per_puzzle = evaluate(puzzles, split)
            results[label]["splits"][split] = {"groups": groups, "per_puzzle": per_puzzle}

    reference = next(iter(results))
    rng = np.random.default_rng(args.seed)
    summary = {}
    for split in args.splits:
        types = [t for t in (*TYPES[:2], MERGED, *TYPES[2:])
                 if any(g["type"] == t for g in results[reference]["splits"][split]["groups"])]
        ref_correct = results[reference]["splits"][split]["per_puzzle"][:, 0]
        n = len(ref_correct)
        print(f"\n### {split}: {n} puzzles, {4 * n} groups, cosine sum + exact decoder\n")
        header = ["Embedding", "Dim", "Groups", f"Δ vs {reference} [95% CI]", "Exact",
                  "Words/16", "Median rank"]
        header += [f"{SHORT[t]} ({sum(g['type'] == t for g in results[reference]['splits'][split]['groups'])})"
                   for t in types]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")
        summary[split] = {}
        for label, result in results.items():
            data = result["splits"][split]
            groups, per_puzzle = data["groups"], data["per_puzzle"]
            correct = per_puzzle[:, 0]
            low, high = bootstrap_diff(correct, ref_correct, args.bootstrap, rng)
            row = {
                "total_groups": int(correct.sum()), "exact_puzzles": int((correct == 4).sum()),
                "mean_matched_words": float(per_puzzle[:, 1].mean()),
                "median_rank": float(np.median([g["rank"] for g in groups])),
                "diff_vs_reference": int(correct.sum() - ref_correct.sum()),
                "diff_ci95": [low, high],
                "by_type": {t: int(sum(g["recovered"] for g in groups if g["type"] == t)) for t in types},
            }
            summary[split][label] = row
            delta = "—" if label == reference else (
                f"{row['diff_vs_reference']:+d} [{low:+.0f}, {high:+.0f}]")
            cells = [label, str(result["dim"]), f"{row['total_groups']}", delta,
                     str(row["exact_puzzles"]), f"{row['mean_matched_words']:.2f}",
                     f"{row['median_rank']:g}"] + [str(row["by_type"][t]) for t in types]
            print("| " + " | ".join(cells) + " |")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "reference": reference, "bootstrap_samples": args.bootstrap, "seed": args.seed,
        "embeddings": {label: {k: v for k, v in r.items() if k != "splits"} for label, r in results.items()},
        "summary": summary,
    }, indent=2) + "\n")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
