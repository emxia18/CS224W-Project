"""Zero-training check of word-similarity sources and their weighted combinations.

Each channel gives a 16x16 word-similarity matrix per puzzle; a quartet's channel
score is the sum of its six pair similarities. A combination scores quartets by
sum_k a_k * score_k, with the weights a_k fit by the partition-level CRF loss on
training puzzles (at most a handful of weights, so little room to overfit), then
decodes exactly. Nothing else is learned.

Channels (name=spec, repeatable --channel):
  cos:DIR          cosine from a single-vector cache (embed_data/embed_static layout)
  max:DIR          max cosine over sense pairs from a build_sense_cache.py cache

(Per-word offsets, e.g. from words with many senses looking similar to everything,
cannot change the decoded partition: every word is in exactly one group, so an
offset adds the same constant to every partition.)

Train numbers are 2-fold cross-validated (fit on half the training puzzles, score
the other half). Validation numbers use weights fit on all training puzzles. The
test split is never loaded.

Example:
  python -m connections.evaluation.combine_similarities \\
    --channel ft=cos:data/embeddings/fasttext-crawl \\
    --channel wn=max:data/embeddings/senses/wordnet-mpnet \\
    --combo ft --combo ft+wn
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from connections.evaluation.analyze_failures import category_type
from connections.features.similarity_channels import parse
from connections.model.crf import crf_loss
from connections.model.decoding import QUARTETS, decode
from connections.paths import ROOT

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]
Q = np.array(QUARTETS)


def load_records(path):
    return json.loads(path.read_text(encoding="utf-8"))


def quartet_scores(sims):
    return sum(sims[Q[:, a], Q[:, b]] for a, b in PAIRS)


def gold(record):
    position = {w: k for k, w in enumerate(record["words"])}
    return [QUARTETS.index(tuple(sorted(position[w] for w in a["words"]))) for a in record["answers"]]


def fit(scores, golds, steps=200, batch=32):
    """CRF-fit channel weights. scores: (puzzles, channels, 1820).

    The CRF's dynamic program has large intermediate tables, so the loss is
    accumulated over batches of puzzles (as in train_gnn.fit_residual_scale).
    """
    x = torch.tensor(scores, dtype=torch.float64)
    y = torch.tensor(golds)
    weights = torch.full((x.shape[1],), 5.0 / x.shape[1], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        total = 0.0
        for i in range(0, len(x), batch):
            loss = crf_loss(torch.einsum("pkq,k->pq", x[i:i + batch], weights), y[i:i + batch]).sum() / len(x)
            loss.backward()
            total += loss.item()
        return torch.tensor(total, dtype=torch.float64)

    optimizer.step(closure)
    return weights.detach().numpy()


def evaluate(scores, golds, weights):
    """Per-group recovery (list of bools per puzzle) and gold rank."""
    out = []
    for s, g in zip(scores, golds):
        combined = np.tensordot(weights, s, axes=1)
        predicted = set(decode(combined))
        out.append([(q in predicted, int(1 + (combined > combined[q]).sum())) for q in g])
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--channel", action="append", required=True, metavar="NAME=SPEC")
    parser.add_argument("--combo", action="append", required=True, metavar="NAME+NAME",
                        help="Channels to combine; the first combo is the reference")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/embeddings/combine.json")
    args = parser.parse_args()

    channels = parse(args.channel)
    records = {split: load_records(args.data_dir / f"{split}.json") for split in ("train", "val")}
    golds = {split: [gold(r) for r in recs] for split, recs in records.items()}
    raw = {split: {name: np.stack([quartet_scores(ch.matrix(r["words"])) for r in recs])
                   for name, ch in channels.items()} for split, recs in records.items()}
    kinds = [[category_type(a["answerDescription"], i, j) for j, a in enumerate(r["answers"])]
             for i, r in enumerate(records["val"])]

    folds = np.arange(len(records["train"])) % 2
    results, reference = {}, None
    print("| Combination | Weights (fit on train) | Val groups /392 | Val exact /98 | Train groups /1824 (2-fold CV) "
          "| Val median gold rank | Val missed with rank > 100 |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for combo in args.combo:
        names = combo.split("+")
        stack = {split: np.stack([raw[split][n] for n in names], axis=1) for split in raw}
        weights = fit(stack["train"], golds["train"])
        val = evaluate(stack["val"], golds["val"], weights)
        train_hits = 0
        for k in (0, 1):
            w = fit(stack["train"][folds != k], [g for g, f in zip(golds["train"], folds) if f != k])
            part = evaluate(stack["train"][folds == k], [g for g, f in zip(golds["train"], folds) if f == k], w)
            train_hits += sum(h for p in part for h, _ in p)
        hits = [h for p in val for h, _ in p]
        ranks = [r for p in val for _, r in p]
        per_puzzle = np.array([sum(h for h, _ in p) for p in val])
        by_type = {}
        for p, ks in zip(val, kinds):
            for (h, _), k in zip(p, ks):
                by_type.setdefault(k, [0, 0])
                by_type[k][0] += h
                by_type[k][1] += 1
        results[combo] = {"weights": dict(zip(names, weights.tolist())), "val_groups": int(sum(hits)),
                          "val_exact": int((per_puzzle == 4).sum()), "train_groups_cv": int(train_hits),
                          "val_median_rank": float(np.median(ranks)),
                          "val_missed_rank_over_100": int(sum((not h) and r > 100 for h, r in zip(hits, ranks))),
                          "val_by_type": by_type, "val_per_puzzle": per_puzzle.tolist()}
        if reference is None:
            reference = per_puzzle
        r = results[combo]
        print(f"| {combo} | {', '.join(f'{n}={w:.2f}' for n, w in zip(names, weights))} | {r['val_groups']} "
              f"| {r['val_exact']} | {r['train_groups_cv']} | {r['val_median_rank']:g} "
              f"| {r['val_missed_rank_over_100']} |", flush=True)

    rng = np.random.default_rng(0)
    print(f"\nPaired bootstrap vs {args.combo[0]} (val groups, 10,000 resamples of puzzles):")
    for combo, r in results.items():
        diff = np.array(r["val_per_puzzle"]) - reference
        totals = diff[rng.integers(0, len(diff), (10000, len(diff)))].sum(1)
        r["val_diff_ci95"] = [float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))]
        types = ", ".join(f"{k.split(' /')[0]} {v[0]}/{v[1]}" for k, v in r["val_by_type"].items())
        print(f"  {combo}: {int(diff.sum()):+d} [{r['val_diff_ci95'][0]:+.0f}, {r['val_diff_ci95'][1]:+.0f}]  ({types})")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
