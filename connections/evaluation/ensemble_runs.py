"""Score trained runs (e.g. seeds 0-4) one by one and as an ensemble.

The ensemble averages the runs' quartet scores, then decodes once. All runs must
come from train_gnn.py with the same inputs (embedding, --similarity sources,
word features); only the seed or training settings may differ.

Every model is first scored on validation and must reproduce its recorded
metrics.json, as a check that the checkpoint and inputs load correctly.

Final test evaluation: `--split test --confirm-final-test`. This is meant to run
once, on a frozen configuration. After the validation check it scores the test
split with each model, the ensemble, and two untrained references (fastText-style
cosine of the run's --embedding-dir, and the run's untrained baseline score),
with paired bootstrap CIs over puzzles. Models trained with --homophone-features
need a homophone cache that covers test words: build one with
`build_homophone_cache --include-test` and pass it as --homophone-cache.

Examples:
  python -m connections.evaluation.ensemble_runs runs/next/wordplay_wiki_seed{0,1,2,3,4}
  python -m connections.evaluation.ensemble_runs runs/next/wordplay_wiki_seed{0,1,2,3,4} \\
    --split test --confirm-final-test --homophone-cache data/embeddings/homophones-test \\
    --output runs/test/final.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from connections.evaluation.baselines import cosine_sum_scores
from connections.features.explicit_features import HomophoneCache
from connections.features.similarity_channels import parse
from connections.features.word_channels import WordTokenizer
from connections.model.decoding import QUARTETS, decode
from connections.model.gnn import QuartetGNN, quartet_pair_sum
from connections.model.train_gnn import load_puzzles

INPUT_KEYS = ("embedding_dir", "similarity", "data_dir", "spelling_features", "pronunciation_features",
              "orthographic_features", "homophone_features", "homophone_cache")
WORD_FEATURES = ("spelling_features", "pronunciation_features", "orthographic_features", "homophone_features")


def load_model(run, input_dim):
    config = json.loads((run / "config.json").read_text())
    unsupported = [key for key in ("hint_features", "contextual_dir", "keep_pronunciation_qualifiers")
                   if config.get(key)]
    if unsupported or config.get("val_limit") or config.get("overfit"):
        raise ValueError(f"{run}: unsupported settings {unsupported or 'val_limit/overfit'}")
    model = QuartetGNN(config["hidden"], config["dropout"], config["pairwise_features"],
                       learned_relations=config.get("learned_relations", 0), graph=config["graph"],
                       spelling_features=config.get("spelling_features", False),
                       pronunciation_features=config.get("pronunciation_features", False),
                       orthographic_features=config.get("orthographic_features", False),
                       homophone_features=config.get("homophone_features", False),
                       input_dim=input_dim, residual=config.get("residual", False),
                       extra_similarities=len(config.get("similarity") or []))
    checkpoint = torch.load(run / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    return model.eval(), config


def load_split(config, split, homophone_cache, vectors, mapping):
    homophones = None
    if config.get("homophone_features"):
        homophones = HomophoneCache(homophone_cache)
        homophones.validate_inputs(config["data_dir"])
    tokenizer = (WordTokenizer(pronunciation=config.get("pronunciation_features", False), strip_qualifiers=True,
                               orthographic_features=config.get("orthographic_features", False),
                               homophone_cache=homophones)
                 if any(config.get(k) for k in WORD_FEATURES) else None)
    puzzles = load_puzzles(Path(config["data_dir"]) / f"{split}.json", vectors, mapping, "cpu", tokenizer)
    channels = parse(config.get("similarity") or [])
    for puzzle in puzzles:
        if channels:
            puzzle["similarities"] = torch.tensor(
                np.stack([c.matrix(puzzle["record"]["words"]) for c in channels.values()]))
    return puzzles


def score(models, puzzles):
    with torch.no_grad():
        return [np.stack([m(p["x"], word_inputs=p.get("word_inputs"),
                            similarities=p.get("similarities")).numpy() for p in puzzles]) for m in models]


def correct(scores, puzzles):
    return np.array([len(set(decode(s)) & p["positives"]) for s, p in zip(scores, puzzles)])


def bootstrap(diff, rng, samples=10000):
    totals = diff[rng.integers(0, len(diff), (samples, len(diff)))].sum(1)
    return [float(np.percentile(totals, 2.5)), float(np.percentile(totals, 97.5))]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--confirm-final-test", action="store_true",
                        help="Required with --split test: the configuration is frozen")
    parser.add_argument("--homophone-cache", type=Path,
                        help="Override the runs' homophone cache (needed for test words)")
    parser.add_argument("--output", type=Path, help="Write results as JSON")
    args = parser.parse_args()
    if args.split == "test" and not args.confirm_final_test:
        parser.error("--split test needs --confirm-final-test; evaluate test once, on a frozen config")

    first = json.loads((args.runs[0] / "config.json").read_text())
    for run in args.runs[1:]:
        other = json.loads((run / "config.json").read_text())
        if any(other.get(k) != first.get(k) for k in INPUT_KEYS):
            parser.error(f"{run} uses different inputs from {args.runs[0]}")
    vectors = np.load(Path(first["embedding_dir"]) / "embeddings.npy", allow_pickle=False)
    mapping = json.loads((Path(first["embedding_dir"]) / "word_to_row.json").read_text())
    homophone_cache = args.homophone_cache or first.get("homophone_cache")
    models = [load_model(run, vectors.shape[1])[0] for run in args.runs]

    # 1. Every model must reproduce its recorded validation score with these inputs.
    val = load_split(first, "val", homophone_cache, vectors, mapping)
    for run, scores in zip(args.runs, score(models, val)):
        got = int(correct(scores, val).sum())
        recorded = round(len(val) * json.loads((run / "metrics.json").read_text())["mean_correct_groups"])
        print(f"{run}: val {got} groups (recorded {recorded}) {'OK' if got == recorded else 'MISMATCH'}", flush=True)
        if got != recorded:
            parser.exit(1, "A model does not reproduce its recorded validation score; check inputs.\n")

    puzzles = val if args.split == "val" else load_split(first, "test", homophone_cache, vectors, mapping)
    n = len(puzzles)
    per_run = score(models, puzzles)
    run_correct = [correct(s, puzzles) for s in per_run]
    ensemble = correct(np.mean(per_run, axis=0), puzzles)
    seed_mean = np.mean(run_correct, axis=0)
    results = {"split": args.split, "puzzles": n, "runs": [str(r) for r in args.runs],
               "per_run_groups": [int(c.sum()) for c in run_correct],
               "per_run_exact": [int((c == 4).sum()) for c in run_correct],
               "ensemble_groups": int(ensemble.sum()), "ensemble_exact": int((ensemble == 4).sum())}

    print(f"\n### {args.split}: {n} puzzles, {4 * n} groups\n")
    groups = np.array(results["per_run_groups"])
    exact = np.array(results["per_run_exact"])
    print(f"Per run: {results['per_run_groups']} groups; exact {results['per_run_exact']}")
    print(f"Mean over runs: {groups.mean():.1f} ± {groups.std(ddof=1) if len(groups) > 1 else 0:.1f} groups "
          f"({groups.mean() / (4 * n):.1%}), {exact.mean():.1f} puzzles solved")
    print(f"Ensemble: {results['ensemble_groups']} groups ({results['ensemble_groups'] / (4 * n):.1%}), "
          f"{results['ensemble_exact']} puzzles solved")

    if args.split == "test":
        # Untrained references on the same puzzles.
        cosine = correct([cosine_sum_scores(p["x"].numpy().astype(np.float64)) for p in puzzles], puzzles)
        config = json.loads((args.runs[0] / "config.json").read_text())
        quartets = torch.tensor(QUARTETS)
        weights = [config.get("initial_residual_scale")] + list((config.get("initial_similarity_scales") or {}).values())
        untrained = correct([weights[0] * cosine_sum_scores(p["x"].numpy().astype(np.float64))
                             + sum(w * quartet_pair_sum(m, quartets).numpy()
                                   for w, m in zip(weights[1:], p.get("similarities", [])))
                             for p in puzzles], puzzles) if weights[0] is not None else None
        rng = np.random.default_rng(0)
        results["cosine_groups"] = int(cosine.sum())
        results["cosine_exact"] = int((cosine == 4).sum())
        results["seed_mean_vs_cosine"] = [float((seed_mean - cosine).sum()), bootstrap(seed_mean - cosine, rng)]
        results["ensemble_vs_cosine"] = [float((ensemble - cosine).sum()), bootstrap(ensemble - cosine, rng)]
        print(f"Embedding cosine, untrained: {results['cosine_groups']} groups, {results['cosine_exact']} solved")
        if untrained is not None:
            results["untrained_groups"] = int(untrained.sum())
            results["untrained_exact"] = int((untrained == 4).sum())
            results["seed_mean_vs_untrained"] = [float((seed_mean - untrained).sum()),
                                                 bootstrap(seed_mean - untrained, rng)]
            print(f"Untrained baseline score (all sources): {results['untrained_groups']} groups, "
                  f"{results['untrained_exact']} solved")
        for key in ("seed_mean_vs_cosine", "ensemble_vs_cosine", "seed_mean_vs_untrained"):
            if key in results:
                diff, (low, high) = results[key]
                print(f"  {key}: {diff:+.1f} groups [95% CI {low:+.1f}, {high:+.1f}]")
    results["per_puzzle_seed_mean"] = seed_mean.tolist()
    results["per_puzzle_ensemble"] = ensemble.tolist()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1) + "\n")
        print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
