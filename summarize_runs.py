"""Aggregate validation metrics over seeds into one Markdown comparison table.

Example:
  python summarize_runs.py \
    --group "GATv2 (complete graph)=runs/compare/gat_seed*" \
    --group "No message passing=runs/compare/nomp_seed*"
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent


def fmt(values, digits=3):
    values = np.asarray(values, dtype=float)
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    return f"{values.mean():.{digits}f} ± {values.std(ddof=1):.{digits}f}"


VAL_PUZZLES = 98  # summarize_runs rejects runs with --val-limit


def row(label, seeds, metrics, decoder, epochs="—", params="—"):
    groups = [m["mean_correct_groups"] for m in metrics]
    exact = [m["exact_accuracy"] * VAL_PUZZLES for m in metrics]
    return (f"| {label} | {decoder} | {seeds} | {fmt(groups)} | {fmt([VAL_PUZZLES * g for g in groups], 1)} "
            f"| {fmt(exact, 1)} | {fmt([m['mean_matched_words'] for m in metrics], 2)} | {epochs} | {params} |")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--group", action="append", default=[], metavar="LABEL=GLOB",
                        help="Run directories (one per seed) sharing a configuration")
    parser.add_argument("--baselines", type=Path, default=ROOT / "runs/baselines/metrics.json")
    parser.add_argument("--output", type=Path, help="Also write the table to this Markdown file")
    args = parser.parse_args()

    lines = ["| Model | Decoder | Seeds | Groups / puzzle | Groups / 392 | Exact / 98 "
             "| Matched words / 16 | Best epoch | Params |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    if args.baselines.exists():
        val = json.loads(args.baselines.read_text())["results"].get("val", {})
        for key, label, decoder in (("random", "Random partition", "—"),
                                    ("cosine_exact", "Cosine sum (no training)", "exact"),
                                    ("cosine_greedy", "Cosine sum (no training)", "greedy")):
            if key in val:
                lines.append(row(label, "—", [val[key]], decoder, params="0"))
    for spec in args.group:
        label, _, pattern = spec.partition("=")
        runs = sorted(Path(p) for p in glob.glob(pattern) if (Path(p) / "metrics.json").exists())
        if not runs:
            parser.error(f"No finished runs match {pattern!r}")
        configs = [json.loads((r / "config.json").read_text()) for r in runs]
        if any(c.get("val_limit") or c.get("overfit") for c in configs):
            parser.error(f"{label}: runs must evaluate the full validation split")
        epochs = [json.loads((r / "history.json").read_text()) for r in runs]
        best_epochs = [max(h, key=lambda e: (e["mean_correct_groups"], e["exact_accuracy"]))["epoch"]
                       for h in epochs]
        params = f"{configs[0].get('parameters', '?'):,}" if "parameters" in configs[0] else "?"
        for decoder, name in (("exact", "metrics.json"), ("greedy", "metrics_greedy.json")):
            if all((r / name).exists() for r in runs):
                metrics = [json.loads((r / name).read_text()) for r in runs]
                lines.append(row(label, len(runs), metrics, decoder, fmt(best_epochs, 1), params))
    table = "\n".join(lines)
    print(table)
    print("\nTrained rows: checkpoint chosen by validation groups, so validation scores are "
          "optimistic relative to the untrained baselines. Mean ± sample std over seeds.")
    if args.output:
        args.output.write_text(table + "\n")


if __name__ == "__main__":
    main()
