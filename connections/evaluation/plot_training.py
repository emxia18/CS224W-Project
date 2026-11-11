"""Plot saved training diagnostics and unassisted validation learning curves."""

import argparse
import json
import os
from pathlib import Path

# Keep plotting caches local, and support machines without a display.
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / "runs/.matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


SERIES = (
    ("full", "Training: no hints", "#2563eb"),
    ("triplets", "Training: triplet hints", "#16a34a"),
    ("pairs", "Training: pair hints", "#d97706"),
    (None, "Validation: no hints", "#dc2626"),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, nargs="?", default=Path("runs/gnn_curriculum_diagnostics"))
    args = parser.parse_args()
    history = json.loads((args.run_dir / "history.json").read_text())
    config_path = args.run_dir / "config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not history:
        parser.error("History contains no epochs")
    epochs = [row["epoch"] for row in history]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    metrics = (
        ("mean_correct_groups", "Fully correct groups per puzzle", "Correct groups (out of 4)", 1, 4),
        ("exact_accuracy", "Entire puzzles solved", "Puzzle accuracy (%)", 100, 100),
        ("mean_matched_words", "Partial credit: matched word overlap", "Matched words (out of 16)", 1, 16),
    )
    for ax, (metric, title, ylabel, factor, maximum) in zip(axes.flat, metrics):
        for task, label, color in SERIES:
            values = [row if task is None else row.get("training_diagnostics", {}).get(task, {})
                      for row in history]
            if not any(metric in value for value in values):
                continue
            ax.plot(epochs, [value.get(metric, float("nan")) * factor for value in values],
                    label=label, color=color, linewidth=2,
                    linestyle="--" if task is None else "-")
        ax.set(title=title, ylabel=ylabel, ylim=(0, maximum * 1.04))

    loss_ax = axes[1, 1]
    loss_ax.plot(epochs, [row["loss"] for row in history], color="#64748b",
                 linestyle="--", label="Overall mixture", linewidth=1.5)
    for task, label, color in (*SERIES[:3], ("sparse", "Training: sparse hints", "#9333ea")):
        loss_ax.plot(epochs, [row.get("task_losses", {}).get(task, float("nan")) for row in history],
                     color=color, label=label, linewidth=1.5)
    loss_ax.set(title="Training BCE by task (lower is better)", ylabel="Balanced BCE", ylim=(0, None))
    loss_ax.legend(fontsize=8)

    # Reproduce the exact epoch boundaries used by task_probabilities.
    stages = []
    if config.get("curriculum") == "progressive":
        total = config.get("epochs", max(epochs))
        starts = [1] + [next((epoch for epoch in range(1, total + 1)
                            if (epoch - 1) / max(total - 1, 1) >= fraction), total + 1)
                        for fraction in (0.25, 0.5, 0.8)]
        names = ["Triplet-heavy", "Pair-heavy", "Sparse / full", "Unassisted only"]
        for index, (start, end, name) in enumerate(zip(starts, starts[1:] + [total + 1], names)):
            if start > max(epochs):
                continue
            left, right = start - 0.5, min(end - 0.5, max(epochs) + 0.5)
            stages.append((left, right, name, index))
    for ax in axes.flat:
        for left, right, name, index in stages:
            ax.axvspan(left, right, color="#e2e8f0" if index % 2 == 0 else "#fef3c7",
                       alpha=0.3, zorder=0)
            if left > 0.5:
                ax.axvline(left, color="#94a3b8", linestyle=":", linewidth=1)
        ax.grid(alpha=0.18)
        ax.set_xlim(0.5, max(epochs) + 0.5)
    if stages:
        caption = "Curriculum: " + " | ".join(
            f"{name} (epochs {int(left + 0.5)}–{int(right - 0.5)})"
            for left, right, name, _ in stages
        )
        fig.text(0.5, 0.855, caption, ha="center", fontsize=9, color="#475569")
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=2, frameon=False)
    fig.suptitle(f"Connections GNN learning curves — {args.run_dir.name}", fontsize=15, y=0.99)
    count = config.get("diagnostic_puzzles", "fixed")
    fig.text(0.5, 0.02,
             f"Training diagnostics: {count} fixed puzzles and fixed hints. Validation: unassisted.\n"
             "Hinted scores include answer information and decoding constraints. Task mixtures change across stages.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, 0.07, 1, 0.88))
    for extension in ("png", "pdf"):
        output = args.run_dir / f"learning_curves.{extension}"
        fig.savefig(output, dpi=180, facecolor="white")
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
