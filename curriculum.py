"""Training-only partial memberships and mixtures of completion tasks."""

import random

import torch


TASKS = ("triplets", "pairs", "sparse", "full")


def task_probabilities(mode, epoch, epochs):
    """Progressive training ends with the final 20% fully unassisted."""
    if mode == "none":
        return (0.0, 0.0, 0.0, 1.0)
    if mode == "fixed":
        return (0.25, 0.25, 0.25, 0.25)
    progress = (epoch - 1) / max(epochs - 1, 1)
    if progress < 0.25:
        return (0.65, 0.15, 0.05, 0.15)
    if progress < 0.5:
        return (0.20, 0.45, 0.15, 0.20)
    if progress < 0.8:
        return (0.05, 0.15, 0.40, 0.40)
    return (0.0, 0.0, 0.0, 1.0)


def sample_hints(official_groups, task, rng: random.Random, device):
    """Reveal whole subsets; hidden words have no membership links at all."""
    groups = [list(group) for group in official_groups]
    if task == "triplets":
        revealed = [rng.sample(group, 3) for group in groups]
    elif task == "pairs":
        revealed = [rng.sample(group, 2) for group in groups]
    elif task == "sparse":
        revealed = [rng.sample(group, 2) for group in rng.sample(groups, rng.randint(1, 2))]
    elif task == "full":
        revealed = []
    else:
        raise ValueError(f"Unknown task: {task}")
    hints = torch.zeros((16, 16), device=device)
    for group in revealed:
        for i in group:
            for j in group:
                if i != j:
                    hints[i, j] = 1.0
    return hints, revealed


def compatible_quartets(quartets, revealed):
    """Known clusters stay intact and different revealed clusters stay apart.

    Returns locally compatible groups. Exact partition decoding additionally
    enforces that every word is used once across all four groups.
    """
    clusters = [set(group) for group in revealed]
    compatible = []
    for index, quartet in enumerate(quartets):
        group = set(quartet)
        touched = [cluster for cluster in clusters if cluster & group]
        if len(touched) <= 1 and all(cluster <= group for cluster in touched):
            compatible.append(index)
    return compatible
