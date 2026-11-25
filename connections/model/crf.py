"""Partition-level CRF loss over all ways to split 16 words into four groups of four.

A partition P is scored as the sum of its four quartet scores, which is exactly
what the exact decoder in decoding.py maximizes. The loss is

    loss = log Z - score(gold),   log Z = log sum_P exp(score(P)),

over all 2,627,625 partitions. log Z uses the same anchored dynamic program as
decode() (the group containing the lowest remaining word is chosen first, so each
partition is counted once), with max replaced by log-sum-exp.
"""

from functools import lru_cache
import math

import torch

from connections.model.decoding import FULL_MASK, QUARTET_IDS, partition_options


NUM_PARTITIONS = 2_627_625  # 16! / (4!^4 * 4!)
LOG_NUM_PARTITIONS = math.log(NUM_PARTITIONS)


def _bits(mask):
    return tuple(i for i in range(16) if mask & (1 << i))


@lru_cache(maxsize=None)
def _tables():
    """Index tables for the bottom-up dynamic program.

    Returns (leaf_ids, steps). leaf_ids maps each reachable 4-word mask to its
    quartet index. Each step covers masks with one more group (8, 12, then 16
    words): a (masks, options) table of quartet indices taken, and a matching
    table of rows of the remaining mask in the previous level.
    """
    levels = [[FULL_MASK]]
    for _ in range(3):
        levels.append(sorted({rest for mask in levels[-1] for _, rest in partition_options(mask)}))
    levels.reverse()  # 4-word masks, 8, 12, 16
    leaf_ids = torch.tensor([QUARTET_IDS[_bits(mask)] for mask in levels[0]], dtype=torch.long)
    steps = []
    for lower, upper in zip(levels, levels[1:]):
        row = {mask: index for index, mask in enumerate(lower)}
        options = [partition_options(mask) for mask in upper]
        quartets = torch.tensor([[q for q, _ in opts] for opts in options], dtype=torch.long)
        rests = torch.tensor([[row[rest] for _, rest in opts] for opts in options], dtype=torch.long)
        steps.append((quartets, rests))
    return leaf_ids, steps


def log_partition(scores):
    """log Z for quartet scores of shape (..., 1820); differentiable."""
    if scores.shape[-1] != 1820:
        raise ValueError("log_partition requires 1,820 quartet scores")
    leaf_ids, steps = _tables()
    device = scores.device
    log_z = scores[..., leaf_ids.to(device)]
    for quartets, rests in steps:
        log_z = torch.logsumexp(scores[..., quartets.to(device)] + log_z[..., rests.to(device)], dim=-1)
    return log_z[..., 0]


def crf_loss(scores, positives):
    """Negative log-likelihood of the gold partition. positives: 4 quartet indices
    (shape (4,) for one puzzle, or (batch, 4) with scores of shape (batch, 1820))."""
    gold = scores.gather(-1, positives).sum(-1)
    return log_partition(scores) - gold

