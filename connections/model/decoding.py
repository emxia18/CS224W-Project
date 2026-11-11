"""Quartet enumeration and partition decoders.

A puzzle has 16 words, so there are C(16,4) = 1,820 candidate quartets. Every
scorer in this project outputs one score per quartet, indexed like QUARTETS.
The decoders turn those 1,820 scores into a valid answer: four disjoint quartets.

- decode:        exact. Dynamic program over subsets of the 16 words that returns
                 the partition with the highest total score (of 2,627,625).
- decode_greedy: repeatedly take the best quartet disjoint from earlier picks.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np


QUARTETS = list(combinations(range(16), 4))
QUARTET_IDS = {group: index for index, group in enumerate(QUARTETS)}
FULL_MASK = (1 << 16) - 1


@lru_cache(maxsize=None)
def partition_options(mask):
    """Anchor each group on the first remaining word to avoid duplicate partitions."""
    remaining = [i for i in range(16) if mask & (1 << i)]
    options = []
    for rest in combinations(remaining[1:], 3):
        group = (remaining[0], *rest)
        group_mask = sum(1 << i for i in group)
        options.append((QUARTET_IDS[group], mask ^ group_mask))
    return options


def decode(scores, allowed_ids=None):
    """Return quartet indices for the maximum-total-logit exact partition."""
    scores = np.asarray(scores)
    if scores.shape != (1820,) or not np.isfinite(scores).all():
        raise ValueError("Decoder requires 1,820 finite scores")
    choices = {}
    allowed = None if allowed_ids is None else set(allowed_ids)

    @lru_cache(maxsize=None)
    def solve(mask):
        if mask == 0:
            return 0.0
        best, choice = max((
            (float(scores[index]) + solve(rest), (index, rest))
            for index, rest in partition_options(mask)
            if allowed is None or index in allowed
        ), default=(-float("inf"), None))
        choices[mask] = choice
        return best

    if not np.isfinite(solve(FULL_MASK)):
        raise ValueError("No complete partition satisfies the revealed memberships")
    groups, mask = [], FULL_MASK
    while mask:
        index, mask = choices[mask]
        groups.append(index)
    return groups


def decode_greedy(scores, allowed_ids=None):
    """Repeatedly take the highest-scoring quartet disjoint from earlier picks.

    With revealed memberships, a greedy pick can strand words in a way no allowed
    quartet completes; that raises, matching the exact decoder's failure mode.
    """
    scores = np.asarray(scores)
    if scores.shape != (1820,) or not np.isfinite(scores).all():
        raise ValueError("Decoder requires 1,820 finite scores")
    allowed = None if allowed_ids is None else set(allowed_ids)
    used, groups = 0, []
    for index in np.argsort(-scores, kind="stable"):
        if allowed is not None and index not in allowed:
            continue
        group_mask = sum(1 << i for i in QUARTETS[index])
        if not used & group_mask:
            used |= group_mask
            groups.append(int(index))
            if used == FULL_MASK:
                return groups
    raise ValueError("Greedy decoding could not complete a partition")


DECODERS = {"exact": decode, "greedy": decode_greedy}
