"""Count complete partitions under an explicit, finite category catalogue."""

from itertools import combinations


def partition_count(words, memberships, limit=2):
    """Count up to limit; unknown semantic categories are outside this check."""
    positions = {w: i for i, w in enumerate(words)}
    candidates = set()
    for members in memberships:
        present = sorted(set(members) & positions.keys())
        for quartet in combinations(present, 4):
            candidates.add(sum(1 << positions[w] for w in quartet))
    by_word = [[] for _ in words]
    for mask in candidates:
        for i in range(len(words)):
            if mask & (1 << i):
                by_word[i].append(mask)

    def search(remaining):
        if not remaining:
            return 1
        first = (remaining & -remaining).bit_length() - 1
        total = 0
        for mask in by_word[first]:
            if mask & remaining == mask:
                total += search(remaining ^ mask)
                if total >= limit:
                    return limit
        return total

    return search((1 << len(words)) - 1)
