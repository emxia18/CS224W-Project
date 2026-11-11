import unittest

import numpy as np
import torch

from baselines import cosine_sum_scores
from train_gnn import QUARTET_IDS, QUARTETS, QuartetGNN, decode, decode_greedy


def covers_all_words(groups):
    return sorted(i for index in groups for i in QUARTETS[index]) == list(range(16))


class GreedyDecoderTest(unittest.TestCase):
    def test_returns_valid_partition(self):
        scores = np.random.default_rng(0).normal(size=1820)
        groups = decode_greedy(scores)
        self.assertEqual(len(groups), 4)
        self.assertTrue(covers_all_words(groups))

    def test_greedy_can_be_worse_than_exact(self):
        scores = np.zeros(1820)
        # Best single quartet straddles two true groups and blocks both of them.
        scores[QUARTET_IDS[(0, 1, 4, 5)]] = 10
        for group in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15)):
            scores[QUARTET_IDS[group]] = 9
        exact, greedy = decode(scores), decode_greedy(scores)
        self.assertEqual(sum(scores[exact]), 36)
        self.assertIn(QUARTET_IDS[(0, 1, 4, 5)], greedy)
        self.assertLess(sum(scores[greedy]), sum(scores[exact]))

    def test_respects_allowed_quartets(self):
        allowed = [QUARTET_IDS[g] for g in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))]
        scores = np.random.default_rng(1).normal(size=1820)
        self.assertEqual(sorted(decode_greedy(scores, allowed)), sorted(allowed))


class CosineBaselineTest(unittest.TestCase):
    def test_matches_pairwise_loop(self):
        x = np.random.default_rng(2).normal(size=(16, 8))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        scores = cosine_sum_scores(x)
        for index in (0, 777, 1819):
            q = QUARTETS[index]
            expected = sum(x[a] @ x[b] for k, a in enumerate(q) for b in q[k + 1:])
            self.assertAlmostEqual(scores[index], expected)


class NoMessagePassingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.nn.functional.normalize(torch.randn(16, 768), dim=-1)
        self.perturbed = self.x.clone()
        self.perturbed[15] = torch.nn.functional.normalize(torch.randn(768), dim=0)
        self.untouched = [i for i, q in enumerate(QUARTETS) if 15 not in q]

    def scores(self, graph):
        torch.manual_seed(1)
        model = QuartetGNN(hidden=16, dropout=0, pairwise_features=True, graph=graph).eval()
        with torch.no_grad():
            return model(self.x), model(self.perturbed)

    def test_same_parameter_count(self):
        counts = [sum(p.numel() for p in QuartetGNN(hidden=16, graph=g).parameters())
                  for g in ("complete", "none")]
        self.assertEqual(counts[0], counts[1])

    def test_none_graph_scores_depend_only_on_members(self):
        before, after = self.scores("none")
        torch.testing.assert_close(before[self.untouched], after[self.untouched])

    def test_complete_graph_passes_messages(self):
        before, after = self.scores("complete")
        self.assertFalse(torch.allclose(before[self.untouched], after[self.untouched]))


if __name__ == "__main__":
    unittest.main()
