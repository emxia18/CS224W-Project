import math
import unittest

import numpy as np
import torch

from connections.evaluation.baselines import cosine_sum_scores
from connections.model.crf import NUM_PARTITIONS, crf_loss, log_partition
from connections.model.decoding import QUARTET_IDS, decode
from connections.model.gnn import QuartetGNN


GOLD = [QUARTET_IDS[g] for g in ((0, 1, 2, 3), (4, 5, 6, 7), (8, 9, 10, 11), (12, 13, 14, 15))]


class LogPartitionTest(unittest.TestCase):
    def test_zero_scores_count_every_partition(self):
        self.assertAlmostEqual(log_partition(torch.zeros(1820, dtype=torch.float64)).item(),
                               math.log(NUM_PARTITIONS), places=9)

    def test_constant_shift(self):
        # Every partition has exactly four quartets, so adding c to all scores adds 4c.
        scores = torch.randn(1820, dtype=torch.float64, generator=torch.Generator().manual_seed(0))
        torch.testing.assert_close(log_partition(scores + 0.7), log_partition(scores) + 2.8)

    def test_marginals_sum_to_four_and_cover_each_word_once(self):
        scores = torch.randn(1820, dtype=torch.float64, generator=torch.Generator().manual_seed(1))
        scores.requires_grad_(True)
        log_partition(scores).backward()
        marginals = scores.grad
        self.assertAlmostEqual(marginals.sum().item(), 4.0, places=9)
        word_0 = [i for g, i in QUARTET_IDS.items() if 0 in g]
        self.assertAlmostEqual(marginals[word_0].sum().item(), 1.0, places=9)

    def test_low_temperature_matches_exact_decoder(self):
        scores = np.random.default_rng(2).normal(size=1820)
        best = scores[decode(scores)].sum()
        beta = 1e4
        soft = log_partition(torch.tensor(scores * beta)).item() / beta
        self.assertAlmostEqual(soft, best, places=2)

    def test_batched_matches_single(self):
        scores = torch.randn(3, 1820, dtype=torch.float64, generator=torch.Generator().manual_seed(3))
        torch.testing.assert_close(log_partition(scores),
                                   torch.stack([log_partition(s) for s in scores]))


class CrfLossTest(unittest.TestCase):
    def test_uniform_scores(self):
        loss = crf_loss(torch.zeros(1820, dtype=torch.float64), torch.tensor(GOLD))
        self.assertAlmostEqual(loss.item(), math.log(NUM_PARTITIONS), places=9)

    def test_loss_is_nonnegative_and_falls_as_gold_rises(self):
        scores = torch.zeros(1820, dtype=torch.float64)
        scores[GOLD] = 5.0
        high = crf_loss(scores, torch.tensor(GOLD))
        scores[GOLD] = 20.0
        low = crf_loss(scores, torch.tensor(GOLD))
        self.assertGreaterEqual(low.item(), 0.0)
        self.assertLess(low.item(), high.item())


class ResidualModelTest(unittest.TestCase):
    def test_untrained_residual_equals_scaled_cosine(self):
        torch.manual_seed(0)
        x = torch.nn.functional.normalize(torch.randn(16, 300), dim=-1)
        model = QuartetGNN(hidden=16, dropout=0.1, pairwise_features=True, input_dim=300,
                           residual=True).eval()
        model.residual_scale.data.fill_(6.97)
        with torch.no_grad():
            scores = model(x).numpy()
        np.testing.assert_allclose(scores, 6.97 * cosine_sum_scores(x.numpy()), rtol=1e-5, atol=1e-5)
        self.assertEqual(decode(scores), decode(cosine_sum_scores(x.numpy())))

    def test_correction_learns_after_one_step(self):
        torch.manual_seed(0)
        x = torch.nn.functional.normalize(torch.randn(16, 300), dim=-1)
        model = QuartetGNN(hidden=16, dropout=0, input_dim=300, residual=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        crf_loss(model(x), torch.tensor(GOLD)).backward()
        self.assertGreater(model.rho[-1].weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertGreater(model.rho[-1].weight.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
