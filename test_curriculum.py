"""Checks for hint leakage, completion constraints, and curriculum schedules."""

import random
import unittest

import numpy as np
import torch

from curriculum import compatible_quartets, sample_hints, task_probabilities
from train_gnn import QUARTETS, QUARTET_IDS, QuartetGNN, decode, training_diagnostics


class CurriculumTests(unittest.TestCase):
    def setUp(self):
        self.official = [tuple(range(i, i + 4)) for i in range(0, 16, 4)]
        self.positive_ids = {QUARTET_IDS[group] for group in self.official}

    def test_hidden_words_have_no_membership_edges(self):
        for task in ("triplets", "pairs", "sparse", "full"):
            hints, revealed = sample_hints(self.official, task, random.Random(42), "cpu")
            visible = {word for group in revealed for word in group}
            for word in set(range(16)) - visible:
                self.assertEqual(hints[word].sum().item(), 0)
                self.assertEqual(hints[:, word].sum().item(), 0)
            self.assertTrue(torch.equal(hints, hints.T))
            self.assertEqual(hints.diag().sum().item(), 0)
            allowed = compatible_quartets(QUARTETS, revealed)
            self.assertTrue(self.positive_ids <= set(allowed))
            self.assertGreater(len(allowed), 4)
            if task == "full":
                self.assertEqual(len(allowed), 1820)

    def test_triplet_completion_is_global(self):
        _, revealed = sample_hints(self.official, "triplets", random.Random(42), "cpu")
        allowed = compatible_quartets(QUARTETS, revealed)
        # Even if incompatible groups have huge scores, constraints exclude them.
        scores = np.full(1820, 1000.0)
        scores[allowed] = 0
        scores[list(self.positive_ids)] = 10
        self.assertEqual(set(decode(scores, allowed)), self.positive_ids)
        scores[allowed] = np.random.default_rng(42).normal(size=len(allowed))
        result = [set(QUARTETS[index]) for index in decode(scores, allowed)]
        self.assertEqual(len(set.union(*result)), 16)
        for cluster in revealed:
            self.assertEqual(sum(set(cluster) <= group for group in result), 1)
        # Four triplets and four hidden words must produce four distinct assignments.
        self.assertTrue(all(sum(set(cluster) <= group for cluster in revealed) == 1
                            for group in result))

    def test_schedule_and_unassisted_forward(self):
        for epoch in range(1, 101):
            probabilities = task_probabilities("progressive", epoch, 100)
            self.assertAlmostEqual(sum(probabilities), 1)
            self.assertGreater(probabilities[-1], 0)
        self.assertEqual(task_probabilities("progressive", 100, 100), (0, 0, 0, 1))
        self.assertEqual(task_probabilities("fixed", 1, 100), (0.25,) * 4)
        torch.set_num_threads(1)
        model = QuartetGNN(hidden=8, dropout=0, hint_features=True).eval()
        x = torch.nn.functional.normalize(torch.randn(16, 768), dim=-1)
        with torch.no_grad():
            self.assertTrue(torch.equal(model(x), model(x, hints=torch.zeros(16, 16))))

    def test_diagnostics_are_repeatable_and_do_not_train(self):
        torch.set_num_threads(1)
        model = QuartetGNN(hidden=8, dropout=0.3, hint_features=True)
        x = torch.nn.functional.normalize(torch.randn(16, 768), dim=-1)
        puzzles = [{"x": x, "positives": self.positive_ids}]
        before = {name: value.clone() for name, value in model.state_dict().items()}
        python_state, torch_state = random.getstate(), torch.random.get_rng_state().clone()
        first = training_diagnostics(model, puzzles, seed=123)
        second = training_diagnostics(model, puzzles, seed=123)
        self.assertEqual(set(first), {"full", "triplets", "pairs"})
        for task in first:
            for metric in ("mean_correct_groups", "mean_matched_words", "exact_accuracy"):
                self.assertEqual(first[task][metric], second[task][metric])
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_state))
        self.assertTrue(all(torch.equal(value, before[name])
                            for name, value in model.state_dict().items()))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))


if __name__ == "__main__":
    unittest.main()
