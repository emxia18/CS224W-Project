import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import sparse
import torch

from connections.features.build_neighbor_cache import category_names
from connections.hetero.concepts import ConceptSource, feature_dim, puzzle_concepts
from connections.hetero.global_graph import pair_targets, same_group_edges
from connections.hetero.model import ConceptQuartetGNN
from connections.model.gnn import QuartetGNN

WORDS = [f"W{i}" for i in range(16)]


def write_source(directory, columns):
    """columns: list of word-index lists, one per neighbor."""
    rows = [i for col in columns for i in col]
    cols = [j for j, col in enumerate(columns) for _ in col]
    matrix = sparse.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(16, len(columns)))
    directory.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(directory / "embeddings.npz", matrix)
    (directory / "word_to_row.json").write_text(json.dumps({w: i for i, w in enumerate(WORDS)}))


class WiktionaryCategoryTest(unittest.TestCase):
    def test_keeps_topical_and_drops_maintenance_categories(self):
        entry = {"senses": [{"categories": ["Footwear", "Terms with Hungarian translations",
                                             {"name": "Entries with translation boxes"}],
                             "topics": ["clothing"]}],
                 "categories": ["English nouns"]}
        self.assertEqual(sorted(category_names(entry)), ["Footwear", "topic:clothing"])


class ConceptGraphTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        # neighbor 0 links words 0-3 (a group), 1 links 0-1 only, 2 links 0-9 (too generic)
        write_source(Path(self.tmp.name), [[0, 1, 2, 3], [0, 1], list(range(10))])
        self.sources = {"a": ConceptSource(self.tmp.name)}

    def tearDown(self):
        self.tmp.cleanup()

    def test_keeps_concepts_within_the_count_range(self):
        graph = puzzle_concepts(WORDS, self.sources, min_count=3, max_count=8)
        self.assertEqual(graph["incidence"].shape, (1, 16))
        self.assertEqual(graph["incidence"][0, :4].tolist(), [1, 1, 1, 1])
        self.assertEqual(graph["features"].shape[1], feature_dim(1, 3, 8))
        graph = puzzle_concepts(WORDS, self.sources, min_count=2, max_count=8)
        self.assertEqual(graph["incidence"].shape[0], 2)

    def test_untrained_concept_model_scores_like_the_base(self):
        torch.manual_seed(0)
        x = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
        base = QuartetGNN(hidden=16, dropout=0.0, pairwise_features=True, input_dim=8, residual=True)
        model = ConceptQuartetGNN(base, 8, 1, feature_dim(1, 2, 8), hidden=8, dropout=0.0).eval()
        graph = puzzle_concepts(WORDS, self.sources, min_count=2, max_count=8)
        with torch.no_grad():
            torch.testing.assert_close(model(x, word_inputs={"concept_graph": graph}), base(x))

    def test_concept_bonus_only_reaches_quartets_it_fully_covers(self):
        torch.manual_seed(0)
        x = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
        base = QuartetGNN(hidden=16, dropout=0.0, pairwise_features=True, input_dim=8, residual=True)
        model = ConceptQuartetGNN(base, 8, 1, feature_dim(1, 3, 8), hidden=8, dropout=0.0).eval()
        torch.nn.init.constant_(model.score[-1].bias, 1.0)  # every concept scores 1
        graph = puzzle_concepts(WORDS, self.sources, min_count=3, max_count=8)
        with torch.no_grad():
            bonus = model(x, word_inputs={"concept_graph": graph}) - base(x)
        covered = base.quartets.tolist().index([0, 1, 2, 3])
        self.assertAlmostEqual(bonus[covered].item(), 1.0, places=5)
        self.assertEqual(int((bonus.abs() > 1e-6).sum()), 1)


class GlobalGraphTest(unittest.TestCase):
    def test_same_group_edges_and_targets_come_only_from_given_puzzles(self):
        index = {w: i for i, w in enumerate(WORDS)}
        record = {"words": WORDS, "answers": [{"words": WORDS[k:k + 4]} for k in range(0, 16, 4)]}
        edges = same_group_edges([record], index)
        self.assertEqual(edges.shape[1], 4 * 12)  # 4 groups x ordered pairs
        self.assertEqual(same_group_edges([], index).shape[1], 0)
        pairs, labels = pair_targets([record], index)
        self.assertEqual(pairs.shape[1], 120)
        self.assertEqual(int(labels.sum()), 24)


if __name__ == "__main__":
    unittest.main()
