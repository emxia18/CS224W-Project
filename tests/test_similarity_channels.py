import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from connections.features.similarity_channels import SimilarityChannel, parse
from connections.model.decoding import QUARTETS
from connections.model.gnn import QuartetGNN, quartet_cosine_sum, quartet_pair_sum
from connections.paths import ROOT

# Four "___ ROLL" words, a pair sharing only "R:pie" (too few in a puzzle), seven
# words sharing "L:big" (too many), and three unrelated words.
ROLL = ["DRUM", "EGG", "HONOR", "SPRING"]
PIE = ["APPLE", "PUMPKIN"]
BIG = ["TOP", "BANG", "DEAL", "SHOT", "WIG", "MOUTH", "TIME"]
OTHER = ["CAT", "RIVER", "BLUE"]
WORDS = ROLL + PIE + BIG + OTHER
KEYS = ([f"{w.lower()}_roll" for w in ROLL] + [f"{w.lower()}_pie" for w in PIE]
        + [f"big_{w.lower()}" for w in BIG] + ["cat", "river"])


def write_splits(directory, words):
    for split in ("train", "val", "test"):
        (directory / f"{split}.json").write_text(json.dumps([{"words": words}]))


class PhraseCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        write_splits(root, WORDS)
        (root / "vocab.vec").write_text(f"{len(KEYS)} 2\n" + "".join(f"{k} 0.1 0.2\n" for k in KEYS))
        cls.cache = root / "phrases"
        subprocess.run([sys.executable, "-m", "connections.features.build_phrase_cache",
                        "--vectors", str(root / "vocab.vec"), "--input-dir", str(root),
                        "--output-dir", str(cls.cache)], check=True, cwd=ROOT, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_partners_come_from_both_sides_of_a_phrase(self):
        partners = json.loads((self.cache / "partners.json").read_text())
        self.assertIn("R:roll", partners)   # drum_roll: DRUM is followed by roll
        self.assertIn("L:big", partners)    # big_top: TOP is preceded by big
        self.assertNotIn("R:cat", partners)  # single words are not phrases

    def test_phrase_kind_links_only_partners_shared_by_three_to_five_words(self):
        sims = SimilarityChannel(f"phrase:{self.cache}").matrix(WORDS)
        roll = [WORDS.index(w) for w in ROLL]
        pie = [WORDS.index(w) for w in PIE]
        big = [WORDS.index(w) for w in BIG]
        self.assertTrue((sims[np.ix_(roll, roll)] == 1).all())
        self.assertEqual(sims[pie[0], pie[1]], 0)       # shared by 2 of 16: too few
        self.assertEqual(sims[big[0], big[1]], 0)       # shared by 7 of 16: too generic
        self.assertEqual(sims[roll[0], WORDS.index("CAT")], 0)
        np.testing.assert_array_equal(sims, sims.T)

    def test_sparse_kind_is_cosine_of_partner_vectors(self):
        sims = SimilarityChannel(f"sparse:{self.cache}").matrix(WORDS)
        pie = [WORDS.index(w) for w in PIE]
        self.assertAlmostEqual(float(sims[pie[0], pie[1]]), 1.0, places=5)  # identical partner sets
        self.assertEqual(sims[WORDS.index("CAT"), WORDS.index("RIVER")], 0)  # no partners at all


class SenseChannelTest(unittest.TestCase):
    def test_max_kind_uses_the_best_matching_senses(self):
        with tempfile.TemporaryDirectory() as tmp:
            senses = np.eye(3, dtype=np.float32)[[0, 1, 1, 2]]  # A: e0,e1  B: e1  C: e2
            np.save(Path(tmp) / "senses.npy", senses)
            (Path(tmp) / "word_to_rows.json").write_text(json.dumps({"A": [0, 1], "B": [2], "C": [3]}))
            sims = SimilarityChannel(f"max:{tmp}").matrix(["A", "B", "C"])
        np.testing.assert_allclose(sims, [[1, 1, 0], [1, 1, 0], [0, 0, 1]])

    def test_parse_rejects_bad_specs(self):
        with self.assertRaises(ValueError):
            parse(["no-equals-sign"])
        with self.assertRaises(ValueError):
            SimilarityChannel("unknown:somewhere")


class ExtraSimilarityModelTest(unittest.TestCase):
    def test_untrained_residual_model_scores_the_weighted_baseline(self):
        torch.manual_seed(0)
        x = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
        similarities = torch.rand(2, 16, 16)
        similarities = (similarities + similarities.transpose(1, 2)) / 2
        model = QuartetGNN(hidden=16, dropout=0.0, pairwise_features=True, input_dim=8,
                           residual=True, extra_similarities=2).eval()
        model.residual_scale.data.fill_(2.0)
        model.similarity_scales.data.copy_(torch.tensor([3.0, 0.5]))
        quartets = torch.tensor(QUARTETS)
        expected = (2.0 * quartet_cosine_sum(x, quartets)
                    + 3.0 * quartet_pair_sum(similarities[0], quartets)
                    + 0.5 * quartet_pair_sum(similarities[1], quartets))
        with torch.no_grad():
            torch.testing.assert_close(model(x, similarities=similarities), expected)

    def test_missing_similarities_raise(self):
        model = QuartetGNN(hidden=16, input_dim=8, residual=True, extra_similarities=1)
        with self.assertRaises(ValueError):
            model(torch.nn.functional.normalize(torch.randn(16, 8), dim=-1))


class TitleListTest(unittest.TestCase):
    def test_lines_format_lowercases_and_drops_disambiguation(self):
        import gzip
        from connections.features.build_phrase_cache import read_lines
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "titles.gz"
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                stream.write("page_title\nMagic_Kingdom_(film)\nMagic_Kingdom\nRock'n'Roll\n!!!\n")
            keys = read_lines(path)
        self.assertIn("magic_kingdom", keys)
        self.assertEqual(keys.count("magic_kingdom"), 1)
        self.assertIn("rocknroll", keys)
        self.assertNotIn("!!!", keys)


class SyntheticSplitTest(unittest.TestCase):
    def _run(self, synthetic_answers):
        root = Path(tempfile.mkdtemp())
        real = [{"words": [f"R{i}" for i in range(16)],
                 "answers": [{"answerDescription": "x", "words": [f"R{i}" for i in range(k, k + 4)]}
                             for k in range(0, 16, 4)]}]
        val = [{"words": [f"V{i}" for i in range(16)],
                "answers": [{"answerDescription": "v", "words": [f"V{i}" for i in range(k, k + 4)]}
                            for k in range(0, 16, 4)]}]
        (root / "train.json").write_text(json.dumps(real))
        (root / "val.json").write_text(json.dumps(val))
        (root / "test.json").write_text(json.dumps([]))
        words = [w for group in synthetic_answers for w in group]
        (root / "syn.json").write_text(json.dumps([{"id": "s1", "tree_id": "s1", "words": words,
            "answers": [{"answerDescription": "s", "words": g} for g in synthetic_answers]}]))
        return subprocess.run([sys.executable, "-m", "connections.data.add_synthetic", "--input-dir", str(root),
                               "--output-dir", str(root / "out"), "--sources", str(root / "syn.json")],
                              cwd=ROOT, capture_output=True, text=True), root

    def test_appends_synthetic_after_real_puzzles(self):
        groups = [[f"S{i}" for i in range(k, k + 4)] for k in range(0, 16, 4)]
        result, root = self._run(groups)
        self.assertEqual(result.returncode, 0, result.stderr)
        train = json.loads((root / "out/train.json").read_text())
        self.assertEqual([p.get("synthetic", False) for p in train], [False, True])
        self.assertEqual((root / "out/val.json").read_bytes(), (root / "val.json").read_bytes())

    def test_rejects_a_group_equal_to_a_validation_group(self):
        groups = [[f"V{i}" for i in range(4)]] + [[f"S{i}" for i in range(k, k + 4)] for k in range(4, 16, 4)]
        result, _ = self._run(groups)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("validation/test group", result.stderr)


if __name__ == "__main__":
    unittest.main()
