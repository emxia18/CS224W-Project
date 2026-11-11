"""Check actual quartet enumeration, anchor selection and global ambiguity."""

import unittest
from itertools import combinations
import copy

import numpy as np

from constraints import partition_count
from mpnet import rank_quartets
from build import audit


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.words = list('ABCDEFGH')
        rng = np.random.default_rng(224)
        self.vectors = rng.normal(size=(8, 12))
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)

    def test_enumerates_seventy_and_scores_six_pairs(self):
        result = rank_quartets(self.words, self.vectors)
        self.assertEqual(len(result['quartets']), 70)
        expected = np.mean([self.vectors[i] @ self.vectors[j]
                            for i, j in combinations(range(4), 2)])
        self.assertAlmostEqual(result['quartets'][0]['similarity'], expected)
        self.assertEqual(result['selections']['yellow']['similarity'], result['max_similarity'])
        self.assertEqual(result['selections']['purple']['similarity'], result['min_similarity'])
        self.assertGreater(result['selections']['green']['target'], result['selections']['blue']['target'])

    def test_anchor_is_preserved_after_full_enumeration(self):
        result = rank_quartets(self.words, self.vectors, ['A'])
        self.assertEqual(len(result['quartets']), 70)
        self.assertEqual(result['eligible_count'], 35)
        for selected in result['selections'].values():
            self.assertIn('A', selected['words'])

    def test_missing_anchor_fails(self):
        with self.assertRaises(ValueError):
            rank_quartets(self.words, self.vectors, ['X'])

    def test_partition_catalogue_detects_real_ambiguity(self):
        words = list('ABCDEFGHIJKLMNOP')
        groups = [words[i:i+4] for i in range(0, 16, 4)]
        self.assertEqual(partition_count(words, groups), 1)
        alternative = [words[i::4] for i in range(4)]
        self.assertEqual(partition_count(words, groups + alternative), 2)

    def test_false_group_does_not_make_complete_second_solution(self):
        words = list('ABCDEFGHIJKLMNOP')
        groups = [words[i:i+4] for i in range(0, 16, 4)]
        self.assertEqual(partition_count(words, groups + [['A', 'E', 'I', 'M']]), 1)


class TrainingDiversityTests(unittest.TestCase):
    @staticmethod
    def puzzle(pid, prefix):
        words = [f'{prefix}{i}' for i in range(16)]
        return {'id': pid, 'tree_id': pid, 'words': words,
                'answers': [{'words': words[i*4:(i+1)*4],
                             'answerDescription': f'{prefix} theme {i}',
                             'category_key': f'{prefix}_concept_{i}', 'color': color}
                            for i, color in enumerate(('yellow', 'green', 'blue', 'purple'))]}

    def test_disjoint_puzzles_pass(self):
        report = audit([self.puzzle('a', 'X'), self.puzzle('b', 'Y')])
        self.assertEqual(report['unique_input_words'], 32)
        self.assertEqual(report['unique_reviewed_category_concepts'], 8)

    def test_case_changes_cannot_bypass_word_exclusion(self):
        a, b = self.puzzle('a', 'X'), self.puzzle('b', 'Y')
        b['words'][0] = a['words'][0].lower()
        b['answers'][0]['words'][0] = a['words'][0].lower()
        with self.assertRaisesRegex(ValueError, 'Cross-puzzle repetition'):
            audit([a, b])

    def test_renaming_same_semantic_category_does_not_bypass_exclusion(self):
        a, b = self.puzzle('a', 'X'), self.puzzle('b', 'Y')
        b['answers'][0]['category_key'] = a['answers'][0]['category_key']
        with self.assertRaisesRegex(ValueError, 'category_concept'):
            audit([a, b])

    def test_same_label_with_new_key_does_not_bypass_exclusion(self):
        a, b = self.puzzle('a', 'X'), self.puzzle('b', 'Y')
        b['answers'][0]['answerDescription'] = a['answers'][0]['answerDescription']
        with self.assertRaisesRegex(ValueError, 'category_label'):
            audit([a, b])


if __name__ == '__main__':
    unittest.main()
