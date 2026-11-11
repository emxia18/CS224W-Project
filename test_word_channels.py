"""Check lexical inputs, gradient flow, and word-order equivariance."""

import unittest

import torch
from torch.nn import functional as F

from train_gnn import QuartetGNN
from word_channels import WordTokenizer, SequenceChannel
from explicit_features import NODE_FEATURES, EDGE_FEATURES, one_edit_apart, HomophoneCache


class WordChannelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)

    def test_variants_homophones_unknown_and_phrase(self):
        tokenizer = WordTokenizer(pronunciation=True, dictionary={
            'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']],
            'red': [['R', 'EH1', 'D']], 'fox': [['F', 'AA1', 'K', 'S']],
        })
        self.assertIn(tokenizer.variants('RED')[0], tokenizer.variants('READ'))
        self.assertEqual(len(tokenizer.variants('READ FOX')), 2)
        self.assertEqual(tokenizer.variants('inventedword'), [])
        inputs = tokenizer.encode(['READ', 'inventedword'])
        self.assertFalse(inputs['phonemes'][1].any())

    def test_padding_does_not_change_features(self):
        encoder = SequenceChannel(257, 8).eval()
        tokens = torch.tensor([[3, 4, 5]])
        torch.testing.assert_close(encoder(tokens), encoder(F.pad(tokens, (0, 9))))
        self.assertTrue(torch.equal(encoder(torch.zeros(1, 3, dtype=torch.long)), torch.zeros(1, 8)))

    def test_qualifier_fix_and_legacy_reproduction(self):
        dictionary = {'tee': [['T', 'IY1']], 'golf': [['G', 'AA1', 'L', 'F']]}
        current = WordTokenizer(pronunciation=True, dictionary=dictionary)
        legacy = WordTokenizer(pronunciation=True, dictionary=dictionary, strip_qualifiers=False)
        self.assertEqual(current.variants('TEE (GOLF)'), current.variants('TEE'))
        self.assertNotEqual(legacy.variants('TEE (GOLF)'), legacy.variants('TEE'))
        self.assertFalse(torch.equal(current.encode(['TEE (GOLF)'])['spelling'], current.encode(['TEE'])['spelling']))

    def test_explicit_relationships(self):
        tokenizer = WordTokenizer(orthographic_features=True, dictionary={})
        words = ['LEVEL', 'PEP', 'REFER', 'TENET', 'EAST', 'SEAT', 'CATS', 'CAT',
                 'DEED', 'DEE', 'READ', 'REED', 'unknown', 'another', 'CANT', "CAN'T"]
        inputs = tokenizer.encode(words)
        nodes, edges = inputs['orthographic_nodes'], inputs['orthographic_edges']
        self.assertTrue(nodes[:4, NODE_FEATURES.index('palindrome')].eq(1).all())
        self.assertEqual(edges[4, 5, EDGE_FEATURES.index('anagram')], 1)
        self.assertEqual(edges[6, 7, EDGE_FEATURES.index('one_edit_apart')], 1)
        self.assertEqual(edges[6, 7, EDGE_FEATURES.index('shared_prefix')], 1)
        self.assertFalse(edges[12, 13, EDGE_FEATURES.index('same_pronunciation')])
        self.assertTrue(one_edit_apart('cat', 'cats'))
        self.assertTrue(one_edit_apart('cat', 'bat'))
        self.assertFalse(one_edit_apart('cat', 'dog'))
        self.assertFalse(one_edit_apart('cat', 'cat'))

    def test_homophone_retrieval_and_missing_semantics(self):
        import json
        from pathlib import Path
        import tempfile
        import numpy as np
        from build_homophone_cache import retrieve_candidates
        tokenizer = WordTokenizer(dictionary={'indy': [['IH1', 'N', 'D', 'IY0']],
            'indie': [['IH1', 'N', 'D', 'IY0']]})
        self.assertEqual(retrieve_candidates(tokenizer, ['INDY', 'xyz']), {'INDY': ['indie'], 'xyz': []})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'metadata.json').write_text(json.dumps({'tokenization': 'strip-parenthetical-stressless-v2'}))
            (root / 'words.json').write_text(json.dumps({'a': {'rows': [0]}, 'b': {'rows': [1]}, 'c': {'rows': []}}))
            vectors = np.zeros((2, 768), dtype=np.float32); vectors[:, 0] = 1
            np.save(root / 'vectors.npy', vectors)
            cache = HomophoneCache(root)
            edges = cache.edges(['a', 'b', 'c'])
            self.assertEqual(edges[0, 1].tolist(), [1, 1])
            self.assertEqual(edges[0, 2].tolist(), [0, 0])
            with self.assertRaises(ValueError):
                cache.edges(['not_in_cache'])

    def test_explicit_features_gradients_and_equivariance(self):
        inputs = WordTokenizer(orthographic_features=True, dictionary={}).encode(
            ['LEVEL', 'PEP', 'REFER', 'TENET', 'EAST', 'SEAT', 'CATS', 'CAT'] * 2)
        inputs['homophone_edges'] = torch.zeros(16, 16, 2)
        model = QuartetGNN(hidden=8, dropout=0, orthographic_features=True,
                           homophone_features=True, hint_features=True,
                           learned_relations=2, spelling_features=True, pronunciation_features=True)
        x = F.normalize(torch.randn(16, 768), dim=-1)
        original = model(x, word_inputs=inputs)
        original.square().mean().backward()
        self.assertGreater(model.orthographic_project.weight.grad.abs().sum(), 0)
        order = torch.randperm(16)
        permuted_inputs = {k: v[order][:, order] if k.endswith('_edges') else v[order]
                           for k, v in inputs.items()}
        permuted = model(x[order], word_inputs=permuted_inputs)
        ids = {tuple(group): i for i, group in enumerate(model.quartets.tolist())}
        reorder = [ids[tuple(sorted(order[group].tolist()))] for group in model.quartets]
        torch.testing.assert_close(permuted, original[reorder], atol=1e-6, rtol=1e-5)

    def test_gradients_and_permutation(self):
        tokenizer = WordTokenizer(pronunciation=True, dictionary={
            'red': [['R', 'EH1', 'D']], 'read': [['R', 'IY1', 'D'], ['R', 'EH1', 'D']],
        })
        inputs = tokenizer.encode(['RED', 'READ', 'unknown', 'a'] * 4)
        model = QuartetGNN(hidden=8, dropout=0, learned_relations=2,
                           spelling_features=True, pronunciation_features=True, hint_features=True)
        x = F.normalize(torch.randn(16, 768), dim=-1)
        scores = model(x, word_inputs=inputs)
        self.assertEqual(scores.shape, (1820,))
        scores.square().mean().backward()
        for component in (model.relations, model.channels.spelling, model.channels.pronunciation):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters()))
        model.eval()
        permutation = torch.randperm(16)
        permuted = model(x[permutation], word_inputs={k: v[permutation] for k, v in inputs.items()})
        original = model(x, word_inputs=inputs)
        ids = {tuple(group): i for i, group in enumerate(model.quartets.tolist())}
        reorder = [ids[tuple(sorted(permutation[group].tolist()))] for group in model.quartets]
        torch.testing.assert_close(permuted, original[reorder], atol=1e-6, rtol=1e-5)
        with self.assertRaises(ValueError):
            model(x)


if __name__ == '__main__':
    unittest.main()
