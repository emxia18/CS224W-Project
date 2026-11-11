"""Check lexical inputs, gradient flow, and word-order equivariance."""

import unittest

import torch
from torch.nn import functional as F

from train_gnn import QuartetGNN
from word_channels import WordTokenizer, SequenceChannel


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
