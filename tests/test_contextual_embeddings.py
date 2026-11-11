"""Check contextual pooling and cache alignment without downloading a model."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from connections.embeddings.contextual_embeddings import ContextualCache, pool_words, puzzle_key, puzzle_text


class ContextualTests(unittest.TestCase):
    def test_canonical_order_and_target_pooling(self):
        self.assertEqual(puzzle_text(['BANK', 'APPLE']), puzzle_text(['APPLE', 'BANK']))
        hidden = torch.tensor([[99., 99.], [1., 0.], [3., 0.], [0., 5.]])
        offsets = torch.tensor([[0, 0], [0, 2], [2, 4], [7, 11]])
        result = pool_words(hidden, offsets, [(0, 4), (7, 11)])
        torch.testing.assert_close(result, torch.eye(2))
        with self.assertRaises(ValueError):
            pool_words(hidden, offsets, [(7, 12)])

    def test_cache_reorders_nodes_and_rejects_stale_data(self):
        words = [f'word{i:02}' for i in range(16)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            split = root / 'train.json'
            split.write_text('[]')
            digest = hashlib.sha256(split.read_bytes()).hexdigest()
            (root / 'metadata.json').write_text(json.dumps({'input_sha256': {'train': digest}}))
            (root / 'index.json').write_text(json.dumps({puzzle_key(words): {'row': 0, 'words': words}}))
            vectors = np.zeros((1, 16, 768), dtype=np.float32)
            vectors[0, np.arange(16), np.arange(16)] = 1
            np.save(root / 'embeddings.npy', vectors)
            cache = ContextualCache(root)
            cache.validate_split(split)
            np.testing.assert_array_equal(cache.lookup(words[::-1]), vectors[0, ::-1])
            split.write_text('[{}]')
            with self.assertRaises(ValueError):
                cache.validate_split(split)


if __name__ == '__main__':
    unittest.main()
