"""Answer-independent spelling and pronunciation inputs and trainable encoders."""

from itertools import product
import re

import torch
from torch import nn
from torch.nn import functional as F


PHONES = 'AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split()
PHONE_IDS = {phone: i + 2 for i, phone in enumerate(PHONES)}


class WordTokenizer:
    """Fixed alphabets; CMUdict variants; unknown pronunciation is explicitly absent.

    UTF-8 bytes preserve spelling without fitting a vocabulary on held-out words.
    Stress is removed so homophones differing only in stress can share tokens.
    Whole-phrase entries are preferred, otherwise component variants are combined.
    """

    def __init__(self, pronunciation=False, dictionary=None, strip_qualifiers=True,
                 orthographic_features=False, homophone_cache=None):
        self.strip_qualifiers = strip_qualifiers
        self.orthographic_features = orthographic_features
        self.homophone_cache = homophone_cache
        self.dictionary = dictionary
        if (pronunciation or orthographic_features or homophone_cache is not None) and dictionary is None:
            import cmudict
            self.dictionary = cmudict.dict()

    def pronunciation_word(self, word):
        word = word.lower().replace('’', "'")
        if self.strip_qualifiers:
            word = re.sub(r'\([^()]*\)', '', word)
        return ' '.join(word.split())

    def variants(self, word):
        word = self.pronunciation_word(word)
        variants = self.dictionary.get(word) if self.dictionary is not None else None
        if not variants and self.dictionary is not None:
            parts = re.findall(r"[a-z]+(?:'[a-z]+)*|[0-9]+", word)
            entries = [self.dictionary.get(part, []) for part in parts]
            if entries and all(entries):
                # Bound phrase expansion; each component keeps up to four variants.
                variants = []
                for combo in product(*(entry[:4] for entry in entries)):
                    variants.append([phone for part in combo for phone in part])
                    if len(variants) == 16:
                        break
        result = []
        for variant in variants or []:
            tokens = [PHONE_IDS.get(re.sub(r'\d', '', phone), 1) for phone in variant]
            if tokens and tokens not in result:
                result.append(tokens)
        return result

    def encode(self, words, device='cpu'):
        chars = [[b + 1 for b in word.lower().encode('utf-8')] for word in words]
        pronunciations = [self.variants(word) for word in words]
        spelling = torch.zeros(len(words), max(1, max(map(len, chars))), dtype=torch.long)
        phonemes = torch.zeros(len(words), max(1, max(map(len, pronunciations))),
                               max([1] + [len(v) for variants in pronunciations for v in variants]),
                               dtype=torch.long)
        for i, tokens in enumerate(chars):
            spelling[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        for i, variants in enumerate(pronunciations):
            for j, tokens in enumerate(variants):
                phonemes[i, j, :len(tokens)] = torch.tensor(tokens, dtype=torch.long)
        result = {'spelling': spelling.to(device), 'phonemes': phonemes.to(device)}
        if self.orthographic_features:
            from connections.features.explicit_features import orthographic_inputs
            nodes, edges = orthographic_inputs(words, pronunciations)
            result.update(orthographic_nodes=nodes.to(device), orthographic_edges=edges.to(device))
        if self.homophone_cache is not None:
            result['homophone_edges'] = self.homophone_cache.edges(words).to(device)
        return result


class SequenceChannel(nn.Module):
    """Local patterns plus first/last tokens; masked pooling ignores padding."""

    def __init__(self, vocabulary, hidden):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary, 24, padding_idx=0)
        self.convs = nn.ModuleList(nn.Conv1d(24, hidden, k, padding=k // 2) for k in (3, 5))
        self.project = nn.Linear(hidden * 2 + 48, hidden)

    def forward(self, tokens):
        valid = tokens.ne(0)
        embedded = self.embedding(tokens)
        encoded = embedded.transpose(1, 2)
        pooled = [F.gelu(conv(encoded)).masked_fill(~valid[:, None], -1e9).amax(-1)
                  for conv in self.convs]
        last = (valid.sum(-1) - 1).clamp_min(0)
        endpoints = [embedded[:, 0], embedded[torch.arange(len(tokens), device=tokens.device), last]]
        output = self.project(torch.cat(pooled + endpoints, dim=-1))
        return output * valid.any(-1, keepdim=True)


class WordChannels(nn.Module):
    def __init__(self, hidden, spelling=False, pronunciation=False):
        super().__init__()
        self.spelling = SequenceChannel(257, hidden) if spelling else None
        self.pronunciation = SequenceChannel(len(PHONE_IDS) + 2, hidden) if pronunciation else None

    def forward(self, inputs):
        channels = []
        if self.spelling is not None:
            channels.append(self.spelling(inputs['spelling']))
        if self.pronunciation is not None:
            phones = inputs['phonemes']
            valid = phones.ne(0).any(-1)
            encoded = self.pronunciation(phones.flatten(0, 1)).reshape(*phones.shape[:2], -1)
            # Keep alternate pronunciations, pooling only over available variants.
            channels.append(encoded.sum(1) / valid.sum(1, keepdim=True).clamp_min(1))
        return channels
