"""QuartetGNN plus a concept-node branch over a per-puzzle heterogeneous graph.

Nodes: the puzzle's 16 words and its concept nodes (concepts.py). Messages run
word -> concept -> word -> concept, with separate weights per concept source
(relation type). Each concept then gets a score s_c, and every quartet gains
sum_c s_c * [concept c links all four of its words]: the "one hidden category
explains exactly these four words" signal, which pairwise similarities cannot
express. The score head starts at zero, so the untrained model equals the base.
"""

import torch
from torch import nn
from torch.nn import functional as F


class ConceptQuartetGNN(nn.Module):
    def __init__(self, base, input_dim, n_sources, concept_features, hidden=64, dropout=0.1):
        super().__init__()
        self.base = base
        self.word_in = nn.Linear(input_dim, hidden)
        self.concept_in = nn.Linear(concept_features, hidden)
        self.word_to_concept = nn.Parameter(torch.empty(2, n_sources, hidden, hidden))
        self.concept_to_word = nn.Parameter(torch.empty(n_sources, hidden, hidden))
        for weight in list(self.word_to_concept.view(-1, hidden, hidden)) + list(self.concept_to_word):
            nn.init.xavier_uniform_(weight)
        self.dropout = nn.Dropout(dropout)
        self.score = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    @property
    def residual_scale(self):
        return self.base.residual_scale

    @property
    def similarity_scales(self):
        return self.base.similarity_scales

    def concept_scores(self, x, graph):
        incidence, kind = graph["incidence"], graph["kind"]
        concept_degree = incidence.sum(1, keepdim=True).clamp(min=1)
        word_degree = incidence.sum(0).clamp(min=1).unsqueeze(-1)
        words = F.relu(self.word_in(x))
        # word -> concept, relation-specific: message from word w to concept c uses W[kind(c)].
        relayed = torch.einsum("wh,khd->kwd", words, self.word_to_concept[0])[kind]  # (C, 16, H)
        concepts = F.relu(self.concept_in(graph["features"])
                          + (incidence.unsqueeze(-1) * relayed).sum(1) / concept_degree)
        concepts = self.dropout(concepts)
        # concept -> word
        sent = torch.einsum("ch,chd->cd", concepts, self.concept_to_word[kind])
        words = F.relu(words + incidence.t() @ sent / word_degree)
        # word -> concept again: a concept now sees which other concepts its words have
        relayed = torch.einsum("wh,khd->kwd", words, self.word_to_concept[1])[kind]
        concepts = F.relu(concepts + (incidence.unsqueeze(-1) * relayed).sum(1) / concept_degree)
        return self.score(self.dropout(concepts)).squeeze(-1)

    def forward(self, x, quartet_ids=None, hints=None, word_inputs=None, similarities=None):
        base_inputs = {k: v for k, v in (word_inputs or {}).items() if k != "concept_graph"} or None
        scores = self.base(x, quartet_ids, hints=hints, word_inputs=base_inputs, similarities=similarities)
        graph = (word_inputs or {}).get("concept_graph")
        if graph is None or graph["incidence"].shape[0] == 0:
            return scores
        groups = self.base.quartets if quartet_ids is None else self.base.quartets[quartet_ids]
        covers_all = graph["incidence"][:, groups].prod(-1)  # (C, quartets)
        return scores + self.concept_scores(x, graph) @ covers_all
