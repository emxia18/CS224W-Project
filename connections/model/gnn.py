"""GATv2 quartet scorer: the graph neural network at the heart of the project.

Each puzzle is a graph with one node per word. Node features are frozen word
embeddings; edge features are their cosine similarities (plus optional learned,
spelling, pronunciation, hint and explicit-relationship channels). One residual
GATv2 layer mixes information across the puzzle, then a permutation-invariant
readout scores each candidate quartet from the mean/variance of its members.
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GATv2Conv

from connections.features.explicit_features import EDGE_FEATURES, NODE_FEATURES
from connections.features.word_channels import WordChannels
from connections.model.decoding import QUARTETS


EDGE_INDEX = torch.tensor(
    [(i, j) for i in range(16) for j in range(16)], dtype=torch.long
).t().contiguous()
# "none" keeps only self-loops: every attention softmax has one entry, so the
# GATv2 layer becomes a nodewise linear map with identical parameters and no
# information flow between words (the no-message-passing control).
GRAPHS = {
    "complete": EDGE_INDEX,
    "none": torch.arange(16, dtype=torch.long).repeat(2, 1),
}



def quartet_cosine_stats(x, groups):
    """Mean, minimum, maximum, and population std of six distinct pairs."""
    members = x[groups]
    pairs = torch.triu_indices(4, 4, offset=1, device=x.device)
    similarities = (members[:, pairs[0]] * members[:, pairs[1]]).sum(dim=-1)
    return torch.stack((
        similarities.mean(dim=1), similarities.amin(dim=1),
        similarities.amax(dim=1), similarities.std(dim=1, unbiased=False),
    ), dim=-1)


def feature_stats(values):
    """Pool members or distinct pair features without depending on word order."""
    return torch.cat((values.mean(1), values.amin(1), values.amax(1),
                      values.std(1, unbiased=False)), dim=-1)


def quartet_edge_stats(edges, groups):
    pairs = torch.triu_indices(4, 4, offset=1, device=groups.device)
    return feature_stats(edges[groups[:, pairs[0]], groups[:, pairs[1]]])


class QuartetGNN(nn.Module):
    def __init__(self, hidden=128, dropout=0.1, pairwise_features=False, hint_features=False,
                 learned_relations=0, spelling_features=False, pronunciation_features=False,
                 orthographic_features=False, homophone_features=False, graph="complete"):
        super().__init__()
        if graph not in GRAPHS:
            raise ValueError(f"Unknown graph: {graph}")
        self.pairwise_features = pairwise_features
        self.hint_features = hint_features
        self.orthographic_features = orthographic_features
        self.homophone_features = homophone_features
        self.project = nn.Linear(768, hidden)
        self.channels = WordChannels(hidden, spelling_features, pronunciation_features)
        channel_count = int(spelling_features) + int(pronunciation_features)
        self.relations = nn.ModuleList(nn.Linear(768, 32) for _ in range(learned_relations))
        explicit_edges = (len(EDGE_FEATURES) if orthographic_features else 0) + (2 if homophone_features else 0)
        extra_edges = learned_relations + channel_count + explicit_edges
        self.gat = GATv2Conv(
            hidden, hidden, heads=1, edge_dim=1 + int(hint_features) + extra_edges, dropout=dropout,
            add_self_loops=False,
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)
        self.phi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.rho = nn.Sequential(
            nn.Linear(hidden * 2 + (4 if pairwise_features else 0) + extra_edges * 4
                      + (len(NODE_FEATURES) * 4 if orthographic_features else 0), hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )
        self.orthographic_project = nn.Linear(len(NODE_FEATURES), hidden) if orthographic_features else None
        self.register_buffer("edge_index", GRAPHS[graph].clone())
        self.register_buffer("quartets", torch.tensor(QUARTETS, dtype=torch.long))

    def forward(self, x, quartet_ids=None, hints=None, word_inputs=None):
        # Cached embeddings are unit length, so dot products are cosine similarities.
        source, target = self.edge_index
        edge_attr = (x[source] * x[target]).sum(dim=-1, keepdim=True)
        if (self.channels.spelling is not None or self.channels.pronunciation is not None
                or self.orthographic_features or self.homophone_features) and word_inputs is None:
            raise ValueError("Enabled word channels require tokenized word_inputs")
        channels = self.channels(word_inputs) if word_inputs is not None else []
        relationship_vectors = [F.normalize(layer(x), dim=-1) for layer in self.relations]
        relationship_vectors += [F.normalize(channel, dim=-1) for channel in channels]
        if relationship_vectors:
            edge_attr = torch.cat([edge_attr] + [
                (z[source] * z[target]).sum(-1, keepdim=True) for z in relationship_vectors
            ], dim=-1)
        explicit = []
        if self.orthographic_features:
            explicit.append(word_inputs['orthographic_edges'])
        if self.homophone_features:
            explicit.append(word_inputs['homophone_edges'])
        if explicit:
            edge_attr = torch.cat([edge_attr] + [values[source, target] for values in explicit], dim=-1)
        if self.hint_features:
            membership = torch.zeros_like(edge_attr[:, :1]) if hints is None else hints[source, target, None]
            edge_attr = torch.cat((edge_attr, membership), dim=-1)
        h = self.project(x) + sum(channels)
        if self.orthographic_project is not None:
            h = h + self.orthographic_project(word_inputs['orthographic_nodes'])
        h = F.relu(h)
        h = self.norm(h + self.dropout(F.elu(self.gat(h, self.edge_index, edge_attr))))
        groups = self.quartets if quartet_ids is None else self.quartets[quartet_ids]
        members = self.phi(h)[groups]
        # Variance supplies explicit within-group interactions. A nearly linear
        # sum-only scorer can assign similar total scores to every partition.
        pooled = torch.cat((members.mean(dim=1), members.var(dim=1, unbiased=False)), dim=-1)
        if self.pairwise_features:
            pooled = torch.cat((pooled, quartet_cosine_stats(x, groups)), dim=-1)
        if relationship_vectors:
            pooled = torch.cat([pooled] + [quartet_cosine_stats(z, groups)
                                          for z in relationship_vectors], dim=-1)
        if self.orthographic_features:
            pooled = torch.cat((pooled, feature_stats(word_inputs['orthographic_nodes'][groups])), dim=-1)
        if explicit:
            pooled = torch.cat([pooled] + [quartet_edge_stats(values, groups) for values in explicit], dim=-1)
        return self.rho(pooled).squeeze(-1)
