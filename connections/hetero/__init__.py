"""Heterogeneous-graph extensions, kept separate from the main model code.

concepts.py / model.py / train_concepts.py: per-puzzle graphs with concept nodes
(Option 1). global_graph.py: one graph over all puzzle words and concepts, with
same-group edges from training puzzles, trained by link prediction (Option 2).
"""
