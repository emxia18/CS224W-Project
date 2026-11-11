# Baselines: cosine similarity, no message passing, greedy decoding

Run date: 2026-11-10. Validation split only (98 puzzles, 2024-09-10 to 2024-12-16).
The test split was not evaluated.

## Summary

| Model | Decoder | Groups / puzzle | Groups / 392 | Exact / 98 | Matched words / 16 | Best epoch |
|---|---|---:|---:|---:|---:|---:|
| Random partition | — | 0.009 | 0.9 | 0.0 | 7.60 | — |
| Cosine sum (no training) | exact | **0.561** | **55** | 2 | **10.41** | — |
| Cosine sum (no training) | greedy | 0.520 | 51 | 1 | 10.04 | — |
| GATv2, complete graph | exact | 0.516 ± 0.039 | 50.6 ± 3.8 | **2.4 ± 0.5** | 10.07 ± 0.09 | 3.2 ± 0.8 |
| GATv2, complete graph | greedy | 0.508 ± 0.057 | 49.8 ± 5.6 | 2.2 ± 0.8 | 10.00 ± 0.15 | 3.2 ± 0.8 |
| No message passing | exact | 0.424 ± 0.054 | 41.6 ± 5.3 | 1.0 ± 0.7 | 9.99 ± 0.08 | 2.4 ± 0.5 |
| No message passing | greedy | 0.457 ± 0.030 | 44.8 ± 2.9 | 2.0 ± 0.7 | 9.88 ± 0.07 | 2.4 ± 0.5 |

Trained rows show the mean ± sample std over seeds 0–4. Random-partition values
are Monte Carlo estimates (200 partitions per puzzle). The analytic chance level
is 4/455 ≈ 0.0088 groups per puzzle.

## Findings

1. **Message passing helps relative to its control.** With exact decoding, GATv2
   beats the parameter-matched no-message-passing model on every seed, by 6 to 13
   groups out of 392 (see per-seed results below).
2. **Neither trained model beats untrained cosine similarity.** Cosine sum with
   exact decoding recovers 55 groups. The best single trained run recovers 57
   groups, and the GATv2 mean is 50.6. The trained numbers are also optimistic:
   each checkpoint was selected on these same validation puzzles, while the
   cosine baseline has no selection step. Both trained models receive the mean
   pairwise cosine in their readout, so they could represent the cosine ranking;
   training on 456 puzzles makes them worse.
3. **The trained models overfit early.** Validation groups peak at epoch 2–4 and
   then fall while training loss keeps decreasing. In the seed-42 reproduction,
   loss went from 0.65 to 0.38 while validation groups went from 0.44 (epoch 4)
   to 0.29 (epoch 8).
4. **Exact decoding helps cosine scores but not consistently for trained
   scorers.** Exact beats greedy by 4 groups for cosine. For GATv2 the
   difference is within seed noise, and the no-message-passing model does
   slightly better with greedy.
5. **Single-seed comparisons are noisy.** Across seeds, the same GATv2
   configuration ranges from 47 to 57 groups. Differences of a few groups
   between single runs (such as those in the README's word-feature table) cannot
   be distinguished from seed variation.

## Per-seed results

| Model | Seed | Best epoch | Groups / 392 (exact) | Exact / 98 (exact) | Matched words (exact) | Groups / 392 (greedy) | Exact / 98 (greedy) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GATv2 | 0 | 3 | 50 | 2 | 10.12 | 54 | 3 |
| GATv2 | 1 | 4 | 57 | 3 | 10.04 | 48 | 1 |
| GATv2 | 2 | 4 | 47 | 2 | 10.01 | 41 | 2 |
| GATv2 | 3 | 3 | 49 | 2 | 10.19 | 55 | 3 |
| GATv2 | 4 | 2 | 50 | 3 | 9.98 | 51 | 2 |
| No message passing | 0 | 3 | 42 | 2 | 9.90 | 46 | 3 |
| No message passing | 1 | 3 | 49 | 1 | 10.04 | 47 | 2 |
| No message passing | 2 | 2 | 37 | 1 | 10.03 | 40 | 2 |
| No message passing | 3 | 2 | 36 | 0 | 9.90 | 44 | 2 |
| No message passing | 4 | 2 | 44 | 1 | 10.07 | 47 | 1 |

Cosine sum on the **training** split, for reference (no fitting involved):
0.711 groups/puzzle and 15/456 exact with exact decoding, and 0.572 groups/puzzle
and 9/456 exact with greedy decoding. The validation period appears harder than
the training period for pure similarity.

## What each model is

- **Embeddings (all models):** frozen `sentence-transformers/all-mpnet-base-v2`
  at revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`. Each word or phrase is
  encoded alone, without puzzle context. Each vector is the mean of the token
  vectors, normalized to unit length, with 768 dimensions. The tokenizer
  lowercases input.
- **Cosine sum:** a quartet's score is the sum of its six pairwise cosine
  similarities. No parameters.
- **GATv2, complete graph:** the baseline configuration from the README
  (`--all-quartets --pairwise-features`). It has a 768→128 projection, one
  residual single-head GATv2 layer with cosine edge features on all 16×16 word
  pairs, and a mean/variance quartet readout plus pairwise cosine statistics.
  The loss is balanced BCE over all 1,820 quartets. It has 182,145 parameters.
- **No message passing (`--graph none`):** identical model, but the graph has
  only self-loops. Each attention softmax has one entry, so the GATv2 layer
  becomes a nodewise linear map and a quartet's score depends only on its own
  four words. It has the same 182,145 parameters.
- **Exact decoder:** the four disjoint quartets with the maximum total score.
- **Greedy decoder:** repeatedly take the highest-scoring quartet that is
  disjoint from earlier picks.

Training settings for both trained models: hidden size 128, dropout 0.1, AdamW
with learning rate 1e-3 and weight decay 1e-4, 8 epochs, patience 8, one puzzle
per update, CPU with one thread, no curriculum. The selection metric is
validation groups, with ties broken by exact accuracy.

## Reproducibility notes

- **Environment:** Python 3.11.3 (arm64, created with `uv`) with
  `requirements-reproduce.txt`. All 18 unit tests pass, 7 of them new
  (`test_baselines.py`).
- **Data:** split hashes match `reproduction_manifest.json`.
- **Embeddings:** regenerated from the pinned MPNet revision. They are **not**
  byte-identical to the recorded cache (`verify_reproduction.py
  --check-embeddings` reports a mismatch). As a result, the seed-42 baseline GAT
  gives 0.439 groups/puzzle (43 groups, 2 exact, best epoch 4), versus 0.500 in
  the README's word-feature table. Results in this file are internally
  consistent because every model uses the same regenerated embeddings.

Commands:

```sh
.venv/bin/python baselines.py

mkdir -p runs/compare
for seed in 0 1 2 3 4; do for graph in complete none; do
  name=$([ $graph = complete ] && echo gat || echo nomp)
  .venv/bin/python train_gnn.py --all-quartets --pairwise-features --seed $seed \
    --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
    --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
    --selection-metric groups --graph $graph \
    --output-dir runs/compare/${name}_seed$seed > runs/compare/${name}_seed$seed.log 2>&1 &
done; done; wait

.venv/bin/python summarize_runs.py \
  --group "GATv2 (complete graph)=runs/compare/gat_seed*" \
  --group "No message passing=runs/compare/nomp_seed*"
```

Raw outputs: `runs/baselines/metrics.json` and `runs/compare/*/` (both gitignored).
