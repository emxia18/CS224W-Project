# Residual scoring, a partition-level loss, and alternative embeddings

Run date: 2026-11-10. Validation split only (98 puzzles, 2024-09-10 to 2024-12-16).
The test split was not evaluated.

This follows up on `results/baselines.md`, where the trained GATv2 (50.6 groups)
lost to untrained cosine similarity (55 groups). This file tests two changes to
the scorer and loss, and repeats the comparison on two other embeddings.

## Summary

Validation groups recovered out of 392 (exact puzzles out of 98 in parentheses).
Trained rows give the mean over seeds 0–4, using the checkpoint selected on validation.

| Embedding | Cosine sum (no training) | GATv2, BCE (original) | GATv2 + residual + CRF | No message passing + residual + CRF |
|---|---:|---:|---:|---:|
| MPNet | 55 (2) | 45.6 (2.0) | 59.8 (3.8) | 58.6 (2.4) |
| **fastText crawl** | **106 (8)** | 81.6 (3.4) | **116.8 (11.0)** | 112.8 (9.4) |
| mxbai-large | 68 (3) | 57.4 (3.0) | 68.0 (3.0) | 68.0 (3.0) |

## Findings

1. **The original training falls below untrained cosine on every embedding.**
   It does so even with validation-selected checkpoints: −9.4 groups on MPNet,
   −24.4 on fastText and −10.6 on mxbai. The training setup is the problem,
   not the embedding.
2. **Starting the scorer at the cosine baseline fixes most of this.** The residual
   scorer learns a correction on top of cosine similarity rather than relearning
   it from 456 puzzles. On MPNet it adds about 13 groups over the original model
   (45.6 → 58.8 with BCE, 59.8 with the CRF loss).
3. **The gain over cosine depends on the embedding.** Without checkpoint selection
   (the best mean over seeds at any one epoch), residual + CRF beats cosine by
   7.2 groups on fastText (113.2 vs 106) and by 2.8 on MPNet (57.8 vs 55). On
   mxbai it never beats cosine: every seed's best epoch is epoch 1, which still
   ranks quartets like cosine, and validation declines after that.
4. **fastText nearly doubles the zero-training baseline.** Its cosine sum recovers
   106 groups vs 55 for MPNet (paired bootstrap over puzzles: +51, 95% CI
   [+29, +73]). The gap holds on the training split (528 vs 324 of 1,824 groups),
   so it is not an artifact of picking the best of eight embeddings on validation.
   The sentence-transformer encoders do not clearly beat MPNet, and the Qwen3
   embedders are worse.
5. **Message passing helps a little, and only on fastText.** With residual + CRF,
   GATv2 beats the no-message-passing control on all five fastText seeds, by 1 to
   8 groups (mean 4.0; 11.0 vs 9.4 exact puzzles). On MPNet the mean difference
   is 1.2 groups with no consistent sign across seeds, and on mxbai it is 0.
   The repository's earlier result (GATv2 50.6 vs control 41.6 on MPNet with BCE)
   did not replicate on this machine: 45.6 vs 44.4.
6. **The models overfit within a few epochs.** Validation peaks within the first
   five epochs in every configuration, while scores on training puzzles keep rising.
7. **Exact decoding beats greedy in every configuration on average.** The gap
   is largest with residual + CRF: +10.2 groups on MPNet, +29.8 on fastText and
   +11.0 on mxbai.

## What changed

**Residual scorer.** The quartet score becomes

```
score(S) = a · cosine_sum(S) + g(S)
```

- `cosine_sum(S)` is the zero-parameter baseline: the sum of the six pairwise
  cosines among S's four words.
- `g(S)` is the existing GATv2 quartet scorer, with its last layer initialized
  to zero. Before training, the model therefore ranks quartets exactly like the
  cosine baseline.
- `a` is a learnable scale. Its initial value is fit on training puzzles only:
  2.41 for BCE on MPNet, and 6.87, 6.97 and 12.0 for CRF on MPNet, fastText and
  mxbai. It barely moves during training (with CRF it ends at 6.9–7.5 for
  MPNet and fastText, and 12.2 for mxbai).

**Partition-level CRF loss.** The original loss classifies each of the 1,820
quartets independently. The CRF loss scores whole partitions:

```
loss = log Σ_P exp(score(P)) − score(gold partition)
```

- The sum runs over all 2,627,625 partitions into four groups of four, and
  `score(P)` is the sum of its four quartet scores. That is the quantity the
  exact decoder maximizes, so training and decoding now optimize the same thing.
- log Z is computed with the same dynamic program over word subsets as
  `decode()`, with max replaced by log-sum-exp. Check: with all scores zero it
  returns log(2,627,625) exactly.

**Embedding dimension.** fastText vectors are 300-dimensional and mxbai's are
1,024, so the first projection layer was resized for those runs. Everything
else is unchanged.

## Embedding comparison (cosine sum, exact decoder, no training)

From `compare_embeddings.py`. MPNet is the reference for the paired bootstrap CIs
(10,000 resamples of puzzles).

| Embedding | Dim | Val groups / 392 | Δ vs MPNet [95% CI] | Val exact / 98 | Train groups / 1824 | Δ vs MPNet [95% CI] | Train exact / 456 |
|---|---:|---:|---:|---:|---:|---:|---:|
| MPNet (all-mpnet-base-v2) | 768 | 55 | — | 2 | 324 | — | 15 |
| fastText crawl-300d-2M | 300 | **106** | **+51 [+29, +73]** | **8** | **528** | **+204 [+158, +252]** | **42** |
| bge-large-en-v1.5 | 1024 | 67 | +12 [−5, +29] | 3 | 311 | −13 [−55, +28] | 13 |
| e5-large-v2 | 1024 | 66 | +11 [−6, +29] | 3 | 316 | −8 [−50, +34] | 16 |
| mxbai-embed-large-v1 | 1024 | 68 | +13 [−3, +30] | 3 | 304 | −20 [−60, +21] | 13 |
| Qwen3-Embedding-0.6B | 1024 | 33 | −22 [−41, −4] | 1 | 207 | −117 [−158, −75] | 5 |
| Qwen3-Embedding-4B | 2560 | 48 | −7 [−24, +10] | 2 | 257 | −67 [−106, −27] | 8 |
| Qwen3-Embedding-8B | 4096 | 53 | −2 [−21, +16] | 1 | 271 | −53 [−92, −14] | 6 |

The Qwen3 caches were built from lowercased words with no instruction prefix.
Qwen3-Embedding is designed to take an instruction on queries, so these numbers
may understate it.

Trained models were run on fastText (best cosine baseline) and mxbai-large (best
sentence-transformer on validation), plus MPNet for continuity.

## Trained models by embedding

Five seeds each (0–4), mean ± sample std.

- **Val-selected:** the best epoch on validation, as in `results/baselines.md`.
  This is optimistic, because the same puzzles choose the checkpoint and report
  the score.
- **Fixed-epoch peak:** the best validation mean over seeds at a single epoch,
  with no per-seed selection. This is the fairer comparison with cosine.
- **Greedy:** the selected checkpoint decoded greedily.

### MPNet (cosine baseline: 55 groups, 2 exact)

| Model | Val-selected groups / 392 | Exact / 98 | Best epoch | Fixed-epoch peak (epoch) | Epoch 8 | Greedy | Per seed |
|---|---:|---:|---:|---:|---:|---:|---|
| GATv2, BCE (original) | 45.6 ± 2.9 | 2.0 ± 1.4 | 3.6 | 42.4 (2) | 35.4 ± 4.3 | 42.4 ± 4.6 | 49, 48, 45, 44, 42 |
| No message passing, BCE | 44.4 ± 3.5 | 1.6 ± 1.1 | 4.0 | 42.6 (3) | 26.0 ± 1.2 | 41.0 ± 3.5 | 42, 40, 45, 49, 46 |
| GATv2 + residual, BCE | 58.8 ± 3.0 | 3.8 ± 1.1 | 3.0 | 56.0 (3) | 43.8 ± 5.6 | 49.0 ± 3.2 | 54, 62, 60, 58, 60 |
| GATv2, CRF | 51.4 ± 6.8 | 2.6 ± 1.7 | 3.0 | 48.8 (2) | 36.6 ± 4.0 | 42.8 ± 4.8 | 50, 52, 61, 42, 52 |
| GATv2 + residual + CRF | 59.8 ± 3.6 | 3.8 ± 1.3 | 5.0 | 57.8 (4) | 53.6 ± 10.4 | 49.6 ± 1.3 | 58, 57, 58, 66, 60 |
| No message passing + residual + CRF | 58.6 ± 3.3 | 2.4 ± 1.1 | 3.2 | 57.0 (4) | 47.8 ± 5.1 | 48.0 ± 4.5 | 58, 64, 58, 55, 58 |

### fastText crawl (cosine baseline: 106 groups, 8 exact)

| Model | Val-selected groups / 392 | Exact / 98 | Best epoch | Fixed-epoch peak (epoch) | Epoch 8 | Greedy | Per seed |
|---|---:|---:|---:|---:|---:|---:|---|
| GATv2, BCE (original) | 81.6 ± 9.1 | 3.4 ± 1.8 | 3.2 | 76.8 (2) | 63.0 ± 6.8 | 70.6 ± 8.7 | 79, 95, 80, 70, 84 |
| No message passing, BCE | 81.2 ± 4.5 | 2.6 ± 1.1 | 2.8 | 77.8 (4) | 54.2 ± 6.8 | 71.2 ± 3.9 | 75, 84, 86, 78, 83 |
| **GATv2 + residual + CRF** | **116.8 ± 3.3** | **11.0 ± 0.7** | 3.6 | **113.2 (3)** | 93.6 ± 5.9 | 87.0 ± 10.2 | 112, 119, 115, 120, 118 |
| No message passing + residual + CRF | 112.8 ± 3.8 | 9.4 ± 1.1 | 2.6 | 111.2 (2) | 85.4 ± 8.6 | 87.0 ± 3.4 | 110, 111, 110, 119, 114 |

### mxbai-large (cosine baseline: 68 groups, 3 exact)

| Model | Val-selected groups / 392 | Exact / 98 | Best epoch | Fixed-epoch peak (epoch) | Epoch 8 | Greedy | Per seed |
|---|---:|---:|---:|---:|---:|---:|---|
| GATv2, BCE (original) | 57.4 ± 11.4 | 3.0 ± 1.7 | 2.8 | 52.4 (1) | 41.2 ± 5.9 | 52.6 ± 9.6 | 59, 61, 38, 68, 61 |
| No message passing, BCE | 49.2 ± 7.0 | 1.6 ± 1.5 | 3.6 | 43.4 (4) | 35.6 ± 6.4 | 45.6 ± 9.2 | 40, 59, 49, 46, 52 |
| GATv2 + residual + CRF | 68.0 ± 0.0 | 3.0 ± 0.0 | 1.0 | 68.0 (1) | 64.2 ± 1.6 | 57.0 ± 0.0 | 68, 68, 68, 68, 68 |
| No message passing + residual + CRF | 68.0 ± 0.0 | 3.0 ± 0.0 | 1.0 | 68.0 (1) | 64.0 ± 1.4 | 57.0 ± 0.0 | 68, 68, 68, 68, 68 |

## Learning curves

Mean validation groups (out of 392) over seeds, by epoch. Epoch 0 is the
untrained model; for residual models, epoch 0 equals the cosine baseline.

| Embedding | Model | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| MPNet | GATv2, BCE (original) | 27.2 | 36.2 | 42.4 | 41.0 | 41.6 | 39.4 | 39.2 | 32.6 | 35.4 |
| MPNet | GATv2 + residual + CRF | 55.0 | 55.0 | 55.0 | 56.4 | 57.8 | 57.8 | 54.8 | 54.2 | 53.6 |
| MPNet | No MP + residual + CRF | 55.0 | 55.0 | 55.6 | 56.0 | 57.0 | 54.8 | 49.6 | 48.8 | 47.8 |
| fastText | GATv2, BCE (original) | 38.8 | 64.0 | 76.8 | 73.4 | 75.6 | 74.2 | 65.8 | 66.8 | 63.0 |
| fastText | GATv2 + residual + CRF | 106.0 | 106.0 | 106.4 | 113.2 | 112.6 | 100.8 | 94.2 | 95.6 | 93.6 |
| fastText | No MP + residual + CRF | 106.0 | 106.4 | 111.2 | 107.6 | 105.8 | 98.0 | 92.6 | 91.2 | 85.4 |

As a fit check, the same models were scored on the last 98 *training* puzzles,
which they were trained on, out of 392 groups. fastText GATv2 + residual + CRF
rises from 104.2 at epoch 1 to 202.8 at epoch 8, while validation falls after
epoch 3. MPNet GATv2 with the CRF loss and no residual rises from 45.0 to 269.0.

## Caveats

- **Validation selection.** The val-selected columns use validation to choose the
  checkpoint. The fixed-epoch peak is the fairer number. No method chose epochs
  from held-out training data.
- **No test evaluation.** Every number is on validation.
- **No significance tests for trained models.** Per-puzzle predictions were not
  saved, so there are no paired bootstrap CIs for residual + CRF vs cosine, or
  for message passing vs none. The five-seed spreads are the only uncertainty
  shown.
- **Platform.** These runs used Linux (Python 3.11.17, torch 2.9.1, CPU, one
  thread per run), not the macOS environment in the README. Unmodified
  `train_gnn.py` at seed 0 gave 48 groups here vs 50 in `results/baselines.md`.
  The MPNet cosine baseline matches exactly (55).
- **Separate training loop.** The runs above used a standalone experiment script
  that imports the model, data loader, decoders and `evaluate()` from
  `train_gnn.py` but has its own copy of the training loop, with the same
  settings. Its seed-0 original-model run gave 49 groups vs 48 for `train_gnn.py`.
- **Embedding sources.** The MPNet training runs used embeddings regenerated from
  the pinned revision (not byte-identical to the recorded cache; cosine baseline
  unchanged at 55). fastText and mxbai used the caches in
  `data/embeddings/fasttext-crawl/` and `data/embeddings/mxbai-large/`.

## Next steps

1. Choose the training epoch without validation (a held-out slice of training
   puzzles, or a fixed epoch set in advance), then evaluate the test split once.
2. Save per-puzzle predictions and run paired bootstraps for residual + CRF vs
   cosine, and GATv2 vs no message passing, on fastText.
3. Regularize: every configuration overfits within five epochs (lower learning rate,
   stronger weight decay, smaller hidden size).
4. Rebuild the Qwen3 caches with an instruction prefix before ruling them out.
5. Add the residual scorer, CRF loss and non-768 input dimension to `train_gnn.py`
   as flags. The experiment script is not yet in the repository.

## Training settings

All trained models use the README baseline configuration: hidden size 128,
dropout 0.1, pairwise cosine features in the readout, AdamW with learning rate
1e-3 and weight decay 1e-4, 8 epochs, one puzzle per update, gradient clipping at
1.0, all 1,820 quartets per puzzle, no curriculum, and CPU with one thread.
Validation uses the exact partition decoder; greedy decoding uses the same
checkpoint. Seeds 0–4.
