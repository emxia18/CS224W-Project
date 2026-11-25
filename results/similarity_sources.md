# Extra similarity sources: WordNet senses and ConceptNet Numberbatch

Run date: 2026-11-24. Validation split only (98 puzzles, 392 groups). The test
split was not evaluated.

**Current best model.** GATv2 + residual + CRF on fastText, with two extra
word-similarity sources: best-matching WordNet senses and ConceptNet Numberbatch.
It recovers **150.6 ± 5.0 of 392 validation groups (38.4%)** and solves
**14.8 ± 1.5 of 98 puzzles**, mean ± sample std over seeds 0–4.

## Why

The previous best model starts from fastText cosine similarity and learns a
correction. Failure analysis showed that most misses are the embedding's fault,
not the model's. A single vector per word stands for the word's most common
meaning, so red herrings win (STAFF, REST, NOTE as everyday words, not sheet
music). Co-occurrence vectors also lack world knowledge for thematic groups.

Two sources address this:

- **WordNet senses (`wn`).** Each word gets one vector per WordNet definition,
  encoded with MPNet as `"<word>: <definition>"`, plus `the word "<word>"`.
  Two words are compared by their best-matching pair of senses.
- **ConceptNet Numberbatch (`nb`).** Word vectors built from a knowledge graph.
  Multi-word answers are looked up as phrases (`deck_of_cards`).

## Results without training

The quartet score is a weighted sum of each source's six pairwise similarities,
with weights fit on training puzzles by the CRF loss and exact decoding.
Nothing else is learned. Train numbers are 2-fold cross-validated (fit on half
the training puzzles, score the other half). The CI is a paired bootstrap over
validation puzzles against fastText alone.

| Sources | Weights (fit on train) | Val groups / 392 | Val exact / 98 | Train groups / 1,824 | Δ val vs fastText [95% CI] |
|---|---|---:|---:|---:|---:|
| fastText | 6.97 | 106 | 8 | 528 | — |
| Numberbatch | 5.81 | 110 | 7 | 574 | +4 [−15, +24] |
| fastText + Numberbatch | 3.39, 3.33 | 121 | 10 | 593 | +15 [−4, +35] |
| fastText + WordNet senses | 3.62, 4.95 | 123 | 11 | 703 | +17 [−1, +37] |
| Numberbatch + WordNet senses | 2.96, 4.85 | 125 | 10 | 691 | +19 [−3, +42] |
| **fastText + WordNet senses + Numberbatch** | **2.48, 4.58, 1.30** | **128** | **12** | **720** | **+22 [+2, +44]** |

Recovered validation groups by category type (from `analyze_failures.py` tags):

| Sources | Synonym (126) | Thematic (193) | Phrase completion (41) | Spelling (19) | Sound (13) |
|---|---:|---:|---:|---:|---:|
| fastText | 56 | 41 | 4 | 1 | 4 |
| fastText + WordNet senses + Numberbatch | 65 | 50 | 6 | 3 | 4 |

Also tried and dropped: separate fastText vectors per casing (`mars`, `Mars`)
with best-match similarity. Alone it fell to 80 val groups, and added to fastText
it gave 105 (train 533 vs 528).

## Results with training

The trained model starts from the three-source score above and learns a
correction. Each source is also a GATv2 edge feature and adds pair statistics
(mean, min, max, std) to the quartet readout. Five seeds, same settings as the
previous best model.

| Seed | Val groups (best epoch) | Exact / 98 | Best epoch | Greedy decoding |
|---:|---:|---:|---:|---:|
| 0 | 146 | 13 | 3 | 129 |
| 1 | 150 | 15 | 3 | 136 |
| 2 | 150 | 15 | 3 | 135 |
| 3 | **159** | **17** | 3 | 132 |
| 4 | 148 | 14 | 2 | 140 |
| **Mean ± std** | **150.6 ± 5.0** | **14.8 ± 1.5** | 2.8 | 134.4 ± 4.2 |

Mean validation groups by epoch (epoch 0 is the untrained model and equals the
three-source score, 128, on every seed):

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean groups | 128.0 | 134.2 | 144.6 | **150.4** | 142.8 | 142.8 | 131.6 | 124.0 | 120.0 |

- **Training adds +22.6 groups** over the untrained three-source score
  (paired bootstrap over puzzles, seed-averaged: 95% CI [+5.6, +39.8]).
- **Checkpoint selection is not driving the result.** The best single epoch
  across seeds (epoch 3, 150.4) is almost the same as the per-seed best (150.6).
- **Compared with the previous best** (fastText only, 116.8 ± 3.3 in
  `results/residual_crf_embeddings.md`), the new model is about +34 groups, and
  every seed is above every previous seed.
- **Fitted weights.** Before training: fastText 2.485, WordNet 4.580,
  Numberbatch 1.299, the same as the no-training fit. After 8 epochs: fastText
  2.55–2.81, WordNet 4.80–4.96, Numberbatch 1.43–1.64.
- **It still overfits.** Validation peaks at epoch 2–3 and falls after; one seed
  ends at 101 at epoch 8.
- The best checkpoint is seed 3: `runs/residual_crf_sim/ft_wn_nb_seed3/best.pt`
  (159 groups, 40.6%; 17 puzzles solved).

## How to reproduce

Run from the repository root, after the environment and data steps in the
README (Python 3.11, `pip install -r requirements-reproduce.txt`,
`pull_data` and `prepare_data --seed 42`).

**1. Download the raw vectors** into `data/vectors/` (gitignored, about 6 GB
unzipped). If Hugging Face downloads stall later, set `HF_HUB_DISABLE_XET=1`.

```sh
mkdir -p data/vectors && cd data/vectors
curl -L -C - -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
curl -L -C - -O https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
unzip crawl-300d-2M.vec.zip && gunzip -k numberbatch-en-19.08.txt.gz
cd ../..
```

**2. Build the three caches.**

```sh
# fastText (5,265 whole-word lookups, 159 token means, 11 OOV)
python -m connections.embeddings.embed_static --vectors data/vectors/crawl-300d-2M.vec \
  --name fasttext-crawl --output-dir data/embeddings/fasttext-crawl

# Numberbatch, multi-word answers looked up as phrases (5,348 whole, 49 token means, 38 OOV)
python -m connections.embeddings.embed_static --vectors data/vectors/numberbatch-en-19.08.txt \
  --name numberbatch-en-19.08 --phrase-joiner _ --output-dir data/embeddings/numberbatch

# WordNet senses encoded with MPNet (30,115 vectors, 5.54 per word, 434 words with one).
# Downloads WordNet (about 10 MB) into data/model_cache/nltk_data first.
python -c "import nltk; nltk.download('wordnet', download_dir='data/model_cache/nltk_data')"
python -m connections.embeddings.build_sense_cache --source wordnet --device mps \
  --output-dir data/embeddings/senses/wordnet-mpnet
```

**3. Check the caches.** `verify_reproduction --check-embeddings` compares every
file with the hashes in `reproduction_manifest.json`.

```sh
python -m connections.data.verify_reproduction --check-embeddings
```

**4. Check the no-training score** (about 10 minutes). The last row should read
128 val groups, 12 exact, 720 train groups.

```sh
python -m connections.evaluation.combine_similarities \
  --channel ft=cos:data/embeddings/fasttext-crawl \
  --channel wn=max:data/embeddings/senses/wordnet-mpnet \
  --channel nb=cos:data/embeddings/numberbatch \
  --combo ft --combo nb --combo ft+nb --combo ft+wn --combo nb+wn --combo ft+wn+nb \
  --output runs/embeddings/combine_numberbatch.json
```

**5. Train five seeds** (about 15 minutes in parallel on a 10-core CPU). Each
log should print `Baseline weights fit on 456 training puzzles: a=2.485 wn=4.580
nb=1.299` and `Epoch 000 (untrained) ... (128 groups)`.

```sh
for s in 0 1 2 3 4; do
  python -m connections.model.train_gnn \
    --embedding-dir data/embeddings/fasttext-crawl --pairwise-features --residual --loss crf \
    --similarity wn=max:data/embeddings/senses/wordnet-mpnet \
    --similarity nb=cos:data/embeddings/numberbatch \
    --seed $s --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 \
    --weight-decay 0.0001 --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
    --selection-metric groups --output-dir runs/residual_crf_sim/ft_wn_nb_seed$s \
    > runs/residual_crf_sim/ft_wn_nb_seed$s.log 2>&1 &
done; wait
python -m connections.evaluation.summarize_runs --group "fastText+WordNet+Numberbatch=runs/residual_crf_sim/ft_wn_nb_seed*"
```

## Environment and caveats

- **Hardware.** macOS on Apple silicon (arm64), Python 3.11.3, torch 2.15.0,
  torch-geometric 2.8.1, sentence-transformers 5.7.0, nltk 3.10.4
  (`requirements-reproduce.txt`). MPNet revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- **CPU and GPU.** All training and evaluation ran on CPU, one thread per run.
  The one exception is the WordNet sense cache, which was encoded on the Apple
  GPU (`--device mps`). Encoding on CPU or other hardware can give slightly
  different vectors. If `verify_reproduction --check-embeddings` reports a
  mismatch for `senses/wordnet-mpnet`, copy that folder from a machine where it
  matches instead of rebuilding it.
- **Exact numbers depend on the machine.** Training is deterministic on one
  machine: rerunning a seed gives identical results at every epoch. On other
  platforms (Linux, other torch versions), the first epochs match and later
  epochs drift by a few groups per seed, as seen when reproducing
  `results/residual_crf_embeddings.md`. Compare 5-seed means, not single seeds.
- **Validation-selected.** Per-seed numbers use the best validation epoch. The
  fixed-epoch peak (150.4 at epoch 3) is the fairer figure, and it is nearly
  identical.
- **Not yet run.** The no-message-passing control (`--graph none`) for this
  configuration, so the share of the gain due to message passing is unknown.
- **Weights fit on train only.** The no-training weights and the training
  model's initial weights are fit on the 456 training puzzles. Validation is
  used only for reporting and checkpoint selection.
