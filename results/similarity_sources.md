# Extra similarity sources: WordNet senses, Numberbatch and phrase completion

Run date: 2026-11-24. All numbers are on the **validation split** (98 puzzles,
392 groups). Models train on the 456 training puzzles only. The test split was
not evaluated.

**Current best model.** GATv2 + residual + CRF on fastText, with three extra
word-similarity sources: best-matching WordNet senses, ConceptNet Numberbatch,
and puzzle-specific phrase completion. It recovers **174.0 ± 3.5 of 392
validation groups (44.4%)** and solves **19.8 ± 0.8 of 98 puzzles (20.2%)**,
mean ± sample std over seeds 0–4.

| Model (validation, seeds 0–4) | Groups / 392 | Accuracy | Puzzles solved / 98 |
|---|---:|---:|---:|
| fastText cosine, no training | 106 | 27.0% | 8 |
| fastText, trained (`results/residual_crf_embeddings.md`) | 116.8 ± 3.3 | 29.8% | 11.0 |
| + WordNet senses + Numberbatch, trained | 150.6 ± 5.0 | 38.4% | 14.8 |
| **+ phrase completion, trained (current best)** | **174.0 ± 3.5** | **44.4%** | **19.8** |

## Why

Most misses of the fastText model are the embedding's fault, not the model's:

- **One vector per word** stands for the most common meaning, so red herrings
  win (STAFF, REST, NOTE as everyday words, not sheet music).
- **Co-occurrence vectors lack world knowledge** for thematic groups.
- **Fill-in-the-blank groups share no meaning at all** ("___ ROLL": DRUM, EGG,
  HONOR, CALIFORNIA), so no word-similarity source can link them.

The three sources each target one of these:

- **WordNet senses (`wn`).** Each word gets one vector per WordNet definition,
  encoded with MPNet as `"<word>: <definition>"`, plus `the word "<word>"`. Two
  words are compared by their best-matching pair of senses.
- **ConceptNet Numberbatch (`nb`).** Word vectors built from a knowledge graph.
  Multi-word answers are looked up as phrases (`deck_of_cards`).
- **Phrase completion (`pc`).** Every two-part phrase in Numberbatch's
  vocabulary (154,636 of its keys, e.g. `drum_roll`, `double_agent`) gives its
  parts a partner: DRUM gets "followed by roll", AGENT gets "preceded by
  double". Within a puzzle, two words score 1 if they share a partner that 3–5 of
  the 16 words share (the group plus at most one red herring), else 0. Partners
  shared by more words are too generic. In 19 of 41 validation phrase-completion
  groups all four words share such a partner, against 10 of the 351 other groups.

## Results without training

The quartet score is a weighted sum of each source's six pairwise similarities,
with weights fit on training puzzles by the CRF loss, then exact decoding.
Nothing else is learned. Train numbers are 2-fold cross-validated (fit on half
the training puzzles, score the other half); with 4.6× more puzzles than
validation, they were used to choose between variants. The CI is a paired
bootstrap over validation puzzles.

| Sources | Weights (fit on train) | Val groups / 392 | Val exact / 98 | Train groups / 1,824 | Δ val [95% CI] |
|---|---|---:|---:|---:|---:|
| fastText | 6.97 | 106 | 8 | 528 | — |
| Numberbatch | 5.81 | 110 | 7 | 574 | +4 [−15, +24] vs fastText |
| fastText + Numberbatch | 3.39, 3.33 | 121 | 10 | 593 | +15 [−4, +35] vs fastText |
| fastText + WordNet senses | 3.62, 4.95 | 123 | 11 | 703 | +17 [−1, +37] vs fastText |
| Numberbatch + WordNet senses | 2.96, 4.85 | 125 | 10 | 691 | +19 [−3, +42] vs fastText |
| fastText + WordNet senses + Numberbatch | 2.48, 4.58, 1.30 | 128 | 12 | 720 | +22 [+2, +44] vs fastText |
| **+ phrase completion** | **2.17, 4.59, 1.35, 0.86** | **139** | **15** | **783** | +11 [−5, +28] vs row above |

Recovered validation groups by category type (from `analyze_failures.py` tags):

| Sources | Synonym (126) | Thematic (193) | Phrase completion (41) | Spelling (19) | Sound (13) |
|---|---:|---:|---:|---:|---:|
| fastText | 56 | 41 | 4 | 1 | 4 |
| fastText + WordNet senses + Numberbatch | 65 | 50 | 6 | 3 | 4 |
| + phrase completion | 67 | 53 | 12 | 3 | 4 |

### Tried and dropped

| Variant | Val groups | Train groups | Why dropped |
|---|---:|---:|---|
| fastText vector per casing (`mars`, `Mars`), best match | 80 alone; 105 with fastText | 409; 533 | Worse than fastText alone |
| Phrase completion with closed compounds (`fire\|ball`) | 132 | 732 | Too many false links: 52 of 351 non-phrase groups share a partner |
| Phrase partners as plain cosine (no puzzle-specific 3–5 rule) | 144 | 737 | +17 on train vs +63 for the puzzle-specific version |
| Both phrase versions together | 140 | 786 | +3 on train over the puzzle-specific version alone |
| WordNet senses encoded with mxbai-embed-large instead of MPNet | 148 | 748 | −35 on train; alone it is also worse (93 / 537 vs 106 / 618) |

## Results with training

The trained model starts from the no-training score above and learns a
correction. Each source is also a GATv2 edge feature and adds pair statistics
(mean, min, max, std) to the quartet readout. Five seeds, same settings as
before. "Best epoch" is chosen on validation, as in the earlier write-ups.

**Current best: fastText + WordNet senses + Numberbatch + phrase completion.**

| Seed | Val groups (best epoch) | Exact / 98 | Best epoch | Greedy decoding |
|---:|---:|---:|---:|---:|
| 0 | 171 | 19 | 3 | 140 |
| 1 | 174 | 20 | 3 | 137 |
| 2 | 172 | 19 | 2 | 134 |
| 3 | 173 | 20 | 2 | 135 |
| 4 | **180** | **21** | 3 | 139 |
| **Mean ± std** | **174.0 ± 3.5** | **19.8 ± 0.8** | 2.6 | 137.0 ± 2.5 |

**Previous step: fastText + WordNet senses + Numberbatch.**

| Seed | 0 | 1 | 2 | 3 | 4 | Mean ± std |
|---|---:|---:|---:|---:|---:|---:|
| Val groups | 146 | 150 | 150 | 159 | 148 | 150.6 ± 5.0 |
| Exact / 98 | 13 | 15 | 15 | 17 | 14 | 14.8 ± 1.5 |

Mean validation groups by epoch. Epoch 0 is the untrained model and equals the
no-training score on every seed.

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| + phrase completion | 139.0 | 151.4 | **172.4** | 171.2 | 164.6 | 149.2 | 137.4 | 138.2 | 129.4 |
| fastText + WordNet + Numberbatch | 128.0 | 134.2 | 144.6 | **150.4** | 142.8 | 142.8 | 131.6 | 124.0 | 120.0 |

- **Phrase completion adds +23.4 validation groups** over the previous step
  (paired bootstrap over puzzles, seed-averaged: 95% CI [+8.2, +40.0]). Every
  seed (171–180) beats every previous seed (146–159).
- **Not an artifact of checkpoint selection.** The fairer comparison, the best
  single epoch with no per-seed choice, gives 172.4 vs 150.4.
- **Training adds more as the inputs get richer:** +35.0 over the untrained
  score now (139 → 174), +22.6 for the previous step (128 → 150.6; 95% CI
  [+5.6, +39.8]) and about +7 on fastText alone.
- **Fitted weights.** Before training: fastText 2.170, WordNet 4.593, Numberbatch
  1.349, phrase 0.865, the same as the no-training fit. After 8 epochs: fastText
  2.35–2.50, WordNet 4.89–4.97, Numberbatch 1.58–1.68, phrase 0.85–0.91.
- **It still overfits.** Validation peaks at epoch 2–3 and falls to 129.4 ± 11.7
  by epoch 8.
- **Best checkpoint:** seed 4, `runs/residual_crf_sim/ft_wn_nb_pc_seed4/best.pt`
  (180 groups, 45.9%; 21 puzzles solved).

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

**2. Build the four caches.**

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

# Phrase partners from Numberbatch's phrase keys (seconds; 4,477 words with
# partners, 17,410 partners shared by at least two puzzle words)
python -m connections.features.build_phrase_cache --vectors data/vectors/numberbatch-en-19.08.txt \
  --output-dir data/embeddings/phrases/numberbatch-open
```

**3. Check the caches.** Every line should print OK (20 files).

```sh
python -m connections.data.verify_reproduction --check-embeddings
```

**4. Check the no-training score** (about 10 minutes). The last two rows should
read 128 / 12 / 720 and 139 / 15 / 783 (val groups, val exact, train groups).

```sh
python -m connections.evaluation.combine_similarities \
  --channel ft=cos:data/embeddings/fasttext-crawl \
  --channel wn=max:data/embeddings/senses/wordnet-mpnet \
  --channel nb=cos:data/embeddings/numberbatch \
  --channel pc=phrase:data/embeddings/phrases/numberbatch-open \
  --combo ft --combo ft+wn+nb --combo ft+wn+nb+pc \
  --output runs/embeddings/combine_phrases.json
```

**5. Train five seeds** (about 15 minutes in parallel on a 10-core CPU). Each
log should print `Baseline weights fit on 456 training puzzles: a=2.170 wn=4.593
nb=1.349 pc=0.865` and `Epoch 000 (untrained) ... (139 groups)`.

```sh
mkdir -p runs/residual_crf_sim
for s in 0 1 2 3 4; do
  python -m connections.model.train_gnn \
    --embedding-dir data/embeddings/fasttext-crawl --pairwise-features --residual --loss crf \
    --similarity wn=max:data/embeddings/senses/wordnet-mpnet \
    --similarity nb=cos:data/embeddings/numberbatch \
    --similarity pc=phrase:data/embeddings/phrases/numberbatch-open \
    --seed $s --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 \
    --weight-decay 0.0001 --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
    --selection-metric groups --output-dir runs/residual_crf_sim/ft_wn_nb_pc_seed$s \
    > runs/residual_crf_sim/ft_wn_nb_pc_seed$s.log 2>&1 &
done; wait
python -m connections.evaluation.summarize_runs \
  --group "fastText+WordNet+Numberbatch+phrase=runs/residual_crf_sim/ft_wn_nb_pc_seed*"
```

Expected: 171, 174, 172, 173, 180 validation groups (174.0 ± 3.5). The previous
step is the same command without the `pc` line (146, 150, 150, 159, 148).

## Environment and caveats

- **Hardware.** macOS on Apple silicon (arm64), Python 3.11.3, torch 2.15.0,
  torch-geometric 2.8.1, sentence-transformers 5.7.0, scipy 1.17.1, nltk 3.10.4
  (`requirements-reproduce.txt`). MPNet revision
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- **CPU and GPU.** All training and evaluation ran on CPU, one thread per run.
  The one exception is the WordNet sense cache, which was encoded on the Apple
  GPU (`--device mps`). Encoding on CPU or other hardware can give slightly
  different vectors. If `verify_reproduction --check-embeddings` reports a
  mismatch for `senses/wordnet-mpnet`, copy that folder from a machine where it
  matches instead of rebuilding it. The other caches rebuild byte-identically.
- **Exact numbers depend on the machine.** Training is deterministic on one
  machine: rerunning a seed gives identical results at every epoch. On other
  platforms (Linux, other torch versions), the first epochs match and later
  epochs drift by a few groups per seed. Compare 5-seed means, not single seeds.
- **Validation is used twice.** It reports the score and picks each seed's best
  epoch, which is slightly optimistic. The fixed-epoch peaks (172.4 and 150.4)
  avoid this and show the same gain. Variants were chosen on train
  cross-validation, not on validation.
- **Phrase completion is a fixed rule, not learned.** It uses only Numberbatch's
  phrase list and the 16 puzzle words; only its weight is fit (on train). The
  3–5 range was set by reasoning (a group of four plus at most one red herring),
  not tuned. Before building it, the coverage check (19 of 41 phrase groups)
  was run on validation answers to see whether the idea was worth pursuing;
  no parameter was set from it. The untouched test split is the clean check.
- **Not yet run.** The no-message-passing control (`--graph none`) for these
  configurations, so the share of the gain due to message passing is unknown.
- **Weights fit on train only.** The no-training weights and the trained
  model's initial weights are fit on the 456 training puzzles.
