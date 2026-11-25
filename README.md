# Solving NYT Connections with Graph Neural Networks

CS 224W project (Emily Xia, Eric Chen).

Each NYT *Connections* puzzle has 16 words that split into four hidden groups of
four. We treat a puzzle as a **graph with one node per word** and ask:
**does message passing across a puzzle improve grouping?**

The pipeline in one line:

```
16 words ─► frozen word embeddings ─► GATv2 over the 16-word graph ─► score all 1,820 quartets ─► pick the best 4 disjoint quartets
```

**Final model:** fastText + WordNet senses + ConceptNet Numberbatch + phrase
completion (Numberbatch and Wikipedia titles) + wordplay features, GATv2 +
residual + CRF. **Test split (evaluated once): 152.4 ± 1.9 of 392 groups
(38.9%), 21.2 of 98 puzzles solved** (5 seeds), vs 112 groups (28.6%) for
untrained fastText cosine (+40.4, 95% CI [+19.0, +61.4]). Validation: 184.6 ±
2.7 (47.1%). Exact steps and values: [`results/final_model.md`](results/final_model.md).

---

## Where things stand

Validation numbers below (98 puzzles, 392 groups) come from `results/`. The test
split was evaluated **once**, for the frozen final model only
(`results/final_model.md`). Do not tune anything on test.

Groups recovered out of 392 (puzzles solved out of 98 in parentheses); trained
columns are the mean over seeds 0–4.

| Word similarity | Cosine baseline (no training) | GATv2 + residual + CRF | Same, no message passing |
|---|---:|---:|---:|
| MPNet    | 55 groups (2 solved)  | 59.8 (3.8)  | 58.6 (2.4) |
| mxbai    | 68 groups (3 solved)  | 68.0 (3.0)  | 68.0 (3.0) |
| fastText | 106 groups (8 solved) | 116.8 (11.0) | 112.8 (9.4) |
| fastText + WordNet senses + Numberbatch | 128 groups (12 solved) | 150.6 (14.8) | not run |
| … + phrase completion (Numberbatch) | 139 groups (15 solved) | 174.0 (19.8) | 167.6 (18.6) |
| **… + Wikipedia phrases + wordplay features (final)** | 153 groups (20 solved) | **184.6 (23.8)** | — |

What we've learned so far:

1. **Untrained cosine similarity is a strong baseline.** The original GATv2 trained
   with BCE does *worse* than it (see `results/baselines.md`).
2. **A residual scorer fixes that.** Starting from the cosine score and learning a
   correction gets a gain over cosine, plus a CRF loss over whole partitions
   (see `results/residual_crf_embeddings.md`).
3. **fastText beats every sentence encoder we tried** (106 vs. 55 groups for MPNet).
4. **Message passing helps modestly, and only clearly on fastText** (+4 groups,
   wins on all 5 seeds).
5. **Exact decoding matters**: 116.8 vs. 87.0 groups with greedy decoding.
6. **Models overfit within about 3–5 epochs.**
7. **Better word similarity is the biggest lever.** Comparing words by their
   best-matching WordNet senses, and adding ConceptNet Numberbatch for world
   knowledge, raises the untrained score from 106 to 128 groups. Training on top
   adds another 22.6 (95% CI [+5.6, +39.8]), about three times what it added on
   fastText alone (see `results/similarity_sources.md`).
8. **Fill-in-the-blank groups need their own signal.** Linking words that share a
   phrase partner within the puzzle ("___ ROLL": DRUM, EGG, HONOR) doubles
   recovered phrase-completion groups and lifts the trained model from 150.6 to
   174.0 groups (95% CI [+8.2, +40.0]). The rule is fixed, not learned.
9. **Message passing helps on the strong model** (+6.4 groups, 95% CI [+0.4,
   +12.8]). Wikipedia-title phrases and the rule-based wordplay features add
   another 10.6 groups; a lower learning rate and 51 synthetic training puzzles
   did not help (see `results/final_model.md`).

The residual scorer, CRF loss and extra similarity sources are flags in
`train_gnn.py` (`--residual`, `--loss crf`, `--similarity`), and any embedding
dimension works.

---

## Repository layout

```
connections/                    all project code (a Python package)
  paths.py                      shared paths: data/, runs/, model cache
  data/
    pull_data.py                download the raw dataset from Hugging Face
    prepare_data.py             validate, split chronologically (456/98/98), shuffle words
    add_synthetic.py            training split with synthetic puzzles appended (leak-checked)
    verify_reproduction.py      check data files against reproduction_manifest.json
  embeddings/
    embed_data.py               cache sentence-transformer vectors (MPNet by default)
    embed_static.py             cache static vectors from a fastText/GloVe/Numberbatch .vec file
    build_sense_cache.py        several vectors per word (WordNet senses) for best-match similarity
    contextual_embeddings.py    per-word MPNet vectors conditioned on the whole puzzle
    build_homophone_cache.py    CMUdict homophones + their MPNet meanings
  features/
    word_channels.py            spelling (bytes) and pronunciation (phoneme) inputs + CNNs
    explicit_features.py        rule-based word relations: anagram, palindrome, homophone...
    similarity_channels.py      extra word-similarity sources for --similarity (cos:, max:, sparse:, phrase:)
    build_phrase_cache.py       phrase partners (drum_roll -> DRUM "followed by roll") for phrase completion
  model/
    decoding.py                 the 1,820 quartets; exact and greedy partition decoders
    gnn.py                      QuartetGNN: GATv2 layer + permutation-invariant quartet scorer
    train_gnn.py                training loop, data loading, evaluation (main entry point)
    curriculum.py               optional "hint" curriculum (revealed partial groups)
  evaluation/
    baselines.py                zero-training baselines: random and cosine-sum
    compare_embeddings.py       rank embedding caches by cosine baseline, per category
    combine_similarities.py     no-training check of similarity sources and weighted combinations
    ensemble_runs.py            score runs per seed and averaged; the one-time test evaluation
    analyze_failures.py         which kinds of categories (synonym, phrase, sound...) we miss
    summarize_runs.py           aggregate multi-seed runs into a Markdown table
    plot_training.py            learning curves from a run's history.json
tests/                          unit tests (python -m unittest)
synthetic_data/                 LLM-generated extra puzzles (see synthetic_data/README.md)
results/                        write-ups of finished experiments (read these!)
docs/EXPERIMENTS.md             full experiment log: every flag, every past run, exact commands
reproduction_manifest.json      SHA-256 hashes of the data splits, raw vectors and embedding caches
requirements.txt                dependency ranges
requirements-reproduce.txt      exact pinned versions used for the recorded results
```

**Not in Git** (see `.gitignore`): `data/` (raw data, splits, downloaded vectors in
`data/vectors/`, embeddings, model cache) and `runs/` (training outputs). You create these locally with the commands below.

---

## Quickstart

Run everything **from the repository root**. Scripts are modules, so use
`python -m connections.<stage>.<script>`. Every script has `--help`.

### 1. Environment (Python 3.11)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-reproduce.txt     # or requirements.txt for looser versions
python -m unittest                            # should print: Ran 39 tests ... OK
```

### 2. Data

```sh
python -m connections.data.pull_data              # -> data/raw/connections.json (652 puzzles)
python -m connections.data.prepare_data --seed 42 # -> data/processed/{train,val,test}.json
python -m connections.data.verify_reproduction    # every line should print OK
```

The splits are **chronological**. Train covers Jun 2023 onward (456 puzzles),
val is Sep 10 – Dec 16 2024 (98), and test is the final 98. Words are shuffled
because the raw file lists them in answer order. Only `words` is a model input.
`answers` is used only as the training label and for scoring.

### 3. Embeddings

```sh
python -m connections.embeddings.embed_data       # MPNet -> data/embeddings/ (downloads model once)
```

Other embedders go into their own directory:

```sh
# Another sentence-transformer
python -m connections.embeddings.embed_data --model mixedbread-ai/mxbai-embed-large-v1 \
  --output-dir data/embeddings/mxbai-large
# fastText: download crawl-300d-2M.vec from fasttext.cc first
python -m connections.embeddings.embed_static --vectors path/to/crawl-300d-2M.vec \
  --name fasttext-crawl --output-dir data/embeddings/fasttext-crawl
```

### 4. Baselines (no training, takes seconds)

```sh
python -m connections.evaluation.baselines                       # MPNet; writes runs/baselines/
python -m connections.evaluation.compare_embeddings \
  data/embeddings data/embeddings/fasttext-crawl                 # compare caches side by side
```

Expected MPNet validation result: cosine + exact decoding recovers **55 / 392 groups,
2 / 98 puzzles**.

### 5. Train the GNN

```sh
# Quick sanity check (a few seconds)
python -m connections.model.train_gnn --epochs 1 --train-limit 5 --val-limit 2 \
  --diagnostic-puzzles 0 --output-dir runs/smoke

# The standard configuration used in our comparisons
python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --seed 0 --device cpu --threads 1 \
  --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --output-dir runs/gat_seed0

# Best configuration (residual + CRF), e.g. on fastText (~9 minutes per seed on CPU)
python -m connections.model.train_gnn \
  --embedding-dir data/embeddings/fasttext-crawl --pairwise-features --residual --loss crf \
  --seed 0 --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 \
  --weight-decay 0.0001 --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --output-dir runs/residual_crf/fasttext_seed0
```

Targets from `results/residual_crf_embeddings.md` (5 seeds, validation):
fastText 116.8 ± 3.3 groups, MPNet 59.8 ± 3.6. With `--residual`, the log line
`Epoch 000 (untrained)` must equal the cosine baseline (106 for fastText, 55 for MPNet).

### 6. Final model

The final model adds Wikipedia-title phrases and the wordplay features to the
model below; its commands, expected values and the test evaluation are in
[`results/final_model.md`](results/final_model.md#how-to-reproduce). The steps
below build the inputs it shares with the previous best.


This needs fastText, ConceptNet Numberbatch, a WordNet sense cache and a phrase
cache built from Numberbatch's phrase list. The full
walkthrough, with expected values at each step, is in
[`results/similarity_sources.md`](results/similarity_sources.md#how-to-reproduce).
In short:

```sh
# 1. Raw vectors into data/vectors/ (about 6 GB unzipped)
mkdir -p data/vectors && cd data/vectors
curl -L -C - -O https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
curl -L -C - -O https://conceptnet.s3.amazonaws.com/downloads/2019/numberbatch/numberbatch-en-19.08.txt.gz
unzip crawl-300d-2M.vec.zip && gunzip -k numberbatch-en-19.08.txt.gz && cd ../..

# 2. Caches
python -m connections.embeddings.embed_static --vectors data/vectors/crawl-300d-2M.vec \
  --name fasttext-crawl --output-dir data/embeddings/fasttext-crawl
python -m connections.embeddings.embed_static --vectors data/vectors/numberbatch-en-19.08.txt \
  --name numberbatch-en-19.08 --phrase-joiner _ --output-dir data/embeddings/numberbatch
python -c "import nltk; nltk.download('wordnet', download_dir='data/model_cache/nltk_data')"
python -m connections.embeddings.build_sense_cache --source wordnet --device mps \
  --output-dir data/embeddings/senses/wordnet-mpnet
python -m connections.features.build_phrase_cache --vectors data/vectors/numberbatch-en-19.08.txt \
  --output-dir data/embeddings/phrases/numberbatch-open
python -m connections.data.verify_reproduction --check-embeddings   # all 20 lines OK

# 3. Train (one seed shown; seeds 0-4 for comparisons)
python -m connections.model.train_gnn \
  --embedding-dir data/embeddings/fasttext-crawl --pairwise-features --residual --loss crf \
  --similarity wn=max:data/embeddings/senses/wordnet-mpnet \
  --similarity nb=cos:data/embeddings/numberbatch \
  --similarity pc=phrase:data/embeddings/phrases/numberbatch-open \
  --seed 0 --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 \
  --weight-decay 0.0001 --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --output-dir runs/residual_crf_sim/ft_wn_nb_pc_seed0
```

Expected: the log prints `a=2.170 wn=4.593 nb=1.349 pc=0.865` and `Epoch 000
(untrained) ... (139 groups)`; seeds 0–4 reach 171, 174, 172, 173, 180 validation
groups (mean 174.0 ± 3.5). Without the `pc` line you get the previous step
(128 untrained; 146, 150, 150, 159, 148 trained).

**Which numbers will match.** Our runs used macOS on Apple silicon with the pinned
environment. All training ran on CPU; only the WordNet sense cache was encoded on
the Apple GPU (`--device mps`). On the same kind of machine, results should match
exactly (training is deterministic: reruns are identical). If the sense-cache hash
check fails on your machine, copy `data/embeddings/senses/wordnet-mpnet/` from
ours instead of rebuilding (the other caches rebuild byte-identically). On Linux
or other torch versions, the untrained score (139) and first epochs match, and later epochs drift by a few groups per seed;
compare 5-seed means. If Hugging Face downloads stall, set `HF_HUB_DISABLE_XET=1`.

### Controls and outputs

Add `--graph none` to get the **no-message-passing control**. It has the same
parameters but the graph keeps only self-loops, so words never see each other.

Each run directory gets `config.json`, `history.json`, `best.pt`, `metrics.json`
(exact decoding), `metrics_greedy.json`, `initial_metrics.json` (the untrained
model) and `predictions.json` (per-puzzle predictions, for paired bootstraps). Then:

```sh
python -m connections.evaluation.plot_training runs/gat_seed0
python -m connections.evaluation.summarize_runs --group "GATv2=runs/gat_seed*"
```

---

## How the model works

1. **Graph.** Each puzzle is a complete graph on 16 nodes (plus self-loops). Node
   features are frozen word embeddings projected to 128 dims. Edge features are
   cosine similarities.
2. **Message passing.** One residual, single-head GATv2 layer (PyG), followed by
   LayerNorm and dropout (`model/gnn.py`).
3. **Quartet scoring.** For every one of the C(16,4) = 1,820 quartets, concatenate
   the mean and variance of its 4 node vectors with stats (mean/min/max/std) of
   its 6 pairwise cosines, then apply an MLP. The scorer is order-invariant, in the
   style of Deep Sets.
4. **Loss.** Balanced binary cross-entropy: the 4 true quartets vs. the other
   1,816 (`--all-quartets`). The better setup adds `--residual` (score = a · cosine
   sum + learned correction, starting at the cosine baseline) and `--loss crf`
   (log-likelihood of the gold partition over all 2.6M partitions; `model/crf.py`).
5. **Decoding.** `model/decoding.py` picks the 4 disjoint quartets with the highest
   total score, exactly, via a dynamic program over subsets of words. Greedy
   decoding is kept for comparison.

**Quartet indexing.** Everything passes around arrays of 1,820 scores indexed
like `decoding.QUARTETS` (all 4-subsets of `range(16)`, lexicographic).
`QUARTET_IDS[(i, j, k, l)]` maps back to an index.

**Extra similarity sources** (`--similarity NAME=KIND:DIR`, repeatable). Each
source gives a 16×16 word-similarity matrix per puzzle: `cos:DIR` is cosine from
a single-vector cache, `max:DIR` is the best cosine over two words' sense vectors,
and `phrase:DIR` links two words that share a phrase partner with 3–5 of the
puzzle's 16 words (`sparse:DIR`, plain cosine of partner vectors, was tried and
not used).
Each one is an extra GATv2 edge feature, adds pair statistics to the quartet
readout and, with `--residual`, adds a baseline term: score = a · cosine sum +
Σ b_k · similarity sum_k + correction. The weights are fit jointly on training
puzzles with the CRF loss before training.

**Optional extras** (all off by default, documented in `docs/EXPERIMENTS.md`):
spelling/pronunciation CNNs (`--spelling-features`, `--pronunciation-features`),
rule-based relations (`--orthographic-features`), homophone semantics
(`--homophone-features`), learned similarity projections (`--learned-relations`),
puzzle-contextual embeddings (`--contextual-dir`), and a hint curriculum
(`--curriculum`). None of them gave a reliable gain in single-seed screening.

---

## Ground rules for experiments

- **The test split has been used once** (2026-11-24, `results/final_model.md`). Further test runs
  would no longer give an unbiased estimate; report any new model on validation.
  The scripts skip test by default.
- **Report validation as "development" numbers.** Checkpoints are selected on the same
  98 validation puzzles we report on.
- **Use 5 seeds (0–4)** for trained comparisons. Single-seed gaps of a few groups are noise.
- **Category descriptions are only for post-hoc error analysis**, never model inputs.
- Give every run its own `--output-dir`.
- Write finished experiments up in `results/`. Include the date, exact commands, and caveats.

## Next steps

1. Thematic groups are the largest remaining gap. WordNet has no senses for 434
   puzzle words (names, brands, slang); LLM-written meanings could fill it.
2. Overfitting: validation still peaks at epoch 3–5. A larger, independent
   selection set would allow honest early stopping.
3. Deliverables: a PyG notebook, a blog post with error analysis, and a demo that
   takes 16 words and returns 4 groups.
