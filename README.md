# Solving NYT Connections with Graph Neural Networks

CS 224W project (Emily Xia, Eric Chen).

Each NYT *Connections* puzzle has 16 words that split into four hidden groups of
four. We treat a puzzle as a **graph with one node per word** and ask:
**does message passing across a puzzle improve grouping?**

The pipeline in one line:

```
16 words ─► frozen word embeddings ─► GATv2 over the 16-word graph ─► score all 1,820 quartets ─► pick the best 4 disjoint quartets
```

---

## Where things stand

Validation numbers below (98 puzzles, 392 groups) come from `results/`. The test
split has **not** been evaluated yet. Do not run on test until the final
configuration is frozen.

| Embedding | Cosine baseline (no training) | GATv2 + residual + CRF | Same, no message passing |
|---|---:|---:|---:|
| MPNet    | 55 groups (2 solved)  | 59.8 (3.8)  | 58.6 (2.4) |
| fastText | 106 groups (8 solved) | **116.8 (11.0)** | 112.8 (9.4) |
| mxbai    | 68 groups (3 solved)  | 68.0 (3.0)  | 68.0 (3.0) |

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

> ⚠️ **Known gap:** the residual scorer and CRF loss were run from a standalone
> script that is **not yet in this repo**. `connections/model/train_gnn.py`
> currently implements the original BCE model and only accepts 768-dim
> embeddings (MPNet). Merging those into `train_gnn.py` is the next to-do.

---

## Repository layout

```
connections/                    all project code (a Python package)
  paths.py                      shared paths: data/, runs/, model cache
  data/
    pull_data.py                download the raw dataset from Hugging Face
    prepare_data.py             validate, split chronologically (456/98/98), shuffle words
    verify_reproduction.py      check data files against reproduction_manifest.json
  embeddings/
    embed_data.py               cache sentence-transformer vectors (MPNet by default)
    embed_static.py             cache static vectors from a fastText/GloVe .vec file
    contextual_embeddings.py    per-word MPNet vectors conditioned on the whole puzzle
    build_homophone_cache.py    CMUdict homophones + their MPNet meanings
  features/
    word_channels.py            spelling (bytes) and pronunciation (phoneme) inputs + CNNs
    explicit_features.py        rule-based word relations: anagram, palindrome, homophone...
  model/
    decoding.py                 the 1,820 quartets; exact and greedy partition decoders
    gnn.py                      QuartetGNN: GATv2 layer + permutation-invariant quartet scorer
    train_gnn.py                training loop, data loading, evaluation (main entry point)
    curriculum.py               optional "hint" curriculum (revealed partial groups)
  evaluation/
    baselines.py                zero-training baselines: random and cosine-sum
    compare_embeddings.py       rank embedding caches by cosine baseline, per category
    analyze_failures.py         which kinds of categories (synonym, phrase, sound...) we miss
    summarize_runs.py           aggregate multi-seed runs into a Markdown table
    plot_training.py            learning curves from a run's history.json
tests/                          unit tests (python -m unittest)
synthetic_data/                 LLM-generated extra puzzles (see synthetic_data/README.md)
results/                        write-ups of finished experiments (read these!)
docs/EXPERIMENTS.md             full experiment log: every flag, every past run, exact commands
reproduction_manifest.json      SHA-256 hashes of the data splits and embedding caches
requirements.txt                dependency ranges
requirements-reproduce.txt      exact pinned versions used for the recorded results
```

**Not in Git** (see `.gitignore`): `data/` (raw data, splits, embeddings, model
cache) and `runs/` (training outputs). You create these locally with the commands below.

---

## Quickstart

Run everything **from the repository root**. Scripts are modules, so use
`python -m connections.<stage>.<script>`. Every script has `--help`.

### 1. Environment (Python 3.11)

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-reproduce.txt     # or requirements.txt for looser versions
python -m unittest                            # should print: Ran 20 tests ... OK
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
```

Add `--graph none` to get the **no-message-passing control**. It has the same
parameters but the graph keeps only self-loops, so words never see each other.

Each run directory gets `config.json`, `history.json`, `best.pt`, `metrics.json`
(exact decoding) and `metrics_greedy.json`. Then:

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
   1,816 (`--all-quartets`). The not-yet-merged follow-up uses a residual on the
   cosine score and a CRF loss over all 2.6M valid partitions.
5. **Decoding.** `model/decoding.py` picks the 4 disjoint quartets with the highest
   total score, exactly, via a dynamic program over subsets of words. Greedy
   decoding is kept for comparison.

**Quartet indexing.** Everything passes around arrays of 1,820 scores indexed
like `decoding.QUARTETS` (all 4-subsets of `range(16)`, lexicographic).
`QUARTET_IDS[(i, j, k, l)]` maps back to an index.

**Optional extras** (all off by default, documented in `docs/EXPERIMENTS.md`):
spelling/pronunciation CNNs (`--spelling-features`, `--pronunciation-features`),
rule-based relations (`--orthographic-features`), homophone semantics
(`--homophone-features`), learned similarity projections (`--learned-relations`),
puzzle-contextual embeddings (`--contextual-dir`), and a hint curriculum
(`--curriculum`). None of them gave a reliable gain in single-seed screening.

---

## Ground rules for experiments

- **Never load the test split** until the final config and checkpoint rule are frozen.
  The scripts skip test by default.
- **Report validation as "development" numbers.** Checkpoints are selected on the same
  98 validation puzzles we report on.
- **Use 5 seeds (0–4)** for trained comparisons. Single-seed gaps of a few groups are noise.
- **Category descriptions are only for post-hoc error analysis**, never model inputs.
- Give every run its own `--output-dir`.
- Write finished experiments up in `results/`. Include the date, exact commands, and caveats.

## Next steps

1. Merge the residual scorer and CRF loss into `train_gnn.py`, with support for
   non-768-dim inputs like fastText's 300.
2. Run matched ablations on fastText: residual only, CRF only, and both.
3. Add stronger regularization against early overfitting.
4. Freeze the config and evaluate the test split once, with paired bootstrap CIs.
5. Deliverables: a PyG notebook, a blog post with error analysis, and a demo that
   takes 16 words and returns 4 groups.
