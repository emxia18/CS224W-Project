# Experiment log and full reproduction guide

> This is the detailed, chronological record of every experiment and flag. For a
> quick orientation, start with the [top-level README](../README.md). All
> commands run from the repository root. Code now lives in the `connections/`
> package, so scripts run as `python -m connections.<stage>.<script>`.

## Reproduce the word-feature experiments

Run all commands from the repository root. These are the seed-42 CPU results on
456 training puzzles and 98 validation puzzles, with all 1,820 quartets per puzzle,
no curriculum, and no training-subset diagnostics. Test puzzles are not evaluated.
The screening budget is eight epochs for every variant. The earlier baseline and
CNN runs used a 100-epoch maximum with patience 15, but their best checkpoints were
already within their first eight epochs; the commands below reproduce that common
eight-epoch comparison rather than those longer training histories.

### 1. Install the recorded environment

The recorded environment is Python **3.11.3 on macOS ARM64**. For the closest
reproduction, use the pinned dependencies rather than the broad version ranges
in `requirements.txt`:

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-reproduce.txt
.venv/bin/python -m unittest
```

The tests should report **20 passing tests** (the count has grown since this section was first written). `requirements-reproduce.txt` records
the installed package versions; it has not been tested in a second clean
environment. Other operating systems, Python versions, or accelerator devices
can produce different numerical results. The experiment commands use CPU and one
training thread explicitly.

### 2. Prepare and verify the data

If the original `data/raw/connections.json` is already present, preserve it and
skip the download command. Otherwise:

```sh
.venv/bin/python -m connections.data.pull_data
```

Then prepare the splits and verify their hashes against the recorded snapshot:

```sh
.venv/bin/python -m connections.data.prepare_data --seed 42
.venv/bin/python -m connections.data.verify_reproduction
```

Every file should print `OK`. The upstream download uses `main` and can change;
if verification fails, use the original 652-puzzle snapshot rather than treating
a changed dataset as a reproduction. Expected validation dates are September 10
through December 16, 2024. Expected hashes are in `reproduction_manifest.json`.

### 3. Prepare the frozen embeddings and homophone cache

If the recorded embeddings, word mapping, model snapshot, and homophone cache
are already present, reuse them. To build them from scratch, pin the MPNet commit:

```sh
.venv/bin/python -m connections.embeddings.embed_data --device cpu --batch-size 32 \
  --revision e8c3b32edf5434bc2275fc9bab85f82640a19130
.venv/bin/python -m connections.embeddings.build_homophone_cache --device cpu --batch-size 32
```

`connections.embeddings.embed_data` must run first: the homophone builder uses its locally cached
MPNet snapshot. The recorded cache has **1,759 of 4,918 train/validation input
words with alternative homophones**, and **3,289 distinct candidate words**.
The original semantic embedding export encodes the words from all three splits
without labels; homophone retrieval uses only train/validation words. Neither
operation uses answer descriptions as features.

To check whether cached arrays and mappings are byte-identical to the original:

```sh
.venv/bin/python -m connections.data.verify_reproduction --check-embeddings
```

Regenerated vectors may differ across environments even with the same model
commit. Reusing the original caches gives the closest comparison. The cache
contains only features; answer groups remain training/evaluation supervision.

### 4. Train the five comparison variants

Use the fresh `runs/reproduce/` directories below to preserve previous runs.
All five commands use the same seed, loss, model width, optimizer settings, and
eight-epoch budget. Keep `--selection-metric groups`: it selects the checkpoint
with the most recovered validation groups, breaking ties by exact puzzle accuracy.

```sh
# Baseline: original semantic vectors and cosine statistics.
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --seed 42 --device cpu --threads 1 \
  --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --output-dir runs/reproduce/baseline

# Earlier augmented model: four semantic projections plus spelling/phoneme CNNs.
# The legacy qualifier flag is necessary to reproduce the earlier preprocessing.
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --seed 42 --device cpu --threads 1 \
  --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --learned-relations 4 \
  --spelling-features --pronunciation-features --keep-pronunciation-qualifiers \
  --output-dir runs/reproduce/original_cnn

# Explicit relationships only: spelling predicates and direct phonetic equality.
# This was the best variant in the eight-epoch screening.
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --seed 42 --device cpu --threads 1 \
  --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --orthographic-features \
  --output-dir runs/reproduce/explicit

# Semantic similarity between retrieved homophone candidates only.
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --seed 42 --device cpu --threads 1 \
  --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --homophone-features \
  --output-dir runs/reproduce/homophones

# Both explicit relationships and homophone-candidate semantics.
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --seed 42 --device cpu --threads 1 \
  --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
  --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
  --selection-metric groups --orthographic-features --homophone-features \
  --output-dir runs/reproduce/combined
```

Each directory contains `config.json`, `history.json`, the selected `best.pt`, and
`metrics.json` for that checkpoint. Compare **best checkpoints**, not epoch 8 alone.
The explicit-feature branch includes an exact-pronunciation feature; this is not
a pure spelling-only experiment. The last three variants omit the CNN channels
and learned semantic projections.

### 5. Check the expected outcomes

| Variant | Best epoch | Correct groups / puzzle | Total groups / 392 | Exact puzzles / 98 | Matched words / 16 |
|---|---:|---:|---:|---:|---:|
| Baseline | 2 | 0.500 | 49 | 2 | 10.296 |
| Earlier CNN augmentation | 1 | 0.367 | 36 | 1 | 9.969 |
| Explicit relationships | 5 | **0.541** | **53** | **4** | 10.214 |
| Homophone semantics | 3 | 0.500 | 49 | 2 | 10.173 |
| Both new branches | 7 | 0.490 | 48 | 2 | 10.153 |

The explicit variant recovered four more groups than baseline, but slightly fewer
matched words. This is a small, single-seed validation result, not an established
general improvement. The test split remains unevaluated.

### 6. Recreate category reports and plots

```sh
for variant in original_cnn explicit homophones combined; do
  .venv/bin/python -m connections.evaluation.analyze_failures \
    --run-dir "runs/reproduce/$variant" \
    --baseline-dir runs/reproduce/baseline \
    --output-dir "runs/reproduce/$variant/failure_analysis"
done

.venv/bin/python -m connections.evaluation.plot_training runs/reproduce/baseline
.venv/bin/python -m connections.evaluation.plot_training runs/reproduce/explicit
```

Each `failure_analysis/` directory contains `REPORT.md`, `summary.json`,
`groups.csv`, `groups.json`, and `category_recovery.png`/`.pdf`. The CSV includes
official words, the descriptive category tag, rank among all 1,820 quartets,
closest predicted group, exact recovery, and missing pronunciations. Tags use
answer descriptions only after model inference; they are not model inputs or
official difficulty colors.

For the explicit variant, expected category recoveries are **29/126 synonyms,
22/193 thematic/knowledge, 0/41 phrase completion, 1/19 spelling/transformation,
and 1/13 sound/pronunciation**. It recovers the “things that sound like T” group.
The palindrome group ranks second but is still excluded from the chosen global
partition. Earlier CNN ranks for those groups were 481 and 1,265 respectively.

## Baselines from the proposal

These separate the contributions of pretrained similarity, learned scoring,
message passing, and joint decoding. Zero-parameter baselines (no training):

```sh
.venv/bin/python -m connections.evaluation.baselines
```

This scores every quartet by the sum of its six pairwise MPNet cosines and
decodes with both the exact partition decoder and a greedy decoder (repeatedly
take the best remaining disjoint quartet). It also reports uniform random
partitions (analytic chance: 4/455 groups per puzzle). Results go to
`runs/baselines/metrics.json`; train and validation are evaluated, test only
with `--splits test`.

The no-message-passing control is `--graph none`: the GATv2 layer keeps only
self-loops, so each attention softmax has a single entry and the layer becomes
a nodewise linear map. Parameter count (182,145), readout, loss, and pairwise
cosine statistics are identical to the complete-graph model; a quartet's score
depends only on its own four words. Every training run now also writes
`metrics_greedy.json`: the same best checkpoint decoded greedily.

Five seeds of each, using the baseline configuration from the reproduction above:

```sh
mkdir -p runs/compare
for seed in 0 1 2 3 4; do for graph in complete none; do
  name=$([ $graph = complete ] && echo gat || echo nomp)
  .venv/bin/python -m connections.model.train_gnn --all-quartets --pairwise-features --seed $seed \
    --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 --weight-decay 0.0001 \
    --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
    --selection-metric groups --graph $graph \
    --output-dir runs/compare/${name}_seed$seed > runs/compare/${name}_seed$seed.log 2>&1 &
done; done; wait

.venv/bin/python -m connections.evaluation.summarize_runs \
  --group "GATv2 (complete graph)=runs/compare/gat_seed*" \
  --group "No message passing=runs/compare/nomp_seed*" \
  --output runs/compare/summary.md
```

Validation results (98 puzzles; mean ± sample std over seeds 0–4). Embeddings
were regenerated from the pinned MPNet revision rather than reused, so they are
not byte-identical to the recorded cache; seed 42 of the baseline GAT gives
0.439 groups here versus 0.500 in the table above.

| Model | Decoder | Groups / puzzle | Groups / 392 | Exact / 98 | Matched words / 16 | Best epoch |
|---|---|---:|---:|---:|---:|---:|
| Random partition | — | 0.009 | 0.9 | 0.0 | 7.60 | — |
| Cosine sum (no training) | exact | **0.561** | **55** | 2 | **10.41** | — |
| Cosine sum (no training) | greedy | 0.520 | 51 | 1 | 10.04 | — |
| GATv2, complete graph | exact | 0.516 ± 0.039 | 50.6 ± 3.8 | **2.4 ± 0.5** | 10.07 ± 0.09 | 3.2 ± 0.8 |
| GATv2, complete graph | greedy | 0.508 ± 0.057 | 49.8 ± 5.6 | 2.2 ± 0.8 | 10.00 ± 0.15 | 3.2 ± 0.8 |
| No message passing | exact | 0.424 ± 0.054 | 41.6 ± 5.3 | 1.0 ± 0.7 | 9.99 ± 0.08 | 2.4 ± 0.5 |
| No message passing | greedy | 0.457 ± 0.030 | 44.8 ± 2.9 | 2.0 ± 0.7 | 9.88 ± 0.07 | 2.4 ± 0.5 |

- **Message passing helps relative to its control.** GATv2 beats the
  parameter-matched no-message-passing model on every seed (exact decoding).
- **Neither trained model beats untrained cosine similarity**, even though the
  trained rows are optimistic (their checkpoints were selected on these same
  validation puzzles). Both receive the mean pairwise cosine in their readout, so
  they could represent the cosine ranking; training makes them worse. Validation
  groups peak by epoch 2–4 and then fall while training loss keeps decreasing.
- **Exact decoding helps cosine scores** (+4 groups over greedy) but makes little
  consistent difference for the trained scorers.
- With 98 validation puzzles, one group is about 0.01 groups/puzzle; differences
  of a few groups between single runs are within seed noise.

## Other training options

Download the [NYT Connections dataset](https://huggingface.co/datasets/tm21cy/NYT-Connections):

```sh
python3 -m connections.data.pull_data
```

Requires Python 3 and no additional packages. The script saves the original JSON
to `data/raw/connections.json` without changing the records. Run it again to
refresh the download, or use `--output path/to/file.json` to choose another path.

Prepare the chronological train/validation/test splits:

```sh
python3 -m connections.data.prepare_data
```

This validates all 652 puzzles, sorts by date, and saves 456 training, 98
validation, and 98 test puzzles to `data/processed/{train,val,test}.json`.
Input words are shuffled with seed 42; answer groups remain word lists, so their
labels stay valid. All original fields are retained. Use only `words` as model
input; answers, category descriptions, and other metadata are for supervision or
evaluation. The script prints split date ranges and one training example.
Optional arguments: `--input`, `--output-dir`, and `--seed`.

Generate frozen word embeddings (the first run downloads the MPNet model):

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m connections.embeddings.embed_data
```

Each unique word or complete phrase is encoded once, without answers or metadata.
The script saves normalized float32 vectors in `data/embeddings/embeddings.npy`,
a `word_to_row.json` mapping, and model/input metadata. Model downloads are cached
in `data/model_cache/`. CPU is the default; optionally pass `--device mps` or
`--device cuda` for supported hardware. Re-running regenerates the embedding files.

Retrieve a puzzle's matrix in its shuffled word order:

```python
import json
import numpy as np

vectors = np.load("data/embeddings/embeddings.npy")
word_to_row = json.load(open("data/embeddings/word_to_row.json"))
puzzle = json.load(open("data/processed/train.json"))[0]
X = vectors[[word_to_row[word] for word in puzzle["words"]]]
print(X.shape)  # (16, 768)
```

Train the initial GATv2 model:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m connections.model.train_gnn
```

The model uses a 128-dimensional projection, one residual GATv2 layer with cosine
edge features on a complete graph, and a permutation-invariant quartet scorer
using the mean and variance of transformed node features.
Each update uses one puzzle, its four positive groups, and 32 sampled negatives
(half share three words with an official group), with balanced binary
cross-entropy. Validation uses all 1,820 quartets and exact
partition decoding. The test split is never loaded. Best weights, configuration,
training history, and metrics are saved under `runs/gnn/` (ignored by Git).

For a quick execution check:

```sh
.venv/bin/python -m connections.model.train_gnn --epochs 1 --train-limit 5 --val-limit 2 --output-dir runs/smoke
```

To check whether the model can memorize a small training subset:

```sh
.venv/bin/python -m connections.model.train_gnn --overfit 5 --dropout 0 --weight-decay 0 --epochs 200 --patience 200 --output-dir runs/overfit
```

Overfit mode evaluates on those same training puzzles; its scores are debugging
results, not held-out performance. It trains against all 1,816 incorrect quartets
per puzzle instead of sampling negatives. Other options include `--device`, `--lr`,
`--hidden`, `--seed`, and `--negatives`. Use a different output directory for each run.

To add explicit pairwise relationship features to the quartet readout:

```sh
.venv/bin/python -m connections.model.train_gnn --pairwise-features --epochs 100 --patience 15 --output-dir runs/gnn_pairwise
```

This appends the mean, minimum, maximum, and population standard deviation of
the six distinct pairwise cosine similarities from the frozen embeddings.
The default 256-dimensional readout input becomes 260 dimensions. Omit the flag
to keep the previous readout for comparison, using the same seed and settings.

### Learned relationships, spelling, and pronunciation

Install the updated requirements, then run:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features --learned-relations 4 \
  --spelling-features --pronunciation-features \
  --epochs 100 --patience 15 --output-dir runs/gnn_word_channels
```

Each learned relationship is a separate trainable 768-to-32 projection followed
by normalization and cosine similarity. Its edge feature supplements the original
frozen cosine. Character and phoneme CNNs each produce a hidden-dimensional node
representation, added to the semantic projection before the GAT. Their normalized
cosines also become edge features. The quartet readout receives mean/min/max/std
similarities for every new relationship and channel, even without
`--pairwise-features` (that flag controls only the original semantic statistics).
All projections and CNNs train jointly under the existing quartet loss; MPNet
remains frozen. The hint channel and exact partition decoder work as before.

Spelling uses lowercase UTF-8 byte tokens with trainable embeddings and width-3
and width-5 convolutions, masked max pooling, and first/last token features.
No vocabulary is fitted on validation or test data. Pronunciation uses the
[CMU Pronouncing Dictionary](https://github.com/cmusphinx/cmudict), bundled in the
`cmudict` Python dependency; no pronunciation API calls or runtime model downloads
are needed. Stress digits are removed. Alternate pronunciations are encoded
separately and averaged; phrase components are combined when there is no direct
phrase entry (up to four variants per component and sixteen combinations).
Unknown pronunciations contribute zero node and similarity features rather than
being inferred from spelling. Coverage by word occurrence is printed per split.
These features use only input words and the external dictionary, never categories
or answer groups. Test puzzles are not loaded by training.

The dictionary is English-focused and incomplete for names, puns, and invented
words. Averaging alternatives can blur a particular sense; this is an initial
pronunciation channel, not puzzle-conditioned pronunciation selection.
[g2p_en](https://github.com/Kyubyong/g2p) is an existing neural grapheme-to-phoneme
model that could supply an out-of-dictionary fallback; it is not enabled here.

Pronunciation preprocessing now removes parenthetical disambiguators, so
`TEE (GOLF)` is pronounced as `TEE`. The original string remains available to
the character encoder. New runs record tokenization v2 in their configuration;
`--keep-pronunciation-qualifiers` reproduces the old preprocessing, and the
failure-analysis loader honors the version saved in older checkpoints.

To use explicit relationships and semantic homophone candidates:

```sh
.venv/bin/python -m connections.embeddings.build_homophone_cache
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features \
  --orthographic-features --homophone-features \
  --epochs 8 --patience 8 --diagnostic-puzzles 0 \
  --output-dir runs/explicit_relationships
```

`--orthographic-features` supplies length, palindrome, vowel/repetition fractions,
apostrophe and space indicators as node features. Pair features detect anagrams,
reversals, shared prefixes/suffixes, a single edit, substrings, and identical
pronunciations. Node features are projected into the GAT; both node and pair
features are pooled by mean/min/max/std into the quartet scorer. In particular,
the scorer can directly observe whether all four words are palindromes. Despite
the flag name, this branch includes direct phonetic equality as well as spelling.

`--homophone-features` retrieves alternative spellings with matching CMUdict
pronunciations and compares their MPNet meanings. For example, INDY retrieves
indie, METTLE retrieves metal, SEOUL includes soul, and WRAP includes rap. Each
pair receives the maximum candidate semantic similarity and an availability
indicator. This branch adds no new node encoder. Candidate meanings remain
frozen, while the GAT and quartet scorer learn how to use their similarities.
The maximum may select different alternatives for different pairs; it does not
enforce a single consistent interpretation across the whole group.

The builder uses only train/validation input words, the external dictionary, and
the locally cached MPNet snapshot. It never reads answer groups or category
descriptions. The cache records dictionary/model versions and input hashes;
training rejects stale caches or missing input words. No test words are loaded.
Each flag works independently and can also be combined with the CNN channels,
but the screening command above avoids their additional parameters. Spelling
predicates alone do not recover the meanings of hidden word fragments or solve
all letter-transformation categories.

For ablations, enable `--learned-relations 4`, `--spelling-features`, and
`--pronunciation-features` independently, keeping the seed, loss, curriculum,
epoch budget, and other settings identical. Omitting all three preserves the
original architecture and checkpoint parameter shapes. New runs need their own
output directory. Check the new channels and existing curriculum with:

```sh
.venv/bin/python -m unittest
```

Analyze validation failures for the saved baseline and augmented comparison runs:

```sh
.venv/bin/python -m connections.evaluation.analyze_failures
```

This writes `REPORT.md`, per-group CSV/JSON, summary JSON, and a category recovery
chart to `runs/word_channels_failure_analysis/`. It compares exact group recovery,
candidate ranks, nearest predicted groups, and pronunciation coverage. Category
mechanisms are descriptive annotations from answer titles after inference, never
model inputs; they are not NYT difficulty colors. The synonym annotations target
the current 98-puzzle validation split. Use `--run-dir`, `--baseline-dir`, and
`--output-dir` to select other checkpoints or destinations on that same split.

Train on all 1,820 quartets per puzzle rather than sampling negatives:

```sh
.venv/bin/python -m connections.model.train_gnn --all-quartets --epochs 100 --patience 15 --output-dir runs/gnn_all_quartets
```

The loss averages the four positive losses and 1,816 negative losses separately,
then combines those averages with equal weight. Each update still uses one
puzzle. `--negatives` is unused in this mode. Add `--pairwise-features` to also
enable the explicit cosine statistics; omit it to isolate the loss change.

Use `--selection-metric overlap` to select the best checkpoint and apply early
stopping based on matched word overlap instead of exact group recovery (the
default `--selection-metric groups`). Overlap tries all 24 one-to-one matchings
of predicted and official groups and counts shared words, out of 16 per puzzle.
All runs report both metrics. This changes validation-based model selection;
training still uses the same balanced quartet BCE and decoding still uses logits.
Overlap gives partial credit and has a nonzero chance level; it does not replace
exact puzzle accuracy when reporting solving performance.

### Partial-membership curriculum

```sh
.venv/bin/python -m connections.model.train_gnn \
  --all-quartets --pairwise-features \
  --curriculum progressive \
  --epochs 50 --patience 15 \
  --output-dir runs/gnn_curriculum
```

This adds a second GAT edge feature: 1 means a revealed same-group membership;
0 means unknown. Hidden words have no revealed membership edges. Hints are sampled
fresh from training answers on each visit. Triplet tasks reveal three members of
each group; pair tasks reveal two; sparse tasks reveal one or two pairs total.
The cosine graph remains fully connected. Candidates cannot split a revealed
cluster or combine distinct revealed clusters. Training uses balanced BCE over
all locally compatible quartets; it remains a quartet classification loss, not
a differentiable global assignment loss. Exact decoding enforces global use of
each word once and supports hint constraints for completion tasks.

The progressive task probabilities are:

| Portion of training | Triplets | Pairs | Sparse pairs | Unassisted |
|---|---:|---:|---:|---:|
| First 25% | 65% | 15% | 5% | 15% |
| Next 25% | 20% | 45% | 15% | 20% |
| Next 30% | 5% | 15% | 40% | 40% |
| Final 20% | 0% | 0% | 0% | 100% |

Validation and checkpoint selection always use zero hints. Progressive mode
starts counting early-stopping patience only in the final unassisted stage;
best-checkpoint selection still considers every epoch. History records task
counts, probabilities, and losses; losses across changing tasks are not directly
comparable. No test puzzles are loaded.

For the fixed-mixture comparison, replace `--curriculum progressive` with
`--curriculum fixed` and choose another output directory. It uses 25% of each
task throughout. For the architecture-matched unassisted comparison, use
`--curriculum none --hint-features`; the hint channel stays zero throughout.
Use the same epochs, seeds, hyperparameters, and a patience at least as large as
the epoch budget for fixed-budget comparisons, since early stopping can give
different numbers of updates. Hinted tasks score fewer candidate quartets, so
equal update counts do not imply equal wall-clock time. This initial curriculum
reduces hints; it does not yet mine increasingly ambiguous completion choices.

Check hint isolation and global completion constraints with:

```sh
.venv/bin/python -m unittest
```

Every epoch also evaluates a fixed random subset of 16 training puzzles under
three conditions: no hints (`full`), triplet hints, and pair hints. Hints stay
identical across epochs, evaluation disables dropout, and hinted conditions use
exact decoding with membership constraints. These are training diagnostics, not
held-out scores; they do not affect checkpoint selection or optimizer updates.
When the hint edge channel is disabled, hinted diagnostics still enforce decoder
constraints but do not supply hints to GAT attention. Hinted scores benefit from
the supplied memberships, so compare trends within each condition rather than
treating them as directly comparable to unassisted scores.

Use `--diagnostic-puzzles 32` to increase the subset or `--diagnostic-puzzles 0`
to disable these checks. The chosen dates and seeds are saved in
`diagnostic_subset.json`; each epoch's metrics are in `history.json` under
`training_diagnostics`. `training_diagnostics.json` contains diagnostics for the
restored best checkpoint. Rising training `full` recovery with falling validation
suggests overfitting; rising hinted recovery with flat training `full` recovery
suggests poor transfer from completion to discovery.

Plot the saved learning curves (PNG and PDF) at any point during training:

```sh
.venv/bin/python -m connections.evaluation.plot_training runs/gnn_curriculum_diagnostics
```

This reads `history.json` and `config.json` and writes `learning_curves.png` and
`learning_curves.pdf` in the run directory. Panels show correct groups, exact
puzzle accuracy, matched-word overlap, and per-task training loss. Curriculum
stages are shaded; only completed epochs are plotted. Re-run to update the plots.

### Puzzle-contextual node embeddings

Encode all sixteen input words jointly with the frozen, locally cached MPNet
transformer, then mean-pool each word's own final-layer tokens and normalize.
This differs from sentence pooling: every node keeps its own representation,
conditioned on the other puzzle words. Words are alphabetically ordered before
encoding for reproducibility; cache lookup restores the puzzle's node order.
Emoji presentation selectors are removed from encoder text. Answers and category
labels are never encoded. Only train and validation splits are cached by default.

```sh
.venv/bin/python -m connections.embeddings.contextual_embeddings
.venv/bin/python -m connections.model.train_gnn \
  --contextual-dir data/embeddings/contextual \
  --all-quartets --pairwise-features \
  --epochs 8 --patience 8 --diagnostic-puzzles 16 \
  --output-dir runs/contextual_seed42
```

The GAT and quartet decoder remain in use. Both semantic node features and cosine
edge/readout features now use contextual vectors. Optional spelling and homophone
branches remain available. Input hashes reject stale caches, and training saves
encoder metadata in its configuration. This is an experimental use of MPNet token
states: its sentence-embedding training does not guarantee that these word-level
vectors improve solving. Compare with the existing static-embedding baseline
under matching settings. Check cache alignment and pooling with:

```sh
.venv/bin/python -m unittest
```
