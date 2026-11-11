# CS224W-Project

Download the [NYT Connections dataset](https://huggingface.co/datasets/tm21cy/NYT-Connections):

```sh
python3 pull_data.py
```

Requires Python 3 and no additional packages. The script saves the original JSON
to `data/raw/connections.json` without changing the records. Run it again to
refresh the download, or use `--output path/to/file.json` to choose another path.

Prepare the chronological train/validation/test splits:

```sh
python3 prepare_data.py
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
.venv/bin/python embed_data.py
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
.venv/bin/python train_gnn.py
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
.venv/bin/python train_gnn.py --epochs 1 --train-limit 5 --val-limit 2 --output-dir runs/smoke
```

To check whether the model can memorize a small training subset:

```sh
.venv/bin/python train_gnn.py --overfit 5 --dropout 0 --weight-decay 0 --epochs 200 --patience 200 --output-dir runs/overfit
```

Overfit mode evaluates on those same training puzzles; its scores are debugging
results, not held-out performance. It trains against all 1,816 incorrect quartets
per puzzle instead of sampling negatives. Other options include `--device`, `--lr`,
`--hidden`, `--seed`, and `--negatives`. Use a different output directory for each run.

To add explicit pairwise relationship features to the quartet readout:

```sh
.venv/bin/python train_gnn.py --pairwise-features --epochs 100 --patience 15 --output-dir runs/gnn_pairwise
```

This appends the mean, minimum, maximum, and population standard deviation of
the six distinct pairwise cosine similarities from the frozen embeddings.
The default 256-dimensional readout input becomes 260 dimensions. Omit the flag
to keep the previous readout for comparison, using the same seed and settings.

### Learned relationships, spelling, and pronunciation

Install the updated requirements, then run:

```sh
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_gnn.py \
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

For ablations, enable `--learned-relations 4`, `--spelling-features`, and
`--pronunciation-features` independently, keeping the seed, loss, curriculum,
epoch budget, and other settings identical. Omitting all three preserves the
original architecture and checkpoint parameter shapes. New runs need their own
output directory. Check the new channels and existing curriculum with:

```sh
.venv/bin/python -m unittest test_curriculum.py test_word_channels.py
```

Train on all 1,820 quartets per puzzle rather than sampling negatives:

```sh
.venv/bin/python train_gnn.py --all-quartets --epochs 100 --patience 15 --output-dir runs/gnn_all_quartets
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
.venv/bin/python train_gnn.py \
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
.venv/bin/python -m unittest test_curriculum.py
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
.venv/bin/python plot_training.py runs/gnn_curriculum_diagnostics
```

This reads `history.json` and `config.json` and writes `learning_curves.png` and
`learning_curves.pdf` in the run directory. Panels show correct groups, exact
puzzle accuracy, matched-word overlap, and per-task training loss. Curriculum
stages are shaded; only completed epochs are plotted. Re-run to update the plots.
