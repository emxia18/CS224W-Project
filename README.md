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

Train on all 1,820 quartets per puzzle rather than sampling negatives:

```sh
.venv/bin/python train_gnn.py --all-quartets --epochs 100 --patience 15 --output-dir runs/gnn_all_quartets
```

The loss averages the four positive losses and 1,816 negative losses separately,
then combines those averages with equal weight. Each update still uses one
puzzle. `--negatives` is unused in this mode. Add `--pairwise-features` to also
enable the explicit cosine statistics; omit it to isolate the loss change.
