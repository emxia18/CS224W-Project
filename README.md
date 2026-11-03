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
