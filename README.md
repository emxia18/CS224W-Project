# CS224W-Project

Download the [NYT Connections dataset](https://huggingface.co/datasets/tm21cy/NYT-Connections):

```sh
python3 pull_data.py
```

Requires Python 3 and no additional packages. The script saves the original JSON
to `data/raw/connections.json` without changing the records. Run it again to
refresh the download, or use `--output path/to/file.json` to choose another path.
