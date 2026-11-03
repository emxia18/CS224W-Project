"""Download the raw NYT Connections dataset from Hugging Face.

Run: python3 pull_data.py
"""

import argparse
import json
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


DATASET_URL = (
    "https://huggingface.co/datasets/tm21cy/NYT-Connections/"
    "resolve/main/ConnectionsFinalDataset.json"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data/raw/connections.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help="Where to save the downloaded JSON file.",
    )
    args = parser.parse_args()

    try:
        with urlopen(DATASET_URL, timeout=60) as response:
            raw_data = response.read()
        records = json.loads(raw_data)
    except (URLError, TimeoutError, json.JSONDecodeError) as exc:
        parser.exit(1, f"Could not download dataset: {exc}\n")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw_data)
    print(f"Downloaded {len(records)} records to {args.output}")
    if records:
        print("\nExample record:")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
