"""Validate the single training dataset, including global lexical diversity."""

import json
from pathlib import Path

from build import audit


if __name__ == '__main__':
    path = Path(__file__).resolve().parent / 'puzzles.json'
    print(json.dumps(audit(json.loads(path.read_text())), indent=2))
