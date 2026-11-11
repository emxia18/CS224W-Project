"""Validate, chronologically split, and shuffle the Connections dataset."""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import random


from connections.paths import ROOT
SPLIT_SIZES = {"train": 456, "val": 98, "test": 98}


def validate_records(records):
    """Reject malformed puzzles and duplicate dates before writing any splits."""
    if not isinstance(records, list) or len(records) != sum(SPLIT_SIZES.values()):
        raise ValueError("Expected a JSON list of exactly 652 puzzles.")

    seen_dates = set()
    for index, record in enumerate(records):
        label = f"Puzzle at index {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{label}: expected an object.")
        try:
            date = datetime.strptime(record["date"], "%Y/%m/%d").date()
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{label}: missing or invalid YYYY/MM/DD date.") from exc
        if date in seen_dates:
            raise ValueError(f"{label}: duplicate puzzle date {date}.")
        seen_dates.add(date)

        words = record.get("words")
        if (not isinstance(words, list) or len(words) != 16
                or not all(isinstance(word, str) and word.strip() for word in words)
                or len(set(words)) != 16):
            raise ValueError(f"{label}: expected 16 distinct, nonempty words.")

        answers = record.get("answers")
        if not isinstance(answers, list) or len(answers) != 4:
            raise ValueError(f"{label}: expected four answer groups.")
        grouped_words = []
        for answer in answers:
            group = answer.get("words") if isinstance(answer, dict) else None
            if (not isinstance(group, list) or len(group) != 4
                    or not all(isinstance(word, str) for word in group)):
                raise ValueError(f"{label}: each answer must contain four words.")
            grouped_words.extend(group)
        if Counter(grouped_words) != Counter(words):
            raise ValueError(f"{label}: answers must cover every input word exactly once.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/raw/connections.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        records = json.loads(args.input.read_text(encoding="utf-8"))
        validate_records(records)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Could not prepare dataset: {exc}\n")

    records.sort(key=lambda record: datetime.strptime(record["date"], "%Y/%m/%d"))
    rng = random.Random(args.seed)
    for record in records:
        rng.shuffle(record["words"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = 0
    for name, size in SPLIT_SIZES.items():
        split = records[start:start + size]
        output = args.output_dir / f"{name}.json"
        output.write_text(
            json.dumps(split, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"{name}: {size} puzzles, {split[0]['date']} to {split[-1]['date']} -> {output}")
        start += size

    print(f"\nShuffled training example (seed={args.seed}):")
    print(json.dumps(records[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
