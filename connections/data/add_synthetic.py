"""Write a training split with synthetic puzzles appended after the real ones.

Reads the processed splits and the LLM-generated synthetic puzzles, and writes
<output-dir>/{train,val,test}.json: train is the real training puzzles followed by
the synthetic ones; val and test are byte-identical copies. Because the real
puzzles come first, `train_gnn.py --train-limit N` (N = number of real training
puzzles) trains on the same caches without the synthetic puzzles, as a control.

Synthetic puzzles used by default: one board per paper_v2 family
(synthetic_data/paper_v2/puzzles.json) plus the first batch
(synthetic_data/puzzles.json), skipping any board whose four groups all appear
in an earlier board. The 688 difficulty variants (paper_v2/variants.json) are
not used: siblings share categories, and six of their groups equal validation
groups. Any synthetic board with a group equal to a validation or test group is
rejected.
"""

import argparse
import json
from pathlib import Path
import shutil

from connections.paths import ROOT

SOURCES = [ROOT / "synthetic_data/paper_v2/puzzles.json", ROOT / "synthetic_data/puzzles.json"]


def groups(record):
    return [frozenset(answer["words"]) for answer in record["answers"]]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed_syn")
    parser.add_argument("--sources", type=Path, nargs="+", default=SOURCES)
    args = parser.parse_args()

    train = json.loads((args.input_dir / "train.json").read_text(encoding="utf-8"))
    held_out = {g for split in ("val", "test")
                for record in json.loads((args.input_dir / f"{split}.json").read_text(encoding="utf-8"))
                for g in groups(record)}
    seen, synthetic, skipped = set(), [], []
    for source in args.sources:
        for puzzle in json.loads(source.read_text(encoding="utf-8")):
            record = {
                "contest": f"Synthetic {puzzle['id']}", "date": None, "synthetic": True,
                "source": str(source.resolve().relative_to(ROOT) if source.resolve().is_relative_to(ROOT)
                              else source), "family": puzzle.get("tree_id"),
                "words": [w.upper() for w in puzzle["words"]],
                "answers": [{"answerDescription": a["answerDescription"],
                             "words": [w.upper() for w in a["words"]]} for a in puzzle["answers"]],
            }
            board = groups(record)
            if len(record["words"]) != 16 or len(set(record["words"])) != 16 or len(board) != 4 \
                    or set().union(*board) != set(record["words"]):
                raise ValueError(f"{puzzle['id']}: not a valid 16-word, four-group board")
            if any(g in held_out for g in board):
                raise ValueError(f"{puzzle['id']}: a group equals a validation/test group")
            if all(g in seen for g in board):
                skipped.append(puzzle["id"])
                continue
            seen.update(board)
            synthetic.append(record)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train.json").write_text(
        json.dumps(train + synthetic, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for split in ("val", "test"):
        shutil.copyfile(args.input_dir / f"{split}.json", args.output_dir / f"{split}.json")
    print(f"train: {len(train)} real + {len(synthetic)} synthetic = {len(train) + len(synthetic)} "
          f"(skipped duplicates: {', '.join(skipped) or 'none'}); val/test copied -> {args.output_dir}")


if __name__ == "__main__":
    main()
