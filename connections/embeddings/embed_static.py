"""Cache normalized static word vectors (fastText/GloVe .vec text format) for puzzle words.

Writes the same embeddings.npy / word_to_row.json / metadata.json layout as
embed_data.py, so baselines.py and compare_embeddings.py read it unchanged.

Lookup per puzzle word, all case-insensitive in effect:
  1. the whole string (lowercase, then Title, then UPPER), which catches
     hyphenated entries such as "flip-flop";
  2. otherwise the mean of its tokens' vectors, each token tried in the same
     three casings (a possessive "'s" falls back to the bare word).
Words with no known token get a seeded random unit vector (cosine ~0 to all
others) and are listed in metadata.json under "oov".
"""

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


from connections.paths import ROOT
TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def casings(text):
    return (text, text.title(), text.upper())


def plan(word):
    """Candidate lookups: one whole-string option, then per-token options."""
    text = word.lower().replace("’", "'").replace("‘", "'")
    tokens = [[*casings(t), *(casings(t.split("'")[0]) if "'" in t else ())]
              for t in TOKEN.findall(text)]
    return list(casings(text)), tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vectors", type=Path, required=True, help="Path to a .vec text file")
    parser.add_argument("--name", required=True, help="Label stored as the model in metadata")
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0, help="Seed for OOV random vectors")
    args = parser.parse_args()

    words, input_hashes = set(), {}
    for split in ("train", "val", "test"):
        raw = (args.input_dir / f"{split}.json").read_bytes()
        words.update(w for record in json.loads(raw) for w in record["words"])
        input_hashes[split] = hashlib.sha256(raw).hexdigest()
    vocabulary = sorted(words)
    plans = {word: plan(word) for word in vocabulary}
    needed = {c for whole, tokens in plans.values() for c in whole + [x for t in tokens for x in t]}

    table = {}
    with args.vectors.open(encoding="utf-8", errors="replace") as stream:
        first = stream.readline().split()
        dim = int(first[1]) if len(first) == 2 else len(first) - 1
        if len(first) != 2 and first[0] in needed:
            table[first[0]] = np.asarray(first[1:], dtype=np.float32)
        for line in stream:
            key, _, rest = line.rstrip("\n").partition(" ")
            if key in needed and key not in table:
                table[key] = np.fromstring(rest, sep=" ", dtype=np.float32)
    print(f"Loaded {len(table)} of {len(needed)} candidate keys, dim {dim}", flush=True)

    rng = np.random.default_rng(args.seed)
    embeddings = np.empty((len(vocabulary), dim), dtype=np.float32)
    oov, partial, sources = [], [], {"whole": 0, "tokens": 0, "oov": 0}
    for row, word in enumerate(vocabulary):
        whole, tokens = plans[word]
        hit = next((c for c in whole if c in table), None)
        if hit is not None:
            vector = table[hit]
            sources["whole"] += 1
        else:
            found = [next((c for c in options if c in table), None) for options in tokens]
            known = [table[c] for c in found if c is not None]
            if known:
                vector = np.mean(known, axis=0)
                sources["tokens"] += 1
                if len(known) < len(tokens):
                    partial.append(word)
            else:
                vector = rng.standard_normal(dim).astype(np.float32)
                sources["oov"] += 1
                oov.append(word)
        embeddings[row] = vector / np.linalg.norm(vector)
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite vectors")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    (args.output_dir / "word_to_row.json").write_text(json.dumps(
        {word: index for index, word in enumerate(vocabulary)}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "model": args.name, "vectors_file": args.vectors.name, "lowercase": True, "prefix": "",
        "normalized": True, "shape": list(embeddings.shape), "dtype": str(embeddings.dtype),
        "lookup": sources, "oov": oov, "partial_token_coverage": partial,
        "input_sha256": input_hashes,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{sources}; {len(partial)} phrases with some unknown tokens")
    print(f"OOV ({len(oov)}): {oov[:30]}")
    print(f"Saved {embeddings.shape} to {args.output_dir}")


if __name__ == "__main__":
    main()
