"""Phrase-completion similarity: words are similar if they complete the same phrases.

Groups like "___ ROLL" (DRUM, EGG, HONOR, CALIFORNIA) or "DOUBLE ___" (AGENT,
DRIBBLE, STANDARD) share no meaning, so no word-similarity source links them.
What they share is a phrase partner. This reads every key of a phrase vocabulary
(ConceptNet Numberbatch stores "drum roll" as drum_roll) and gives each puzzle
word a set of partners:

  drum_roll      DRUM gets "R:roll" (followed by roll), ROLL gets "L:drum"
  double_agent   AGENT gets "L:double", DOUBLE gets "R:agent"

With --closed, single-word compounds split into two known words count too
(fireball: FIRE gets "R:ball", BALL gets "L:fire").

--format lines reads a plain or gzipped list with one phrase per line instead,
such as Wikipedia article titles (enwiki-latest-all-titles-in-ns0.gz): keys are
lowercased, a trailing disambiguation like "_(film)" is dropped, and punctuation
is removed ("Magic_Kingdom_(film)" -> magic_kingdom).

Each word becomes a sparse vector over partners, weighted by inverse document
frequency over the puzzle vocabulary (a partner shared by few words says more),
then L2-normalized. Cosine between two words is then their weighted share of
partners; words with no partners have a zero vector (similarity 0). Partners
linked to fewer than two puzzle words are dropped, since they cannot connect
anything.

Output (--output-dir): embeddings.npz (scipy CSR), word_to_row.json,
partners.json, metadata.json. Use with train_gnn.py / combine_similarities.py
as --similarity NAME=sparse:DIR.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np
from scipy import sparse

from connections.paths import ROOT

TOKEN = re.compile(r"[a-z0-9]+")


def form(word):
    """Puzzle word -> underscore-joined lowercase tokens ("DECK OF CARDS" -> deck_of_cards)."""
    text = word.lower().replace("’", "'").replace("‘", "'").replace("'", "")
    return "_".join(TOKEN.findall(text))


DISAMBIGUATION = re.compile(r"_\([^)]*\)$")


def read_lines(path):
    """One phrase per line (optionally gzipped), normalized to lowercase_tokens."""
    keys = set()
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            text = DISAMBIGUATION.sub("", line.strip()).lower().replace("'", "").replace("’", "")
            tokens = TOKEN.findall(text.replace("_", " "))
            if tokens:
                keys.add("_".join(tokens))
    return sorted(keys)


def read_keys(path):
    keys = []
    with path.open(encoding="utf-8", errors="replace") as stream:
        first = stream.readline().split()
        if len(first) != 2:  # no "count dim" header
            keys.append(first[0])
        for line in stream:
            keys.append(line[:line.find(" ")])
    return keys


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vectors", type=Path, required=True,
                        help="Phrase vocabulary: .vec text format (only the keys are read), "
                             "or with --format lines one phrase per line")
    parser.add_argument("--format", choices=("vec", "lines"), default="vec")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--closed", action="store_true",
                        help="Also split single-word compounds into two known words (fire|ball)")
    parser.add_argument("--min-part", type=int, default=3,
                        help="--closed: minimum length of each half (default 3)")
    args = parser.parse_args()

    words, hashes = set(), {}
    for split in ("train", "val", "test"):
        raw = (args.input_dir / f"{split}.json").read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        for record in json.loads(raw):
            words.update(record["words"])
    vocabulary = sorted(words)
    forms = defaultdict(list)
    for word in vocabulary:
        if form(word):
            forms[form(word)].append(word)

    keys = read_lines(args.vectors) if args.format == "lines" else read_keys(args.vectors)
    singles = {key for key in keys if "_" not in key}
    partners = defaultdict(set)  # puzzle word -> partner labels
    open_links = closed_links = 0
    for key in keys:
        tokens = key.split("_")
        if len(tokens) >= 2:
            for i in range(1, len(tokens)):
                head, tail = "_".join(tokens[:i]), "_".join(tokens[i:])
                for word in forms.get(head, ()):
                    partners[word].add("R:" + tail)
                    open_links += 1
                for word in forms.get(tail, ()):
                    partners[word].add("L:" + head)
                    open_links += 1
        elif args.closed and len(key) >= 2 * args.min_part and key.isalpha():
            for j in range(args.min_part, len(key) - args.min_part + 1):
                head, tail = key[:j], key[j:]
                if head in singles and tail in singles:
                    for word in forms.get(head, ()):
                        partners[word].add("R:" + tail)
                        closed_links += 1
                    for word in forms.get(tail, ()):
                        partners[word].add("L:" + head)
                        closed_links += 1

    frequency = defaultdict(int)
    for labels in partners.values():
        for label in labels:
            frequency[label] += 1
    kept = sorted(label for label, count in frequency.items() if count >= 2)
    column = {label: index for index, label in enumerate(kept)}
    idf = np.log(len(vocabulary) / np.array([frequency[label] for label in kept]))
    rows, cols, values = [], [], []
    for row, word in enumerate(vocabulary):
        for label in partners.get(word, ()):
            if label in column:
                rows.append(row)
                cols.append(column[label])
                values.append(idf[column[label]])
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(vocabulary), len(kept)), dtype=np.float32)
    norms = np.sqrt(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel())
    matrix = sparse.diags(np.where(norms > 0, 1 / np.maximum(norms, 1e-12), 0)).astype(np.float32) @ matrix
    matrix = matrix.tocsr()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output_dir / "embeddings.npz", matrix)
    (args.output_dir / "word_to_row.json").write_text(json.dumps(
        {word: index for index, word in enumerate(vocabulary)}, ensure_ascii=False) + "\n")
    (args.output_dir / "partners.json").write_text(json.dumps(kept, ensure_ascii=False) + "\n")
    covered = int((norms > 0).sum())
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update({"words": len(vocabulary), "words_with_partners": covered, "partners": len(kept),
                     "nonzeros": int(matrix.nnz), "open_links": open_links, "closed_links": closed_links,
                     "phrase_keys": sum("_" in key for key in keys), "input_sha256": hashes})
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{covered}/{len(vocabulary)} words have shared partners; {len(kept)} partners, "
          f"{matrix.nnz} links -> {args.output_dir}")


if __name__ == "__main__":
    main()
