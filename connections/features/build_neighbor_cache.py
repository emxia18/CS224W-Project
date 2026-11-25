"""Shared-neighbor caches from knowledge sources, for thematic groups.

Thematic groups ("things that are yellow": CANARY, LEMON, MINION, MUSTARD;
"kinds of shoes": FLAT, MULE, PUMP, SLIDE) are linked by a shared fact, often
about a rare sense. This gives every puzzle word a set of neighbors from a
knowledge source:

  conceptnet  concepts linked to the word by a ConceptNet 5.7 edge between two
              English terms (canary -HasProperty-> yellow gives CANARY the
              neighbor "yellow"); relation-agnostic, so lemon -RelatedTo-> yellow
              also gives LEMON "yellow". Morphological and negative relations
              are skipped.
  wiktionary  topical categories of the word's Wiktionary entries in the
              kaikki.org English extract (pump: "en:Footwear"), all senses and
              parts of speech; grammatical and maintenance categories skipped.

The output has the same layout as build_phrase_cache.py (embeddings.npz,
word_to_row.json, partners.json), so it is used the same way:
--similarity NAME=phrase:DIR links two words in a puzzle if they share a
neighbor that 3-5 of the puzzle's 16 words share.
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
SKIP_RELATIONS = {"FormOf", "DerivedFrom", "EtymologicallyRelatedTo", "EtymologicallyDerivedFrom",
                  "ExternalURL", "Antonym", "DistinctFrom", "NotDesires", "NotCapableOf",
                  "NotHasProperty", "NotUsedFor", "ObstructedBy"}
SKIP_CATEGORY = re.compile(
    r"(^English |terms|entries|lemmas|nouns|verbs|adjectives|adverbs|pages|requests|"
    r"rhymes|derived from|borrowed|inherited|etymolog|pronunciation|spelling|"
    r"abbreviations|acronyms|initialisms|ethnic slurs|with |redlinks|links)", re.I)


def form(text):
    """Puzzle word or source term -> lowercase tokens joined by "_"."""
    text = text.lower().replace("’", "'").replace("‘", "'").replace("'", "")
    return "_".join(TOKEN.findall(text))


def conceptnet(path, forms):
    neighbors = defaultdict(set)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            parts = line.split("\t", 4)
            if len(parts) < 4 or not parts[2].startswith("/c/en/") or not parts[3].startswith("/c/en/"):
                continue
            relation = parts[1].removeprefix("/r/")  # e.g. IsA, HasProperty, dbpedia/genre
            if relation in SKIP_RELATIONS:
                continue
            start, end = form(parts[2].split("/")[3]), form(parts[3].split("/")[3])
            if start == end:
                continue
            for word in forms.get(start, ()):
                neighbors[word].add(end)
            for word in forms.get(end, ()):
                neighbors[word].add(start)
    return neighbors


def category_names(entry):
    names = []
    for item in entry.get("categories", []) or []:
        names.append(item.get("name") if isinstance(item, dict) else item)
    for sense in entry.get("senses", []) or []:
        for item in sense.get("categories", []) or []:
            names.append(item.get("name") if isinstance(item, dict) else item)
        names.extend(f"topic:{t}" for t in sense.get("topics", []) or [])
    return [n for n in names if n and not SKIP_CATEGORY.search(n)]


def wiktionary(path, forms):
    neighbors = defaultdict(set)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            entry = json.loads(line)
            if entry.get("lang_code", "en") != "en":
                continue
            for word in forms.get(form(entry.get("word", "")), ()):
                neighbors[word].update(category_names(entry))
    return neighbors


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("conceptnet", "wiktionary"), required=True)
    parser.add_argument("--path", type=Path, required=True,
                        help="conceptnet-assertions-5.7.0.csv.gz or kaikki.org-dictionary-English.jsonl(.gz)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
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

    neighbors = (conceptnet if args.source == "conceptnet" else wiktionary)(args.path, forms)

    frequency = defaultdict(int)
    for labels in neighbors.values():
        for label in labels:
            frequency[label] += 1
    kept = sorted(label for label, count in frequency.items() if count >= 2)
    column = {label: index for index, label in enumerate(kept)}
    rows, cols = [], []
    for row, word in enumerate(vocabulary):
        for label in neighbors.get(word, ()):
            if label in column:
                rows.append(row)
                cols.append(column[label])
    matrix = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                               shape=(len(vocabulary), len(kept)))
    covered = int((np.asarray(matrix.sum(axis=1)).ravel() > 0).sum())

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(args.output_dir / "embeddings.npz", matrix)
    (args.output_dir / "word_to_row.json").write_text(json.dumps(
        {word: index for index, word in enumerate(vocabulary)}, ensure_ascii=False) + "\n")
    (args.output_dir / "partners.json").write_text(json.dumps(kept, ensure_ascii=False) + "\n")
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "source": args.source, "path": args.path.name, "words": len(vocabulary),
        "words_with_neighbors": covered, "neighbors": len(kept), "links": int(matrix.nnz),
        "input_sha256": hashes}, indent=2) + "\n")
    print(f"{covered}/{len(vocabulary)} words have shared neighbors; {len(kept)} neighbors, "
          f"{matrix.nnz} links -> {args.output_dir}")


if __name__ == "__main__":
    main()
