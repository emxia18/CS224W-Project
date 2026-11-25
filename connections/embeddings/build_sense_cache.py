"""Cache several vectors per puzzle word, one per sense, for max-over-senses similarity.

A single vector stands for a word's most common meaning, while Connections relies
on the others (STAFF as in sheet music, not employees). This writes every sense
vector of every puzzle word, so two words can be compared by their best-matching
senses instead.

Sources (--source):
  fasttext-case  the fastText vector of each distinct casing found (mars, Mars,
                 MARS). Words with no whole-string entry keep their row from the
                 single-vector cache given by --fallback-dir (mean of tokens).
  wordnet        each WordNet definition, encoded as "<word>: <definition>" with
                 a sentence encoder, plus the word itself in a template. Words
                 WordNet does not know keep only the template vector.
  wiktionary     the same with Wiktionary definitions from the kaikki.org
                 English extract (--wiktionary), which also covers slang, brand
                 names and proper nouns, all parts of speech.

Output (--output-dir): senses.npy (unit rows), word_to_rows.json (word -> list of
row indices; the first row is the word's default vector), labels.json (what each
row encodes) and metadata.json.
"""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from connections.embeddings.embed_static import casings
from connections.paths import ROOT


def vocabulary(input_dir):
    words, hashes = set(), {}
    for split in ("train", "val", "test"):
        raw = (input_dir / f"{split}.json").read_bytes()
        hashes[split] = hashlib.sha256(raw).hexdigest()
        for record in json.loads(raw):
            words.update(record["words"])
    return sorted(words), hashes


def normalized(text):
    return text.lower().replace("’", "'").replace("‘", "'")


def fasttext_case(words, args):
    """Distinct casing vectors from a .vec file; the cache row is the default."""
    fallback = np.load(args.fallback_dir / "embeddings.npy")
    fallback_rows = json.loads((args.fallback_dir / "word_to_row.json").read_text())
    wanted = {word: list(dict.fromkeys(casings(normalized(word)))) for word in words}
    needed = {c for options in wanted.values() for c in options}
    table = {}
    with args.vectors.open(encoding="utf-8", errors="replace") as stream:
        stream.readline()
        for line in stream:
            key, _, rest = line.rstrip("\n").partition(" ")
            if key in needed and key not in table:
                table[key] = np.fromstring(rest, sep=" ", dtype=np.float32)
    senses = {}
    for word in words:
        default = fallback[fallback_rows[word]]
        found = [(c, table[c] / np.linalg.norm(table[c])) for c in wanted[word] if c in table]
        rows = [("default", default)]
        rows += [(c, v) for c, v in found if not np.allclose(v, default, atol=1e-5)]
        senses[word] = rows
    return senses


def wiktionary_glosses(path, words, limit):
    """word -> list of (label, gloss) from a kaikki.org JSONL extract (optionally .gz)."""
    import gzip
    import re

    def key(text):
        return " ".join(re.findall(r"[a-z0-9]+", normalized(text).replace("'", "")))

    wanted = defaultdict(list)
    for word in words:
        wanted[key(word)].append(word)
    glosses = defaultdict(list)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            entry = json.loads(line)
            matches = wanted.get(key(entry.get("word", "")))
            if not matches or entry.get("lang_code", "en") != "en":
                continue
            for sense in entry.get("senses", []) or []:
                for gloss in (sense.get("glosses") or [])[-1:]:  # most specific gloss
                    for word in matches:
                        if gloss not in [g for _, g in glosses[word]] and len(glosses[word]) < limit:
                            glosses[word].append((f"{entry.get('pos', '?')}:{len(glosses[word])}", gloss))
    return glosses


def encode_senses(texts, args):
    import torch
    from sentence_transformers import SentenceTransformer

    flat = [t for rows in texts.values() for _, t in rows]
    model = SentenceTransformer(args.model, device=args.device,
                                cache_folder=str(ROOT / "data/model_cache"),
                                model_kwargs={"dtype": torch.float32})
    vectors = model.encode(flat, batch_size=64, show_progress_bar=True,
                           convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    senses, i = {}, 0
    for word in texts:
        senses[word] = [(label, vectors[i + k]) for k, (label, _) in enumerate(texts[word])]
        i += len(texts[word])
    return senses


def wordnet(words, args):
    """WordNet definitions plus a template, encoded with a sentence encoder."""
    import nltk

    nltk.data.path.insert(0, str(ROOT / "data/model_cache/nltk_data"))
    from nltk.corpus import wordnet as wn

    texts = {}
    for word in words:
        text = normalized(word)
        lemma = "_".join(text.replace("-", " ").split())
        synsets = wn.synsets(lemma)[:args.max_senses]
        texts[word] = ([("template", args.template.replace("{word}", text))]
                       + [(s.name(), f"{text}: {s.definition()}") for s in synsets])
    return encode_senses(texts, args)


def wiktionary(words, args):
    """Wiktionary definitions plus a template, encoded with a sentence encoder."""
    glosses = wiktionary_glosses(args.wiktionary, words, args.max_senses)
    texts = {}
    for word in words:
        text = normalized(word)
        texts[word] = ([("template", args.template.replace("{word}", text))]
                       + [(label, f"{text}: {gloss}") for label, gloss in glosses.get(word, [])])
    return encode_senses(texts, args)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", choices=("fasttext-case", "wordnet", "wiktionary"), required=True)
    parser.add_argument("--wiktionary", type=Path,
                        help="wiktionary: kaikki.org-dictionary-English.jsonl(.gz)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--vectors", type=Path, help="fasttext-case: .vec file")
    parser.add_argument("--fallback-dir", type=Path, default=ROOT / "data/embeddings/fasttext-crawl",
                        help="fasttext-case: single-vector cache used as each word's default")
    parser.add_argument("--model", default="sentence-transformers/all-mpnet-base-v2",
                        help="wordnet: sentence encoder for definitions")
    parser.add_argument("--template", default='the word "{word}"', help="wordnet: default sense text")
    parser.add_argument("--max-senses", type=int, default=12, help="wordnet: synsets kept per word")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.source == "fasttext-case" and args.vectors is None:
        parser.error("--source fasttext-case needs --vectors")
    if args.source == "wiktionary" and args.wiktionary is None:
        parser.error("--source wiktionary needs --wiktionary")

    words, hashes = vocabulary(args.input_dir)
    senses = {"fasttext-case": fasttext_case, "wordnet": wordnet, "wiktionary": wiktionary}[args.source](words, args)

    rows, labels, word_to_rows = [], [], {}
    for word in words:
        word_to_rows[word] = list(range(len(rows), len(rows) + len(senses[word])))
        for label, vector in senses[word]:
            rows.append(vector)
            labels.append(label)
    matrix = np.stack(rows).astype(np.float32)
    if not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
        raise ValueError("Expected finite unit-length sense vectors")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "senses.npy", matrix, allow_pickle=False)
    (args.output_dir / "word_to_rows.json").write_text(json.dumps(word_to_rows, ensure_ascii=False) + "\n")
    (args.output_dir / "labels.json").write_text(json.dumps(labels, ensure_ascii=False) + "\n")
    counts = np.array([len(r) for r in word_to_rows.values()])
    metadata = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata.update({"words": len(words), "rows": len(rows), "dim": int(matrix.shape[1]),
                     "mean_senses": float(counts.mean()), "single_sense_words": int((counts == 1).sum()),
                     "input_sha256": hashes})
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"{len(words)} words, {len(rows)} sense vectors (mean {counts.mean():.2f}, "
          f"{(counts == 1).sum()} with one) -> {args.output_dir}")


if __name__ == "__main__":
    main()
