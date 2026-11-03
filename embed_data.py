"""Cache frozen, normalized MPNet embeddings for unique puzzle words."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parent
MODEL = "sentence-transformers/all-mpnet-base-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda (default: cpu)")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    words = set()
    input_hashes = {}
    for split in ("train", "val", "test"):
        path = args.input_dir / f"{split}.json"
        try:
            raw = path.read_bytes()
            records = json.loads(raw)
            for record in records:
                words.update(record["words"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            parser.exit(1, f"Could not load {path}: {exc}\n")
        input_hashes[split] = hashlib.sha256(raw).hexdigest()

    vocabulary = sorted(words)
    if not vocabulary:
        parser.error("No words found in processed splits")
    print(f"Encoding {len(vocabulary)} unique words/phrases with {MODEL}", flush=True)
    model = SentenceTransformer(
        MODEL, device=args.device, cache_folder=str(ROOT / "data/model_cache")
    )
    model.eval()
    embeddings = model.encode(
        vocabulary, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    if (embeddings.shape != (len(vocabulary), 768)
            or not np.isfinite(embeddings).all()
            or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)):
        raise ValueError("Expected finite, unit-length 768-dimensional embeddings")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    word_to_row = {word: index for index, word in enumerate(vocabulary)}
    (args.output_dir / "word_to_row.json").write_text(
        json.dumps(word_to_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "model": MODEL, "normalized": True, "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype), "input_sha256": input_hashes,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved embeddings {embeddings.shape} and word mapping to {args.output_dir}")


if __name__ == "__main__":
    main()
