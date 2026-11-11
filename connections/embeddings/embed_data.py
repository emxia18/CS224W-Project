"""Cache frozen, normalized sentence-transformer embeddings for unique puzzle words.

The default model is MPNet; pass --model to try another embedder and write it to
its own --output-dir so caches for different models do not overwrite each other.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
import torch


from connections.paths import ROOT
MODEL = "sentence-transformers/all-mpnet-base-v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--revision", help="Pin the Hugging Face model commit for reproduction")
    parser.add_argument("--device", default="cpu", help="cpu, mps, or cuda (default: cpu)")
    parser.add_argument("--model", default=MODEL, help=f"Hugging Face model id (default: {MODEL})")
    parser.add_argument("--prefix", default="",
                        help="Text prepended to every word, e.g. 'query: ' for E5 models")
    parser.add_argument("--lowercase", action="store_true",
                        help="Lowercase words before encoding (puzzle words are all caps; "
                             "matters only for cased models)")
    parser.add_argument("--trust-remote-code", action="store_true")
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
    print(f"Encoding {len(vocabulary)} unique words/phrases with {args.model}", flush=True)
    model = SentenceTransformer(
        args.model, device=args.device, cache_folder=str(ROOT / "data/model_cache"),
        revision=args.revision, trust_remote_code=args.trust_remote_code,
        # Some checkpoints default to fp16/bf16; use fp32 so models compare like for like.
        model_kwargs={"dtype": torch.float32},
    )
    model.eval()
    texts = [args.prefix + (word.lower() if args.lowercase else word) for word in vocabulary]
    embeddings = model.encode(
        texts, batch_size=args.batch_size, show_progress_bar=True,
        convert_to_numpy=True, normalize_embeddings=True,
    ).astype(np.float32)
    if (embeddings.ndim != 2 or embeddings.shape[0] != len(vocabulary)
            or not np.isfinite(embeddings).all()
            or not np.allclose(np.linalg.norm(embeddings, axis=1), 1, atol=1e-5)):
        raise ValueError("Expected finite, unit-length embeddings, one row per word")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "embeddings.npy", embeddings, allow_pickle=False)
    word_to_row = {word: index for index, word in enumerate(vocabulary)}
    (args.output_dir / "word_to_row.json").write_text(
        json.dumps(word_to_row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    metadata = {
        "model": args.model, "prefix": args.prefix, "lowercase": args.lowercase, "normalized": True, "shape": list(embeddings.shape),
        "dtype": str(embeddings.dtype), "input_sha256": input_hashes,
    }
    ref = ROOT / "data/model_cache" / f"models--{args.model.replace('/', '--')}" / "refs/main"
    metadata['revision'] = args.revision or (ref.read_text().strip() if ref.exists() else None)
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved embeddings {embeddings.shape} and word mapping to {args.output_dir}")


if __name__ == "__main__":
    main()
