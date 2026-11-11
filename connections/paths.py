"""Repository-relative paths shared by every script.

All defaults resolve from the repository root, so commands work no matter which
directory you run them from (as long as you use `python -m connections.<...>`).
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"                 # downloaded dataset (gitignored)
PROCESSED = DATA / "processed"     # train/val/test splits (gitignored)
EMBEDDINGS = DATA / "embeddings"   # cached frozen word vectors (gitignored)
MODEL_CACHE = DATA / "model_cache" # Hugging Face downloads (gitignored)
RUNS = ROOT / "runs"               # training outputs (gitignored)
