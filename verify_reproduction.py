"""Verify the input snapshot, optionally including byte-identical cached vectors."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-embeddings', action='store_true',
                        help='Also require the original embedding/cache file hashes')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'reproduction_manifest.json').read_text())
    expected = dict(manifest['data_sha256'])
    if args.check_embeddings:
        expected.update(manifest['embedding_sha256'])
    failed = []
    for name, digest in expected.items():
        path = ROOT / name
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
        if actual != digest:
            failed.append(name)
        print(f'{"OK" if actual == digest else "MISMATCH/MISSING"} {name}')
    if failed:
        parser.exit(1, 'Inputs differ from the recorded experiment. Check the dataset snapshot '
                    'and preparation seed; regenerated vectors can differ across platforms.\n')


if __name__ == '__main__':
    main()
