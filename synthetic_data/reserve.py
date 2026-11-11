"""Atomically reserve candidate words/concepts across generator agents.

Usage: python3 'data generation/reserve.py' OWNER proposal.json
Proposal is {"words": [...], "categories": [canonical semantic keys]}.
"""

import argparse
import fcntl
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('owner')
    parser.add_argument('proposal', type=Path)
    args = parser.parse_args()
    proposal = json.loads(args.proposal.read_text())
    with (HERE / 'reservations.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = HERE / 'reservations.json'
        registry = json.loads(path.read_text())
        conflicts = {}
        for kind in ('words', 'categories'):
            items = proposal.get(kind, [])
            for item in items:
                owner = registry[kind].get(item)
                if owner is not None and owner != args.owner:
                    conflicts.setdefault(kind, {})[item] = owner
        if conflicts:
            parser.exit(1, json.dumps({'conflicts': conflicts}, indent=2) + '\n')
        for kind in ('words', 'categories'):
            for item in proposal.get(kind, []):
                registry[kind][item] = args.owner
        path.write_text(json.dumps(registry, indent=2) + '\n')
        print('Reserved', {kind: len(proposal.get(kind, [])) for kind in ('words', 'categories')})
