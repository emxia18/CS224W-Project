"""Assemble MPNet-selected tree variants and apply separate LLM review records."""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path
import random
import re

from mpnet import COLORS, score_pools
from constraints import partition_count


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def read(name):
    return json.loads((HERE / name).read_text())


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def check_tree(tree):
    if len(tree['groups']) != 4:
        raise ValueError(f"{tree['id']}: four groups required")
    used = []
    for group in tree['groups']:
        words = group['words']
        if len(words) != 8 or any(not re.fullmatch('[A-Z]+', w) for w in words):
            raise ValueError(f"{group['id']}: eight uppercase single words required")
        if len(set(words)) != 8 or not set(group.get('required', [])) <= set(words):
            raise ValueError(f"{group['id']}: invalid pool/anchors")
        used.extend(words)
    if len(set(used)) != len(used):
        raise ValueError(f"{tree['id']}: pools must be disjoint")
    if tree['method'] == 'false_connection':
        anchors = tree['false_group']['words']
        if len(set(anchors)) != 4:
            raise ValueError('Four distinct false anchors required')
        for anchor, group in zip(anchors, tree['groups']):
            if anchor not in group.get('required', []):
                raise ValueError('False anchor must be preserved')
    elif tree['method'] == 'intentional_overlap':
        if len(tree['links']) != 3 or {l['to_group'] for l in tree['links']} != {1, 2, 3}:
            raise ValueError('Three successive overlap links required')
        for link in tree['links']:
            src, dst = link['from_group'], link['to_group']
            if not 0 <= src < dst <= 3:
                raise ValueError('Overlap must point to a later group')
            if link['word'] not in tree['groups'][src].get('required', []):
                raise ValueError('Overlap source must be preserved')
    else:
        raise ValueError('Unknown method')


def assemble(trees, scores):
    variants, duplicate_variants = [], []
    seen = set()
    for tree in trees:
        check_tree(tree)
        for number, assignment in enumerate(itertools.permutations(COLORS), 1):
            answers = []
            for group, color in zip(tree['groups'], assignment):
                choice = scores['pools'][group['id']]['selections'][color]
                answers.append({'group_id': group['id'], 'answerDescription': group['category'],
                                'words': choice['words'], 'selection_color': color,
                                'similarity': choice['similarity']})
            words = [w for a in answers for w in a['words']]
            if len(set(words)) != 16:
                raise ValueError('Invalid board')
            key = tuple(sorted(tuple(sorted(a['words'])) for a in answers))
            pid = f"{tree['id']}-v{number:02}"
            if key in seen:
                duplicate_variants.append(pid)
            seen.add(key)
            random.Random(pid).shuffle(words)
            variants.append({'id': pid, 'tree_id': tree['id'], 'method': tree['method'],
                             'contest': f'Synthetic Connections {pid}', 'words': words,
                             'answers': answers, 'review_status': 'awaiting_independent_editor',
                             'seed_words': tree['seed_words'],
                             'distractors': tree.get('false_group', tree.get('links'))})
    return variants, duplicate_variants


def finalize(trees, variants):
    edits = read('editor_review.json')
    ranks = read('difficulty_review.json')
    edit_map = edits['groups']
    rank_map = ranks['trees']
    tree_map = {t['id']: t for t in trees}
    exported, quarantined, seen = [], [], set()
    for original in variants:
        p = json.loads(json.dumps(original))
        reasons = []
        for a in p['answers']:
            e = edit_map[a['group_id']]
            a['answerDescription'] = e['category']
            if e.get('reject', False):
                reasons.append(f"{a['group_id']}: {e['note']}")
            bad = set(e.get('invalid_words', [])) & set(a['words'])
            if bad:
                reasons.append(f"{a['group_id']}: invalid words {sorted(bad)}")
            if len(a['answerDescription'].split()) > 4:
                reasons.append('Category label exceeds four words')
            tokens = set(re.findall(r'[A-Z]+', a['answerDescription'].upper()))
            if tokens & set(a['words']):
                reasons.append('Category name contains an answer word')
        rank_record = rank_map[p['tree_id']]
        order = rank_record.get('variant_overrides', {}).get(p['id'], rank_record['order'])
        group_ids = [a['group_id'] for a in p['answers']]
        if len(order) != 4 or set(order) != set(group_ids):
            raise ValueError(f"{p['id']}: incomplete independent difficulty ranking")
        for color, gid in zip(COLORS, order):
            next(a for a in p['answers'] if a['group_id'] == gid)['color'] = color
        tree = tree_map[p['tree_id']]
        memberships = [set(g['words']) | set(edit_map[g['id']].get('additional_members', []))
                       for g in tree['groups']]
        if tree['method'] == 'false_connection':
            memberships.append(set(tree['false_group']['words']))
        else:
            for link in tree['links']:
                memberships[link['to_group']].add(link['word'])
        p['catalogue_partition_count_capped_at_two'] = partition_count(p['words'], memberships)
        if p['catalogue_partition_count_capped_at_two'] != 1:
            reasons.append('Nonunique partition under documented category memberships')
        p['answers'].sort(key=lambda a: COLORS.index(a['color']))
        p['review_status'] = 'llm_edited_and_ranked; human_review_pending'
        key = tuple(sorted(tuple(sorted(a['words'])) for a in p['answers']))
        if key in seen:
            reasons.append('Duplicate partition among generated variants')
        if reasons:
            quarantined.append({'id': p['id'], 'reasons': reasons})
        else:
            exported.append(p)
            seen.add(key)
    write('variants.json', exported)
    write('quarantine.json', quarantined)
    historical = read_historical()
    historical_groups = {frozenset(a['words']) for p in historical for a in p['answers']}
    historical_boards = {frozenset(p['words']) for p in historical}
    # One deterministic representative per family. Selection is not a human
    # quality judgement. Prefer no exact historical answer-group reuse.
    representatives = []
    for tree in trees:
        available = [p for p in exported if p['tree_id'] == tree['id']]
        if available:
            representatives.append(min(available, key=lambda p: (
                sum(frozenset(a['words']) in historical_groups for a in p['answers']), p['id'])))
    write('representatives.json', representatives)
    write('puzzles.json', representatives)
    old = json.loads((HERE.parent / 'puzzles.json').read_text())
    old_boards = {frozenset(p['words']) for p in old}
    exact_groups = [{'id': p['id'], 'group': a['group_id']} for p in exported for a in p['answers']
                    if frozenset(a['words']) in historical_groups]
    for p in exported:
        if len(p['words']) != 16 or len(set(p['words'])) != 16:
            raise ValueError('Invalid board')
        if Counter(p['words']) != Counter(w for a in p['answers'] for w in a['words']):
            raise ValueError('Invalid answer coverage')
        if {a['color'] for a in p['answers']} != set(COLORS):
            raise ValueError('Invalid color assignment')
    report = {
        'trees': len(trees), 'methods': dict(Counter(t['method'] for t in trees)),
        'generated_variant_slots': len(variants), 'exported_distinct_partitions': len(exported),
        'main_puzzles': len(representatives), 'representatives': len(representatives), 'quarantined': len(quarantined),
        'all_quartets_scored': sum(len(s['quartets']) for s in read('quartet_scores.json')['pools'].values()),
        'structural_checks': 'passed', 'max_category_words': max((len(a['answerDescription'].split()) for p in exported for a in p['answers']), default=0),
        'historical_board_matches': [p['id'] for p in exported if frozenset(p['words']) in historical_boards],
        'prior_batch_board_matches': [p['id'] for p in exported if frozenset(p['words']) in old_boards],
        'historical_group_matches': exact_groups,
        'representative_historical_group_matches': [
            {'id': p['id'], 'group': a['group_id']} for p in representatives for a in p['answers']
            if frozenset(a['words']) in historical_groups],
        'semantic_uniqueness': 'not proven; human review required',
        'documented_category_catalogue_partition_counts': dict(Counter(p['catalogue_partition_count_capped_at_two'] for p in exported)),
        'human_evaluation': 'not performed',
    }
    write('validation_report.json', report)
    print(json.dumps(report, indent=2))


def read_historical():
    return json.loads((ROOT / 'data/raw/connections.json').read_text())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--reuse-scores', action='store_true')
    args = parser.parse_args()
    trees = read('false_trees.json') + read('overlap_trees.json')
    if len({t['id'] for t in trees}) != len(trees):
        raise ValueError('Duplicate tree IDs')
    for tree in trees:
        check_tree(tree)
    pools = [g for t in trees for g in t['groups']]
    if args.reuse_scores:
        scores = read('quartet_scores.json')
        digest = hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest()
        if digest != scores['input_sha256']:
            raise ValueError('Stale scores; run without --reuse-scores')
    else:
        scores = score_pools(pools)
        write('quartet_scores.json', scores)
    variants, duplicates = assemble(trees, scores)
    write('unedited_variants.json', variants)
    write('duplicate_variant_slots.json', duplicates)
    if args.finalize:
        finalize(trees, variants)
    else:
        print(f'{len(trees)} trees; {len(variants)} variant slots; {len(duplicates)} duplicate partitions')
