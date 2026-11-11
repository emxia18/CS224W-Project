"""Build the single synthetic training dataset with global repetition checks.

Saved selections, trees, and independent reviews live in generation_sources.json.
Run with the project virtual environment; --rescore reruns frozen MPNet.
"""

from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

from constraints import partition_count
from mpnet import COLORS, score_pools


HERE = Path(__file__).resolve().parent


def normalize(value):
    return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())


def audit(puzzles):
    words, labels, concepts, ids, families = [], [], [], [], []
    for p in puzzles:
        pid = p['id']
        ids.append(pid)
        families.append(p['tree_id'])
        board = p['words']
        if len(board) != 16 or any(not isinstance(w, str) or not w.strip() for w in board):
            raise ValueError(f'{pid}: sixteen nonempty words required')
        if len({normalize(w) for w in board}) != 16:
            raise ValueError(f'{pid}: repeated words within board')
        answers = p['answers']
        if len(answers) != 4 or any(len(a['words']) != 4 for a in answers):
            raise ValueError(f'{pid}: four quartets required')
        if Counter(board) != Counter(w for a in answers for w in a['words']):
            raise ValueError(f'{pid}: answer coverage mismatch')
        if {a['color'] for a in answers} != set(COLORS):
            raise ValueError(f'{pid}: all four colors required')
        for a in answers:
            label = a['answerDescription']
            if not 1 <= len(label.split()) <= 4:
                raise ValueError(f'{pid}: label must have one to four words')
            if set(re.findall(r'[A-Z]+', label.upper())) & {w.upper() for w in a['words']}:
                raise ValueError(f'{pid}: label contains an answer word')
            labels.append(normalize(label))
            if not a.get('category_key'):
                raise ValueError(f'{pid}: missing reviewed semantic category key')
            concepts.append(normalize(a['category_key']))
        words.extend(normalize(w) for w in board)
    repeated = {}
    for name, values in [('word', words), ('category_label', labels),
                         ('category_concept', concepts), ('id', ids), ('family', families)]:
        duplicates = {k: n for k, n in Counter(values).items() if n > 1}
        if duplicates:
            repeated[name] = duplicates
    if repeated:
        raise ValueError('Cross-puzzle repetition: ' + json.dumps(repeated, sort_keys=True))
    return {'puzzles': len(puzzles), 'input_word_occurrences': len(words),
            'unique_input_words': len(set(words)), 'repeated_input_words': {},
            'answer_groups': len(concepts), 'unique_category_labels': len(set(labels)),
            'unique_reviewed_category_concepts': len(set(concepts)),
            'repeated_category_labels': {}, 'repeated_category_concepts': {},
            'max_category_words': max((len(a['answerDescription'].split()) for p in puzzles for a in p['answers']), default=0),
            'structural_checks': 'passed'}


def assemble(sources, scores):
    puzzles = []
    seen_source_ids = set()
    for item in sources['families']:
        tree = item['tree']
        if tree['id'] in seen_source_ids:
            raise ValueError('Repeated source family')
        seen_source_ids.add(tree['id'])
        answers = []
        memberships = []
        for group in tree['groups']:
            gid = group['id']
            edit = sources['editor_review']['groups'][gid]
            choice = item['selections'][gid]
            target = choice['selection_color']
            selected = scores['pools'][gid]['selections'][target]
            if selected['words'] != choice['words']:
                raise ValueError(f'{gid}: saved selection disagrees with scored pool')
            if edit.get('reject') or set(edit.get('invalid_words', [])) & set(selected['words']):
                raise ValueError(f'{gid}: independent editor rejected selected words')
            if not set(group.get('required', [])) <= set(selected['words']):
                raise ValueError(f'{gid}: distractor anchor removed')
            memberships.append(set(group['words']) | set(edit.get('additional_members', [])))
            answers.append({'answerDescription': edit['category'], 'words': selected['words'],
                            'category_key': choice['category_key'], 'group_id': gid,
                            'selection_color': target, 'similarity': selected['similarity']})
        rank = sources['difficulty_review']['trees'][tree['id']]
        order = rank.get('variant_overrides', {}).get(item['selected_variant_id'], rank['order'])
        if len(order) != 4 or set(order) != {a['group_id'] for a in answers}:
            raise ValueError('Invalid independent difficulty ranking')
        for color, gid in zip(COLORS, order):
            next(a for a in answers if a['group_id'] == gid)['color'] = color
        answers.sort(key=lambda a: COLORS.index(a['color']))
        words = [w for a in answers for w in a['words']]
        if tree['method'] == 'false_connection':
            memberships.append(set(tree['false_group']['words']))
        else:
            for link in tree['links']:
                memberships[link['to_group']].add(link['word'])
        if partition_count(words, memberships) != 1:
            raise ValueError(f"{tree['id']}: alternative partition under reviewed memberships")
        random.Random(tree['id']).shuffle(words)
        puzzles.append({'id': tree['id'], 'tree_id': tree['id'],
                        'contest': f"Synthetic training Connections {tree['id']}",
                        'method': tree['method'], 'words': words, 'answers': answers,
                        'review_status': 'independent_llm_editor_and_ranker; human_review_pending'})
    return puzzles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rescore', action='store_true')
    args = parser.parse_args()
    source_path = HERE / 'generation_sources.json'
    sources = json.loads(source_path.read_text())
    pools = [g for item in sources['families'] for g in item['tree']['groups']]
    digest = hashlib.sha256(json.dumps(pools, sort_keys=True).encode()).hexdigest()
    score_path = HERE / 'quartet_scores.json'
    if args.rescore:
        scores = score_pools(pools)
        score_path.write_text(json.dumps(scores, indent=2) + '\n')
    else:
        scores = json.loads(score_path.read_text())
    if scores['input_sha256'] != digest:
        raise ValueError('Pool hash mismatch; run with --rescore')
    puzzles = assemble(sources, scores)
    report = audit(puzzles)
    report.update({'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
                   'all_quartets_scored': sum(len(x['quartets']) for x in scores['pools'].values()),
                   'semantic_category_scope': 'independently reviewed category concepts; broad reasoning styles may recur',
                   'semantic_solution_uniqueness': 'one partition under documented memberships; unrestricted semantics and fairness require human review'})
    registry = {'words': sorted(w for p in puzzles for w in p['words']),
                'category_keys': sorted(a['category_key'] for p in puzzles for a in p['answers']),
                'category_labels': sorted(a['answerDescription'] for p in puzzles for a in p['answers'])}
    # All checks finish before replacing the dataset or registry.
    for name, data in [('puzzles.json', puzzles), ('validation_report.json', report),
                       ('training_registry.json', registry)]:
        (HERE / name).write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
