"""Analyze validation group recovery by manually reviewable category mechanisms.

Answer descriptions are used only AFTER scoring for analysis, never as model input.
This is a descriptive validation diagnostic, not an independent test evaluation.
"""

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from train_gnn import QuartetGNN, QUARTETS, QUARTET_IDS, decode, load_puzzles
from word_channels import WordTokenizer
from explicit_features import HomophoneCache


ROOT = Path(__file__).resolve().parent
TYPES = ('Semantic / synonyms', 'Thematic / knowledge', 'Phrase completion',
         'Spelling / transformation', 'Sound / pronunciation')

# Manually reviewed synonym/near-synonym categories in the current validation set.
# Others receive a visible rule-based mechanism tag; the CSV supports reannotation.
SYNONYM_ROWS = {
    0: (0, 1), 1: (0, 1), 2: (0, 1), 3: (0,), 5: (0, 1, 2), 6: (0, 2),
    7: (0,), 8: (1,), 10: (0, 1), 11: (0, 1, 2, 3), 13: (0,), 14: (0, 1, 3),
    16: (0, 1), 17: (0, 1), 18: (0, 1), 19: (0,), 21: (0, 1), 22: (0,),
    24: (0, 1), 26: (0,), 27: (0,), 28: (1,), 29: (0, 1), 30: (1,),
    31: (0, 1, 2, 3), 32: (1,), 34: (0,), 36: (0,), 37: (0, 1),
    38: (0, 1), 39: (0,), 40: (0, 1), 41: (1,), 42: (0, 1), 44: (0,),
    45: (0,), 46: (0, 1), 47: (0, 1, 2), 48: (1,), 49: (0,), 50: (0,),
    51: (0,), 52: (0,), 53: (1,), 54: (0, 1), 55: (0, 1), 57: (1, 2),
    58: (1,), 59: (0, 2), 60: (0, 1), 61: (0, 1), 62: (1,), 63: (0,),
    65: (0, 1), 66: (0,), 68: (0, 1), 70: (0,), 71: (1,), 73: (0, 1),
    74: (0,), 75: (0,), 76: (0, 1), 77: (1,), 78: (0,), 79: (0, 1, 2),
    80: (0, 1), 81: (1, 2), 82: (0,), 83: (0,), 85: (0,), 86: (0, 1),
    87: (0, 1), 88: (0,), 89: (0, 1), 90: (0,), 91: (0, 2), 92: (0,),
    93: (2,), 96: (1, 2),
}


def category_type(description, puzzle_index, answer_index):
    """One primary tag per group; priorities are explicit and inspectable."""
    title = description.upper()
    if any(term in title for term in ('HOMOPHONE', 'ONOMATOPOEIA', 'SOUNDS OF',
                                      'SOUND LIKE', 'SOUND MORE', 'SOUND LONGER',
                                      'LONGER WRITTEN THAN SPOKEN')):
        return TYPES[4]
    if any(term in title for term in ('PALINDROME', 'ANAGRAM', 'MINUS', 'REMOVED',
                                      'LETTER CHANGED', 'PLUS STARTING LETTER',
                                      'STARTING WITH', 'STARTS OF', 'ENDING WITH',
                                      'ENDING IN', 'PORTMANTEAU', 'PREFIX', 'SUFFIX')):
        return TYPES[3]
    if '___' in title or 'WORDS AFTER' in title or 'WORDS BEFORE' in title:
        return TYPES[2]
    if answer_index in SYNONYM_ROWS.get(puzzle_index, ()):
        return TYPES[0]
    return TYPES[1]


@torch.no_grad()
def score_run(run_dir):
    checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
    config = checkpoint['config']
    if config.get('overfit') or config.get('val_limit'):
        raise ValueError('Use checkpoints selected on the full validation set')
    model = QuartetGNN(**{key: config[key] for key in (
        'hidden', 'dropout', 'pairwise_features', 'hint_features', 'learned_relations',
        'spelling_features', 'pronunciation_features', 'orthographic_features',
        'homophone_features', 'graph') if key in config})
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    tokenizer = WordTokenizer(pronunciation=True,
        strip_qualifiers=config.get('word_channel_tokenization') == 'strip-parenthetical-stressless-v2',
        orthographic_features=config.get('orthographic_features', False),
        homophone_cache=HomophoneCache(config['homophone_cache']) if config.get('homophone_features') else None)
    if tokenizer.homophone_cache is not None:
        tokenizer.homophone_cache.validate_inputs(config['data_dir'])
        if tokenizer.homophone_cache.metadata != config.get('homophone_cache_metadata'):
            raise ValueError('Homophone cache differs from the checkpoint metadata')
    vectors = np.load(Path(config['embedding_dir']) / 'embeddings.npy', allow_pickle=False)
    mapping = json.loads((Path(config['embedding_dir']) / 'word_to_row.json').read_text())
    puzzles = load_puzzles(Path(config['data_dir']) / 'val.json', vectors, mapping, 'cpu', tokenizer)
    results = []
    for puzzle in puzzles:
        scores = model(puzzle['x'], word_inputs=puzzle['word_inputs']).numpy()
        predicted = decode(scores)
        results.append((scores, predicted))
    return puzzles, results, checkpoint['epoch']


def summarize(rows):
    result = {}
    for name in ('All', *TYPES):
        subset = rows if name == 'All' else [r for r in rows if r['type'] == name]
        if not subset:
            continue
        result[name] = {'groups': len(subset)}
        for model in ('augmented', 'baseline'):
            result[name][model] = {
                'recovered': sum(r[f'{model}_recovered'] for r in subset),
                'recovery_rate': np.mean([r[f'{model}_recovered'] for r in subset]).item(),
                'median_rank': float(np.median([r[f'{model}_rank'] for r in subset])),
                'top4_scored': sum(r[f'{model}_rank'] <= 4 for r in subset),
                'three_word_near_misses': sum(not r[f'{model}_recovered'] and r[f'{model}_closest_overlap'] == 3 for r in subset),
            }
        result[name]['all_words_pronounced'] = sum(not r['missing_pronunciations'] for r in subset)
        result[name]['gained'] = sum(r['augmented_recovered'] and not r['baseline_recovered'] for r in subset)
        result[name]['lost'] = sum(r['baseline_recovered'] and not r['augmented_recovered'] for r in subset)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'runs/compare_word_channels_seed42')
    parser.add_argument('--baseline-dir', type=Path, default=ROOT / 'runs/compare_baseline_seed42')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'runs/word_channels_failure_analysis')
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_config = json.loads((args.run_dir / 'config.json').read_text())
    tokenizer = WordTokenizer(pronunciation=True, strip_qualifiers=
        run_config.get('word_channel_tokenization') == 'strip-parenthetical-stressless-v2')
    all_results = {}
    epochs = {}
    reference_records = None
    for name, folder in (('augmented', args.run_dir), ('baseline', args.baseline_dir)):
        print(f'Scoring {name}', flush=True)
        puzzles, all_results[name], epochs[name] = score_run(folder)
        current_records = [p['record'] for p in puzzles]
        if reference_records is not None and current_records != reference_records:
            raise ValueError('Runs must use identical validation records and word orders')
        reference_records = current_records
    records = [p['record'] for p in puzzles]
    # Guard the hand-reviewed synonym map against a different validation dataset.
    if len(records) != 98 or records[0]['date'] != '2024/09/10' or records[-1]['date'] != '2024/12/16':
        raise ValueError('Review category annotations before using a different validation set')
    train = json.loads((ROOT / 'data/processed/train.json').read_text())
    train_words = {word for puzzle in train for word in puzzle['words']}
    rows = []
    for i, record in enumerate(records):
        positions = {word: j for j, word in enumerate(record['words'])}
        for j, answer in enumerate(record['answers']):
            words = answer['words']
            group = tuple(sorted(positions[word] for word in words))
            official_id = QUARTET_IDS[group]
            row = dict(date=record['date'], puzzle_index=i, answer_index=j,
                       category=answer['answerDescription'], type=category_type(answer['answerDescription'], i, j),
                       words=words, missing_pronunciations=[w for w in words if not tokenizer.variants(w)],
                       unseen_train_words=[w for w in words if w not in train_words])
            for name, results in all_results.items():
                scores, predicted = results[i]
                closest = max(predicted, key=lambda q: (len(set(QUARTETS[q]) & set(group)), scores[q]))
                row.update({
                    f'{name}_recovered': official_id in predicted,
                    f'{name}_rank': int(1 + (scores > scores[official_id]).sum()),
                    f'{name}_score': float(scores[official_id]),
                    f'{name}_closest_overlap': len(set(QUARTETS[closest]) & set(group)),
                    f'{name}_closest_prediction': [record['words'][k] for k in QUARTETS[closest]],
                    f'{name}_top_scored': [record['words'][k] for k in QUARTETS[int(scores.argmax())]],
                })
            rows.append(row)
    summary = summarize(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'groups.json').write_text(json.dumps(rows, indent=2) + '\n')
    with (args.output_dir / 'groups.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()} for row in rows)
    coverage = {}
    for key, predicate in (
        ('all_pronunciations_known', lambda r: not r['missing_pronunciations']),
        ('some_pronunciations_missing', lambda r: bool(r['missing_pronunciations'])),
        ('all_words_seen_in_training', lambda r: not r['unseen_train_words']),
        ('some_words_unseen_in_training', lambda r: bool(r['unseen_train_words'])),
    ):
        subset = [r for r in rows if predicate(r)]
        coverage[key] = summarize(subset).get('All', {})
    metadata = dict(split='validation', puzzles=len(records), groups=len(rows), best_epochs=epochs,
                    category_summary=summary, coverage=coverage,
                    caveat='Descriptive post-hoc mechanism tags, not NYT difficulty colors. Combined model; cannot attribute differences to individual channels.')
    (args.output_dir / 'summary.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lines = ['# Validation failures by category', '',
             '98 puzzles, 392 official groups. Best checkpoints: augmented epoch '
             f'{epochs["augmented"]}, baseline epoch {epochs["baseline"]}.', '',
             'Categories are post-hoc mechanism annotations from answer descriptions. '
             'Descriptions never enter the model. Semantic tags are manually reviewed; '
             'sound, spelling, and phrase tags use the explicit rules in analyze_failures.py. '
             'Remaining groups are thematic/knowledge. These are not difficulty colors. '
             'Ranks are among all 1,820 candidate quartets before exact partition decoding.', '',
             '| Category | Groups | Augmented recovered | Baseline recovered | Median rank: aug / base | Gained / lost |',
             '|---|---:|---:|---:|---:|---:|']
    for name, values in summary.items():
        a, b = values['augmented'], values['baseline']
        lines.append(f'| {name} | {values["groups"]} | {a["recovered"]} ({a["recovery_rate"]:.1%}) | '
                     f'{b["recovered"]} ({b["recovery_rate"]:.1%}) | {a["median_rank"]:g} / {b["median_rank"]:g} | '
                     f'{values["gained"]} / {values["lost"]} |')
    lines += ['', '## Failure examples', '', 'Closest prediction is chosen independently for each official group; '
              'it is not a one-to-one overlap matching and must not be summed as puzzle accuracy.']
    for kind in TYPES:
        lines += ['', f'### {kind}', '']
        examples = [r for r in rows if r['type'] == kind and not r['augmented_recovered']]
        # Show baseline regressions, then high-scoring missed groups.
        examples.sort(key=lambda r: (not r['baseline_recovered'], r['augmented_rank']))
        for row in examples[:5]:
            lines += [f'- **{row["category"]}** ({row["date"]}): {", ".join(row["words"])}. '
                      f'Augmented rank {row["augmented_rank"]}; closest prediction '
                      f'{", ".join(row["augmented_closest_prediction"])} '
                      f'({row["augmented_closest_overlap"]}/4 overlap). '
                      f'Baseline {"recovered" if row["baseline_recovered"] else "missed"}. '
                      f'Missing pronunciations: {", ".join(row["missing_pronunciations"]) or "none"}.']
    sound = summary[TYPES[4]]
    by_category = {r['category']: r for r in rows}
    missed = [r for r in rows if not r['augmented_recovered']]
    lines += ['', '## Interpretation', '',
              f'- The combined augmentation gains {summary["All"]["gained"]} groups overall but loses '
              f'{summary["All"]["lost"]} baseline recoveries. See the table for the mechanism breakdown.',
              f'- Sound coverage: {sound["all_words_pronounced"]}/{sound["groups"]} sound groups '
              'have pronunciations for all four words. Check coverage before attributing misses to absent entries.',
              f'- Of {len(missed)} missed groups, {sum(r["augmented_rank"] > 100 for r in missed)} '
              'rank below the top 100 candidates, while '
              f'{sum(r["augmented_rank"] <= 4 for r in missed)} rank in the top four but are excluded '
              'by the selected global partition. Both weak group scoring and partition competition occur.',
              '- Transformation categories require mapping each word to an external concept, not merely '
              'finding similar strings or phonemes. INDY, METTLE, SEOUL, WRAP map to indie, metal, soul, rap; '
              'their input pronunciations are not mutually similar. Likewise APP, GOO, MOTOR, SAM require '
              'completion to phone-maker names. Homophone-candidate features address the first mapping; '
              'explicit spelling predicates do not by themselves generate semantic completions.',
              '- The direct palindrome group LEVEL, PEP, REFER, TENET ranks '
              f'{by_category["PALINDROMES FEATURING “E”"]["augmented_rank"]}/1820. '
              f'Explicit orthographic features enabled: {run_config.get("orthographic_features", False)}. '
              'Without explicit features, the short-window character CNN has no direct reversal operation.',
              '- Pronunciation qualifiers stripped in this run: '
              f'{tokenizer.strip_qualifiers}. Legacy preprocessing encodes the qualifier in TEE (GOLF); '
              'v2 preprocessing retains only the pronounced word. Display/spelling inputs retain the qualifier.',
              '- High ranks can still be rejected by the global partition. Inspect rank and recovered flags '
              'together: competition among complete partitions can exclude an individually high-scoring '
              'group. That alone is not evidence that the exact decoder is implemented incorrectly.',
              '', '## Next experiments', '',
              '1. Isolate semantic relationships, spelling, and pronunciation in matched retraining ablations; '
              'this combined model cannot identify which component caused regressions.',
              '2. Add explicit character predicates (palindrome, suffix/prefix relations) and test them '
              'separately from learned character similarities.',
              '3. Preserve the display word but strip parenthetical disambiguators for pronunciation inputs. '
              'For homophone-of-category tasks, explore phonetic candidate lookup followed by semantic '
              'scoring of the candidate words, rather than only pairwise input-phoneme similarity.',
              '', '## Limits', '', 'One seed, validation already used for checkpoint selection, and small wordplay '
              'buckets. Inspect config.json for the enabled channels in each run. '
              'Differences cannot be attributed to a particular channel without retraining ablations. '
              'Coverage comparisons are observational and confounded by category difficulty. '
              'All rows, tags, scores, ranks, coverage, and closest predictions are in groups.csv and groups.json.']
    (args.output_dir / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'runs/.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 5))
    xs = np.arange(len(TYPES))
    for name, offset, color in (('baseline', -.2, '#64748b'), ('augmented', .2, '#2563eb')):
        bars = ax.bar(xs + offset, [summary[t][name]['recovery_rate'] * 100 for t in TYPES], .4,
                      label=name.title(), color=color)
        ax.bar_label(bars, [f'{summary[t][name]["recovered"]}/{summary[t]["groups"]}' for t in TYPES], fontsize=9, padding=3)
    ax.set(xticks=xs, xticklabels=[t.replace(' / ', '/\n') for t in TYPES], ylabel='Official groups recovered (%)',
           title='Validation recovery by relationship mechanism', ylim=(0, 50))
    ax.legend(); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(args.output_dir / 'category_recovery.png', dpi=180)
    fig.savefig(args.output_dir / 'category_recovery.pdf')
    plt.close(fig)
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
