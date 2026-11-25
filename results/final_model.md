# Final model: wordplay features, two phrase sources, and the test split

Run date: 2026-11-24. This continues `results/similarity_sources.md` (174.0
validation groups). Models train on the 456 training puzzles; development
numbers are on the **validation split** (98 puzzles, 392 groups). The
configuration below was **frozen before the test split was loaded**, and the
test split was evaluated once: **152.4 ± 1.9 of 392 test groups (38.9%), 21.2
of 98 puzzles solved**, vs 112 for untrained fastText cosine (see "Test split").

**Final model.** GATv2 + residual + CRF on fastText with four extra similarity
sources (WordNet senses, ConceptNet Numberbatch, Numberbatch phrase completion,
Wikipedia-title phrase completion) plus the existing rule-based wordplay
features (`--orthographic-features --homophone-features`). Validation: **184.6 ±
2.7 of 392 groups (47.1%), 23.8 ± 0.8 of 98 puzzles solved** (seeds 0–4).

## Development experiments (validation, 5 seeds each)

All rows use the same settings as before (hidden 128, dropout 0.1, lr 1e-3,
8 epochs, best epoch chosen on validation). Paired bootstrap CIs are over
validation puzzles, seed-averaged.

| Configuration | Groups / 392 | Exact / 98 | Δ [95% CI] | Decision |
|---|---:|---:|---|---|
| Previous best (WordNet + Numberbatch + Numberbatch phrases) | 174.0 ± 3.5 | 19.8 | — | — |
| + wordplay features | 178.8 ± 1.8 | 22.2 | +4.8 [−1.2, +11.6] vs previous | Kept |
| **+ Wikipedia phrases (final)** | **184.6 ± 2.7** | **23.8** | +5.8 [−8.6, +20.2] vs wordplay | **Kept** |
| Wordplay, lr 3e-4, 12 epochs | 166.0 ± 2.9 | 19.0 | −12.8 [−20.8, −5.8] vs wordplay | Dropped |
| No message passing (`--graph none`), previous best | 167.6 ± 5.3 | 18.6 | −6.4 [−12.8, −0.4] vs previous | Control |
| Final + 51 synthetic training puzzles | 185.0 ± 3.0 | 24.0 | −0.4 [−5.4, +4.0] vs control | Dropped |
| Control for the row above (same caches, real puzzles only) | 185.4 ± 1.5 | 24.2 | — | Control |
| Final, epoch chosen on 46 held-out training puzzles | 177.2 ± 7.7 | 22.0 | — | Not used |
| Final, 5-seed score average (ensemble) | 185 | 24 | — | Reported alongside |

- **Wordplay features** (anagrams, palindromes, hidden words, homophones; until
  now only tested on the old BCE model) help and **reduce overfitting**: mean
  validation at epoch 8 is 161.2 instead of 129.4, and seeds vary less.
- **Wikipedia phrases.** Titles such as `Magic_Kingdom` fill gaps in Numberbatch:
  all four words of 31 of 41 validation phrase-completion groups share a
  Wikipedia partner (19 with Numberbatch), but 112 of 351 other groups do too
  (10 with Numberbatch). On its own it is noisier than Numberbatch; added to it,
  it helps: every seed (182–189) beats every seed without it (177–181), and
  the best single epoch across seeds rises from 177.6 to 183.2.
- **Message passing helps.** Removing it (`--graph none`, same parameters) costs
  6.4 groups with a CI excluding zero.
- **Lower learning rate** peaks later but lower, and still overfits.
- **Synthetic puzzles** (51 boards from `synthetic_data/`, about 11% more
  training data) make no difference. The 688 difficulty variants were not used:
  siblings share categories and six of their groups equal validation groups.
- **Held-out epoch selection.** The 46 most recent training puzzles always
  preferred epoch 1–2 while validation peaks at 2–4, and holding them out costs
  10% of the training data. The final model keeps the standard protocol: choose
  on validation, report on test.
- **Ensembling** the five seeds' scores gives 185 groups, about the seed mean:
  the seeds already agree, so there is little noise to average out.

## Final model in detail (validation)

| Seed | Groups | Exact / 98 | Best epoch | Greedy decoding |
|---:|---:|---:|---:|---:|
| 0 | 185 | 24 | 4 | 126 |
| 1 | **189** | **25** | 5 | 131 |
| 2 | 184 | 24 | 4 | 128 |
| 3 | 183 | 23 | 3 | 127 |
| 4 | 182 | 23 | 4 | 111 |
| **Mean ± std** | **184.6 ± 2.7** | **23.8 ± 0.8** | 4.0 | 124.6 ± 7.8 |

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Mean val groups | 153.0 | 173.0 | 179.6 | 180.8 | **183.2** | 174.6 | 169.2 | 176.2 | 171.4 |

- Epoch 0 (untrained) is 153 groups, 20 solved: the weighted sum of all
  sources, with weights fit on train (fastText 2.051, WordNet 4.593,
  Numberbatch 1.302, Numberbatch phrases 0.840, Wikipedia phrases 0.517).
  Training adds 31.6 groups.
- Parameters: 134,534. Final weights after 8 epochs: fastText 2.03–2.25,
  WordNet 4.72–4.89, Numberbatch 1.36–1.55, phrases 0.77–0.82 and 0.44–0.47.

**Frozen for the test split:** the five checkpoints
`runs/next/wordplay_wiki_seed{0..4}/best.pt` (best validation epoch per seed),
reported per seed, as their mean, and as a 5-seed score average.

## Test split

Evaluated once, on 2026-11-24, after the configuration above was frozen and
documented. Test is the last 98 puzzles chronologically (392 groups), never
used for training, selection or design. Before scoring test, every checkpoint
reproduced its recorded validation score.

| Model (test) | Groups / 392 | Accuracy | Puzzles solved / 98 |
|---|---:|---:|---:|
| fastText cosine, no training | 112 | 28.6% | 12 |
| All similarity sources, no training (weights fit on train) | 139 | 35.5% | 19 |
| **Final model, mean of 5 seeds** | **152.4 ± 1.9** | **38.9%** | **21.2** |
| Final model, 5-seed score average | 152 | 38.8% | 21 |

Per seed: 151, 155, 154, 151, 151 groups; 21, 21, 22, 21, 21 puzzles solved.

Paired bootstrap over test puzzles (10,000 resamples, seed-averaged):

- **vs fastText cosine: +40.4 groups, 95% CI [+19.0, +61.4].**
- vs the untrained all-sources score: +13.4 groups, 95% CI [−0.6, +27.8].

**Reading the result.**

- **The final model generalizes:** +40 groups over the plain embedding baseline
  on unseen puzzles, with a CI well clear of zero.
- **Validation overstated it.** Test is 152.4 vs 184.6 on validation. The test
  puzzles are not harder for the baseline (fastText cosine scores 112 on test vs
  106 on validation), so the gap reflects selection on validation: choices of
  sources and features (untrained score 153 on validation vs 139 on test) and of
  each seed's checkpoint (training adds 31.6 on validation, 13.4 on test).
- **Most of the test gain comes from the similarity sources** (+27 untrained);
  training on top adds 13.4, a CI that only just includes zero.

## How to reproduce

Run from the repository root after the README's environment and data steps.

**1. Inputs.** Everything from `results/similarity_sources.md` steps 1–2 (fastText,
Numberbatch, WordNet senses, Numberbatch phrases), plus Wikipedia titles. Use the
dated 2026-11-01 dump (the "latest" link changes every few weeks):

```sh
curl -L -C - -o data/vectors/enwiki-latest-all-titles-in-ns0.gz \
  https://dumps.wikimedia.org/enwiki/20261101/enwiki-20261101-all-titles-in-ns0.gz
python -m connections.features.build_phrase_cache --format lines \
  --vectors data/vectors/enwiki-latest-all-titles-in-ns0.gz --output-dir data/embeddings/phrases/wikipedia
# Homophone cache for the wordplay features (train/val words) and a copy that
# also covers test words, used only for the final test evaluation
python -m connections.embeddings.build_homophone_cache
python -m connections.embeddings.build_homophone_cache --include-test --output-dir data/embeddings/homophones-test
python -m connections.data.verify_reproduction --check-embeddings    # all lines OK
```

**2. Train five seeds** (about 20 minutes in parallel on a 10-core CPU). Each log
prints `Baseline weights fit on 456 training puzzles: a=2.051 wn=4.593 nb=1.302
pc=0.840 pw=0.517` and `Epoch 000 (untrained) ... (153 groups)`.

```sh
mkdir -p runs/next
for s in 0 1 2 3 4; do
  python -m connections.model.train_gnn \
    --embedding-dir data/embeddings/fasttext-crawl --pairwise-features --residual --loss crf \
    --orthographic-features --homophone-features \
    --similarity wn=max:data/embeddings/senses/wordnet-mpnet \
    --similarity nb=cos:data/embeddings/numberbatch \
    --similarity pc=phrase:data/embeddings/phrases/numberbatch-open \
    --similarity pw=phrase:data/embeddings/phrases/wikipedia \
    --seed $s --device cpu --threads 1 --hidden 128 --dropout 0.1 --lr 0.001 \
    --weight-decay 0.0001 --epochs 8 --patience 8 --curriculum none --diagnostic-puzzles 0 \
    --selection-metric groups --output-dir runs/next/wordplay_wiki_seed$s \
    > runs/next/wordplay_wiki_seed$s.log 2>&1 &
done; wait
```

Expected validation: 185, 189, 184, 183, 182 (184.6 ± 2.7).

**3. Validation summary and ensemble** (also checks every checkpoint reproduces
its recorded score):

```sh
python -m connections.evaluation.ensemble_runs runs/next/wordplay_wiki_seed{0,1,2,3,4}
```

**4. Test split, once:**

```sh
python -m connections.evaluation.ensemble_runs runs/next/wordplay_wiki_seed{0,1,2,3,4} \
  --split test --confirm-final-test --homophone-cache data/embeddings/homophones-test \
  --output runs/test/final.json
```

**Experiments in the table** use the same command with: `--graph none` (no
message passing), `--lr 0.0003 --epochs 12 --patience 12`, `--holdout 46`, or,
for synthetic data, `python -m connections.data.add_synthetic` followed by
caches rebuilt with `--input-dir data/processed_syn` into `*-syn` directories
and `--data-dir data/processed_syn` (`--train-limit 456` for the control).

## Caveats

- Same environment and hardware notes as `results/similarity_sources.md`
  (macOS arm64, CPU training, WordNet sense cache encoded with `--device mps`).
- Validation chose both the configuration and each seed's checkpoint, so the
  validation numbers are optimistic. The test split is the unbiased estimate.
- The Wikipedia phrase source and the wordplay features were added after looking
  at validation results; the decision rule was "keep if every seed improves".
