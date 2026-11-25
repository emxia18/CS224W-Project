# Thematic knowledge and heterogeneous graphs (development round)

Run date: 2026-11-24. **Validation only** (98 puzzles, 392 groups). The test
split was already used once for the frozen final model (`results/final_model.md`)
and was **not** evaluated again. Validation has now guided many decisions, so the
gains below are development numbers, weaker evidence than earlier rounds; a fresh
test set (newer puzzles) is needed to confirm them.

**Current development best: per-puzzle heterogeneous graph with concept nodes
(Option 1).** 198.2 ± 2.5 of 392 validation groups (50.6%), 28.6 ± 1.1 of 98
puzzles solved, seeds 0–4. Previous: final model 184.6 (test 152.4).

| Model (validation, seeds 0–4) | Groups / 392 | Exact / 98 | Thematic / 193 | Δ vs row above [95% CI] |
|---|---:|---:|---:|---:|
| Final model (`results/final_model.md`) | 184.6 ± 2.7 | 23.8 | 70.6 | — |
| + ConceptNet similarity | 190.2 ± 2.3 | 25.8 | 78.8 | +5.6 [−8.6, +20.4] |
| **+ concept nodes (Option 1, development best)** | **198.2 ± 2.5** | **28.6** | **83.4** | +8.0 [−4.2, +20.2] |
| Global graph (Option 2) | no gain | — | — | negative result, see below |

## 1. What the final model still missed

The final model missed 122 of 193 validation thematic groups. They fall into
three kinds:

- **Rare sense in a "kinds of X" category:** shoes (FLAT, MULE, PUMP, SLIDE),
  fish (FLUKE, MULLET, SOLE, TANG), currencies (POUND, REAL, SOL, YUAN).
- **Properties and associations:** things that are yellow, things with shells,
  made of keratin, seen in a kitchen.
- **Pop culture and entities:** NBA legends, car models, HBO shows, Sopranos.

No LLM API was available, so all new sources are offline knowledge bases.

## 2. New knowledge sources

All are used with the shared-neighbor rule from phrase completion
(`phrase:DIR`): two words in a puzzle are linked if they share a neighbor that
3–5 of the puzzle's 16 words share.

| Source | What a word's neighbors are | Missed thematic groups linked | False links (other groups) |
|---|---|---:|---:|
| Wikipedia phrases (already in the final model) | title partners | 41 / 122 | 72 / 199 |
| **ConceptNet 5.7** (`neighbors/conceptnet`) | concepts joined by any semantic edge (IsA, HasProperty, PartOf, UsedFor, AtLocation, ...) | 16 / 122 | 51 / 199 |
| **Wiktionary categories** (`neighbors/wiktionary`) | topical categories of every sense ("Footwear", "Fish") | 11 / 122 | 9 / 199 |
| WordNet hypernyms, depth 3 (not kept) | "is a" ancestors | 5 / 122 | 33 / 199 |

Also built: **Wiktionary definitions** as sense vectors (`senses/wiktionary-mpnet`,
56,872 senses; only 63 words left without a definition, vs 434 with WordNet).

**No-training check** (weights fit on train, exact decoding; train numbers 2-fold
cross-validated):

| Sources added to the final model's 5 | Val groups | Train groups / 1,824 |
|---|---:|---:|
| none | 153 | 796 |
| + Wiktionary categories | 154 | 800 |
| + ConceptNet | 148 | **836** |
| + Wiktionary definitions | 148 | 796 |
| + all three | 160 | 828 |

**Trained (5 seeds):** + ConceptNet gives 190.2 ± 2.3 (per seed 192, 188, 188,
190, 193; 25.8 solved; thematic 70.6 → 78.8; best single epoch across seeds 188.4
vs 183.2). + all three gives 190.2 ± 2.6 (27.4 solved, thematic 80.2, less
overfitting) but no more groups, so ConceptNet alone was kept as the base.

## 3. Option 1: per-puzzle heterogeneous graph (development best)

Code: `connections/hetero/` (`concepts.py`, `model.py`, `train_concepts.py`).

- **Graph per puzzle.** The 16 word nodes plus a concept node for every neighbor
  shared by 3–8 of the words, from four sources: Numberbatch phrases, Wikipedia
  phrases, ConceptNet, Wiktionary categories. Median 181 concept nodes per
  puzzle (max 1,807).
- **Concept features:** source (one-hot), number of puzzle words linked
  (one-hot), rarity across the vocabulary (log inverse frequency).
- **Message passing:** word → concept → word → concept, relation-specific
  weights per source (64 hidden units).
- **Group-level readout.** Each concept gets a learned score; a candidate
  quartet gains the scores of all concepts linked to **all four** of its words.
  Pairwise similarities cannot express "one hidden category explains exactly
  these four words".
- **Start at the base model.** The score head is zero-initialized, so epoch 0
  equals the base (148 groups untrained). Everything else (base model, inputs,
  baseline-weight fit, optimizer, loop, evaluation) matches `train_gnn.py`;
  with no concept sources the script reproduces `train_gnn.py` **identically**
  at every epoch (checked for seed 0).

| Concept nodes kept | Groups / 392 | Exact / 98 | Thematic | Best single epoch | Greedy |
|---|---:|---:|---:|---:|---:|
| none (base: final + ConceptNet) | 190.2 ± 2.3 | 25.8 | 78.8 | 188.4 | 135.2 |
| shared by 2–8 words (median 864 nodes) | 197.6 ± 6.3 | 27.8 | 85.0 | 193.6 | 143.8 |
| **shared by 3–8 words (median 181 nodes)** | **198.2 ± 2.5** | **28.6** | 83.4 | **194.6** | 146.6 |

Per seed (3–8): 198, 196, 202, 196, 199 groups; 28, 29, 30, 29, 27 solved; best
epochs 3, 5, 5, 2, 3. Every seed beats every base seed (188–193). Mean by epoch:

| Epoch | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Concept nodes (3–8) | 148.0 | 171.0 | 192.8 | **194.6** | 193.0 | 192.2 | 183.8 | 183.8 | 176.4 |
| Base | 148.0 | 165.8 | 184.4 | 188.0 | **188.4** | 185.4 | 182.6 | 182.2 | 169.0 |

Parameters: 208,584 (base 135,175 + concept branch).

## 4. Option 2: one global graph (negative result)

Code: `connections/hetero/global_graph.py`.

- **Graph.** All 5,435 puzzle words (fastText + Numberbatch features) and about
  153,000 concept nodes (Numberbatch phrases, ConceptNet, Wiktionary; 605,000
  with Wikipedia phrases), plus **same-group edges from training answers only**.
- **Training.** Heterogeneous GraphSAGE or GAT, link prediction on all 120
  within-puzzle pairs (the negatives are each puzzle's own red herrings), with
  message-passing and target puzzles disjoint at every step.
- **No leakage.** Out-of-fold stacking: a training puzzle's matrices come from a
  model that never saw its answers. Validation and test answers are never edges.

**Is there a history signal?** Same-group pairs in validation were grouped
together in a training puzzle 6.8% of the time, against 0.5% for different-group
pairs (14× enrichment). But only 89 of 392 validation groups have even one such
pair, and 906 of the 1,568 validation word slots are words seen in training.

Pair AUC: the probability that a random same-group pair scores above a random
different-group pair. Reference on validation: fastText 0.753, Numberbatch
0.758, WordNet senses 0.734. Tuning used held-out training puzzles, not validation.

| Global model | Held-out training pair AUC |
|---|---:|
| Untrained (skip connection: cosine of input features) | 0.780 |
| First version (no skip connection, 60 steps) | 0.556 (validation 0.555) |
| SAGE, concepts + history, 80 steps | 0.782 |
| SAGE, lr 1e-3 / dropout 0.5 + weight decay 1e-3 | 0.779 / 0.778 |
| GAT (attention), 1 or 2 layers | 0.782 |
| History only (no concept nodes), SAGE / GAT | 0.772 / 0.778 |

**Training never moves the model beyond its own input features**, so it was not
added to the per-puzzle model. The likely reason, and the main lesson of the
comparison:

- **A global graph gives each word one embedding for all puzzles,** but the useful
  knowledge is puzzle-specific: "shoe" matters for PUMP only when three other
  words in the same puzzle are shoes.
- **The per-puzzle graph expresses that directly** (a concept linked to exactly
  four of these 16 words). A global embedding has to mix all of a word's meanings,
  the single-vector problem WordNet senses were introduced to fix.
- **The direct history is too sparse** to learn from on its own (the history-only
  graph also stays at input level).

Possible follow-up: learn concept embeddings on the global graph and use them as
Option 1's concept-node features, combining global knowledge with puzzle context.

## How to reproduce

From the repository root, after `results/final_model.md`'s inputs exist.

**1. Downloads** (into `data/vectors/`, about 1 GB):

```sh
curl -L -C - -o data/vectors/conceptnet-assertions-5.7.0.csv.gz \
  https://s3.amazonaws.com/conceptnet/downloads/2019/edges/conceptnet-assertions-5.7.0.csv.gz
curl -L -C - -o data/vectors/kaikki.org-dictionary-English.jsonl.gz \
  https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz
```

The kaikki.org extract is regenerated regularly and has no dated archive, so a
fresh download will differ from ours (downloaded 2026-11-24; hash in
`reproduction_manifest.json`). To reproduce exactly, copy our file.

**2. Caches:**

```sh
python -m connections.features.build_neighbor_cache --source conceptnet \
  --path data/vectors/conceptnet-assertions-5.7.0.csv.gz --output-dir data/embeddings/neighbors/conceptnet
python -m connections.features.build_neighbor_cache --source wiktionary \
  --path data/vectors/kaikki.org-dictionary-English.jsonl.gz --output-dir data/embeddings/neighbors/wiktionary
# optional (tested, not in the best model): Wiktionary definitions as sense vectors
python -m connections.embeddings.build_sense_cache --source wiktionary --device mps --max-senses 16 \
  --wiktionary data/vectors/kaikki.org-dictionary-English.jsonl.gz --output-dir data/embeddings/senses/wiktionary-mpnet
python -m connections.data.verify_reproduction --check-embeddings
```

**3. Development best (Option 1), five seeds.** Each log prints `Baseline weights
fit on 456 training puzzles: a=1.960 wn=4.243 nb=1.081 pc=0.817 pw=0.516
cn=0.478`, `Concept nodes per puzzle: median 181` and `Epoch 000 (untrained) ...
(148 groups)`.

```sh
E=data/embeddings; mkdir -p runs/hetero
for s in 0 1 2 3 4; do
  python -m connections.hetero.train_concepts --orthographic-features --homophone-features \
    --similarity wn=max:$E/senses/wordnet-mpnet --similarity nb=cos:$E/numberbatch \
    --similarity pc=phrase:$E/phrases/numberbatch-open --similarity pw=phrase:$E/phrases/wikipedia \
    --similarity cn=phrase:$E/neighbors/conceptnet \
    --concept pc=$E/phrases/numberbatch-open --concept pw=$E/phrases/wikipedia \
    --concept cn=$E/neighbors/conceptnet --concept wc=$E/neighbors/wiktionary \
    --concept-min 3 --seed $s --output-dir runs/hetero/concepts3_seed$s > runs/hetero/concepts3_seed$s.log 2>&1 &
done; wait
```

Expected: 198, 196, 202, 196, 199. Without the `--concept` lines it reproduces
the base, `train_gnn.py` with the same `--similarity` flags: 192, 188, 188, 190, 193.

**4. Option 2** (tuning mode, held-out training AUC; `--folds 5` writes
out-of-fold matrices for `--similarity NAME=puzzle:DIR`):

```sh
python -m connections.hetero.global_graph --extra-word-dir $E/numberbatch \
  --source pc=$E/phrases/numberbatch-open --source cn=$E/neighbors/conceptnet \
  --source wc=$E/neighbors/wiktionary --conv gat --epochs 10 --chunks 8 --folds 0 --output-dir runs/hetero/global
```

## Caveats

- Same environment as earlier rounds (macOS arm64, CPU training). The Wiktionary
  sense cache was encoded on the Apple GPU (`--device mps`); it is not used by the
  development best.
- All gains here are on heavily reused validation, with CIs that include zero.
  Treat 198.2 as a development number until checked on new puzzles.
