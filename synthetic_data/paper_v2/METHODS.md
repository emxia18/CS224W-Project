# Generation protocol and departures

This batch implements the generation method in [Merino et al. (2024),
arXiv:2407.11240v1](https://arxiv.org/html/2407.11240v1), checked against the
[authors' released prompts](https://github.com/TimMerino1710/making-new-connections/blob/7e30c702291d2fc73bfe5fdb31c9b2da5c41bf40/prompts.py).
The released repository at this commit has prompts and puzzle datasets but no
tree-building/scoring code. This is a documented implementation, not a claim of
bit-for-bit reproduction.

## Stages

1. Sample four words per tree from the vocabulary of published puzzles, using
   Python seed 224. Use only the project's historical training split as the
   source. Save the words in `seeds.json`.
2. A creator agent writes a four-sentence inspiration story, proposes two root
   categories, and creates an eight-word pool for the chosen category. Categories
   use the paper's styles: synonyms, names, shared features, wordplay, cultural
   references, and phrase completion. Store literal-fit explanations.
3. In false-connection trees, score the root's 70 quartets and use its
   maximum-similarity quartet as the misleading group. Create four new pools
   based on alternate meanings of its four words, one anchor per actual answer.
   The root category is not an answer.
4. In overlap trees, score a pool before generating the next one. Use the
   selected yellow quartet as context, pick an alternate meaning of a word
   already selected, and create a new pool excluding earlier pool words. Repeat
   to obtain four answer pools. Record the context and three directional links.
5. Encode lowercase pool words with frozen `all-mpnet-base-v2`, normalize
   embeddings, and score every four-word subset by the mean of its six pairwise
   cosines. Store all 70 scores per pool, including ineligible subsets.
6. Select four representative quartets per pool. Enumerate the 24 permutations
   assigning the four selection targets to the four pools. These are related
   difficulty variants of one family, **not 24 independent category designs**.
7. A separate editor agent reviews all candidate group alternatives, their
   complete-puzzle context, literal fits, and distractors. It writes short labels
   and flags invalid words/groups; it cannot silently replace words. Rejected
   variants are quarantined. Pool repairs require actual rescoring.
8. A separate difficulty agent reviews the edited group alternatives in puzzle
   context. Its rankings, not embedding scores, assign final colors. A family
   order may apply to all its variants; explicit overrides can differ per board.
9. Check structural validity, duplicate partitions, historical reuse, and
   alternative complete partitions under documented category memberships.
   Human selection and human evaluation remain pending.

## Explicit implementation choices

- **LLM:** creator/editor/ranker are separate instances of the current session's
  model, not the paper's `gpt-4-1106-preview`. No API temperature of 1 is claimed.
  Prompts are operational adaptations of the released instructions, not identical
  API message histories. Model generation is not reproducible from Python seed.
- **Tree expansion:** 15 false-connection families and 15 overlap families; one
  chosen category path per family, followed by 24 selection-target permutations.
  We do not implement an exhaustive search over all possible category proposals.
  The optional historical-seeded false-group and one-step comparison arms are
  outside this batch.
- **Anchors:** every final quartet must retain words required for its documented
  distractor links. All 70 subsets are scored, then the targets are selected from
  anchor-compatible subsets (35 when one anchor is required). This extra explicit
  constraint prevents a difficulty variant from deleting its own distractor. The
  article's illustrations require anchors, whereas the released followup prompts
  exclude the fixed words from newly proposed pools; no public assembly code
  resolves how the original pipeline reinserts them. False-connection pools here
  include their designated anchor and seven new words.
- **Intermediate targets:** yellow is the maximum cosine and purple the minimum.
  Green is `min + 2*(max-min)/3`, blue `min + (max-min)/3`. The printed formula
  omits the minimum offset and lists intermediate fractions in an order contrary
  to the stated difficulty direction. We use equally spaced points between the
  extrema. Extrema/targets use the anchor-compatible subset range, not all 70.
- **Context:** overlap followups use actual MPNet-selected yellow construction
  quartets. Later anchor requirements can change those choices when assembling
  variants; saved construction contexts preserve exactly what the creator used.
- **Model revision:** cached MPNet snapshot
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`, CPU, one Torch thread. This pins the
  local run; the paper did not supply a matching MPNet revision.
- **Review scale:** the editor reviews eight-word pools and all selected quartet
  alternatives together in each family's full-puzzle context. Difficulty ranking
  is also grouped by family, rather than issuing a separate API call per variant.
- **Names:** prefer 1–4 words, following the released editor prompt. Shortening
  never justifies making a category inaccurate.

## Interpreting checks

`puzzles.json` contains one edited, ranked, structurally valid candidate per
family. `variants.json` contains the full retained set of related difficulty
variants; do not count those as independent puzzle designs.
`representatives.json` is an alias of `puzzles.json`. Assembly deterministically
takes one variant per family, preferring
variants with no exact historical answer-group reuse and breaking ties by ID;
this is a convenient smaller batch, not the paper's human preference selection.
`quarantine.json` records exclusions. Exact historical group reuse is reported,
because common valid quartets can recur; matching whole historical boards are
separately reported. The original batch remains available one directory above.

The finite partition check includes the intended categories, documented false
group/overlap memberships, and any extra memberships the editor records. Passing
it proves uniqueness only under that catalogue. Natural-language categories are
open-ended: fairness, global semantic uniqueness, player difficulty, and novelty
outside the local archive require human review.

The historical duplicate audit reads the local archive after generation. The new
creator inputs use training-only inspiration vocabulary. However, the false-group
creator inherited the earlier conversation, which included historical examples
and archive duplicate comparisons. This run is therefore not certified as blind
to the historical holdouts. Synthetic siblings share pools/categories and should
stay together in any future split. This batch is not automatically merged into
the historical experiments.

## Re-run

From the repository root, with the existing virtual environment and MPNet cache:

```sh
.venv/bin/python -m unittest discover -s synthetic_data/paper_v2 -p 'test_*.py'
.venv/bin/python synthetic_data/paper_v2/build.py --finalize
```

This reproduces scoring and assembly from saved creator outputs and independent
review records. It does not regenerate agent outputs. `--reuse-scores` avoids
re-encoding and verifies the pool hash before reuse.
