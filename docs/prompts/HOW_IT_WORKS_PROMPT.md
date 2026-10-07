# Prompt: rebuild the How It Works page

Work in `C:\projects\newschecker - with rag`, branch `feat/rag-upgrade`. Commit
locally only; never push. Read `CLAUDE.md` first.

## Goal

`client/src/components/HowItWorks.jsx` must let someone who has never seen the
code understand **everything** NewsChecker does, from paste to verdict, by
reading one page. Today it is nine short accordion cards and one summary
strip. Rebuild it as a guided, illustrated walkthrough.

## What the page must explain (all of it, accurately)

Source of truth is the code, not the README; check each claim against it.

1. **Normalising the submission** — `claim_normalizer.py`: strips "Is it true
   that…", "BREAKING:", pasted URLs, wire tags; the user's words are kept for
   display, the normalised claim drives the machinery.
2. **Triage** — `claim_triage.py`: checkable / prospective (future) / opinion /
   not a claim / unsupported language; salience (would a true version
   necessarily be reported?) and negation. Nothing is searched for non-claims.
3. **Deterministic answers** — `knowledge_verifier.py`: a small table of
   textbook facts and arithmetic answered instantly (why "the Great Wall is
   visible from the Moon" returns in seconds); negated/quoted forms are
   declined and sent to the evidence pipeline.
4. **Claim splitting and understanding** — `claim_verifier.extract_claims`,
   `claim_decomposer.py`: up to 3 atomic claims; entities, predicate,
   negation, attribution, modality, time.
5. **Coverage mode** — `claim_recency.py`: recent (last 30 days) vs historical
   decided automatically from the wording; stale articles cannot confirm a
   claim about today.
6. **Query generation** — `query_generator.py`: several query shapes per claim.
7. **Retrieval** — `providers/registry.py`: six providers in parallel under a
   45 s budget (`EVIDENCE_BUDGET_SECONDS`), per-provider diagnostics,
   dedup; failures are reported, never treated as "no evidence".
8. **Relevance filtering** — `relevance_filter.py`: entity + action +
   coherence scoring; reserved slots for credible sources
   (`RESERVED_TIER_SLOTS`) so rumour posts cannot crowd them out.
9. **Article extraction + hybrid passage selection** — `article_extractor.py`,
   `passage_retriever.py`: boilerplate stripping; lexical overlap + MiniLM
   embeddings merged by reciprocal rank fusion (show the formula
   Σ 1/(60+rank)); the paraphrased-debunk example.
10. **NLI** — `nli_service.py`: premise = passage, hypothesis = claim;
    entailment / contradiction / neutral; label-order safety; float32 on CPU.
11. **Per-document stance** — `evidence_pipeline.decide_stance`: threshold
    0.35, dominance 1.6×; quoted-claim frames excluded from support; the
    checks that only *withdraw* a position: different number
    (`numeric_consistency`), stale article (`claim_recency`), not about the
    claim (aboutness gate, `SEMANTIC_MIN_SIMILARITY`).
12. **Aggregation** — `evidence_aggregator.py`: source tiers and weights,
    independent publishers (not article counts), 2× dominance for a
    direction to win outright, otherwise "mixed".
13. **Absence of coverage** — `assess_coverage`: the six conditions under
    which "no credible source reports this" is allowed.
14. **Verdicts** — all nine statuses and what each means; categorical
    confidence and what scales it.
15. **Grounded explanation** — `explainer.py`: runs after the verdict, cites
    `[n]`, every sentence NLI-checked, dropped reasons, fallback model,
    "never changes the verdict".
16. **What it deliberately does not do** and **known limitations** (dates,
    same-kind-different-event, weak queries, English only).

## Diagrams to include (inline SVG or styled HTML, no new dependencies)

- **End-to-end flow** with the three exits marked: no-search verdict (triage /
  deterministic), evidence verdict, and the explanation branch that is drawn
  as display-only.
- **Funnel** showing a real example's numbers shrinking: candidates →
  relevant → classified → supporting/contradicting (use illustrative numbers,
  labelled as an example).
- **Hybrid retrieval**: two ranked lists merging into one via RRF, with the
  paraphrased debunk rising.
- **NLI**: premise/hypothesis box with the three-way score bar.
- **Stance decision**: a small decision tree for `decide_stance` + the
  withdrawal checks.
- **Aggregation**: sources weighted by tier, grouped by publisher, the 2× rule.
- **Explanation filter**: LLM sentences → citation check → NLI check → kept /
  dropped.

## Page structure

1. Short intro + "the one-minute version" (5 bullets).
2. Sticky or top table of contents linking to each stage.
3. One section per stage: plain-English what/why, the diagram, a small
   "worked example" box, and "where in the code" (file names).
4. Verdict glossary table (all statuses).
5. Design principles (short) and limitations.
6. Tech stack (keep the current card).

## Constraints

- Plain React + existing CSS conventions (`hiw-*` classes, CSS variables in
  `index.css`); no new npm packages. Lucide icons are available.
- Readable on mobile (diagrams scroll horizontally rather than shrink).
- Accurate: every number (thresholds, budgets, slot counts) must match the
  code. If the code and this prompt disagree, the code wins — say so.
- Split into small components (one per diagram) in
  `client/src/components/howItWorks/`.
- `npm run lint` and `npm run build` must pass; check the page in the browser
  at desktop and phone widths; commit with a descriptive message.
