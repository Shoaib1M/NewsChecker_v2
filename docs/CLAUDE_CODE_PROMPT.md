# Working prompt: NewsChecker RAG upgrade

This is the prompt Claude works from in `C:\projects\newschecker - with rag`,
on branch `feat/rag-upgrade`. It is the original mega prompt with every
decision since resolved, so a fresh session can pick the work up mid-way.
Hard rules live in the repo-root `CLAUDE.md` and are not repeated in full here.

## Decisions already made (2026-10-07)

- Work in this folder only. Commit **locally**; never push. `origin` still
  points at the public repo — do not push to it.
- Run all four phases end to end without stopping for approval between them.
- Explanation model: **`gemini-3.8-flash`** (stable, on Gemini's free tier).
  Env `EXPLAIN_MODEL` overrides it.
- `GOOGLE_API_KEY` lives in `ml-service/.env` (gitignored). Never print it.
- Use `ml-service/.venv` for everything. System Python has a broken
  `torchvision` (`operator torchvision::nms does not exist`) which stops
  `transformers` loading any model, so real NLI cannot run there.
- The two stale tests in `tests/test_provider_registry.py` (live Google News
  calls) are fixed by patching `KEYLESS_PROVIDERS` out — approved.
- Dense-ranking availability is reported as `passage_ranking:
  {enabled, available, status, model, error}`, mirroring `nli`, in both the
  pipeline outcome (`/api/check` → `retrieval.passage_ranking`) and
  `/api/health`.

### Decided later the same day

- `gemini-3.8-flash` returned 503 "high demand" all day, so
  `EXPLAIN_FALLBACK_MODELS` (default `gemini-3.5-flash-lite`) is tried when the
  primary fails. The response records which model actually wrote the text.
  Gemini refuses request deadlines under 10 s; the default timeout is 10 s.
- Benchmark findings: hybrid retrieval is a **null result** on wrong-answer
  rate (within run-to-run spread). Most wrong answers came from the NLI model,
  not passage selection: `nli-deberta-v3-small` scores unrelated pairs as
  1.00 contradiction.
- Added an **aboutness gate** (`evidence_pipeline._not_about_claim`): a
  document below `SEMANTIC_MIN_SIMILARITY` on title, snippet and decisive
  passage cannot support or contradict. Only active when dense ranking is.
- User approved switching `NLI_MODEL` (in `.env` / `.env.example`, not the
  code default) to `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli`:
  stance_sweep accuracy 0.61 → 0.91, invented positions 6 → 2 of 23.
- The GNews / NewsAPI free daily quotas were exhausted by benchmarking. Do
  not run the benchmark on demo day.

## Phase 1 — hybrid passage retrieval

- `ml-service/passage_retriever.py`
  - `semantic_order(claim, sentences) -> list[int] | None` — model
    `PASSAGE_EMBED_MODEL` (default `sentence-transformers/all-MiniLM-L6-v2`),
    lazy + cached, CPU, normalised embeddings, cosine. Keep only similarity ≥
    `SEMANTIC_MIN_SIMILARITY` (0.30). Cap 150 sentences. `None` = disabled /
    failed; `[]` = nothing similar enough.
  - `reciprocal_rank_fusion(*orders, k=60)` — ties keep original order.
  - `SEMANTIC_PASSAGES` (default true; false in tests).
- `extract_passages`: title + snippet first; fuse dense + lexical when dense
  is non-empty; else exact old behaviour; unfused sentences follow in lexical
  order.
- Tests `tests/test_dense_passages.py`: paraphrased debunk reaches passages;
  None ⇒ lexical; [] ⇒ lexical; title/snippet first; RRF tie order; unrelated
  claim keeps document order.
- Measure warm latency on one real claim.

## Phase 2 — NLI-checked explanations

- `ml-service/explainer.py`: input = final status + classified supporting /
  contradicting evidence (best_sentence, publisher, stance, tier). Skip for
  not_a_claim, not_objectively_verifiable, insufficient_evidence,
  not_verifiable_yet, unsupported language. Gemini via `google-genai`,
  temperature 0, short timeout, runs after the evidence budget.
- Prompt: 2–4 sentences, only the numbered evidence, cite every sentence
  `[n]`, no new facts, never restate/soften/contradict the verdict.
- Faithfulness filter: split sentences; drop uncited / invalid citations;
  `get_nli_service().score_many(sentence, [evidence_text])` per citation;
  keep only if entailment > `STANCE_THRESHOLD`. NLI unavailable ⇒
  `available: false`.
- Response `explanation: {available, reason, text, sentences: [{text,
  citations, kept, entailment}], dropped_count, model}`. Schema round-trip
  test + README API reference.
- Client: explanation under the verdict, `[n]` scrolls to + highlights the
  EvidenceCard, note "Each sentence checked against its source by the NLI
  model." Nothing shown when unavailable.
- Tests (mocked): unsupported dropped, uncited dropped, status never changed,
  missing key ⇒ unavailable, NLI unavailable ⇒ unavailable, skip statuses
  never call the LLM.

## Phase 3 — measure + document

- `news_benchmark.py`: `--save` once, then `--from-file` with
  `SEMANTIC_PASSAGES=false` and `true`, twice each; table of wrong-answer
  rate, confirmation recall, abstention rate, mean latency. Sweep
  `SEMANTIC_MIN_SIMILARITY` 0.25 / 0.30 / 0.40. Report honestly, null
  results included.
- 10 varied claims: explanation sentences kept / dropped, spot-check.
- README section "Hybrid retrieval & grounded explanations", both Mermaid
  diagrams, env tables. Two resume bullets from real numbers only.

## Phase 4 — demo prep

- `scripts/demo_warmup.ps1` (Windows): start all three services, send one
  warm-up check.
- `docs/DEMO.md`: 3 clean demo claims (supported / contradicted /
  not_objectively_verifiable or not_verifiable_yet), 2.5-minute script with
  timestamps, recording checklist. README "Demo video" placeholder.

## Definition of done

Original + new tests pass (counts reported); both flags false reproduce the
original behaviour; honest benchmark table in README; explanation in UI with
working citations, never unverified; `.env.example` files; `docs/DEMO.md`;
final summary with run instructions and top-10 interview Q&As.
