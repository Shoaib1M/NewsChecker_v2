# NewsChecker RAG upgrade — final summary

Branch `feat/rag-upgrade` in `C:\projects\newschecker - with rag` (local only,
not pushed). Written 2026-10-07.

## 1. What changed

| Area | Change | Files |
|---|---|---|
| Hybrid retrieval | MiniLM sentence embeddings + word-overlap ranking, merged with reciprocal rank fusion, choose the passages NLI reads. Falls back to the old ranking exactly when off or broken. | `ml-service/passage_retriever.py`, `article_extractor.py` |
| Aboutness gate | A document whose title, snippet and decisive passage are all semantically unrelated to the claim cannot support or contradict it (stance withdrawn to *unclear* with a note). | `evidence_pipeline.py` |
| Grounded explanations | After the verdict is final, Gemini writes 2–4 cited sentences; each is kept only if the NLI model finds its cited source entails it. Fallback model on 503s. Never changes a verdict. | `ml-service/explainer.py`, `main.py` |
| UI | "Why this verdict" panel with clickable `[n]` citations that scroll to and highlight the evidence card; How It Works page updated. | `client/src/components/ExplanationPanel.jsx`, `EvidenceCard.jsx`, `App.jsx`, `HowItWorks.jsx`, `App.css` |
| History | Explanation + passage-ranking diagnostics persisted and replayed. | `server/models/Check.js`, `server/routes/check.js`, `client/src/App.jsx` |
| NLI speed bug | NLI pipeline pinned to float32 on CPU (a float16 checkpoint ran ~100× slower). | `nli_service.py` |
| NLI model | `.env.example` recommends `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (stance_sweep accuracy 0.61 → 0.91). | `.env.example`, README |
| Test hygiene | 4 stale tests no longer hit the live internet; developer API keys can no longer leak into tests; HF offline mode in tests. | `tests/conftest.py`, `test_provider_registry.py`, `test_pipeline_budget.py` |
| Measurement | Benchmark reports recall, abstention, latency and search failures; explanation quality script; Windows scripts. | `news_benchmark.py`, `explanation_check.py`, `scripts/*.ps1` |
| Legacy classifier removed | The LIAR-trained MLP (wording-only, never read by the verdict), its training/evaluation scripts, dataset, tests, the Evaluation and Comparison pages and the `ml` API fields. | `main.py`, `server/`, `client/` |
| UI cleanup | How It Works spacing and diagram, current tech stack, Auto/Recent/Historical picker removed. | `client/src/` |
| Docs | README section, diagrams, env tables, limitations; CLAUDE.md; demo guide; this file. | `README.md`, `CLAUDE.md`, `docs/` |

Tests: **ml-service 480 passed** (517 before the legacy classifier and its 37
tests were removed; the baseline was 465 passed + 2 failing), **server 24
passed**, client lint + build clean.

## 2. How to run it

See the bottom of this file, or `docs/DEMO.md` for the recording workflow.

## 3. Results (honest)

Full table and discussion: README → *Hybrid retrieval & grounded explanations → Results*.

- **Hybrid retrieval: null result.** 17 fixed cases, re-scored repeatedly: lexical 4/3/3 wrong, hybrid 3/3, hybrid + gate 3/3. All within the lexical baseline's own run-to-run spread.
- **Latency:** ~2.0 s of embedding per check (≈5% of a 42 s check); changed NLI's passages for 5 of 8 articles on a real claim.
- **NLI model:** labelled stance corpus accuracy 0.61 → 0.91, invented positions 6 → 2 of 23. No measurable change on the 17-case benchmark (fixed two wrong answers, introduced two).
- **Explanations:** live end-to-end check passed (3/3 sentences kept, fallback model used while `gemini-3.8-flash` was overloaded). The 10-claim quality check (`ml-service/explanation_check.py`) was stopped before completion — run it before quoting a keep rate.

## 4. Resume bullets

- Added hybrid dense + lexical passage retrieval (MiniLM, reciprocal rank fusion) and NLI-verified LLM explanations to an evidence-first fact-checker; benchmarked it honestly (no significant change on 17 live claims) and used per-claim error analysis to find an NLI failure mode, lifting stance accuracy on a labelled corpus from 61% to 91%.
- Found and fixed a silent ~100× CPU slowdown (float16 checkpoint on CPU) and test-isolation leaks that sent real API calls from the test suite; the suite runs 480 tests with no network or model downloads.

## 5. Top 10 interview questions

**1. Walk me through what happens when I submit a claim.**
Triage decides whether it is even a checkable claim (questions, opinions,
future events are answered without searching). A deterministic checker handles
textbook facts. Otherwise: query generation → six search providers in parallel
under a 45 s budget → relevance filtering (entity + action, not keyword
overlap) → passage selection (hybrid) → DeBERTa NLI per passage → stance per
document (with numeric, staleness and aboutness checks that can only withdraw a
position) → aggregation weighted by source tier and counted by independent
publisher → verdict with categorical confidence → optional NLI-checked
explanation.

**2. Why NLI rather than asking an LLM whether the claim is true?**
An LLM answers from memory, sounds equally sure when wrong, and cannot say
which source it relied on. NLI is a narrow, checkable question — does this
passage entail, contradict or neither — that I can measure (`stance_sweep.py`)
and threshold. The verdict is then an aggregation over sources I can show the
user.

**3. What is reciprocal rank fusion and why use it here?**
Each item scores Σ 1/(k + rank) over the ranked lists it appears in (k = 60).
It merges rankings without putting their scores on one scale — cosine
similarity and word-overlap counts are not comparable — and has no weight to
tune. A sentence near the top of either list rises; one near the top of both
rises most.

**4. Why no vector database?**
Each check embeds a few hundred freshly fetched sentences that are never
queried again. An index costs more than it saves; embeddings live in memory
for one request.

**5. How do you stop the LLM from inventing facts or changing the verdict?**
Structurally, not by prompt: it runs after every verdict field is computed,
writes only `explanation`, and nothing reads that back — a test runs the full
API with an LLM told to argue the opposite verdict and asserts every verdict
field is identical. For facts: every sentence must cite a provided source, and
is kept only if NLI (premise = source, hypothesis = sentence) finds it
entailed under the same rule that classifies evidence. NLI down ⇒ no
explanation at all.

**6. Did hybrid retrieval improve accuracy?**
No measurable change on the benchmark — verdicts were identical on 16 of 17
claims, and the one difference was a search failure. I report that as a null
result. Measuring it is what revealed the real problem: the NLI model.

**7. What was the real problem, and how did you find it?**
Per-claim inspection of wrong answers showed off-topic Guardian articles
(oil pipelines, a walking guide) "contradicting" a car-crash headline at 1.00.
Testing the NLI model directly showed `nli-deberta-v3-small` scores *unrelated*
text as contradiction — an SNLI labelling convention. I added an aboutness
gate using the embeddings I had just built, and measured a checkpoint trained
without SNLI: accuracy 0.61 → 0.91 on the labelled corpus.

**8. Tell me about a bug you found by measuring rather than by an error.**
Two: (a) one benchmark run reported 0% wrong answers — because every search
had been rate-limited and the benchmark counted outages as abstentions; it now
records retrieval status. (b) The better NLI checkpoint ships a float16 config;
on CPU that made each pair take 13.7 s instead of 0.18 s. Nothing errored —
checks just got slow. Fixed by pinning float32, with a regression test.

**9. How do you keep ~500 tests deterministic when the system depends on live
news, models and an LLM?**
Every external dependency is behind a seam that tests replace: providers,
NLI, the embedder, Gemini. `conftest.py` turns the new features off, forces
Hugging Face offline mode, and blanks API keys — which mattered: the API tests
were loading my real keys from `.env` and later tests silently queried live
providers.

**10. What are the biggest remaining limitations?**
NLI cannot tell "same kind of event" from "same event" (Route 44 vs Route 209)
or handle dates well ("gained ground in August" vs "invaded in February 2022").
Short headlines sometimes produce weak queries. The benchmark is built from
today's headlines and corrupted versions of them, not real misinformation.
Next step: a labelled set of real misinformation, and a date/referent-aware
contradiction rule.

## 6. Running locally (Windows / PowerShell)

```powershell
# One step: start all three services in their own windows and warm the models
cd "C:\projects\newschecker - with rag"
powershell -ExecutionPolicy Bypass -File .\scripts\demo_warmup.ps1
# then open http://localhost:5173
```

By hand, three terminals:

```powershell
cd "C:\projects\newschecker - with rag\ml-service"; .\.venv\Scripts\python.exe main.py
cd "C:\projects\newschecker - with rag\server"; npm run dev
cd "C:\projects\newschecker - with rag\client"; npm run dev
```

Tests:

```powershell
cd "C:\projects\newschecker - with rag\ml-service"; .\.venv\Scripts\python.exe -m pytest -q
cd "C:\projects\newschecker - with rag\server"; npm test
```
