# NewsChecker — demo guide

Everything needed to record a clean 2.5-minute demo. NewsChecker runs locally
(DeBERTa + PyTorch need 1 GB+ RAM), so the demo is a recording, not a link.

## 1. Before you record

```powershell
# From the repo root. Starts all three services in their own windows,
# waits for both models, and sends one warm-up check.
powershell -ExecutionPolicy Bypass -File .\scripts\demo_warmup.ps1
```

Wait for `Ready.` — the first start after a reboot downloads nothing (models
are cached) but still takes ~30–60 s to load them.

### Recording checklist

- [ ] `demo_warmup.ps1` printed `Ready.`, with NLI and passage ranking `ready`
- [ ] **No `.env` file, API key or terminal showing a key is on screen.** Close
      editors with `.env` open; the ML service window is safe (it never prints keys)
- [ ] Browser zoom at **125%**, window maximised, bookmarks bar hidden
- [ ] History panel cleared (sign out, or delete old checks)
- [ ] Windows notifications off (Focus assist → Alarms only), Slack/Discord closed
- [ ] Run each demo claim once off-camera first — news changes daily, and the
      second run of a claim is faster
- [ ] A terminal ready at `ml-service\` for the pytest shot
- [ ] `docs/benchmarks/` results table open in the README for the benchmark shot

## 2. Demo claims

News changes daily, so **run each claim once off-camera on the day** and swap
in the backup if the verdict is not the expected one.

| # | Claim | Expected | Why this one |
|---|---|---|---|
| 1 | *Tropical Storm Isaias forms; forecast to strengthen over next few days* | `supported` | Came back supported in every benchmark run on 2026-10-07. Backup: any headline from today's Google News front page. |
| 2 | *Morning crash did not shut down Route 209* | `contradicted` | Contradicted by the WGAL and PennLive reports themselves (with the recommended NLI model). Backup: *5G mobile networks spread the coronavirus*. |
| 3 | *Pineapple is the best pizza topping* | `not_objectively_verifiable` | Triage answers it without searching — instant and deterministic. Backup: *India will win the 2030 FIFA World Cup* → `not_verifiable_yet`. |

Do **not** run `news_benchmark.py` on demo day: it exhausts the free GNews and
NewsAPI daily quotas, and thin retrieval looks like a broken fact-checker.

## 3. Script (2:30)

| Time | On screen | Say |
|---|---|---|
| 0:00–0:15 | Home page | "Most fact-checkers either guess from the wording or ask an LLM, which can sound sure and be wrong. NewsChecker does neither: it finds evidence, reads it with an NLI model, and abstains when the evidence isn't there." |
| 0:15–0:45 | Claim 1 (supported) | Paste it, run. "It searched live news, kept the articles that are actually about this event, and an NLI model read the passages. Three independent publishers entail the claim — so: supported, with the sources right here." Click a `[n]` citation in the explanation so the card highlights. |
| 0:45–1:10 | Claim 2 (contradicted) | "This one is viral and false. Note the verdict comes from credible sources contradicting it — not from counting how many posts repeat it." Point at the fact-check card and its key passage. |
| 1:10–1:25 | Claim 3 (subjective / future) | "And when there's nothing to verify, it says so instead of inventing a verdict. No search was run." |
| 1:25–1:55 | Back on claim 1/2: open *How this was checked*; scroll explanation | "Under the hood: claim triage, multi-query search across providers, relevance filtering, then passage selection — now hybrid: sentence embeddings fused with keyword overlap, so a debunk written in different words still gets read. DeBERTa scores each passage. The explanation is written by Gemini *after* the verdict, and every sentence is checked by the same NLI model against the source it cites — anything not entailed is dropped. The LLM can't change the verdict; there's a test for exactly that." |
| 1:55–2:10 | Terminal: `.\.venv\Scripts\python.exe -m pytest -q` | "500+ tests, no network, no model downloads — including one where the LLM argues the opposite verdict and nothing changes." |
| 2:10–2:22 | README benchmark table | "I measured hybrid retrieval against the old ranking on today's headlines and corrupted versions of them. No setting moved the wrong-answer rate beyond run-to-run noise — I report that as a null result. But inspecting each wrong answer found two real defects, which I fixed and measured." |
| 2:22–2:30 | README *Known limitations* | "Biggest limitation: retrieval depends on what free news sources return today. Next step: a labelled set of real misinformation, so the benchmark isn't built from corrupted headlines." |

## 4. If something goes wrong on camera

- **Verdict is `insufficient evidence` for a claim that worked before** — a
  provider is rate-limited. Run `.\.venv\Scripts\python.exe check_providers.py`
  and swap to the backup claim.
- **No explanation appears** — Gemini's free tier returns 503 under load; the
  fallback model usually answers. If neither does, the verdict and evidence are
  unaffected, which is itself worth saying: "the explanation is optional by
  design."
