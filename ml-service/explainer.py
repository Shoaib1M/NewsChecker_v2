"""Retrieval-augmented, NLI-checked explanations of a verdict.

WHAT THIS DOES:
After the evidence pipeline has decided a verdict, an LLM (Gemini Flash)
writes two to four sentences explaining why the classified evidence leads to
it, citing each source as [n]. Every sentence it writes is then checked by
the same NLI model that classified the evidence: a sentence survives only if
the source it cites actually entails it. What reaches the user is the
surviving sentences — or nothing.

THE ONE RULE THIS MODULE IS BUILT AROUND:
The LLM never decides or changes a verdict. It is handed a status that is
already final, it returns a dict that nothing downstream reads back, and
``main.py`` calls it after every verdict field has been computed. If the
LLM is down, rate-limited, slow, wrong or malicious, the worst outcome is
that no explanation is shown. That property is pinned by
``tests/test_explainer.py`` rather than left to code review.

WHY A FAITHFULNESS FILTER AND NOT JUST A CAREFUL PROMPT:
A prompt saying "use only the evidence" lowers the rate of invented facts;
it does not make it zero, and a fact-checker that occasionally adds a
plausible unsourced sentence under its verdict is worse than one that
explains nothing. The filter turns "the LLM was asked to be faithful" into
"every displayed sentence was entailed by the source it cites, according to
a model that does not share the LLM's failure modes". The cost is that true
but awkwardly-phrased sentences are sometimes dropped — the same trade the
rest of the pipeline makes: a missing sentence is a shortfall, an
unsupported one is a failure.

WHICH EVIDENCE THE LLM SEES:
Only classified sources taking the verdict's side — supporting for
``supported`` / ``reported_plan``, contradicting for ``contradicted``, both
for ``mixed``. Neutral coverage addresses neither side and has nothing to
explain. Opposite-side sources are withheld from a directional verdict on
purpose: a sentence like "posts report the US banned Google [3]" is
faithful to its source, would pass the filter, and reads as an argument
against the verdict printed above it.

Each source is given to the LLM as exactly the text NLI judged decisive
(``best_sentence``), and the explanation is checked against exactly that
text. The LLM is never shown anything the filter cannot check it against.

CITATION NUMBERS:
[n] is the source's 1-based position in ``top_evidence`` — the list the UI
renders as evidence cards — so a citation can link straight to its card.
Numbers can therefore have gaps ([1], [3]) when neutral sources sit between
them; that is cosmetic, and keeping one numbering everywhere is worth it.
"""

from __future__ import annotations

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Callable, Sequence

from article_extractor import protect_abbreviations, restore_abbreviations
from evidence_pipeline import decide_stance
from nli_service import get_nli_service

DEFAULT_MODEL = "gemini-3.8-flash"
# Tried, in order, only when the model before it errors or times out. Free-tier
# Flash models return 503 "high demand" for minutes at a time — measured on
# 2026-10-07, gemini-3.8-flash failed four calls in a row while
# gemini-3.5-flash-lite answered in 1.1s. Without a fallback the explanation
# would be missing exactly when someone is watching. Set
# EXPLAIN_FALLBACK_MODELS= (empty) to use the primary model only.
DEFAULT_FALLBACK_MODELS = "gemini-3.5-flash-lite"
# Per attempt. The explanation runs after the evidence budget, so this is
# extra latency on every explained check, bounded at
# timeout × (1 + number of fallbacks).
DEFAULT_TIMEOUT_SECONDS = 10.0
# The Gemini API rejects any request deadline below 10s with a 400
# ("Manually set deadline 8s is too short"), so the deadline SENT to it is
# clamped up to this. A shorter EXPLAIN_TIMEOUT_SECONDS is still honoured by
# the wall-clock limit in generate_text — the request is just abandoned
# locally instead of being refused remotely.
GEMINI_MIN_DEADLINE_SECONDS = 10.0

# The explanation is a few sentences over a handful of sources. More sources
# than this make the prompt longer without making the explanation better,
# and every one is an NLI call when its sentences are checked.
MAX_SOURCES = 6

# Statuses with no classified evidence to explain. The triage and
# deterministic paths never searched; insufficient_evidence and
# not_verifiable_yet found nothing that takes a side; unsupported_no_coverage
# is a finding about ABSENCE, so there is no source to cite for it.
SKIP_STATUSES = frozenset({
    "not_a_claim",
    "not_objectively_verifiable",
    "insufficient_evidence",
    "not_verifiable_yet",
    "unsupported_no_coverage",
})
SKIP_CLAIM_TYPES = frozenset({"unsupported language"})

# Which classified stances an explanation of each verdict may draw on.
_SIDES = {
    "supported": {"supports"},
    "reported_plan": {"supports"},
    "contradicted": {"contradicts"},
    "mixed": {"supports", "contradicts"},
}

_STATUS_WORDS = {
    "supported": "the evidence SUPPORTS the claim",
    "reported_plan": "the evidence reports the claimed plan as ANNOUNCED, not as done",
    "contradicted": "the evidence CONTRADICTS the claim",
    "mixed": "credible evidence points BOTH WAYS on the claim",
}


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def explanations_enabled() -> bool:
    """Read at call time, so tests and the benchmark can flip it."""
    return _flag("EXPLANATIONS_ENABLED", default=True)


def explain_model() -> str:
    return os.getenv("EXPLAIN_MODEL", DEFAULT_MODEL)


def explain_models() -> list[str]:
    """Primary model first, then the fallbacks, without duplicates."""
    fallbacks = os.getenv("EXPLAIN_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS)
    models = [explain_model()] + [m.strip() for m in fallbacks.split(",") if m.strip()]
    return list(dict.fromkeys(models))


def explain_timeout() -> float:
    try:
        return float(os.getenv("EXPLAIN_TIMEOUT_SECONDS", DEFAULT_TIMEOUT_SECONDS))
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def status() -> dict:
    """Explanation-layer readiness for ``/api/health``. Never includes the key."""
    enabled = explanations_enabled()
    has_key = bool(os.getenv("GOOGLE_API_KEY"))
    if not enabled:
        state, error = "disabled", "explanations disabled via EXPLANATIONS_ENABLED"
    elif not has_key:
        state, error = "no_key", "GOOGLE_API_KEY not set"
    else:
        state, error = "ready", None
    return {"enabled": enabled, "model": explain_model(),
            "fallback_models": explain_models()[1:], "status": state, "error": error}


# ── Input ────────────────────────────────────────────────────────────

@dataclass
class Source:
    """One classified source the explanation may cite."""
    number: int            # 1-based position in top_evidence
    text: str              # the passage NLI judged decisive
    publisher: str
    stance: str            # supports | contradicts
    tier: str


def select_sources(status: str, top_evidence: Sequence) -> list[Source]:
    """The classified, verdict-side sources an explanation may cite.

    ``top_evidence`` items are read by attribute (``EvidenceItem``) or key
    (a plain dict), so the function is testable without the API models.
    """
    sides = _SIDES.get(status, set())

    def get(item, name, default=""):
        if isinstance(item, dict):
            return item.get(name, default)
        return getattr(item, name, default)

    sources: list[Source] = []
    for position, item in enumerate(top_evidence, start=1):
        if not get(item, "nli_available", False):
            continue
        stance = get(item, "stance", "")
        if stance not in sides:
            continue
        text = (get(item, "best_sentence", "") or get(item, "title", "")).strip()
        if not text:
            continue
        sources.append(Source(
            number=position,
            text=text,
            publisher=get(item, "publisher", "") or get(item, "source", "") or "unknown",
            stance=stance,
            tier=get(item, "source_tier", "unclassified"),
        ))
        if len(sources) >= MAX_SOURCES:
            break
    return sources


# ── Prompt ───────────────────────────────────────────────────────────

def build_prompt(claim: str, status: str, sources: Sequence[Source]) -> str:
    """The instruction sent to the LLM.

    The verdict is stated as settled, and the model is told it is not being
    asked for an opinion on it. Restating or hedging the verdict is
    forbidden for the same reason the verdict is computed elsewhere: two
    statements of the outcome that can disagree are worse than one.
    """
    lines = [
        f"[{s.number}] ({s.publisher}; {s.tier}; {s.stance} the claim) {s.text}"
        for s in sources
    ]
    # The format example uses a number that really is in the list: a fixed
    # "[2]" invited citing a source the model was never shown.
    example = sources[0].number if sources else 1
    return (
        "You explain fact-check verdicts that have ALREADY been decided by an "
        "evidence pipeline. You are not asked whether the verdict is right.\n\n"
        f"Claim: {claim}\n"
        f"Decided verdict: {_STATUS_WORDS.get(status, status)}.\n\n"
        "Evidence:\n" + "\n".join(lines) + "\n\n"
        "Write 2 to 4 sentences explaining why this evidence leads to the "
        "decided verdict.\n"
        "Rules:\n"
        "- Use ONLY the numbered evidence above. Do not add any fact, number, "
        "date, name or background that is not stated in it.\n"
        "- End EVERY sentence with the citation of the evidence that states "
        f"it, in square brackets before the full stop, e.g. \"... [{example}].\" "
        "Cite only numbers listed above.\n"
        "- Do not restate, soften, hedge or contradict the verdict, and do not "
        "say whether the claim is true or false in your own words.\n"
        "- Plain prose only: no lists, headings, markdown or preamble."
    )


# ── LLM call ─────────────────────────────────────────────────────────

def _call_gemini(prompt: str, model: str, api_key: str, timeout: float) -> str:
    """One deterministic Gemini call. Raises on any failure.

    The seam tests patch. The SDK's own HTTP timeout is in milliseconds and
    bounds a single attempt; ``generate_text`` adds a wall-clock bound around
    the whole call, retries included.
    """
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=int(max(timeout, GEMINI_MIN_DEADLINE_SECONDS) * 1000),
        ),
    )
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=1024,
            # No tools are declared, so there is nothing to call; disabling
            # it explicitly also silences the SDK's per-call AFC warning.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    return (response.text or "").strip()


_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="explainer")


def generate_text(prompt: str, model: str, api_key: str, timeout: float) -> str:
    """``_call_gemini`` under a hard wall-clock limit."""
    future = _pool.submit(_call_gemini, prompt, model, api_key, timeout)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout:
        future.cancel()
        raise TimeoutError(f"explanation model did not answer within {timeout:.0f}s")


# ── Faithfulness filter ──────────────────────────────────────────────

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
# A citation placed after the full stop ("…exists. [2]") belongs to the
# sentence before it. Moved inside first, or the split hands it to the next.
_TRAILING_CITATION_RE = re.compile(r"([.!?])((?:\s*\[\d+(?:\s*,\s*\d+)*\])+)")
# Digits included: "Two sources agree [1]. 3 outlets report X [2]." was one
# "sentence" checked against both sources jointly.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_explanation(text: str) -> list[str]:
    """Sentences of the LLM's output, each keeping its own citations."""
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if not cleaned:
        return []
    cleaned = _TRAILING_CITATION_RE.sub(lambda m: m.group(2) + m.group(1), cleaned)
    protected = protect_abbreviations(cleaned)
    return [
        restore_abbreviations(part).strip()
        for part in _SENTENCE_SPLIT_RE.split(protected)
        if part.strip()
    ]


def citations_in(sentence: str) -> list[int]:
    numbers: list[int] = []
    for group in _CITATION_RE.findall(sentence):
        for piece in group.split(","):
            number = int(piece.strip())
            if number not in numbers:
                numbers.append(number)
    return numbers


def _without_citations(sentence: str) -> str:
    stripped = _CITATION_RE.sub("", sentence)
    return re.sub(r"\s+([.!?,;:])", r"\1", re.sub(r"\s{2,}", " ", stripped)).strip()


class NLIUnavailable(RuntimeError):
    """NLI could not score a sentence; the explanation must not be shown."""


def check_sentence(sentence: str, sources_by_number: dict[int, Source]) -> dict:
    """Score one explanation sentence against the sources it cites.

    NLI direction matters and is easy to get backwards. ``score_many(claim,
    passages)`` sends each PASSAGE as the premise and its first argument as
    the hypothesis — so here the SENTENCE goes first (hypothesis) and the
    cited evidence goes in the list (premise). The question asked is "does
    the source imply this sentence?", which is the one that matters.

    A sentence citing several sources is checked against each alone and
    against all of them together: "Reuters [1] and AP [2] both report X" is
    entailed by either, while "X [1], and later Y [2]" is only entailed by
    the two read together. Either way the premise is nothing but cited text.

    Kept only if ``decide_stance`` — the same rule that classifies the
    evidence itself — would call the strongest premise "supports".
    """
    citations = citations_in(sentence)
    result = {
        "text": sentence, "citations": citations, "kept": False,
        "entailment": 0.0, "drop_reason": "",
    }
    if not citations:
        result["drop_reason"] = "no citation"
        return result
    if any(number not in sources_by_number for number in citations):
        result["drop_reason"] = "cites a source that was not provided"
        return result
    hypothesis = _without_citations(sentence)
    if len(hypothesis.split()) < 3:
        result["drop_reason"] = "nothing to check"
        return result

    premises = [sources_by_number[n].text for n in citations]
    if len(premises) > 1:
        premises.append(" ".join(premises))

    scores = get_nli_service().score_many(hypothesis, premises)
    if not scores or not all(score.get("available") for score in scores):
        raise NLIUnavailable("NLI could not score an explanation sentence")

    best = max(scores, key=lambda score: score["entailment"])
    result["entailment"] = round(float(best["entailment"]), 3)
    if decide_stance(best["entailment"], best["contradiction"]) == "supports":
        result["kept"] = True
    else:
        result["drop_reason"] = "not entailed by the cited source"
    return result


# ── Entry point ──────────────────────────────────────────────────────

@dataclass
class Explanation:
    available: bool = False
    reason: str = ""
    text: str = ""
    sentences: list[dict] = field(default_factory=list)
    dropped_count: int = 0
    model: str = ""

    def to_dict(self) -> dict:
        return {
            "available": self.available, "reason": self.reason,
            "text": self.text, "sentences": self.sentences,
            "dropped_count": self.dropped_count, "model": self.model,
        }


def explain(
    claim: str,
    status: str,
    top_evidence: Sequence,
    claim_type: str = "",
    generate: Callable[[str, str, str, float], str] | None = None,
) -> Explanation:
    """Explain an already-final verdict, or say plainly why there is none.

    Never raises: every failure becomes ``available: False`` with a reason,
    the same contract the NLI service and the dense ranker follow.
    ``status`` is read, never returned — callers cannot get a verdict back
    out of this function because it does not produce one.
    """
    model = explain_model()
    unavailable = lambda reason: Explanation(available=False, reason=reason, model=model)  # noqa: E731

    if not explanations_enabled():
        return unavailable("explanations disabled via EXPLANATIONS_ENABLED")
    if status in SKIP_STATUSES or claim_type in SKIP_CLAIM_TYPES:
        return unavailable(f"no classified evidence to explain for status '{status}'")
    if status not in _SIDES:
        return unavailable(f"no explanation defined for status '{status}'")

    sources = select_sources(status, top_evidence)
    if not sources:
        return unavailable("no classified source on the verdict's side")

    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        return unavailable("GOOGLE_API_KEY not set")
    # Checked before spending an LLM call: without NLI nothing it writes
    # could be shown anyway.
    if not get_nli_service().is_available:
        return unavailable("NLI unavailable — explanations cannot be checked")

    started = time.monotonic()
    prompt = build_prompt(claim, status, sources)
    text, failures = "", []
    for candidate in explain_models():
        try:
            text = (generate or generate_text)(prompt, candidate, api_key, explain_timeout())
        except Exception as exc:  # noqa: BLE001 - any LLM failure is an abstention
            message = str(exc).replace(api_key, "<redacted>")
            failures.append(f"{candidate}: {message[:120]}")
            continue
        if text:
            model = candidate
            break
        failures.append(f"{candidate}: returned no text")
    if not text:
        return unavailable("explanation model failed — " + "; ".join(failures))

    by_number = {source.number: source for source in sources}
    try:
        checked = [check_sentence(s, by_number) for s in split_explanation(text)]
    except NLIUnavailable as exc:
        return unavailable(str(exc))

    kept = [entry["text"] for entry in checked if entry["kept"]]
    dropped = sum(1 for entry in checked if not entry["kept"])
    print(f"Explanation: kept {len(kept)}/{len(checked)} sentences "
          f"in {time.monotonic() - started:.1f}s ({model})")
    if not kept:
        return Explanation(
            available=False,
            reason="no sentence survived the faithfulness check",
            sentences=checked, dropped_count=dropped, model=model,
        )
    return Explanation(
        available=True, reason="", text=" ".join(kept),
        sentences=checked, dropped_count=dropped, model=model,
    )
