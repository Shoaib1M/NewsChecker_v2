"""
FILE PURPOSE:
This is the main entry point for the Python ML Service.
It runs a FastAPI web server that exposes the machine learning model to the Node.js backend.

FLOW:
1. `lifespan`: When the server starts, it loads the trained MLP model and TF-IDF vectorizer from disk into memory.
2. Registers two endpoints: `/api/health` and `/api/check`.
3. When `/api/check` is hit:
   - It returns the legacy MLP score as an experimental claim prior.
   - It extracts atomic claims, retrieves evidence, and uses NLI to compare
     each claim with relevant passages.
   - It produces a verdict only when enough classified evidence is available.

USED BY:
- The Node.js Express server (`server/routes/check.js`) sends POST requests here.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# PATH CONFIGURATION
# Make sure sibling modules (binary_truth_mlp, evidence_scraper, tfidf, etc.)
# are importable when this file is executed.
# ---------------------------------------------------------------------------
SERVICE_DIR = Path(__file__).resolve().parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from binary_truth_mlp import (
    MODEL_FILE,
    explain_probability,
    load_artifacts,
    make_prediction_features,
)
from claim_normalizer import normalize_claim
from claim_recency import resolve_mode
from claim_triage import triage_claim
from claim_verifier import extract_claims
from evidence_aggregator import assess_coverage, count_independent_groups
import explainer
from evidence_pipeline import (
    run_pipeline,
    passage_ranking_report,
    EvidenceResult as PipelineEvidenceResult,
)
from knowledge_verifier import assess_claim
from nli_service import get_nli_service
from passage_retriever import get_passage_ranker


def _env_flag(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_env_file():
    """Load .env files to populate API keys into os.environ."""
    env_paths = [
        SERVICE_DIR / ".env",
        SERVICE_DIR.parent / ".env",
    ]
    for path in env_paths:
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key, value = key.strip(), value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value

# ---------------------------------------------------------------------------
# GLOBAL STATE
# We load the model once when the server starts and keep it in memory.
# Loading a model on every single request would be incredibly slow.
# ---------------------------------------------------------------------------
_model = None
_vectorizer = None
_train_max_values = None

# Hard ceiling on the evidence phase of a single /api/check request, shared
# across all extracted claims. Kept well under the Node proxy's own timeout
# (ML_SERVICE_TIMEOUT_MS) so a slow provider surfaces as partial evidence
# with honest diagnostics rather than a dead request.
EVIDENCE_BUDGET_SECONDS = float(os.getenv("EVIDENCE_BUDGET_SECONDS", "45"))

# Retrieval outcomes ordered worst to best. A multi-claim statement takes the
# WORST status across its claims, because the status gates absence-of-coverage
# reasoning: reporting "no credible source reports this" on the strength of
# one claim's successful search, while another claim's search failed, asserts
# something about the world that was never checked.
#
# Previously the loop simply overwrote the status each iteration, so a
# three-claim statement reported whatever happened to the last claim.
_RETRIEVAL_SEVERITY = {
    "SEARCH_FAILED": 0,
    "NO_RESULTS": 1,
    "NO_RELEVANT_RESULTS": 2,
    "SEARCH_PARTIAL": 3,
    "SEARCH_SUCCESS": 4,
}


def _worst_retrieval_status(statuses: list[str]) -> str:
    """The most pessimistic retrieval outcome among several claims."""
    if not statuses:
        return "NO_RESULTS"
    return min(statuses, key=lambda s: _RETRIEVAL_SEVERITY.get(s, 0))


"""
PURPOSE:
This function runs exactly once when the FastAPI server starts up.

WHY THIS EXISTS:
Neural network files can be hundreds of megabytes. We use this startup block to 
read the `.pkl` file from the hard drive into RAM so it's ready to instantly serve requests.
"""
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _vectorizer, _train_max_values

    # Step 1: Locate the trained model file
    model_path = SERVICE_DIR / "binary_truth_mlp.pkl"
    if not model_path.exists():
        model_path = SERVICE_DIR / "saved_models" / "binary_truth_mlp.pkl"

    if not model_path.exists():
        raise RuntimeError(
            f"Cannot find the trained model. Looked in:\n"
            f"  {SERVICE_DIR / 'binary_truth_mlp.pkl'}\n"
            f"  {SERVICE_DIR / 'saved_models' / 'binary_truth_mlp.pkl'}"
        )

    # Step 2: Load the model, vectorizer, and scaling values
    print(f"Loading model from {model_path} …")
    _model, _vectorizer, _train_max_values = load_artifacts(model_path)
    print(
        f"Model loaded — input_size={_model.input_size}, "
        f"threshold={_model.best_threshold:.2f}"
    )

    # Step 3: Load API keys for the web scraper
    load_env_file()

    # Step 4: Warm the NLI model.
    # Loading it lazily meant the first evidence-requiring request paid for
    # the model download (hundreds of MB on a cold cache) inside the HTTP
    # request — unbounded by the evidence budget, and silent, because
    # nothing logs until the load finishes. Doing it here makes that cost
    # visible at boot and keeps /api/health honest before traffic arrives.
    if _env_flag("NLI_PRELOAD", default=True):
        nli = get_nli_service()
        if nli.status["enabled"]:
            print(
                f"Preloading NLI model ({nli.model_name}) — "
                f"the first run downloads it, which can take a few minutes …"
            )
            state = nli.warm_up()
            if state["status"] == "ready":
                print(f"NLI model ready: {nli.model_name}")
            else:
                # Not fatal: the service still runs and abstains rather than
                # guessing, and /api/health reports exactly why.
                print(f"NLI unavailable ({state['status']}): {state['error']}")

        # The passage embedder, for the same reason. Much smaller (~90 MB),
        # but its first use would otherwise also be a download inside a
        # request. Skipped entirely when SEMANTIC_PASSAGES is off.
        ranker = get_passage_ranker()
        if ranker.status["enabled"]:
            print(f"Preloading passage embedding model ({ranker.model_name}) …")
            state = ranker.warm_up()
            if state["status"] != "ready":
                # Not fatal either: passage selection falls back to lexical.
                print(f"Dense passage ranking unavailable ({state['status']}): "
                      f"{state['error']}")

    # Yield hands control back to FastAPI to start accepting requests
    yield

    # Code below here runs when the server shuts down
    print("Shutting down ML service.")


# ---------------------------------------------------------------------------
# FASTAPI APPLICATION SETUP
# ---------------------------------------------------------------------------
app = FastAPI(
    title="NewsChecker ML Service",
    description="Evidence-first fact checking with NLI-backed claim verification.",
    version="2.0.0",
    lifespan=lifespan, # Attach our startup function
)

# Enable CORS so our frontend/backend can communicate with this service
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS (Pydantic)
# These define exactly what the JSON data coming IN and going OUT should look like.
# FastAPI will automatically validate incoming requests against these.
# ---------------------------------------------------------------------------
class CheckRequest(BaseModel):
    statement: str = Field(..., min_length=5, max_length=2000)
    # "recent" restricts retrieval to the last month and refuses to let older
    # articles confirm the claim; "historical" searches without a date limit.
    # "auto" reads the claim's own wording, which is a convenience rather than
    # a substitute: "the PM resigned" is a claim about today for someone who
    # just saw it on television, and no parser can tell.
    #
    # Deliberately NOT pattern-constrained. `resolve_mode()` already falls back
    # to "auto" for anything it does not recognise, and a regex here made that
    # fallback unreachable — an unknown value rejected the whole fact-check
    # with a raw Pydantic 422 instead. This is a search hint with a safe
    # default; failing an entire check over a malformed hint is the wrong
    # trade, and it disagreed with the Express proxy, which coerces. Values
    # outside {auto, recent, historical} are normalised, not refused.
    mode: str = Field(
        default="auto",
        description="auto | recent | historical (anything else is read as auto)",
    )


# ── Nested response models ───────────────────────────────────────────
class MLInfo(BaseModel):
    """Legacy MLP signal — advisory only, never drives the verdict."""
    available: bool = True
    auxiliary_only: bool = True
    score: float = 0.0
    verdict: str = ""
    threshold: float = 0.5


class RetrievalInfo(BaseModel):
    """What happened during the search phase."""
    status: str = "NO_RESULTS"
    candidate_count: int = 0
    relevant_count: int = 0
    diagnostics: list[dict] = Field(default_factory=list)
    # Dense passage ranking: the embedder's status (same shape as `nli`) plus
    # how many classified documents it actually ranked. Empty when nothing
    # was searched.
    passage_ranking: dict = Field(default_factory=dict)


class NLIInfo(BaseModel):
    """NLI model state and classification outcome."""
    available: bool = False
    status: str = "disabled"
    classified_count: int = 0


class EvidenceSummary(BaseModel):
    """Aggregated evidence after NLI classification."""
    supporting_count: int = 0
    contradicting_count: int = 0
    neutral_count: int = 0
    # Distinct publishers across *all* classified evidence — a measure of how
    # broadly the search actually reached.
    independent_groups: int = 0
    # Distinct publishers on each side. These are what back a verdict: four
    # copies of one wire story are one confirmation, and only these counts
    # can tell you that.
    independent_supporting: int = 0
    independent_contradicting: int = 0


class VerificationInfo(BaseModel):
    """The core verification outcome."""
    status: str = "insufficient_evidence"
    reasoning: str = ""
    # What kind of proposition this is, decided before retrieval. Exposed so
    # a caller can tell "we could not verify this" apart from "there is
    # nothing here to verify" — the pipeline used to report both identically.
    claim_kind: str = "checkable"
    # "high" when a true version of the claim would necessarily have been
    # reported. Only high-salience claims can reach unsupported_no_coverage.
    salience: str = "normal"


class EvidenceItem(BaseModel):
    title: str
    url: str
    similarity: float
    stance: str
    source: str
    best_sentence: str
    support_score: float
    contradiction_score: float
    source_tier: str = "unclassified"
    nli_available: bool = False
    # Set when the pipeline overrode what the NLI scores implied — currently
    # only when the article states a different figure from the claim's. Empty
    # for every source whose stance came straight from the scores.
    stance_note: str = ""
    # ISO 8601, or None when the provider gave no date (Wikipedia, DuckDuckGo).
    # Shown on the card: a date next to a claim about today is the fastest way
    # for a reader to see why a source did or did not count.
    published: str | None = None
    # Who actually published this, resolved from the aggregator link where
    # needed. The UI shows this rather than the raw URL host, which for a
    # Google News link would read "news.google.com" for every source.
    publisher: str = ""


class ClaimAssessment(BaseModel):
    claim: str
    status: str
    verdict: str
    support: float
    contradiction: float
    evidence_count: int


class CoverageMode(BaseModel):
    """Which slice of coverage was searched, and why."""
    mode: str = "historical"          # recent | historical
    requested: str = "auto"           # what the caller asked for
    window_days: int | None = None    # days of coverage searched, None = no limit
    reason: str = ""
    stale_evidence_count: int = 0     # found, but too old to be reporting it


class ExplanationSentence(BaseModel):
    """One sentence the LLM wrote, and what the faithfulness check made of it."""
    text: str
    citations: list[int] = Field(default_factory=list)   # 1-based top_evidence positions
    kept: bool = False
    entailment: float = 0.0                              # best NLI entailment vs. its cited source(s)
    drop_reason: str = ""


class ExplanationInfo(BaseModel):
    """An LLM explanation of the verdict, filtered sentence by sentence by NLI.

    Explains a verdict; never sets one. ``text`` holds only sentences whose
    cited source entails them, and is empty whenever ``available`` is false.
    """
    available: bool = False
    reason: str = "not generated"
    text: str = ""
    sentences: list[ExplanationSentence] = Field(default_factory=list)
    dropped_count: int = 0
    model: str = ""


class CheckResponse(BaseModel):
    statement: str
    coverage: CoverageMode = Field(default_factory=CoverageMode)
    claim_type: str = "general factual"
    verdict: str = "insufficient evidence"
    confidence: str = "low"

    verification: VerificationInfo = Field(default_factory=VerificationInfo)
    ml: MLInfo = Field(default_factory=MLInfo)
    retrieval: RetrievalInfo = Field(default_factory=RetrievalInfo)
    nli: NLIInfo = Field(default_factory=NLIInfo)
    evidence: EvidenceSummary = Field(default_factory=EvidenceSummary)

    # Legacy fields kept for backward compatibility with the frontend
    ml_score: float = 0.0
    ml_verdict: str = ""
    ml_threshold: float = 0.5
    evidence_score: float = 0.0
    evidence_stance: dict = Field(default_factory=dict)
    combined_score: int = 50
    combined_verdict: str = "insufficient evidence"
    assessment_status: str = "insufficient_evidence"
    claim_assessments: list[ClaimAssessment] = Field(default_factory=list)
    top_evidence: list[EvidenceItem] = Field(default_factory=list)
    processing_time_seconds: float = 0.0
    reasoning: str = ""
    external_evidence_available: bool = False
    external_evidence_checked: bool = False
    explanation: ExplanationInfo = Field(default_factory=ExplanationInfo)


# ---------------------------------------------------------------------------
# SCORING HELPERS
# ---------------------------------------------------------------------------
# One canonical phrase per outcome. The whole point of the rewrite is that
# these are genuinely different answers — "we couldn't check" and "nobody
# reports this" used to be the same string, which is what made a fabricated
# headline look identical to a broken search.
_STATUS_VERDICTS: dict[str, str] = {
    "supported": "evidence supports the claim",
    "contradicted": "evidence contradicts the claim",
    "mixed": "claims have mixed evidence",
    "reported_plan": "reported as planned — not yet done",
    "unsupported_no_coverage": "no credible source reports this",
    "not_verifiable_yet": "not yet verifiable — describes a future event",
    "not_a_claim": "no verifiable claim found",
    "not_objectively_verifiable": "subjective — not objectively verifiable",
    "insufficient_evidence": "insufficient evidence",
}

# Outcomes where the 0-100 evidence-balance dial is meaningless and must not
# be drawn as a number. Each of these is a statement about the *claim*, not a
# measurement of evidence weight.
NON_NUMERIC_STATUSES = frozenset({
    "insufficient_evidence", "not_a_claim", "not_objectively_verifiable",
    "not_verifiable_yet", "unsupported_no_coverage",
    # reported_plan too: the dial reads "evidence balance" for the claim, and
    # a claim about a future event has none. The evidence measures how well
    # the *announcement* is attested, so showing 90 beside "reported as
    # planned — not yet done" invites reading it as 90% true.
    "reported_plan",
})


def evidence_verdict_score(stance: dict) -> int:
    """A visual evidence balance only; it is not a probability of truth.

    The legacy MLP is returned for transparency but is intentionally excluded:
    it was trained on historical US political statements and is not a reliable
    universal-news classifier.
    """
    if stance.get("status") in NON_NUMERIC_STATUSES:
        return 50
    net = max(-1.0, min(1.0, float(stance.get("net", 0.0))))
    return int(round(max(5, min(95, 50 + 45 * net))))


# Statuses that describe the *claim* rather than the evidence for it. They
# come from triage or the deterministic knowledge check, always as the single
# summary for the whole submission, and must pass through the multi-claim
# merge untouched — folding them into the support/contradiction averages
# below would turn "this is a question, not a claim" into "contradicted".
_NON_EVIDENTIAL_STATUSES = frozenset({
    "not_a_claim", "not_objectively_verifiable",
})


def merge_claim_summaries(summaries: list[dict]) -> dict:
    """Use a conservative overall status for multi-claim submissions."""
    if len(summaries) == 1 and summaries[0].get("status") in _NON_EVIDENTIAL_STATUSES:
        return dict(summaries[0])

    assessed = [
        summary for summary in summaries
        if summary.get("status") not in _NON_EVIDENTIAL_STATUSES
        and summary.get("status") != "insufficient_evidence"
    ]
    if not assessed:
        return {
            "support": 0.0, "contradiction": 0.0, "net": 0.0,
            "verdict": "insufficient evidence", "status": "insufficient_evidence",
            "nli_available": any(summary.get("nli_available") for summary in summaries),
            "evidence_count": 0,
            "supporting_count": sum(s.get("supporting_count", 0) for s in summaries),
            "contradicting_count": sum(s.get("contradicting_count", 0) for s in summaries),
            "neutral_count": sum(s.get("neutral_count", 0) for s in summaries),
        }
    support = sum(summary["support"] for summary in assessed) / len(assessed)
    contradiction = sum(summary["contradiction"] for summary in assessed) / len(assessed)
    net = support - contradiction
    statuses = {summary["status"] for summary in assessed}
    if len(statuses) > 1 or "mixed" in statuses:
        status, verdict = "mixed", "claims have mixed evidence"
    elif "supported" in statuses:
        status, verdict = "supported", "evidence supports the claim"
    else:
        status, verdict = "contradicted", "evidence contradicts the claim"
    return {
        "support": support, "contradiction": contradiction, "net": net,
        "verdict": verdict, "status": status, "nli_available": True,
        "evidence_count": sum(summary.get("evidence_count", 0) for summary in assessed),
        # These must survive the merge. assess_coverage decides whether an
        # absence of support is a finding by reading them, so dropping them
        # here silently reported "nobody supports this claim" for claims that
        # had supporting evidence.
        "supporting_count": sum(s.get("supporting_count", 0) for s in summaries),
        "contradicting_count": sum(s.get("contradicting_count", 0) for s in summaries),
        "neutral_count": sum(s.get("neutral_count", 0) for s in summaries),
        # Weakest link, not the sum: a multi-claim statement is only as well
        # sourced as its least-supported claim, and summing across claims
        # would also double-count a publisher that covered several of them.
        "independent_supporting": min(
            (s.get("independent_supporting", 0) for s in assessed), default=0
        ),
        "independent_contradicting": min(
            (s.get("independent_contradicting", 0) for s in assessed), default=0
        ),
    }


def _compute_confidence(
    status: str,
    independent_sources: int,
    nli_available: bool,
    candidate_count: int = 0,
) -> str:
    """Categorical confidence — avoids fake-precision percentages.

    Scaled by the number of *independent publishers* backing the verdict, not
    by how many articles were classified. Those differ exactly when it
    matters: a wire story carried by four outlets under one masthead used to
    read as four confirmations and earn "high" confidence. One story is one
    confirmation however many times it is reprinted.

    For ``unsupported_no_coverage`` the confidence instead comes from how much
    of the press we actually looked at — the finding *is* that nothing was
    classified as supporting, so counting supporting sources would be
    circular. A wide search that came back empty is a stronger negative than
    a narrow one.
    """
    if status in {"not_a_claim", "not_objectively_verifiable"}:
        return "high"  # nothing uncertain about "there is no claim here"
    if status == "unsupported_no_coverage":
        if candidate_count >= 15:
            return "high"
        return "medium" if candidate_count >= 8 else "low"
    if status in {"insufficient_evidence", "not_verifiable_yet"}:
        return "low"
    if not nli_available:
        return "low"
    if independent_sources >= 3:
        return "high"
    if independent_sources >= 2:
        return "medium"
    return "low"


def _build_reasoning(
    triage,
    status: str,
    searched: bool,
    candidate_count: int,
    relevant_count: int,
    classified_count: int,
    independent_groups: int,
    retrieval_status: str,
    nli_status: str,
) -> str:
    """Explain, in plain English, how this verdict was reached.

    This is the field the user actually reads, so it states what was searched
    and what was found rather than restating the verdict. The pipeline used
    to emit one of two fixed sentences here regardless of what happened,
    which is why a correct abstention was indistinguishable from a bug.
    """
    if not searched:
        return triage.reason

    # How much of a search actually happened, phrased the same way each time
    # so two results are comparable.
    scope = (
        f"Searched {candidate_count} source"
        f"{'' if candidate_count == 1 else 's'} across the configured news, "
        f"reference and web providers"
    )
    if relevant_count:
        scope += (
            f"; {relevant_count} discussed this claim and "
            f"{classified_count} " +
            ("was" if classified_count == 1 else "were") +
            " compared against it by the NLI model"
        )
    scope += "."

    if retrieval_status == "SEARCH_FAILED":
        return (
            "No verdict: every search provider failed or timed out, so nothing "
            "was retrieved to check this against. This is a retrieval failure "
            "on our side, not a finding about the claim."
        )

    if nli_status not in {"ready", "loading"} and status not in {
        "not_a_claim", "not_objectively_verifiable"
    }:
        return (
            f"{scope} No verdict: the NLI model is unavailable ({nli_status}), "
            "so retrieved sources could not be compared against the claim. "
            "Sources are shown as candidates only."
        )

    if status == "unsupported_no_coverage":
        subject = "a plan of this kind" if triage.is_prospective else "an event of this kind"
        return (
            f"{scope} Coverage of the subjects in this claim was found, but no "
            f"source reports what it asserts. {subject.capitalize()} would be "
            "reported widely if it were real, so the absence of any coverage "
            "is itself evidence against the claim — note this is different "
            "from a failed search, which is reported separately."
        )

    if status == "reported_plan":
        return (
            f"{scope} Sources report this as announced or planned. That "
            "confirms the plan was reported — it does not confirm that it has "
            "happened or will happen."
        )

    if status == "not_verifiable_yet":
        return (
            f"{scope} {triage.reason} Nothing in the retrieved coverage "
            "reports it as announced either way."
        )

    if status in {"supported", "contradicted", "mixed"}:
        independence = (
            f" from {independent_groups} independent publisher"
            f"{'' if independent_groups == 1 else 's'}"
            if independent_groups else ""
        )
        direction = {
            "supported": "support", "contradicted": "contradict",
            "mixed": "point both ways on",
        }[status]
        # Count the sources that take the verdict's direction, not every
        # classified source: saying "5 sources support this claim" when three
        # of the five were neutral is the same overstatement the Related
        # coverage split exists to prevent.
        return (
            f"{scope} Sources{independence} {direction} this claim."
        )

    if relevant_count == 0 and candidate_count:
        return (
            f"{scope} None of the retrieved sources discussed what this claim "
            "asserts closely enough to count as evidence either way, so no "
            "verdict is given."
        )

    return (
        f"{scope} Nothing retrieved was strong enough to support or contradict "
        "this claim, so no verdict is given."
    )


def _providers_answered(diagnostics: list[dict]) -> int:
    """Distinct providers whose query completed, whether or not it found anything.

    This is what makes silence mean something. A provider reporting
    `no_results` looked and found nothing; a provider reporting `failed` is
    blind, and counting it would turn an outage into a finding about the
    world. Counted per provider rather than per query, because the same
    provider answering four queries is still one view of the press.
    """
    return len({
        diagnostic.get("provider")
        for diagnostic in diagnostics or []
        if diagnostic.get("status") in {"success", "no_results"}
    })


def _independent_backing(stance: dict) -> int:
    """Distinct publishers backing the verdict's direction.

    For a mixed verdict both sides matter, so take the larger; for a
    directional one only that side counts toward confidence.
    """
    supporting = stance.get("independent_supporting", 0)
    contradicting = stance.get("independent_contradicting", 0)
    if stance.get("status") == "contradicted":
        return contradicting
    if stance.get("status") == "supported":
        return supporting
    return max(supporting, contradicting)


def _count_evidence_by_stance(evidence_list: list) -> dict:
    """Count supporting/contradicting/neutral from classified evidence."""
    supporting = sum(1 for e in evidence_list if e.nli_available and e.stance == "supports")
    contradicting = sum(1 for e in evidence_list if e.nli_available and e.stance == "contradicts")
    neutral = sum(1 for e in evidence_list if e.nli_available and e.stance in {"mixed", "unclear"})
    return {"supporting": supporting, "contradicting": contradicting, "neutral": neutral}


# ---------------------------------------------------------------------------
# API ENDPOINTS
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Render's default probe endpoint."""
    return {"status": "ok", "service": "newschecker-ml"}


@app.get("/api/health")
async def health():
    """Comprehensive health check — reports live state of every subsystem."""
    nli_svc = get_nli_service()

    # Search provider readiness. Keyed providers report whether their key is
    # present; keyless ones report whether they have been switched off. This
    # is the first thing to check when results look thin — an unconfigured
    # checkout retrieves less, and that must be visible rather than inferred.
    providers = {
        name: {
            "enabled": bool(os.getenv(key)),
            "status": "ready" if os.getenv(key) else "no_key",
            "requires_key": True,
        }
        for name, key in (
            ("gnews", "GNEWS_API_KEY"),
            ("guardian", "GUARDIAN_API_KEY"),
            ("newsapi", "NEWSAPI_KEY"),
        )
    }
    for name, flag in (
        ("google_news", "GOOGLE_NEWS_ENABLED"),
        ("wikipedia", "WIKIPEDIA_ENABLED"),
        ("duckduckgo", "DUCKDUCKGO_ENABLED"),
    ):
        on = _env_flag(flag, default=True)
        providers[name] = {
            "enabled": on,
            "status": "ready" if on else "disabled",
            "requires_key": False,
        }

    return {
        "status": "ok",
        "service": "newschecker-ml",
        "model_loaded": _model is not None,
        "input_size": _model.input_size if _model else None,
        "threshold": _model.best_threshold if _model else None,
        "nli": nli_svc.status,
        "passage_ranking": get_passage_ranker().status,
        "explanations": explainer.status(),
        "search_providers": providers,
    }


"""
PURPOSE:
The primary fact-checking logic.

FLOW:
1. Validates the legacy MLP is loaded and returns its claim-only prior.
2. Extracts atomic claims, retrieves candidate evidence, and runs NLI.
3. Abstains if evidence is unavailable, weak, or from unclassified sources.
4. Returns evidence, claim-level outcomes, and a conservative overall status.
"""
@app.post("/api/check", response_model=CheckResponse)
def check_statement(request: CheckRequest):
    # Deliberately a sync `def`, not `async def`: this handler does blocking
    # work throughout (urllib search/article fetches, NumPy and PyTorch
    # inference). On an `async def` handler that work runs directly on the
    # event loop and freezes the entire service for its duration — including
    # /api/health. As a sync def, FastAPI runs it in its threadpool instead.
    if _model is None:
        # If the server is still booting up and loading the massive .pkl file
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.time()
    submitted = request.statement.strip()

    # --- 0. Normalise the submission into the proposition it contains ---
    # People submit what they saw, with the framing they saw it in: "is it
    # true that X?", "BREAKING: X!!!", a pasted URL in front of X, a headline
    # in quotes with "- Reuters, March 2024" after it. Every stage below reads
    # the claim — triage, entity extraction, query generation, and NLI, which
    # uses it as the hypothesis — so the packaging reached all of them. "Is it
    # true that the prime minister of India resigned?" was triaged as
    # `not_a_claim` and never searched: the most natural way to ask a
    # fact-checker a question, answered with "no verifiable claim found".
    #
    # `submitted` stays the user's own words. It is what the response echoes,
    # what the UI shows and what history stores; only the machinery sees the
    # normalised form.
    statement = normalize_claim(submitted)
    # The ORIGINAL text, not just the normalised claim: normalisation
    # strips the wire label off the front, and that label is the strongest
    # thing a pasted headline says about when the event happened.
    claim_time = resolve_mode(statement, request.mode, submitted=submitted)

    # --- 1. ML Prediction Phase ---
    # Convert text into a numerical array (TF-IDF features)
    features = make_prediction_features(
        vectorizer=_vectorizer,
        train_max_values=_train_max_values,
        statement=statement,
    )
    # Ask the Neural Network for its prediction (returns a 0 to 1 probability)
    ml_score = float(_model.predict_proba(features)[0])
    ml_verdict = explain_probability(ml_score)
    knowledge_assessment = assess_claim(statement)

    # --- 2. Triage: what kind of question does this claim even pose? ---
    # Done before any network work. Sending "is the earth flat?" or a string
    # of keyboard mash through four search queries and an NLI model wastes
    # the request budget and — worse — dresses the result up as a failed
    # verification rather than saying there is nothing here to verify.
    triage = triage_claim(statement)

    # --- 3. Claim extraction and evidence phase ---
    # A long submission may contain several independently checkable claims.
    claims = extract_claims(statement, max_claims=3)
    claim_summaries = []
    all_evidence = []
    retrieval_status = "NO_RESULTS"
    retrieval_diagnostics = []
    candidate_count = 0
    relevant_count = 0
    searched = False
    passage_documents = 0
    hybrid_documents = 0

    if knowledge_assessment:
        status = knowledge_assessment["status"]
        support = 1.0 if status == "supported" else 0.0
        contradiction = 1.0 if status == "contradicted" else 0.0
        claim_summaries = [(statement, {
            "support": support, "contradiction": contradiction,
            "net": support - contradiction,
            "verdict": knowledge_assessment["verdict"],
            "status": status, "nli_available": False, "evidence_count": 0,
        })]
    elif not triage.search_worthwhile:
        # Nothing external could settle this — a question, a fragment, a
        # value judgment. Report that plainly instead of running a search
        # whose emptiness would then be misread as a verification failure.
        status = "not_objectively_verifiable" if triage.kind == "opinion" else "not_a_claim"
        claim_summaries = [(statement, {
            "support": 0.0, "contradiction": 0.0, "net": 0.0,
            "verdict": _STATUS_VERDICTS[status], "status": status,
            "nli_available": False, "evidence_count": 0,
        })]
    else:
        searched = True
        # One budget shared across every claim, so a multi-claim statement
        # can't multiply the worst case by the number of claims.
        pipeline_deadline = time.monotonic() + EVIDENCE_BUDGET_SECONDS
        claim_statuses: list[str] = []
        try:
            for claim in claims:
                outcome = run_pipeline(
                    claim, max_results=8, fetch_articles=True,
                    deadline=pipeline_deadline, claim_time=claim_time,
                )
                claim_summaries.append((claim, outcome.stance))
                all_evidence.extend(outcome.evidence)
                # Accumulate, never overwrite: the status gates absence
                # reasoning for the whole statement, and the diagnostics are
                # how a thin result is traced back to a provider.
                claim_statuses.append(outcome.retrieval_status)
                retrieval_diagnostics.extend(outcome.diagnostics)
                retrieval_status = _worst_retrieval_status(claim_statuses)
                candidate_count += outcome.candidate_count
                relevant_count += outcome.relevant_count
                ranking = outcome.passage_ranking or {}
                passage_documents += ranking.get("documents", 0)
                hybrid_documents += ranking.get("hybrid_documents", 0)
        except Exception as err:
            print(f"Evidence pipeline failed: {err}")
            claim_summaries = [(
                claim,
                {
                    "support": 0.0, "contradiction": 0.0, "net": 0.0,
                    "verdict": "insufficient evidence", "status": "insufficient_evidence",
                    "nli_available": False, "evidence_count": 0,
                },
            ) for claim in claims]
            all_evidence = []
            retrieval_status = "SEARCH_FAILED"

    ev_stance = merge_claim_summaries([summary for _, summary in claim_summaries])

    # --- 4. Reinterpret the aggregate in light of the triage -----------
    # Two adjustments, both about not conflating different kinds of silence:
    #
    #   a) Absence of coverage. A search that ran correctly across several
    #      providers and turned up nothing supporting a claim whose true
    #      version would have been front-page news is a finding, not a
    #      failure. It becomes "no credible source reports this" — never
    #      "true" or "false", and never when the search itself broke.
    #
    #   b) Future events. Nothing can make a prospective claim true today.
    #      Supporting coverage means the plan was *reported*, not that it
    #      happened; the absence of it means no such plan is on record.
    nli_svc = get_nli_service()
    if searched:
        override = assess_coverage(
            ev_stance,
            retrieval_status=retrieval_status,
            candidate_count=candidate_count,
            salience=triage.salience,
            providers_answered=_providers_answered(retrieval_diagnostics),
            prospective=triage.is_prospective,
            nli_ready=nli_svc.is_available,
            negated=triage.negated,
        )
        if override is not None:
            ev_stance = override
        elif triage.is_prospective:
            if ev_stance["status"] == "supported":
                ev_stance = {**ev_stance, "status": "reported_plan",
                             "verdict": _STATUS_VERDICTS["reported_plan"]}
            elif ev_stance["status"] == "insufficient_evidence":
                ev_stance = {**ev_stance, "status": "not_verifiable_yet",
                             "verdict": _STATUS_VERDICTS["not_verifiable_yet"]}

    # This is the amount of strong, classified evidence available, not a
    # similarity score and not a probability that the claim is true.
    ev_score = min(
        1.0,
        (ev_stance["support"] + ev_stance["contradiction"])
        * min(ev_stance.get("evidence_count", 0) / 2, 1.0),
    )
    combined = evidence_verdict_score(ev_stance)
    combined_verdict = (
        knowledge_assessment["verdict"] if knowledge_assessment else ev_stance["verdict"]
    )

    # --- 5. Build top evidence list ---
    top_evidence = []
    seen_urls = set()
    ranked_evidence = sorted(
        all_evidence,
        key=lambda result: max(result.support_score, result.contradiction_score) * result.source_weight,
        reverse=True,
    )
    for result in ranked_evidence:
        if result.url in seen_urls:
            continue
        seen_urls.add(result.url)
        top_evidence.append(EvidenceItem(
            title=result.title or "",
            url=result.url or "",
            similarity=round(result.similarity, 3),
            stance=result.stance or "unclear",
            source=result.source or "",
            best_sentence=result.best_sentence or "",
            support_score=round(result.support_score, 3),
            contradiction_score=round(result.contradiction_score, 3),
            source_tier=result.source_tier,
            nli_available=result.nli_available,
            stance_note=getattr(result, "stance_note", "") or "",
            published=(
                result.published.isoformat()
                if getattr(result, "published", None) else None
            ),
            publisher=result.publisher or "",
        ))
        if len(top_evidence) == 8:
            break

    # --- 6. Compute evidence counts ---
    evidence_counts = _count_evidence_by_stance(all_evidence)
    nli_classified = sum(1 for e in all_evidence if e.nli_available)

    confidence = (
        knowledge_assessment["confidence"] if knowledge_assessment
        else _compute_confidence(
            ev_stance["status"],
            _independent_backing(ev_stance),
            ev_stance.get("nli_available", False),
            candidate_count=candidate_count,
        )
    )

    reasoning = (
        knowledge_assessment["reasoning"] if knowledge_assessment
        else _build_reasoning(
            triage=triage,
            status=ev_stance["status"],
            searched=searched,
            candidate_count=candidate_count,
            relevant_count=relevant_count,
            classified_count=nli_classified,
            independent_groups=_independent_backing(ev_stance),
            retrieval_status=retrieval_status,
            nli_status=nli_svc.status["status"],
        )
    )

    # --- 7. Explain the verdict (cannot change it) ---
    # Runs last, after every verdict field above is final, and its result is
    # only ever placed in `explanation`. It has its own short timeout outside
    # EVIDENCE_BUDGET_SECONDS, and any failure is an abstention, never an
    # error: a check never fails because its explanation did.
    if searched and not knowledge_assessment:
        try:
            explanation = explainer.explain(
                statement, ev_stance["status"], top_evidence,
                claim_type=triage.claim_type,
            )
        except Exception as err:  # noqa: BLE001
            # explain() is written never to raise; this is the guarantee that
            # an unforeseen bug in it still cannot turn a finished check into
            # a 500. The verdict above is already computed and stands alone.
            print(f"Explanation failed: {err}")
            explanation = explainer.Explanation(
                available=False, reason="explanation failed unexpectedly",
                model=explainer.explain_model(),
            )
    else:
        explanation = explainer.Explanation(
            available=False,
            reason="nothing was searched, so there is no evidence to explain",
            model=explainer.explain_model(),
        )

    elapsed = round(time.time() - start, 2)

    # --- 5. Build response ---
    return CheckResponse(
        statement=submitted,
        coverage=CoverageMode(
            mode=claim_time.mode,
            requested=request.mode,
            window_days=claim_time.window_days,
            reason=claim_time.reason,
            stale_evidence_count=ev_stance.get("stale_evidence_count", 0),
        ),
        claim_type=(
            knowledge_assessment["claim_type"] if knowledge_assessment
            else triage.claim_type
        ),
        verdict=combined_verdict,
        confidence=confidence,

        verification=VerificationInfo(
            status=ev_stance["status"],
            reasoning=reasoning,
            claim_kind=triage.kind,
            salience=triage.salience,
        ),
        ml=MLInfo(
            available=True,
            auxiliary_only=True,
            score=round(ml_score, 4),
            verdict=ml_verdict,
            threshold=round(_model.best_threshold, 4),
        ),
        retrieval=RetrievalInfo(
            status=retrieval_status,
            candidate_count=candidate_count,
            relevant_count=relevant_count,
            diagnostics=retrieval_diagnostics,
            passage_ranking=(
                passage_ranking_report(passage_documents, hybrid_documents)
                if searched else {}
            ),
        ),
        nli=NLIInfo(
            available=nli_svc.is_ready or nli_svc.is_available,
            status=nli_svc.status["status"],
            classified_count=nli_classified,
        ),
        evidence=EvidenceSummary(
            supporting_count=evidence_counts["supporting"],
            contradicting_count=evidence_counts["contradicting"],
            neutral_count=evidence_counts["neutral"],
            independent_groups=count_independent_groups(all_evidence),
            independent_supporting=ev_stance.get("independent_supporting", 0),
            independent_contradicting=ev_stance.get("independent_contradicting", 0),
        ),

        # Legacy fields for backward compatibility
        ml_score=round(ml_score, 4),
        ml_verdict=ml_verdict,
        ml_threshold=round(_model.best_threshold, 4),
        evidence_score=round(ev_score, 4),
        evidence_stance={
            **ev_stance,
            "retrieval_status": retrieval_status,
            "retrieval_diagnostics": retrieval_diagnostics,
            "candidate_count": candidate_count,
            "relevant_source_count": relevant_count,
            "evidence_used_count": ev_stance.get("evidence_count", 0),
        },
        combined_score=combined,
        combined_verdict=combined_verdict,
        assessment_status=ev_stance["status"],
        claim_assessments=[
            ClaimAssessment(
                claim=claim,
                status=summary["status"],
                verdict=summary["verdict"],
                support=round(summary["support"], 4),
                contradiction=round(summary["contradiction"], 4),
                evidence_count=summary.get("evidence_count", 0),
            )
            for claim, summary in claim_summaries
        ],
        top_evidence=top_evidence,
        processing_time_seconds=elapsed,
        reasoning=reasoning,
        external_evidence_available=any(
            result.nli_available and result.stance in {"supports", "contradicts", "mixed"}
            for result in all_evidence
        ),
        external_evidence_checked=searched,
        explanation=ExplanationInfo(**explanation.to_dict()),
    )


# ---------------------------------------------------------------------------
# LOCAL DEVELOPMENT RUNNER
# Run `python main.py` to start the server on port 8000 locally.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
