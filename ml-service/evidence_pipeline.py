"""Evidence pipeline — the clean replacement for evidence_scraper's collect_evidence.

Pipeline stages:
  1. Query generation (via query_generator.py)
  2. Multi-provider search with diagnostics (via providers/registry.py)
  3. Deduplication (via providers/registry.py)
  4. Relevance filtering (via relevance_filter.py)
  5. Article extraction + passage selection (via article_extractor.py)
  6. NLI classification (via nli_service.py)
  7. Source quality classification (via claim_verifier.py)
  8. Evidence aggregation (via evidence_aggregator.py)

A source only becomes "evidence" after step 6.  Before that it is a "candidate".
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import NamedTuple

from article_extractor import (
    extract_article,
    extract_passages,
    strip_boilerplate,
)
from claim_recency import (
    ClaimTime,
    describe_staleness,
    document_is_stale,
    resolve_mode,
)
from claim_verifier import classify_source, resolve_publisher_host
from evidence_aggregator import ClassifiedEvidence, compute_stance
from nli_service import get_nli_service
from passage_retriever import get_passage_ranker
from numeric_consistency import conflicting_quantity
from providers import SearchResult
from providers.registry import search_all_providers
from query_generator import QueryGenerator
from relevance_filter import RelevanceFilter


@dataclass
class EvidenceResult:
    """A source that has passed through the complete pipeline."""
    url: str
    title: str
    snippet: str
    similarity: float
    text_length: int
    provider: str
    source: str
    support_score: float = 0.0
    contradiction_score: float = 0.0
    stance: str = "unclear"
    best_sentence: str = ""
    source_tier: str = "unclassified"
    source_weight: float = 0.0
    nli_available: bool = False
    # Why the stance is not what the NLI scores alone would imply. Empty
    # whenever the scores were taken at face value.
    stance_note: str = ""
    # When the article was published, or None when the provider gave no date.
    published: datetime | None = None
    # Host of whoever actually published the article. Differs from the URL's
    # host for aggregator links (Google News), and it is this value — not the
    # link — that decides source tier and counts independent groups.
    publisher: str = ""


class PipelineOutcome(NamedTuple):
    """Everything the API needs from one claim's evidence pipeline run."""
    stance: dict
    evidence: list[EvidenceResult]
    retrieval_status: str
    diagnostics: list[dict]
    candidate_count: int
    relevant_count: int
    # How passages were chosen for NLI: the dense ranker's status plus how
    # many documents were actually ranked hybrid. Reported like the provider
    # diagnostics, because "the embedding model was down" and "it ran and
    # changed nothing" look identical in the verdict and must not in the logs.
    passage_ranking: dict | None = None


# ── Singletons ───────────────────────────────────────────────────────
_query_generator = QueryGenerator()
_relevance_filter = RelevanceFilter()


DEFAULT_BUDGET_SECONDS = 45.0
# Share of the budget retrieval may consume before article extraction starts.
_SEARCH_BUDGET_SHARE = 0.5

# Below this many words, a candidate's text is a headline and a sentence or
# two, and the page is fetched for the real body. It was 30, which sat just
# under the ~31-word excerpt the news APIs return — so the best-quality
# providers were the ones whose articles were never fetched, and NLI judged
# them on a two-line stub. Judging a claim needs a paragraph or two.
MIN_WORDS_WITHOUT_FETCH = 120

# Slots reserved, out of `max_results`, for candidates from a classified
# source — primary, fact-check, reporting or reference.
#
# Only `max_results` documents are ever NLI-classified, and they were chosen by
# lexical relevance alone. For a viral false claim that is exactly backwards:
# the posts repeating the claim use its precise wording, while the debunkings
# describe the situation in their own. Measured on a realistic pool for "The
# United States banned Google across all its cities", eight rumour blogs scored
# 0.78-0.94 and a PolitiFact fact-check scored 0.735 — so the fact-check ranked
# NINTH and never reached NLI. The system would have classified eight copies of
# the rumour and reported the claim supported.
#
# Reserving slots rather than adding a score bonus keeps relevance ranking
# untouched: a credible source still has to pass the relevance filter to be a
# candidate at all, and the reserved seats are filled in relevance order. There
# is no magic constant weighing "authority" against "aboutness" — the rule is
# simply that if credible sources were found, some of them get read.
RESERVED_TIER_SLOTS = 3

# A passage must reach this entailment/contradiction score before it counts as
# taking a side at all.
STANCE_THRESHOLD = 0.35

# When BOTH directions clear that threshold, the document argues both ways.
# One side must be this many times stronger to be called the document's
# position; otherwise the honest label is "unclear". Without it, 0.88 against
# 0.72 read as a confident "supports".
STANCE_DOMINANCE = 1.6


def decide_stance(
    support_score: float,
    contradiction_score: float,
    threshold: float = STANCE_THRESHOLD,
    dominance: float = STANCE_DOMINANCE,
) -> str:
    """Turn a document's two NLI scores into the position it takes.

    A document that both entails and contradicts the claim somewhere is not
    evidence for either side — it is a document discussing the dispute, and
    the honest label is "unclear". Requiring a margin is what makes that
    possible: without one, 0.80 against 0.75 read as a confident "supports".

    The thresholds are parameters rather than constants read from the module
    so that `stance_sweep.py` measures THIS rule at different settings instead
    of a reimplementation of it. Three separate rules in this codebase have
    drifted from their copies; a sweep against a copy would report on a
    decision procedure the system does not use.
    """
    supports = support_score > threshold
    contradicts = contradiction_score > threshold
    if supports and contradicts:
        # Both directions present: one must clearly dominate.
        if support_score >= contradiction_score * dominance:
            return "supports"
        if contradiction_score >= support_score * dominance:
            return "contradicts"
        return "unclear"
    if supports:
        return "supports"
    if contradicts:
        return "contradicts"
    return "unclear"

# Passages that REPORT a claim rather than assert it. Debunking articles are
# built out of these — "Posts claim the US banned Google in all its cities" —
# and an NLI model scores them as strongly entailing the claim, because the
# claim is right there in the sentence. Nothing in the passage says it is
# true; the article exists to say the opposite.
#
# Deliberately narrow: it matches the frames misinformation coverage uses, not
# ordinary attribution. "Officials said the minister resigned" is a newspaper
# reporting a fact and must keep counting as evidence.
_CLAIM_REPORTING_FRAME = re.compile(
    r"\b(?:posts?|users?|rumou?rs?|memes?|videos?|tweets?)\s+(?:that\s+)?"
    r"(?:claim|claims|claimed|allege|alleges|alleged|say|says|said|suggest)\b"
    r"|\b(?:social media|viral|circulating|widely shared|misleading|"
    r"unfounded|baseless|debunk\w*|fact[- ]check\w*|false claim)\b"
    r"|\bclaims?\s+(?:that\s+)?(?:have|has)\s+(?:been\s+)?circulat",
    re.IGNORECASE,
)


def _select_for_classification(included: list[dict], max_results: int) -> list[dict]:
    """Choose which relevant candidates get spent on NLI.

    Relevance order, except that up to ``RESERVED_TIER_SLOTS`` places are held
    for candidates from a classified source, so a crowd of near-identical
    low-tier posts cannot push every credible source past the cut. See
    ``RESERVED_TIER_SLOTS`` for the measurement behind this.

    With no classified sources among the candidates this returns exactly the
    same list as a plain relevance-ordered slice.
    """
    if len(included) <= max_results:
        return included

    tiered = [
        doc for doc in included
        if classify_source(doc["url"], doc.get("source", "")).weight > 0
    ]
    reserved = tiered[:min(RESERVED_TIER_SLOTS, max_results)]
    reserved_urls = {doc["url"] for doc in reserved}

    remaining = max_results - len(reserved)
    filler = [doc for doc in included if doc["url"] not in reserved_urls][:remaining]

    chosen_urls = reserved_urls | {doc["url"] for doc in filler}
    # Return in the original relevance order so downstream ranking is stable.
    return [doc for doc in included if doc["url"] in chosen_urls]


def run_pipeline(
    claim: str,
    max_results: int = 8,
    fetch_articles: bool = True,
    deadline: float | None = None,
    claim_time: ClaimTime | None = None,
) -> PipelineOutcome:
    """Execute the complete evidence pipeline for a single claim.

    ``deadline`` is an absolute ``time.monotonic()`` value bounding the whole
    run. Every network stage checks it, so a blocked search provider or a
    stalling news site degrades this to partial results instead of running
    until the caller's HTTP timeout fires. Defaults to
    ``DEFAULT_BUDGET_SECONDS`` from now.

    Returns a PipelineOutcome containing the stance summary, classified
    evidence, retrieval status, and per-provider diagnostics.
    """
    if deadline is None:
        deadline = time.monotonic() + DEFAULT_BUDGET_SECONDS

    # ── Stage 1: Query generation ────────────────────────────────────
    query_variants = _query_generator.generate_queries(claim)
    queries = [q["query"] for q in query_variants[:4]]
    if not queries:
        queries = [claim]

    # ── Stage 2: Multi-provider search ───────────────────────────────
    # Cap search at half the budget so extraction and NLI still get a turn.
    search_deadline = min(
        deadline,
        time.monotonic() + (deadline - time.monotonic()) * _SEARCH_BUDGET_SHARE,
    )
    # In recent mode every provider that supports a date filter is asked for
    # the last month only, and Wikipedia sits the round out. Without this the
    # feeds are recency-RANKED but not recency-FILTERED, so a query about
    # today's story still returns last year's coverage of the same subject
    # whenever that older coverage happens to match the words better.
    claim_time = claim_time or resolve_mode(claim)
    raw_results, diagnostics = search_all_providers(
        queries, deadline=search_deadline, recent_days=claim_time.window_days,
    )
    diagnostics_dicts = [d.to_dict() for d in diagnostics]
    candidate_count = len(raw_results)

    if not raw_results:
        # Determine if providers failed or simply returned nothing
        any_success = any(d.status == "success" for d in diagnostics)
        status = "NO_RESULTS" if any_success else "SEARCH_FAILED"
        return PipelineOutcome(
            stance=compute_stance([]),
            evidence=[],
            retrieval_status=status,
            diagnostics=diagnostics_dicts,
            candidate_count=0,
            relevant_count=0,
        )

    # ── Stage 3: Relevance filtering ─────────────────────────────────
    documents = [
        {
            "url": r.url,
            "title": r.title,
            "snippet": r.snippet,
            "text": r.text or r.snippet,
            "source": r.source,
            "provider": r.provider,
            "published": r.published,
        }
        for r in raw_results
    ]
    included, _excluded = _relevance_filter.filter_documents(
        claim, documents, strict=True
    )
    relevant_count = len(included)

    if not included:
        return PipelineOutcome(
            stance=compute_stance([]),
            evidence=[],
            retrieval_status="NO_RELEVANT_RESULTS",
            diagnostics=diagnostics_dicts,
            candidate_count=candidate_count,
            relevant_count=0,
        )

    # ── Stage 4: Article extraction ──────────────────────────────────
    # Fetched concurrently: search-result snippets are usually too short to
    # judge, so nearly every candidate needs its page pulled. Doing that one
    # at a time meant a handful of stalling news sites serialized into
    # minutes of wall time.
    selected = _select_for_classification(included, max_results)
    fetched_articles: dict[str, tuple[str, str]] = {}

    if fetch_articles:
        needs_fetch = [
            d for d in selected
            if len((d.get("text") or "").split()) < MIN_WORDS_WITHOUT_FETCH
        ]
        if needs_fetch and time.monotonic() < deadline:
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    (d["url"], pool.submit(extract_article, d["url"], 6))
                    for d in needs_fetch
                ]
                for url, future in futures:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        future.cancel()
                        continue
                    try:
                        fetched_articles[url] = future.result(timeout=remaining)
                    except Exception:
                        pass  # Fall back to the snippet we already have

    nli_service = get_nli_service()
    evidence_results: list[EvidenceResult] = []
    hybrid_documents = 0
    # Documents that were about the claim but too old to be reporting it.
    # This is the difference between 'nobody reported this' and 'nobody
    # reported it THIS MONTH', and the absence verdict must not confuse them.
    stale_count = 0

    for doc in selected:
        url = doc["url"]
        title = doc.get("title", "")
        snippet = doc.get("snippet", "")
        full_text = doc.get("text", "")
        publisher_name = doc.get("source", "")
        source_profile = classify_source(url, publisher_name)
        publisher_host = resolve_publisher_host(url, publisher_name)

        fetched_title, fetched_text = fetched_articles.get(url, ("", ""))
        # Compare what each version actually SAYS, not how long it is. A
        # paywalled page ships a few hundred words of subscription pitch and
        # none of the story, which outweighed the one real sentence a provider
        # snippet had given us — so the fetch replaced the only usable text in
        # the document with marketing copy.
        if fetched_text:
            fetched_content = strip_boilerplate(fetched_text, claim)
            existing_content = strip_boilerplate(full_text or "", claim)
            if len(fetched_content.split()) > len(existing_content.split()):
                full_text = fetched_text
        if fetched_title and not title:
            title = fetched_title

        # ── Stage 5: NLI classification ──────────────────────────────
        # The claim steers passage selection: NLI only sees what this
        # returns, so the sentences that mention the claim's subject matter
        # must not be crowded out by the article's opening paragraphs.
        ranking_info: dict = {}
        passages = extract_passages(
            title, snippet, full_text, claim=claim, ranking_info=ranking_info,
        )
        if ranking_info.get("ranking") == "hybrid":
            hybrid_documents += 1
        support_score = 0.0
        contradiction_score = 0.0
        best_sentence = ""
        nli_available = False

        if passages and nli_service.is_available:
            nli_scores = nli_service.score_many(claim, passages)
            # Availability is per passage, so it must be read per passage.
            # Testing only nli_scores[0] discarded a document that had been
            # classified cleanly seven times over because the first passage
            # happened to fail — and, in the other direction, let unavailable
            # entries (which report 0.0/0.0/1.0) into the score comparison.
            usable = [i for i, score in enumerate(nli_scores) if score.get("available")]
            if usable:
                nli_available = True
                # The strongest entailment and the strongest contradiction are
                # taken INDEPENDENTLY, across all passages.
                #
                # Reading both scores off whichever single passage had the
                # highest max inverted fact-checks. A debunking article quotes
                # the claim it is refuting — "Posts claim the US banned Google
                # in all its cities" entails at 0.88 — and then refutes it —
                # "This is false; no such ban exists" contradicts at 0.80. The
                # quote scored higher, so that passage was chosen, and its
                # near-zero contradiction score was read off with it. The
                # article was recorded as SUPPORTING the claim it exists to
                # debunk, at the highest source weight in the system.
                # A passage that merely reports the claim cannot count as the
                # article endorsing it, so it is excluded from the entailment
                # maximum. It stays eligible for contradiction: an article
                # saying the claim is false is refuting it either way.
                assertive = [
                    i for i in usable
                    if not _CLAIM_REPORTING_FRAME.search(passages[i])
                ] or usable

                support_idx = max(assertive, key=lambda i: nli_scores[i]["entailment"])
                contradiction_idx = max(
                    usable, key=lambda i: nli_scores[i]["contradiction"]
                )
                support_score = nli_scores[support_idx]["entailment"]
                contradiction_score = nli_scores[contradiction_idx]["contradiction"]
                best_idx = (
                    support_idx if support_score >= contradiction_score
                    else contradiction_idx
                )
                best_sentence = passages[best_idx] if best_idx < len(passages) else ""

        stance = decide_stance(support_score, contradiction_score) if nli_available \
            else "unclear"

        # A document that states a DIFFERENT figure cannot be the source that
        # confirms the claim's, however strongly it entails on wording. This
        # only withdraws support — see numeric_consistency for why it never
        # converts one into a contradiction.
        stance_note = ""
        if stance == "supports":
            conflict = conflicting_quantity(claim, passages)
            if conflict:
                claimed, stated = conflict
                stance = "unclear"
                stance_note = (
                    f"states {stated.text} where the claim says {claimed.text}"
                )

        # Entailment has no clock. "India's prime minister resigned on Tuesday"
        # entails "the prime minister resigned this morning" perfectly — and if
        # that article is from 2014 the entailment is a coincidence, not a
        # confirmation. Support is withdrawn, never turned into a
        # contradiction: an old article is not evidence that today's event did
        # not happen, it is just not evidence that it did.
        published = doc.get("published")
        if stance == "supports" and document_is_stale(claim_time, published):
            stance = "unclear"
            stance_note = describe_staleness(published)
            stale_count += 1

        evidence_results.append(EvidenceResult(
            url=url,
            title=title,
            snippet=snippet,
            similarity=0.0,  # TF-IDF similarity not used as a score
            text_length=len(full_text or ""),
            provider=doc.get("provider", "unknown"),
            source=doc.get("source", ""),
            support_score=support_score,
            contradiction_score=contradiction_score,
            stance=stance,
            best_sentence=best_sentence,
            source_tier=source_profile.tier,
            source_weight=source_profile.weight,
            nli_available=nli_available,
            publisher=publisher_host,
            stance_note=stance_note,
            published=published,
        ))

    # ── Stage 6: Evidence aggregation ────────────────────────────────
    classified = [
        ClassifiedEvidence(
            url=e.url,
            publisher=e.publisher,
            source=e.source,
            source_tier=e.source_tier,
            source_weight=e.source_weight,
            support_score=e.support_score,
            contradiction_score=e.contradiction_score,
            nli_available=e.nli_available,
            stance=e.stance,
        )
        for e in evidence_results
    ]
    stance = compute_stance(classified)

    # Determine overall retrieval status
    any_success = any(d.status == "success" for d in diagnostics)
    all_success = all(
        d.status in {"success", "no_results"} or not d.enabled
        for d in diagnostics
    )
    if all_success:
        retrieval_status = "SEARCH_SUCCESS"
    elif any_success:
        retrieval_status = "SEARCH_PARTIAL"
    else:
        retrieval_status = "SEARCH_FAILED"

    # Add retrieval metadata to stance
    stance["retrieval_status"] = retrieval_status
    stance["retrieval_diagnostics"] = diagnostics_dicts
    stance["candidate_count"] = candidate_count
    # "Nobody reports this" and "nobody reports it THIS MONTH" are different
    # findings, and only the first is a statement about the world.
    stance["stale_evidence_count"] = stale_count
    stance["mode"] = claim_time.mode
    stance["relevant_source_count"] = relevant_count

    return PipelineOutcome(
        stance=stance,
        evidence=evidence_results,
        retrieval_status=retrieval_status,
        diagnostics=diagnostics_dicts,
        candidate_count=candidate_count,
        relevant_count=relevant_count,
        passage_ranking=passage_ranking_report(len(selected), hybrid_documents),
    )


def passage_ranking_report(documents: int = 0, hybrid_documents: int = 0) -> dict:
    """The dense ranker's live status, plus what it did for this claim.

    ``hybrid_documents < documents`` is normal: an article whose sentences are
    all below the similarity floor falls back to lexical ranking by design.
    """
    return {
        **get_passage_ranker().status,
        "documents": documents,
        "hybrid_documents": hybrid_documents,
    }
