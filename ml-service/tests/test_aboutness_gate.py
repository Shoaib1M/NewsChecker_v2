"""A document that is not about the claim cannot support or contradict it.

WHY THIS EXISTS:
Measured live on 2026-10-07: for "Morning crash shuts down Route 209", the
weak query "Route shuts" pulled a Guardian walking guide through relevance
filtering, the NLI model scored one of its sentences at 1.00 contradiction,
and at reporting-tier weight it outvoted the local news confirming the crash.
The claim's negation was "contradicted" by the same articles. Both answers
cannot be right; the walking guide is evidence for neither.

The pipeline now withdraws the stance of a document whose title, snippet and
decisive passage are all semantically unrelated to the claim. These tests pin
that the gate removes the unrelated document, keeps related ones — including
a debunk whose refuting sentence alone looks unrelated — and does nothing at
all when dense ranking is unavailable.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import evidence_pipeline  # noqa: E402
from evidence_pipeline import run_pipeline  # noqa: E402
from providers import ProviderDiagnostic, SearchResult  # noqa: E402

CLAIM = "Morning crash shuts down Route 209"

LOCAL_NEWS = ("wgal.com", "Morning crash shuts down Route 209 in Dauphin County",
              "A crash shut down Route 209 on Tuesday morning, police said.")
WALKING_GUIDE = ("theguardian.com", "A weekend walk on the Shropshire Way",
                 "The path shuts in winter and the scenery is at its best in spring.")
DEBUNK = ("politifact.com", "Fact check: Route 209 was not closed by a crash",
          "This is false. Traffic moved normally all day.")


class StubNLI:
    """Entails for the local report; contradicts for everything else."""

    is_available = True

    def score_many(self, claim, passages):
        out = []
        for passage in passages:
            if "shut down Route 209" in passage or "shuts down Route 209" in passage:
                out.append(self._s(0.95, 0.01))
            else:
                out.append(self._s(0.01, 0.97))
        return out

    @staticmethod
    def _s(entail, contradict):
        return {"entailment": entail, "contradiction": contradict,
                "neutral": max(0.0, 1 - entail - contradict), "available": True}


def similarity_by_keyword(claim, texts):
    """Related iff the text mentions Route 209 at all."""
    return [0.8 if "209" in (t or "") else 0.05 for t in texts]


def run(rows, similarity):
    results = [
        SearchResult(url=f"https://{domain}/story-{i}", title=title, snippet=body,
                     text=(body + " ") * 30, provider="google_news", source=domain)
        for i, (domain, title, body) in enumerate(rows)
    ]
    diagnostics = [ProviderDiagnostic(
        provider="google_news", query="q", enabled=True, status="success",
        raw_result_count=len(results), new_result_count=len(results),
    )]
    with patch.object(evidence_pipeline, "search_all_providers",
                      lambda q, **k: (list(results), diagnostics)), \
         patch.object(evidence_pipeline, "get_nli_service", lambda: StubNLI()), \
         patch.object(evidence_pipeline, "claim_similarity", similarity), \
         patch.object(evidence_pipeline._relevance_filter, "filter_documents",
                      lambda claim, docs, strict=True: (docs, [])):
        return run_pipeline(CLAIM, max_results=8, fetch_articles=False)


def by_publisher(outcome):
    return {e.publisher or e.source: e for e in outcome.evidence}


class TestAboutnessGate(unittest.TestCase):

    def test_an_unrelated_document_cannot_contradict_the_claim(self):
        outcome = run([LOCAL_NEWS, WALKING_GUIDE], similarity_by_keyword)
        guide = by_publisher(outcome)["theguardian.com"]
        self.assertEqual(guide.stance, "unclear")
        self.assertIn("not about the claim", guide.stance_note)
        self.assertEqual(outcome.stance["status"], "supported")

    def test_without_the_gate_the_walking_guide_wins(self):
        """The failure being fixed, reproduced with dense ranking off."""
        outcome = run([LOCAL_NEWS, WALKING_GUIDE], lambda claim, texts: None)
        self.assertEqual(by_publisher(outcome)["theguardian.com"].stance, "contradicts")
        self.assertNotEqual(outcome.stance["status"], "supported")

    def test_a_debunk_counts_through_its_headline(self):
        """'This is false.' is unrelated on its own; the article is not."""
        outcome = run([DEBUNK], similarity_by_keyword)
        self.assertEqual(by_publisher(outcome)["politifact.com"].stance, "contradicts")

    def test_related_documents_are_untouched(self):
        outcome = run([LOCAL_NEWS], similarity_by_keyword)
        local = by_publisher(outcome)["wgal.com"]
        self.assertEqual(local.stance, "supports")
        self.assertEqual(local.stance_note, "")

    def test_dense_ranking_unavailable_changes_nothing(self):
        gated_off = run([LOCAL_NEWS, WALKING_GUIDE, DEBUNK], lambda claim, texts: None)
        for evidence in gated_off.evidence:
            self.assertNotIn("not about the claim", evidence.stance_note)


if __name__ == "__main__":
    unittest.main()
