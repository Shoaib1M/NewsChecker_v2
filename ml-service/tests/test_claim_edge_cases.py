"""End-to-end verdict behaviour across the full range of claim shapes.

WHY THIS EXISTS:
The pipeline was correct on the kind of claim it was built against — a
well-formed, past-tense, LIAR-dataset-shaped political statement — and
produced the same "insufficient evidence" answer for everything else:
fabricated headlines, future predictions, questions, keyboard mash. Those are
four different situations and this file pins each one to a distinct outcome.

Every test drives the real FastAPI endpoint with search and NLI replaced by
deterministic doubles, so what is asserted is the verdict logic itself, not
the behaviour of any live provider.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from fastapi.testclient import TestClient  # noqa: E402

import evidence_pipeline  # noqa: E402
import main  # noqa: E402
from providers import ProviderDiagnostic, SearchResult  # noqa: E402


# ── Test doubles ─────────────────────────────────────────────────────
def make_results(specs: list[tuple[str, str, str]]) -> list[SearchResult]:
    """Build SearchResults from (domain, title, body) triples."""
    return [
        SearchResult(
            url=f"https://{domain}/story-{i}",
            title=title,
            snippet=body,
            # Long enough that the pipeline never tries to fetch the page.
            text=(body + " ") * 30,
            provider="gnews",
            source=domain,
        )
        for i, (domain, title, body) in enumerate(specs)
    ]


class StubNLI:
    """NLI double driven by a keyword rule, so tests state their intent."""

    def __init__(self, entail_on: str | None = None, contradict_on: str | None = None,
                 available: bool = True):
        self.entail_on = entail_on
        self.contradict_on = contradict_on
        self._available = available
        self.status = {"status": "ready" if available else "failed",
                       "enabled": True, "model": "stub", "error": None}
        self.is_ready = available

    @property
    def is_available(self) -> bool:
        return self._available

    def score_many(self, claim, passages):
        out = []
        for passage in passages:
            lowered = passage.lower()
            if not self._available:
                out.append({"entailment": 0.0, "contradiction": 0.0,
                            "neutral": 1.0, "available": False})
            elif self.entail_on and self.entail_on in lowered:
                out.append({"entailment": 0.93, "contradiction": 0.02,
                            "neutral": 0.05, "available": True})
            elif self.contradict_on and self.contradict_on in lowered:
                out.append({"entailment": 0.02, "contradiction": 0.91,
                            "neutral": 0.07, "available": True})
            else:
                out.append({"entailment": 0.04, "contradiction": 0.03,
                            "neutral": 0.93, "available": True})
        return out


class VerdictCaseMixin:
    """Shared harness: run one statement through /api/check with doubles."""

    @classmethod
    def setUpClass(cls):
        # TestClient without the context manager skips lifespan, leaving the
        # MLP unloaded and every request a 503.
        cls._client_cm = TestClient(main.app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def check(
        self,
        statement: str,
        results: list[SearchResult] | None = None,
        nli: StubNLI | None = None,
        search_status: str = "success",
        providers: tuple[str, ...] = ("gnews", "google_news"),
    ) -> dict:
        """``providers`` names the providers that ANSWERED.

        Two by default because that is what a working search looks like. This
        harness used to stub exactly one, which meant every absence-of-coverage
        test was asserting that a single provider is enough to say "no credible
        source reports this" — and the code agreed with it.
        """
        results = results or []
        diagnostics = [
            ProviderDiagnostic(
                provider=name, query="q", enabled=True, status=search_status,
                raw_result_count=len(results), new_result_count=len(results),
            )
            for name in providers
        ]
        nli = nli or StubNLI()

        def fake_search(queries, **kwargs):
            return list(results), diagnostics

        with patch.object(evidence_pipeline, "search_all_providers", fake_search), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
             patch.object(main, "get_nli_service", lambda: nli):
            response = self.client.post("/api/check", json={"statement": statement})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()


# ── 1. Inputs that are not checkable claims ──────────────────────────
class TestNonClaims(VerdictCaseMixin, unittest.TestCase):
    """These must never reach the search phase or report a verification failure."""

    def test_gibberish_is_not_a_failed_verification(self):
        body = self.check("asdkjh asdkjh asdkjh qwe")
        self.assertEqual(body["verification"]["status"], "not_a_claim")
        self.assertFalse(body["external_evidence_checked"])
        self.assertEqual(body["retrieval"]["candidate_count"], 0)

    def test_question_is_rejected_as_a_question(self):
        body = self.check("Is the earth actually flat?")
        self.assertEqual(body["verification"]["status"], "not_a_claim")
        self.assertIn("question", body["reasoning"].lower())

    def test_bare_noun_phrase_has_no_assertion(self):
        body = self.check("unemployment rate figures")
        self.assertEqual(body["verification"]["status"], "not_a_claim")

    def test_opinion_is_not_objectively_verifiable(self):
        body = self.check("Pizza is the best food in the world")
        self.assertEqual(body["verification"]["status"], "not_objectively_verifiable")
        self.assertFalse(body["external_evidence_checked"])

    def test_non_claims_never_show_a_numeric_score(self):
        for statement in ("asdkjh asdkjh asdkjh qwe", "Pizza is the best food ever"):
            with self.subTest(statement=statement):
                body = self.check(statement)
                self.assertIn(body["verification"]["status"], main.NON_NUMERIC_STATUSES)


# ── 2. Fabricated but highly newsworthy claims ───────────────────────
class TestAbsenceOfCoverage(VerdictCaseMixin, unittest.TestCase):
    """The case that made the system look broken: a claim nobody reported."""

    # Real coverage of the right subjects that says nothing about the claim.
    BACKGROUND = [
        ("reuters.com", "Musk's xAI raises new funding round",
         "The company closed a funding round led by existing investors."),
        ("apnews.com", "Paris tourism numbers rebound",
         "Visitor numbers to the French capital returned to pre-pandemic levels."),
        ("bbc.com", "Eiffel Tower closes for scheduled maintenance",
         "The monument shut briefly for routine structural work."),
        ("theguardian.com", "Tesla shares slip on delivery miss",
         "Investors reacted to quarterly delivery figures."),
        ("npr.org", "France debates monument funding",
         "Lawmakers discussed budgets for national landmarks."),
    ]

    def test_fabricated_headline_is_reported_as_unsupported(self):
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=make_results(self.BACKGROUND),
        )
        self.assertEqual(body["verification"]["status"], "unsupported_no_coverage")
        self.assertEqual(body["verdict"], "no credible source reports this")
        self.assertIn("absence of any coverage", body["reasoning"])

    def test_absence_verdict_is_never_reached_when_search_failed(self):
        """The whole point: a broken search must not look like a finding."""
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=[],
            search_status="failed",
        )
        self.assertEqual(body["verification"]["status"], "insufficient_evidence")
        self.assertEqual(body["retrieval"]["status"], "SEARCH_FAILED")
        self.assertIn("retrieval failure", body["reasoning"])

    def test_absence_verdict_needs_more_than_one_provider(self):
        """"No credible source reports this" is a claim about the press. One
        working provider while the rest are blocked or down is not a canvass
        of it — it is a quarter of one."""
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=make_results(self.BACKGROUND),
            providers=("gnews",),
        )
        self.assertNotEqual(body["verification"]["status"], "unsupported_no_coverage")
        self.assertEqual(body["verification"]["status"], "insufficient_evidence")

    def test_a_provider_that_found_nothing_still_counts_as_having_looked(self):
        """It is the difference between silence and blindness: `no_results`
        looked and found nothing, `failed` never looked."""
        from evidence_aggregator import assess_coverage
        empty = {"supporting_count": 0, "contradicting_count": 0}
        self.assertIsNotNone(assess_coverage(
            empty, retrieval_status="SEARCH_SUCCESS", candidate_count=5,
            salience="high", providers_answered=2))
        self.assertIsNone(assess_coverage(
            empty, retrieval_status="SEARCH_SUCCESS", candidate_count=5,
            salience="high", providers_answered=1))

    def test_absence_verdict_needs_a_real_pool_of_candidates(self):
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=make_results(self.BACKGROUND[:2]),
        )
        self.assertNotEqual(body["verification"]["status"], "unsupported_no_coverage")

    def test_low_salience_claim_does_not_get_the_absence_verdict(self):
        """Ordinary claims go unreported all the time; that proves nothing."""
        body = self.check(
            "A regional bakery chain updated its supplier contracts last quarter",
            results=make_results(self.BACKGROUND),
        )
        self.assertNotEqual(body["verification"]["status"], "unsupported_no_coverage")
        self.assertEqual(body["verification"]["salience"], "normal")

    def test_contradicting_evidence_wins_over_absence(self):
        results = make_results(self.BACKGROUND + [
            ("politifact.com", "No, Elon Musk did not buy the Eiffel Tower",
             "The claim is false; the monument remains owned by the city of Paris."),
        ])
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=results,
            nli=StubNLI(contradict_on="remains owned by the city"),
        )
        self.assertEqual(body["verification"]["status"], "contradicted")

    def test_absence_confidence_scales_with_how_much_was_searched(self):
        wide = make_results(self.BACKGROUND * 3)  # 15 candidates
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=wide,
        )
        self.assertEqual(body["verification"]["status"], "unsupported_no_coverage")
        self.assertEqual(body["confidence"], "high")


# ── 3. Claims about the future ───────────────────────────────────────
class TestProspectiveClaims(VerdictCaseMixin, unittest.TestCase):
    """Nothing can make a future event true today — say so, don't abstain."""

    BACKGROUND = [
        ("reuters.com", "Google reports quarterly earnings",
         "Revenue rose on advertising strength."),
        ("apnews.com", "Google expands advertising tools in the United States",
         "New ad products were announced for US businesses."),
        ("bbc.com", "US tech regulation debate continues",
         "Lawmakers discussed oversight of large platforms."),
        ("npr.org", "Cloud market share shifts",
         "Providers competed for enterprise customers."),
        ("theguardian.com", "Search engine competition examined",
         "Regulators looked at the search market."),
    ]

    def test_future_claim_is_not_reported_as_unverifiable_by_accident(self):
        body = self.check(
            "The United States is Going to ban Google across all its cities",
            results=make_results(self.BACKGROUND),
        )
        self.assertEqual(body["verification"]["claim_kind"], "prospective")
        self.assertIn(
            body["verification"]["status"],
            {"unsupported_no_coverage", "not_verifiable_yet"},
        )
        self.assertNotEqual(body["verdict"], "insufficient evidence")

    def test_reported_plan_is_distinguished_from_a_done_deal(self):
        results = make_results(self.BACKGROUND + [
            ("reuters.com", "US lawmakers propose nationwide ban on Google services",
             "A bill introduced this week would ban Google across all US cities."),
        ])
        body = self.check(
            "The United States is Going to ban Google across all its cities",
            results=results,
            nli=StubNLI(entail_on="would ban google"),
        )
        self.assertEqual(body["verification"]["status"], "reported_plan")
        self.assertIn("not yet done", body["verdict"])
        self.assertIn("does not confirm", body["reasoning"])

    def test_announced_future_action_is_treated_as_checkable(self):
        """"X announced it will Y" is about the announcement, which happened."""
        body = self.check(
            "Apple announced it will move all manufacturing to India by 2027",
            results=make_results(self.BACKGROUND),
        )
        self.assertEqual(body["verification"]["claim_kind"], "checkable")


# ── 4. Ordinary checkable claims must still work ─────────────────────
class TestCheckableClaims(VerdictCaseMixin, unittest.TestCase):
    """Regression guard: none of the above may break the normal path."""

    def test_supported_claim_reports_supporting_evidence(self):
        results = make_results([
            ("reuters.com", "Indian PM steps down after coalition collapse",
             "The prime minister resigned on Tuesday morning."),
            ("apnews.com", "India's prime minister resigns",
             "The prime minister resigned, ending weeks of speculation."),
            ("bbc.com", "Political turmoil in Delhi",
             "The prime minister resigned amid coalition infighting."),
        ])
        body = self.check(
            "The prime minister of India resigned this morning",
            results=results,
            nli=StubNLI(entail_on="prime minister resigned"),
        )
        self.assertEqual(body["verification"]["status"], "supported")
        self.assertGreater(body["evidence"]["supporting_count"], 0)
        self.assertGreater(body["evidence"]["independent_groups"], 1)

    def test_deterministic_knowledge_check_still_wins(self):
        body = self.check("Water freezes at 0°C at sea level")
        self.assertEqual(body["verdict"], "true")
        self.assertFalse(body["external_evidence_checked"])

    def test_relevant_sources_beat_merely_topical_ones(self):
        """The filter must prefer the article about the claim's action."""
        results = make_results([
            ("apnews.com", "Google expands advertising tools in the United States",
             "Google announced new ad products for US businesses."),
            ("reuters.com", "US lawmakers propose nationwide ban on Google services",
             "A bill would ban Google across all US cities."),
        ])
        body = self.check(
            "US lawmakers banned Google across all cities",
            results=results,
            nli=StubNLI(entail_on="would ban google"),
        )
        titles = [e["title"] for e in body["top_evidence"]]
        self.assertNotIn("Google expands advertising tools in the United States", titles)


# ── 5. Degraded infrastructure must never fabricate a verdict ────────
class TestDegradedModes(VerdictCaseMixin, unittest.TestCase):

    def test_nli_unavailable_yields_no_verdict_and_says_why(self):
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=make_results(TestAbsenceOfCoverage.BACKGROUND),
            nli=StubNLI(available=False),
        )
        self.assertNotEqual(body["verification"]["status"], "unsupported_no_coverage")
        self.assertEqual(body["nli"]["classified_count"], 0)
        self.assertIn("NLI model is unavailable", body["reasoning"])

    def test_no_wording_based_classifier_score_is_returned(self):
        """The LIAR-trained MLP was removed: verdicts come from evidence only,
        and no score derived from the claim's wording alone is reported."""
        body = self.check("The prime minister of India resigned this morning")
        for field in ("ml", "ml_score", "ml_verdict", "ml_threshold"):
            self.assertNotIn(field, body)

    def test_no_results_at_all_is_not_a_finding(self):
        body = self.check(
            "Elon Musk bought the Eiffel Tower for 3 trillion dollars",
            results=[],
            search_status="no_results",
        )
        self.assertEqual(body["verification"]["status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── 6. Triage misclassification traps ────────────────────────────────
class TestTriageTraps(unittest.TestCase):
    """Inputs that broke triage in ways only a wide sweep surfaced.

    Each of these produced a wrong classification at some point during
    development; they are pinned here because every one of them silently
    changes the verdict a user sees.
    """

    def triage(self, statement):
        from claim_triage import triage_claim
        return triage_claim(statement)

    def test_factual_superlatives_are_not_opinions(self):
        """"the best-selling car" is a sales fact, not a value judgment."""
        for statement in (
            "The best-selling car in 2024 was the Tesla Model Y",
            "The greatest earthquake in recorded history hit Chile in 1960",
            "Reuters is the best-known wire service by revenue",
        ):
            with self.subTest(statement=statement):
                self.assertEqual(self.triage(statement).kind, "checkable")

    def test_predicative_superlatives_still_are_opinions(self):
        for statement in ("Pizza is the best food", "The Model Y is the best car ever made"):
            with self.subTest(statement=statement):
                self.assertEqual(self.triage(statement).kind, "opinion")

    def test_irregular_past_tense_counts_as_an_assertion(self):
        """No copula, no -ed ending — the shape test alone rejects these."""
        for statement in (
            "The greatest earthquake in history hit Chile in 1960",
            "India won the cricket world cup",
            "The company bought a rival chipmaker",
        ):
            with self.subTest(statement=statement):
                self.assertEqual(self.triage(statement).kind, "checkable")

    def test_a_bare_link_is_not_a_claim(self):
        result = self.triage("https://example.com/some-article")
        self.assertEqual(result.kind, "not_a_claim")
        self.assertFalse(result.search_worthwhile)

    def test_negation_is_detected(self):
        self.assertTrue(self.triage("The United States did not ban Google").negated)
        self.assertFalse(self.triage("The United States banned Google").negated)


class TestNegatedClaims(VerdictCaseMixin, unittest.TestCase):
    """Silence cannot count against a claim that something did NOT happen."""

    def test_absence_of_coverage_never_contradicts_a_negative_claim(self):
        body = self.check(
            "The United States did not ban Google",
            results=make_results(TestProspectiveClaims.BACKGROUND),
        )
        self.assertTrue(body["verification"].get("salience") == "high")
        self.assertNotEqual(body["verification"]["status"], "unsupported_no_coverage")

    def test_the_positive_form_of_the_same_claim_does_get_it(self):
        """Control: the guard is about negation, not about this subject."""
        body = self.check(
            "The United States banned Google across all its cities",
            results=make_results(TestProspectiveClaims.BACKGROUND),
        )
        self.assertEqual(body["verification"]["status"], "unsupported_no_coverage")


# ── 7. Contradicting evidence must survive relevance filtering ───────
class TestContradictingEvidenceSurvives(VerdictCaseMixin, unittest.TestCase):
    """A refutation is written in its own words, not the claim's.

    The relevance filter scores whether a document discusses the claim's
    action. Applied naively that rejects contradicting coverage — which
    describes the *opposite* outcome and so shares almost none of the claim's
    vocabulary — and leaves a one-sided "supported" verdict on a claim that is
    actually contested. This is the failure mode that matters most: it is the
    one where the system states something confidently and wrongly.
    """

    def test_a_contested_claim_comes_back_mixed_not_supported(self):
        results = make_results([
            ("bbc.com", "Trial finds four-day week improves productivity",
             "The pilot found a four-day workweek improves productivity."),
            ("wsj.com", "Study casts doubt on four-day week gains",
             "Researchers said output fell under the shorter schedule."),
        ])
        body = self.check(
            "A four-day workweek improves productivity",
            results=results,
            nli=StubNLI(entail_on="improves productivity", contradict_on="output fell"),
        )
        self.assertEqual(body["verification"]["status"], "mixed")
        self.assertGreater(body["evidence"]["supporting_count"], 0)
        self.assertGreater(body["evidence"]["contradicting_count"], 0)

    def test_the_antonym_of_the_claims_action_still_counts_as_on_topic(self):
        from relevance_filter import RelevanceFilter
        filt = RelevanceFilter()
        score = filt.assess_document_relevance(
            "The United States is going to ban Google across all its cities",
            "Court declines to outlaw Google search deals",
            "Judges rejected calls to prohibit the arrangements.",
            "Judges rejected calls to prohibit the arrangements. " * 20,
        )
        self.assertEqual(score.action_match_score, 1.0)
        self.assertTrue(filt.should_include_document(score, strict=True))

    def test_an_off_topic_article_is_still_rejected(self):
        """Control: relaxing the filter must not readmit the original bug."""
        from relevance_filter import RelevanceFilter
        filt = RelevanceFilter()
        score = filt.assess_document_relevance(
            "The United States is going to ban Google across all its cities",
            "Google expands advertising tools in the United States",
            "Google announced new ad products for US businesses.",
            "Google announced new ad products for US businesses. " * 20,
        )
        self.assertEqual(score.action_match_score, 0.0)
        self.assertFalse(filt.should_include_document(score, strict=True))


class TestKnowledgeVerifierTraps(unittest.TestCase):
    """knowledge_verifier runs before triage and had its own copy of a bug."""

    def test_factual_superlatives_are_not_caught_as_subjective(self):
        from knowledge_verifier import assess_claim
        self.assertIsNone(assess_claim("The best-selling car in 2024 was the Tesla Model Y"))
        self.assertIsNone(assess_claim("Reuters is the best-known wire service by revenue"))

    def test_genuine_value_judgments_are_still_caught(self):
        from knowledge_verifier import assess_claim
        result = assess_claim("Pizza is the best food in the world")
        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "not_objectively_verifiable")


# ── 8. The deterministic layer must not answer the wrong question ────
class TestDeterministicLayerQualifiers(unittest.TestCase):
    """This layer returns "very high" confidence and skips evidence entirely.

    Its pattern tables match substrings, so a sentence that *negates* or
    *comments on* a known proposition matched the proposition and ignored the
    frame around it. That produces the most confidently wrong output the
    system can emit — no search, no NLI, no way for anything downstream to
    correct it.
    """

    def assess(self, statement):
        from knowledge_verifier import assess_claim
        return assess_claim(statement)

    def test_a_denial_of_a_known_falsehood_is_not_itself_false(self):
        """"It is false that a triangle has four sides" is a TRUE statement."""
        self.assertIsNone(self.assess("It is false that a triangle has four sides"))

    def test_a_denial_of_a_known_fact_is_not_itself_true(self):
        """"Nobody claims WWII ended in 1945" is a FALSE statement."""
        self.assertIsNone(self.assess("Nobody claims world war ii ended in 1945"))

    def test_commentary_frames_are_declined(self):
        for statement in (
            "The sun revolves around the earth, a belief long since disproven",
            "A triangle has four sides according to the debunked claim",
            "It is a myth that the Great Wall of China is visible from the Moon",
            "Water does not freeze at 0°C at sea level",
        ):
            with self.subTest(statement=statement):
                self.assertIsNone(self.assess(statement))

    def test_plain_statements_are_still_answered(self):
        for statement, verdict in (
            ("Water freezes at 0°C at sea level", "true"),
            ("A triangle has four sides", "false"),
            ("World War II ended in 1945", "true"),
            ("2 + 2 = 5", "false"),
            ("2 + 2 = 4", "true"),
        ):
            with self.subTest(statement=statement):
                result = self.assess(statement)
                self.assertIsNotNone(result, f"{statement} should still be answered")
                self.assertEqual(result["verdict"], verdict)

    def test_a_declined_claim_falls_through_to_the_evidence_pipeline(self):
        """Declining must mean "search this", not "no verdict"."""
        from claim_triage import triage_claim
        triage = triage_claim("It is false that a triangle has four sides")
        self.assertTrue(triage.search_worthwhile)

    def test_arithmetic_cannot_hang_the_request(self):
        """Huge exponents overflow in float rather than computing a bignum."""
        from knowledge_verifier import _safe_arithmetic
        self.assertIsNone(_safe_arithmetic("9 ** 9 ** 9"))
        self.assertIsNone(_safe_arithmetic("2 ** 200000"))


# ── 9. Multi-claim statements must not report one claim's retrieval ──
class TestMultiClaimRetrievalState(VerdictCaseMixin, unittest.TestCase):
    """The loop overwrote retrieval status per claim, keeping only the last.

    That matters because the status gates absence-of-coverage reasoning. With
    the last claim's search succeeding and an earlier one having failed, the
    system could report "no credible source reports this" about a statement
    whose retrieval never actually ran.
    """

    TWO_CLAIMS = ("The prime minister resigned this morning. "
                  "The finance minister was arrested yesterday.")

    def run_with_outcomes(self, *outcomes):
        """Drive /api/check with a scripted PipelineOutcome per claim."""
        from evidence_pipeline import PipelineOutcome
        from evidence_aggregator import compute_stance

        scripted = iter(outcomes)
        fallback = PipelineOutcome(compute_stance([]), [], "SEARCH_SUCCESS", [], 5, 0)
        with patch.object(main, "run_pipeline", lambda *a, **k: next(scripted, fallback)):
            return self.check(self.TWO_CLAIMS)

    def test_a_failure_on_any_claim_is_not_hidden_by_a_later_success(self):
        from evidence_pipeline import PipelineOutcome
        from evidence_aggregator import compute_stance

        body = self.run_with_outcomes(
            PipelineOutcome(compute_stance([]), [], "SEARCH_FAILED", [], 0, 0),
            PipelineOutcome(compute_stance([]), [], "SEARCH_SUCCESS",
                            [{"provider": "gnews", "status": "success"}], 9, 0),
        )

        self.assertEqual(body["retrieval"]["status"], "SEARCH_FAILED")
        self.assertNotEqual(
            body["verification"]["status"], "unsupported_no_coverage",
            "a failed search on any claim must block the absence verdict",
        )

    def test_diagnostics_from_every_claim_are_kept(self):
        from evidence_pipeline import PipelineOutcome
        from evidence_aggregator import compute_stance

        body = self.run_with_outcomes(
            PipelineOutcome(compute_stance([]), [], "SEARCH_SUCCESS",
                            [{"provider": "gnews", "status": "success"}], 5, 0),
            PipelineOutcome(compute_stance([]), [], "SEARCH_SUCCESS",
                            [{"provider": "wikipedia", "status": "success"}], 5, 0),
        )

        providers = {d.get("provider") for d in body["retrieval"]["diagnostics"]}
        self.assertEqual(providers, {"gnews", "wikipedia"})


# ── 10. Language scope must be stated, not disguised as gibberish ────
class TestUnsupportedLanguage(unittest.TestCase):
    """Every stage here is English-only — say so rather than implying nonsense.

    The NLI model, the event vocabulary, the abbreviation and demonym tables,
    and the providers (queried with lang=en) are all English. A Hindi or
    Spanish claim cannot be checked, and used to be reported as "no verifiable
    claim found" — which tells the user their claim was unintelligible rather
    than out of scope. Those are very different messages to receive.
    """

    def triage(self, statement):
        from claim_triage import triage_claim
        return triage_claim(statement)

    def test_non_latin_scripts_are_reported_as_out_of_scope(self):
        for language, statement in (
            ("Hindi", "भारत के प्रधानमंत्री ने आज सुबह इस्तीफा दे दिया"),
            ("Arabic", "رئيس الوزراء الهندي استقال هذا الصباح"),
            ("Chinese", "印度总理今天早上辞职了"),
            ("Japanese", "インドの首相が今朝辞任した"),
            ("Russian", "Премьер-министр Индии подал в отставку"),
        ):
            with self.subTest(language=language):
                result = self.triage(statement)
                self.assertEqual(result.claim_type, "unsupported language")
                self.assertIn("English", result.reason)

    def test_latin_script_languages_are_recognised_too(self):
        for language, statement in (
            ("Spanish", "El primer ministro de India renunció esta mañana"),
            ("French", "Le premier ministre a démissionné ce matin"),
            ("German", "Der Premierminister ist heute Morgen zurückgetreten"),
        ):
            with self.subTest(language=language):
                self.assertEqual(self.triage(statement).claim_type, "unsupported language")

    def test_english_is_never_misread_as_foreign(self):
        """The property that matters: no false positives on real English."""
        for statement in (
            "The prime minister of India resigned this morning",
            "Google banned all US cities immediately",
            "Modi resigned Tuesday morning citing health",
            "Angela Merkel resigned as chancellor",
            "Marine Le Pen won the presidential election in France",
            "Le Monde reported the resignation on Tuesday",
            "Rio de Janeiro hosted the summit last week",
            "The café owner resigned after the exposé",
            "The coup de grace came as the vote failed",
        ):
            with self.subTest(statement=statement):
                self.assertNotEqual(
                    self.triage(statement).claim_type, "unsupported language"
                )

    def test_out_of_scope_is_not_searched(self):
        self.assertFalse(self.triage("印度总理今天早上辞职了").search_worthwhile)


# ── 11. The gauge must not put a number on a non-measurement ─────────
class TestGaugeNumbers(unittest.TestCase):
    """`combined_score` is an evidence-balance dial, not a truth percentage.

    Only outcomes that genuinely measure evidence for the claim may show a
    number. `reported_plan` is the subtle one: real evidence exists, but it
    attests the *announcement*, not the event — so "90" beside "reported as
    planned — not yet done" reads as "90% true" for something that is not yet
    true or false at all.
    """

    def test_outcomes_that_measure_nothing_show_no_number(self):
        for status in (
            "insufficient_evidence", "not_a_claim", "not_objectively_verifiable",
            "not_verifiable_yet", "unsupported_no_coverage", "reported_plan",
        ):
            with self.subTest(status=status):
                self.assertIn(status, main.NON_NUMERIC_STATUSES)

    def test_evidence_outcomes_still_show_a_number(self):
        for status in ("supported", "contradicted", "mixed"):
            with self.subTest(status=status):
                self.assertNotIn(status, main.NON_NUMERIC_STATUSES)

    def test_the_dial_tracks_the_direction_of_the_evidence(self):
        self.assertGreater(main.evidence_verdict_score({"status": "supported", "net": 0.88}), 80)
        self.assertLess(main.evidence_verdict_score({"status": "contradicted", "net": -0.85}), 20)

    def test_every_status_has_a_verdict_phrase(self):
        """A status with no phrase would surface as an empty verdict line."""
        for status in main.NON_NUMERIC_STATUSES:
            with self.subTest(status=status):
                self.assertIn(status, main._STATUS_VERDICTS)
                self.assertTrue(main._STATUS_VERDICTS[status])
