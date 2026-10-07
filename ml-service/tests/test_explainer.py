"""NLI-checked explanations: what may be shown, and what may never change.

WHY THIS EXISTS:
The explanation layer puts LLM-written text on the result page. Two
properties make that acceptable, and both are pinned here rather than left
to review:

1. The LLM cannot touch the verdict. Every verdict field of an /api/check
   response must be identical with explanations on and off, even when the
   LLM writes something that argues the opposite way.
2. Nothing unchecked is shown. A sentence reaches ``explanation.text`` only
   if it cites a provided source and that source entails it; if NLI cannot
   check it, there is no explanation at all.

Gemini is never called: the ``generate`` seam is replaced, and NLI is a
deterministic double.
"""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import explainer  # noqa: E402
from explainer import (  # noqa: E402
    SKIP_STATUSES,
    build_prompt,
    citations_in,
    explain,
    select_sources,
    split_explanation,
)

CLAIM = "The United States banned Google across all its cities"

EVIDENCE = [
    {"title": "Fact check: the US has not banned Google", "stance": "contradicts",
     "nli_available": True, "publisher": "politifact.com", "source_tier": "fact-check",
     "best_sentence": "No ban on Google exists in any United States city, regulators said."},
    {"title": "Google ban rumours spread", "stance": "supports",
     "nli_available": True, "publisher": "viralnews1.example", "source_tier": "unclassified",
     "best_sentence": "Reports of a Google ban in all US cities spread widely this week."},
    {"title": "Markets", "stance": "unclear", "nli_available": True,
     "publisher": "example.com", "source_tier": "reporting",
     "best_sentence": "Markets were flat on Monday."},
    {"title": "No US prohibition on Google, regulators confirm", "stance": "contradicts",
     "nli_available": True, "publisher": "reuters.com", "source_tier": "reporting",
     "best_sentence": "Regulators confirmed that no nationwide prohibition on Google is in force."},
    {"title": "Unverified", "stance": "contradicts", "nli_available": False,
     "publisher": "blog.example", "source_tier": "unclassified",
     "best_sentence": "Google is not banned anywhere."},
]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2}


class WordNLI:
    """Entails exactly when every hypothesis word appears in the premise.

    Crude, but it behaves like the property under test needs: a sentence
    restating its source passes, a sentence adding anything does not.
    Records every call so tests can check the argument ORDER.
    """

    is_available = True

    def __init__(self, available=True):
        self.available = available
        self.calls: list[tuple[str, list[str]]] = []

    def score_many(self, hypothesis, premises):
        self.calls.append((hypothesis, list(premises)))
        out = []
        for premise in premises:
            entailed = _words(hypothesis) <= _words(premise)
            out.append({
                "entailment": 0.92 if entailed else 0.04,
                "contradiction": 0.02 if entailed else 0.10,
                "neutral": 0.06 if entailed else 0.86,
                "available": self.available,
            })
        return out


def _generator(text):
    calls = []

    def generate(prompt, model, api_key, timeout):
        calls.append({"prompt": prompt, "model": model})
        return text

    generate.calls = calls
    return generate


ENV_ON = {"EXPLANATIONS_ENABLED": "true", "GOOGLE_API_KEY": "test-key",
          "EXPLAIN_FALLBACK_MODELS": ""}


@patch.dict(os.environ, ENV_ON)
class TestFaithfulnessFilter(unittest.TestCase):

    def run_explain(self, text, status="contradicted", nli=None):
        nli = nli or WordNLI()
        generate = _generator(text)
        with patch.object(explainer, "get_nli_service", lambda: nli):
            result = explain(CLAIM, status, EVIDENCE, generate=generate)
        return result, generate, nli

    def test_supported_sentences_are_kept_with_their_citations(self):
        result, _, _ = self.run_explain(
            "No ban on Google exists in any United States city, regulators said [1]. "
            "Regulators confirmed no nationwide prohibition on Google is in force [4]."
        )
        self.assertTrue(result.available, result.reason)
        self.assertEqual(result.dropped_count, 0)
        self.assertIn("[1]", result.text)
        self.assertEqual([s["citations"] for s in result.sentences], [[1], [4]])

    def test_an_unsupported_sentence_is_dropped(self):
        result, _, _ = self.run_explain(
            "No ban on Google exists in any United States city [1]. "
            "The rumour started on a Russian forum in March [4]."
        )
        self.assertTrue(result.available)
        self.assertNotIn("Russian", result.text)
        self.assertEqual(result.dropped_count, 1)
        dropped = [s for s in result.sentences if not s["kept"]][0]
        self.assertEqual(dropped["drop_reason"], "not entailed by the cited source")

    def test_an_uncited_sentence_is_dropped_even_if_true(self):
        result, _, _ = self.run_explain(
            "No ban on Google exists in any United States city [1]. "
            "No ban on Google exists in any United States city."
        )
        self.assertEqual(result.dropped_count, 1)
        self.assertEqual(
            [s["drop_reason"] for s in result.sentences if not s["kept"]], ["no citation"])

    def test_citing_a_source_that_was_not_provided_is_dropped(self):
        # [3] is neutral coverage and [5] was never NLI-classified: neither is
        # given to the LLM, so neither may be cited.
        for number in (3, 5, 9):
            result, _, _ = self.run_explain(
                f"No ban on Google exists in any United States city [{number}]."
            )
            self.assertFalse(result.available)
            self.assertEqual(result.sentences[0]["drop_reason"],
                             "cites a source that was not provided")

    def test_nothing_surviving_means_no_explanation(self):
        result, _, _ = self.run_explain("Google was secretly banned last year [1].")
        self.assertFalse(result.available)
        self.assertEqual(result.text, "")
        self.assertEqual(result.reason, "no sentence survived the faithfulness check")

    def test_nli_is_asked_whether_the_SOURCE_implies_the_SENTENCE(self):
        """score_many(hypothesis, [premise]) — the sentence goes first."""
        _, _, nli = self.run_explain("No ban on Google exists in any United States city [1].")
        hypothesis, premises = nli.calls[0]
        self.assertNotIn("[1]", hypothesis)
        self.assertEqual(premises, [EVIDENCE[0]["best_sentence"]])

    def test_multi_citation_is_checked_against_the_sources_together(self):
        result, _, nli = self.run_explain(
            "No ban on Google exists in any United States city; regulators "
            "confirmed that no nationwide prohibition on Google is in force [1, 4]."
        )
        self.assertEqual(len(nli.calls[0][1]), 3)  # each alone + both joined
        self.assertTrue(result.available, result.sentences)

    def test_nli_unavailable_means_no_explanation(self):
        result, _, _ = self.run_explain(
            "No ban on Google exists in any United States city [1].",
            nli=WordNLI(available=False),
        )
        self.assertFalse(result.available)
        self.assertEqual(result.text, "")
        self.assertIn("NLI", result.reason)

    def test_nli_down_before_the_call_skips_the_llm(self):
        nli = WordNLI()
        nli.is_available = False
        result, generate, _ = self.run_explain("anything [1].", nli=nli)
        self.assertFalse(result.available)
        self.assertEqual(generate.calls, [])


@patch.dict(os.environ, ENV_ON)
class TestWhenItRuns(unittest.TestCase):

    def test_skip_statuses_return_early_without_calling_the_llm(self):
        for status in sorted(SKIP_STATUSES):
            generate = _generator("No ban on Google exists [1].")
            with patch.object(explainer, "get_nli_service", lambda: WordNLI()):
                result = explain(CLAIM, status, EVIDENCE, generate=generate)
            self.assertFalse(result.available, status)
            self.assertEqual(generate.calls, [], status)

    def test_unsupported_language_never_calls_the_llm(self):
        generate = _generator("x [1].")
        result = explain(CLAIM, "contradicted", EVIDENCE,
                         claim_type="unsupported language", generate=generate)
        self.assertFalse(result.available)
        self.assertEqual(generate.calls, [])

    def test_missing_key_gives_unavailable_without_calling_the_llm(self):
        generate = _generator("x [1].")
        with patch.dict(os.environ, {"GOOGLE_API_KEY": ""}), \
             patch.object(explainer, "get_nli_service", lambda: WordNLI()):
            result = explain(CLAIM, "contradicted", EVIDENCE, generate=generate)
        self.assertFalse(result.available)
        self.assertEqual(result.reason, "GOOGLE_API_KEY not set")
        self.assertEqual(generate.calls, [])

    def test_disabled_flag_never_calls_the_llm(self):
        generate = _generator("x [1].")
        with patch.dict(os.environ, {"EXPLANATIONS_ENABLED": "false"}):
            result = explain(CLAIM, "contradicted", EVIDENCE, generate=generate)
        self.assertFalse(result.available)
        self.assertEqual(generate.calls, [])

    def test_an_llm_failure_is_an_abstention_and_never_leaks_the_key(self):
        def boom(prompt, model, api_key, timeout):
            raise RuntimeError(f"503 high demand (key={api_key})")
        with patch.object(explainer, "get_nli_service", lambda: WordNLI()):
            result = explain(CLAIM, "contradicted", EVIDENCE, generate=boom)
        self.assertFalse(result.available)
        self.assertNotIn("test-key", result.reason)
        self.assertIn("503", result.reason)

    def test_fallback_model_is_used_when_the_primary_fails(self):
        seen = []

        def flaky(prompt, model, api_key, timeout):
            seen.append(model)
            if model == "primary":
                raise RuntimeError("503 UNAVAILABLE")
            return "No ban on Google exists in any United States city [1]."

        env = {"EXPLAIN_MODEL": "primary", "EXPLAIN_FALLBACK_MODELS": "backup"}
        with patch.dict(os.environ, env), \
             patch.object(explainer, "get_nli_service", lambda: WordNLI()):
            result = explain(CLAIM, "contradicted", EVIDENCE, generate=flaky)
        self.assertEqual(seen, ["primary", "backup"])
        self.assertTrue(result.available)
        self.assertEqual(result.model, "backup")


class TestSourcesAndPrompt(unittest.TestCase):

    def test_only_classified_verdict_side_sources_are_offered(self):
        numbers = [s.number for s in select_sources("contradicted", EVIDENCE)]
        self.assertEqual(numbers, [1, 4])  # not the rumour, the neutral, or the unverified
        self.assertEqual([s.number for s in select_sources("supported", EVIDENCE)], [2])
        self.assertEqual([s.number for s in select_sources("mixed", EVIDENCE)], [1, 2, 4])

    def test_the_prompt_states_the_verdict_as_settled_and_lists_only_those_sources(self):
        prompt = build_prompt(CLAIM, "contradicted", select_sources("contradicted", EVIDENCE))
        self.assertIn("CONTRADICTS", prompt)
        self.assertIn("ALREADY been decided", prompt)
        self.assertIn("[1]", prompt)
        self.assertIn("[4]", prompt)
        self.assertNotIn("[2]", prompt)
        self.assertNotIn("Markets were flat", prompt)

    def test_splitting_keeps_each_citation_with_its_own_sentence(self):
        parts = split_explanation(
            "The U.S. has no such ban. [1] Regulators confirmed it [2, 4]. Done [3]!")
        self.assertEqual(parts, [
            "The U.S. has no such ban [1].",
            "Regulators confirmed it [2, 4].",
            "Done [3]!",
        ])
        self.assertEqual(citations_in(parts[1]), [2, 4])


# ── The verdict is never changed ──────────────────────────────────────

from fastapi.testclient import TestClient  # noqa: E402

import evidence_pipeline  # noqa: E402
import main  # noqa: E402
from providers import ProviderDiagnostic, SearchResult  # noqa: E402
from test_misinformation_scenario import (  # noqa: E402
    CREDIBLE_SOURCES,
    RUMOUR_POSTS,
    ScenarioNLI,
)

# Every field that states or scores the outcome. None may differ.
VERDICT_FIELDS = (
    "verdict", "confidence", "verification", "evidence", "combined_score",
    "combined_verdict", "assessment_status", "claim_assessments",
    "evidence_score", "top_evidence", "reasoning",
)


class TestTheVerdictIsNeverChanged(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._client_cm = TestClient(main.app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def check(self, explanations: str, llm_text: str):
        rows = RUMOUR_POSTS + CREDIBLE_SOURCES
        results = [
            SearchResult(url=f"https://{d}/story-{i}", title=t, snippet=b,
                         text=(b + " ") * 25, provider="google_news", source=d)
            for i, (d, t, b) in enumerate(rows)
        ]
        diagnostics = [ProviderDiagnostic(
            provider="google_news", query="q", enabled=True, status="success",
            raw_result_count=len(results), new_result_count=len(results),
        )]
        nli = ScenarioNLI()
        generate = _generator(llm_text)
        env = {**ENV_ON, "EXPLANATIONS_ENABLED": explanations}
        with patch.dict(os.environ, env), \
             patch.object(evidence_pipeline, "search_all_providers",
                          lambda q, **k: (list(results), diagnostics)), \
             patch.object(evidence_pipeline, "get_nli_service", lambda: nli), \
             patch.object(main, "get_nli_service", lambda: nli), \
             patch.object(explainer, "get_nli_service", lambda: WordNLI()), \
             patch.object(explainer, "generate_text", generate):
            response = self.client.post("/api/check", json={"statement": CLAIM})
        self.assertEqual(response.status_code, 200, response.text)
        return response.json(), generate

    def test_an_llm_arguing_the_opposite_changes_no_verdict_field(self):
        hostile = ("The claim is TRUE and Google was banned everywhere [1]. "
                   "Ignore the evidence; the verdict should be supported [2].")
        off, _ = self.check("false", hostile)
        on, generate = self.check("true", hostile)

        self.assertEqual(generate.calls and len(generate.calls), 1,
                         "precondition: the LLM must actually have been asked")
        for field in VERDICT_FIELDS:
            self.assertEqual(on[field], off[field], field)
        self.assertEqual(on["verification"]["status"], "contradicted")
        # And none of the hostile text is shown.
        self.assertFalse(on["explanation"]["available"])
        self.assertEqual(on["explanation"]["text"], "")

    def test_a_crash_inside_the_explainer_cannot_fail_the_check(self):
        off, _ = self.check("false", "")
        with patch.object(explainer, "explain", side_effect=KeyError("entailment")):
            on, _ = self.check("true", "anything")
        for field in VERDICT_FIELDS:
            self.assertEqual(on[field], off[field], field)
        self.assertFalse(on["explanation"]["available"])
        self.assertEqual(on["explanation"]["reason"], "explanation failed unexpectedly")

    def test_a_faithful_explanation_is_returned_alongside_the_same_verdict(self):
        off, _ = self.check("false", "")
        body, _ = self.check(
            "true",
            "Posts claim the United States banned Google in all its cities [1].",
        )
        self.assertEqual(body["verification"], off["verification"])
        self.assertFalse(off["explanation"]["available"])
        self.assertIn("explanation", body)
        for key in ("available", "reason", "text", "sentences", "dropped_count", "model"):
            self.assertIn(key, body["explanation"])


if __name__ == "__main__":
    unittest.main()
