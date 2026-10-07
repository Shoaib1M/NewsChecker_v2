"""The benchmark's corpus construction and scoring.

WHY THIS EXISTS:
`news_benchmark.py` needs live network and cannot run in this suite. What it
reports, though, rests entirely on two things that are pure functions: whether
a corrupted headline is genuinely FALSE and readable, and whether an outcome is
scored as wrong rather than merely short. Both are tested here, because a
benchmark that is wrong in either respect produces a number that is worse than
having no number — it looks like evidence.

Every case below was a real defect, found by running the benchmark end to end
against stubbed retrieval and reading its output:

  - "The central bank lower interest rates" — the antonym was substituted in
    its base form regardless of the original's tense. A fact-checker asked to
    rule on a broken sentence is being tested on its parser.
  - "India's PM resigned after coalition talks launch" — an arbitrary event
    from anywhere in the headline was corrupted, so a subordinate clause was
    flipped and the claim stayed substantially TRUE. Scoring the system wrong
    for confirming it would have blamed the pipeline for a bad label.
  - "Regulators repealed the merger" — grammatical, but odd, because the
    antonym family's first entry won rather than its canonical verb.
  - "A magnitude 7 did not earthquake struck northern Japan" — negation was
    applied to a noun.
"""

from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import news_benchmark as nb  # noqa: E402


def corrupt_all(headline, seeds=range(8)):
    """Every corruption the strategies can produce for one headline."""
    out = []
    for seed in seeds:
        result = nb.corrupt(headline, random.Random(seed))
        if result:
            out.append(result)
    return out


class TestCorruptionsAreReadable(unittest.TestCase):

    HEADLINES = [
        "The central bank raised interest rates by 25 basis points",
        "India's prime minister resigned after coalition talks collapsed",
        "Regulators approved the merger between the two carriers",
        "The court rejected the appeal from the tech giant",
        "Officials banned the app across the country",
        "A magnitude 7 earthquake struck northern Japan",
        "The chief executive was dismissed by the board",
        "Shares soared after the earnings report",
        "Wildfires spread across the valley overnight",
    ]

    def test_no_corruption_produces_a_dangling_verb(self):
        """"did not earthquake" is not English, and rules on nothing."""
        for headline in self.HEADLINES:
            for claim, strategy in corrupt_all(headline):
                with self.subTest(claim=claim):
                    if " did not " in claim:
                        verb = claim.split(" did not ", 1)[1].split()[0]
                        self.assertIn(verb, nb.SURFACE_TO_EVENT,
                                      f"{verb!r} is not a verb this project knows")

    def test_the_antonym_keeps_the_original_tense(self):
        """Past tense in, past tense out. The original substituted the
        family's base form: "The central bank lower interest rates"."""
        claim, _ = nb.corrupt(
            "The central bank raised interest rates", random.Random(0))
        replacement = claim.lower().split()[3]
        self.assertTrue(replacement.endswith("ed"), claim)
        self.assertEqual(nb.SURFACE_TO_EVENT.get(replacement), "decrease", claim)

    def test_the_antonym_uses_the_family_s_canonical_verb(self):
        claim, _ = nb.corrupt(
            "Regulators approved the merger between the two carriers",
            random.Random(0))
        self.assertIn("rejected", claim.lower())

    def test_every_corruption_actually_changes_the_claim(self):
        for headline in self.HEADLINES:
            for claim, _strategy in corrupt_all(headline):
                with self.subTest(headline=headline):
                    self.assertNotEqual(claim.lower(), headline.lower())

    def test_a_headline_that_cannot_be_corrupted_cleanly_is_skipped(self):
        """Returning None costs a sample; mangling one costs the number."""
        self.assertIsNone(nb.corrupt("Paris", random.Random(0)))
        self.assertIsNone(nb.corrupt("", random.Random(0)))


class TestOnlyTheMainEventIsCorrupted(unittest.TestCase):
    """Corrupting a subordinate clause leaves the claim true, and then the
    benchmark punishes the system for being right."""

    def test_a_trailing_clause_is_not_the_one_flipped(self):
        headline = "India's prime minister resigned after coalition talks collapsed"
        for claim, strategy in corrupt_all(headline):
            with self.subTest(claim=claim):
                if strategy == "antonym":
                    self.fail(f"the main event has no antonym; got {claim!r}")

    def test_the_main_verb_is_the_one_negated(self):
        claim, _ = nb.corrupt(
            "India's prime minister resigned after coalition talks collapsed",
            random.Random(0))
        self.assertIn("did not resign", claim)


class TestHeadlineFiltering(unittest.TestCase):
    """Scoring the system on non-assertions measures nothing."""

    def test_assertions_are_kept(self):
        for headline in (
            "India's prime minister resigned after coalition talks collapsed",
            "The central bank raised interest rates by 25 basis points",
        ):
            with self.subTest(headline=headline):
                self.assertTrue(nb.usable_headline(headline))

    def test_questions_opinion_and_listicles_are_dropped(self):
        for headline in (
            "Will the prime minister resign this week?",
            "Opinion: the ban would be a mistake for everyone",
            "Analysis: what the ruling means for the tech industry",
            "Here's what to know about the new interest rate decision",
            "The 10 best laptops you can buy right now",
            "Rates",
        ):
            with self.subTest(headline=headline):
                self.assertFalse(nb.usable_headline(headline))


class TestScoring(unittest.TestCase):
    """The distinction the whole report rests on: wrong versus short."""

    def case(self, truth, status):
        c = nb.Case(claim="c", truth=truth, origin="o")
        c.status = status
        return nb.classify_outcome(c)

    def test_confirming_what_was_reported_is_correct(self):
        self.assertEqual(self.case("reported", "supported"), "correct")

    def test_failing_to_establish_a_real_headline_is_short_not_wrong(self):
        self.assertEqual(self.case("reported", "insufficient_evidence"), "missed")

    def test_calling_a_real_headline_false_is_wrong(self):
        self.assertEqual(self.case("reported", "contradicted"), "wrong")
        self.assertEqual(self.case("reported", "unsupported_no_coverage"), "wrong")

    def test_confirming_a_corrupted_headline_is_wrong(self):
        self.assertEqual(self.case("corrupted", "supported"), "wrong")

    def test_rejecting_a_corrupted_headline_is_correct(self):
        self.assertEqual(self.case("corrupted", "contradicted"), "correct")

    def test_abstaining_on_a_corrupted_headline_is_short_not_correct(self):
        """Refusing for want of evidence is not the same as finding the
        refutation, and the report must not present it as one."""
        self.assertEqual(self.case("corrupted", "insufficient_evidence"), "missed")

    def test_the_wrong_answer_rate_counts_only_confident_falsehoods(self):
        cases = []
        for truth, status in (("reported", "supported"),
                              ("reported", "insufficient_evidence"),
                              ("corrupted", "supported"),
                              ("corrupted", "contradicted")):
            c = nb.Case(claim="c", truth=truth, origin="o")
            c.status = status
            c.outcome = nb.classify_outcome(c)
            cases.append(c)
        summary = nb.summarise(cases)
        self.assertEqual(summary["wrong_answers"], 1)
        self.assertAlmostEqual(summary["wrong_answer_rate"], 0.25)

    def test_abstention_recall_and_latency_are_reported(self):
        cases = []
        for truth, status, seconds in (("reported", "supported", 10.0),
                                       ("reported", "insufficient_evidence", 20.0),
                                       ("corrupted", "mixed", 30.0),
                                       ("corrupted", "unsupported_no_coverage", 40.0)):
            c = nb.Case(claim="c", truth=truth, origin="o", seconds=seconds)
            c.status = status
            c.outcome = nb.classify_outcome(c)
            cases.append(c)
        summary = nb.summarise(cases)
        self.assertAlmostEqual(summary["confirmation_recall"], 0.5)
        # insufficient_evidence and mixed commit to neither direction.
        self.assertAlmostEqual(summary["abstention_rate"], 0.5)
        self.assertAlmostEqual(summary["mean_seconds"], 25.0)

    def test_the_two_directions_are_never_merged(self):
        """They measure different things and fail differently; a single
        blended 'accuracy' would hide that."""
        summary = nb.summarise([])
        self.assertIn("reported", summary)
        self.assertIn("corrupted", summary)
        self.assertNotIn("accuracy", summary)


if __name__ == "__main__":
    unittest.main(verbosity=2)
