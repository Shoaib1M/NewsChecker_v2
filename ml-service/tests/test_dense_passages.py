"""Hybrid (dense + lexical) passage selection.

WHY THIS EXISTS:
Dense ranking was added to rescue one specific failure: a debunking sentence
that paraphrases the claim shares no words with it, scores zero on overlap,
and never reaches NLI. These tests pin that rescue, and — just as important —
pin that every way the dense ranker can be absent (disabled, broken, found
nothing) leaves passage selection exactly as it was before it existed.

No model is ever loaded here: ``article_extractor.semantic_order`` is patched
with a stand-in, and ``PassageRanker`` is given a fake embedding model with
hand-built vectors so similarities are known exactly.
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import article_extractor  # noqa: E402
from article_extractor import extract_passages, split_sentences  # noqa: E402
from passage_retriever import (  # noqa: E402
    MAX_SENTENCES,
    PassageRanker,
    reciprocal_rank_fusion,
)

CLAIM = "The United States banned Google across all its cities"

# Six sentences that share the claim's words and say nothing decisive, then
# the one sentence that settles it — in words of its own.
DEBUNK = "Washington has not prohibited the search giant anywhere in the country."
ARTICLE = " ".join([
    "Google shares rose slightly on Monday in early trading.",
    "Google announced a new phone at an event in California.",
    "Analysts in the United States expect strong holiday sales.",
    "Several cities hosted technology fairs over the weekend.",
    "Google executives spoke about artificial intelligence plans.",
    "The United States economy added jobs last month.",
    "Officials from several departments attended the briefing.",
    DEBUNK,
])

# From test_article_extraction: seven sentences of scene-setting, then the fact.
BURIED_LEDE = " ".join([
    "The meeting began early on Tuesday in the capital city.",
    "Officials from several departments attended the session.",
    "Reporters gathered outside the building from dawn.",
    "The agenda covered a wide range of economic topics.",
    "Analysts had expected a routine set of announcements.",
    "Markets were broadly flat ahead of the statement.",
    "Security was tight around the perimeter all morning.",
    "The prime minister resigned, effective immediately, citing health reasons.",
    "His deputy will serve in an acting capacity.",
])


def _dense_finding(word: str):
    """A stand-in for semantic_order that 'understands' one word."""
    def fake(claim, sentences):
        return [i for i, s in enumerate(sentences) if word in s.lower()]
    return fake


def _lexical(title, snippet, text, **kwargs):
    """The pre-upgrade behaviour: dense ranking unavailable."""
    with patch.object(article_extractor, "semantic_order", return_value=None):
        return extract_passages(title, snippet, text, **kwargs)


class TestHybridPassageSelection(unittest.TestCase):

    def test_a_paraphrased_debunk_with_no_shared_words_reaches_nli(self):
        lexical = _lexical("Tech news roundup", "Markets and gadgets.", ARTICLE, claim=CLAIM)
        self.assertNotIn(
            DEBUNK, lexical,
            "precondition: overlap ranking alone must miss the paraphrase, "
            "or this test proves nothing",
        )

        with patch.object(article_extractor, "semantic_order",
                          side_effect=_dense_finding("prohibited")):
            hybrid = extract_passages(
                "Tech news roundup", "Markets and gadgets.", ARTICLE, claim=CLAIM,
            )
        self.assertIn(DEBUNK, hybrid)

    def test_dense_unavailable_is_identical_to_lexical(self):
        """None = disabled or broken: behave as if the module did not exist."""
        expected = _lexical("T", "S", BURIED_LEDE, claim="prime minister resigned")
        with patch.dict(os.environ, {"SEMANTIC_PASSAGES": "false"}):
            flag_off = extract_passages("T", "S", BURIED_LEDE, claim="prime minister resigned")
        self.assertEqual(flag_off, expected)
        # And the lexical order itself is the one test_article_extraction pins.
        self.assertEqual(expected[2], "The prime minister resigned, effective immediately, citing health reasons.")

    def test_dense_finding_nothing_is_identical_to_lexical(self):
        """[] = the model ran and nothing was close enough."""
        for claim in ("prime minister resigned", CLAIM, ""):
            expected = _lexical("T", "S", ARTICLE, claim=claim)
            with patch.object(article_extractor, "semantic_order", return_value=[]):
                got = extract_passages("T", "S", ARTICLE, claim=claim)
            self.assertEqual(got, expected, claim)

    def test_title_and_snippet_still_lead(self):
        with patch.object(article_extractor, "semantic_order",
                          side_effect=_dense_finding("prohibited")):
            passages = extract_passages("Headline here", "Snippet here.", ARTICLE, claim=CLAIM)
        self.assertEqual(passages[:2], ["Headline here", "Snippet here."])

    def test_an_unrelated_claim_still_keeps_document_order(self):
        """The guarantee test_article_extraction pins must survive fusion."""
        for dense in (None, []):
            with patch.object(article_extractor, "semantic_order", return_value=dense):
                passages = extract_passages(
                    "T", "S", BURIED_LEDE, claim="quantum chromodynamics lattice",
                )
            self.assertEqual(passages[2], "The meeting began early on Tuesday in the capital city.")
            self.assertEqual(passages[3], "Officials from several departments attended the session.")

    def test_unnominated_sentences_follow_in_lexical_order(self):
        sentences = split_sentences(ARTICLE)
        debunk_index = sentences.index(DEBUNK)
        with patch.object(article_extractor, "semantic_order", return_value=[debunk_index]):
            passages = extract_passages("T", "S", ARTICLE, claim=CLAIM, max_passages=20)
        body = passages[2:]
        # Every sentence still appears exactly once.
        self.assertEqual(sorted(body), sorted(sentences))
        # The zero-overlap, unnominated sentence comes last, after everything
        # either signal nominated.
        self.assertEqual(body[-1], "Officials from several departments attended the briefing.")

    def test_ranking_info_reports_which_ranking_decided(self):
        info: dict = {}
        with patch.object(article_extractor, "semantic_order", return_value=None):
            extract_passages("T", "S", ARTICLE, claim=CLAIM, ranking_info=info)
        self.assertEqual(info, {"ranking": "lexical"})
        with patch.object(article_extractor, "semantic_order", return_value=[0]):
            extract_passages("T", "S", ARTICLE, claim=CLAIM, ranking_info=info)
        self.assertEqual(info, {"ranking": "hybrid"})


class TestReciprocalRankFusion(unittest.TestCase):

    def test_ties_keep_original_article_order(self):
        # Each item is first in one list and second in the other: equal scores.
        self.assertEqual(reciprocal_rank_fusion([5, 2], [2, 5]), [2, 5])
        self.assertEqual(reciprocal_rank_fusion([7], [3]), [3, 7])

    def test_agreement_between_lists_beats_either_alone(self):
        fused = reciprocal_rank_fusion([4, 1], [9, 1])
        self.assertEqual(fused[0], 1)

    def test_an_item_from_only_one_list_still_competes(self):
        self.assertEqual(set(reciprocal_rank_fusion([0, 1], [2])), {0, 1, 2})

    def test_empty_inputs(self):
        self.assertEqual(reciprocal_rank_fusion(), [])
        self.assertEqual(reciprocal_rank_fusion([], []), [])


class _FakeEmbedder:
    """Hand-built 2-D vectors: cosine with the claim = the x component."""

    def __init__(self, similarities: dict[str, float]):
        self.similarities = similarities
        self.calls: list[list[str]] = []

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        assert kwargs.get("normalize_embeddings") is True
        rows = [[1.0, 0.0]]  # the claim
        for text in texts[1:]:
            x = self.similarities.get(text, 0.0)
            rows.append([x, float(np.sqrt(max(0.0, 1 - x * x)))])
        return np.array(rows)


def _ranker(similarities=None, raises=None):
    embedder = _FakeEmbedder(similarities or {})

    def factory(name, device):
        assert device == "cpu"
        if raises:
            raise raises
        return embedder

    return PassageRanker(model_factory=factory, model_name="fake"), embedder


@patch.dict(os.environ, {"SEMANTIC_PASSAGES": "true"})
class TestPassageRanker(unittest.TestCase):

    def test_orders_by_similarity_and_applies_the_floor(self):
        ranker, _ = _ranker({"a": 0.31, "b": 0.90, "c": 0.29, "d": 0.31})
        self.assertEqual(ranker.semantic_order("claim", ["a", "b", "c", "d"]), [1, 0, 3])
        self.assertEqual(ranker.status["status"], "ready")

    def test_floor_is_configurable(self):
        ranker, _ = _ranker({"a": 0.31, "b": 0.45})
        with patch.dict(os.environ, {"SEMANTIC_MIN_SIMILARITY": "0.40"}):
            self.assertEqual(ranker.semantic_order("claim", ["a", "b"]), [1])

    def test_nothing_close_enough_is_an_empty_list_not_none(self):
        ranker, _ = _ranker({"a": 0.1})
        self.assertEqual(ranker.semantic_order("claim", ["a"]), [])

    def test_only_the_first_max_sentences_are_embedded(self):
        sentences = [f"s{i}" for i in range(MAX_SENTENCES + 25)]
        ranker, embedder = _ranker({s: 0.5 for s in sentences})
        order = ranker.semantic_order("claim", sentences)
        self.assertEqual(len(embedder.calls[0]), MAX_SENTENCES + 1)
        self.assertEqual(max(order), MAX_SENTENCES - 1)

    def test_disabled_returns_none_and_never_loads(self):
        ranker, embedder = _ranker({"a": 0.9})
        with patch.dict(os.environ, {"SEMANTIC_PASSAGES": "false"}):
            self.assertIsNone(ranker.semantic_order("claim", ["a"]))
            self.assertEqual(ranker.status["status"], "disabled")
        self.assertEqual(embedder.calls, [])

    def test_a_model_that_cannot_load_reports_failed_and_returns_none(self):
        ranker, _ = _ranker(raises=OSError("no such model"))
        self.assertIsNone(ranker.semantic_order("claim", ["a"]))
        self.assertEqual(ranker.status["status"], "failed")
        self.assertIn("no such model", ranker.status["error"])
        self.assertFalse(ranker.is_available)

    def test_an_encode_failure_skips_one_call_without_disabling_the_ranker(self):
        ranker, embedder = _ranker({"a": 0.9})
        working = embedder.encode
        embedder.encode = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom"))
        self.assertIsNone(ranker.semantic_order("claim", ["a"]))
        self.assertEqual(ranker.status["status"], "ready")
        self.assertIn("oom", ranker.status["error"])
        # The next article is ranked normally, and the error clears.
        embedder.encode = working
        self.assertEqual(ranker.semantic_order("claim", ["a"]), [0])
        self.assertIsNone(ranker.status["error"])

    def test_empty_claim_or_sentences(self):
        ranker, embedder = _ranker()
        self.assertEqual(ranker.semantic_order("", ["a"]), [])
        self.assertEqual(ranker.semantic_order("claim", []), [])
        self.assertEqual(embedder.calls, [])


if __name__ == "__main__":
    unittest.main()
