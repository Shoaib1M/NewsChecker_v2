"""Dense (embedding) passage ranking, fused with the lexical ranking.

WHY THIS EXISTS:
``article_extractor.extract_passages`` decides which handful of sentences NLI
ever sees, and it ranked them by exact content-word overlap with the claim.
That is blind in exactly the case a fact-checker exists for. Posts repeating
a false claim use its wording; the sources refuting it use their own. For
"The United States banned Google across all its cities", the sentence
"Washington has not prohibited the search giant anywhere in the country"
shares no word with the claim — not "banned", not "Google", not "cities" — so
it scored zero and lost its slot to any sentence that merely mentioned Google.
The article was then scored neutral, and the refutation never counted.

A sentence embedding places that sentence next to the claim because it MEANS
something close to it. This module provides that ranking, and nothing else:

  * ``semantic_order``         which sentences are semantically close to the
                               claim, best first
  * ``reciprocal_rank_fusion`` merge that ordering with the lexical one

WHY FUSE RATHER THAN REPLACE:
Lexical overlap is not wrong, it is incomplete. Exact matches on a name or a
number are the strongest relevance signal there is, and an embedding model
blurs them — "raised rates by 0.25%" and "raised rates by 0.75%" embed almost
identically. Reciprocal rank fusion keeps both signals without a weight
constant to tune: a sentence near the top of either list rises, a sentence
near the top of both rises furthest, and the two scores never have to be put
on a common scale (cosine similarity and word counts are not comparable).

WHY THERE IS NO VECTOR DATABASE:
Each check reads a few hundred freshly-fetched sentences that will never be
queried again. Indexing them would cost more than embedding them, so they are
embedded in memory per request and thrown away.

DEGRADING:
Same contract as ``nli_service``: a state machine with observable status, and
a model that cannot load is reported, never guessed around. ``semantic_order``
returns ``None`` when dense ranking is disabled or broken, and the caller
falls back to exactly the old lexical behaviour. ``[]`` means the model ran
and found nothing close enough — also a lexical fallback, but a different
fact, which is why the two are distinguished.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Sequence

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Below this cosine similarity a sentence is not considered semantically
# related to the claim. MiniLM places almost any two English news sentences
# around 0.05-0.20 apart from chance alone, so a low floor would let the dense
# list nominate filler — and every filler sentence it nominates outranks a
# genuine lexical match it displaced. Tuned in news_benchmark (see README).
DEFAULT_MIN_SIMILARITY = 0.30

# Sentences embedded per article. A long page is mostly comments, related
# links and footer that survived extraction; the reporting is near the top.
# The cap bounds the per-request cost (MiniLM on CPU embeds a few hundred
# short sentences a second) so one huge page cannot eat the evidence budget.
MAX_SENTENCES = 150

# k in 1 / (k + rank). 60 is the constant from the original RRF paper
# (Cormack et al., 2009) and the de facto default; it flattens the curve so
# that rank 1 vs rank 2 in one list does not outweigh agreement between lists.
RRF_K = 60


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def semantic_passages_enabled() -> bool:
    """Read at call time, not import time.

    The benchmark compares SEMANTIC_PASSAGES=false against true, and tests
    flip it per case; a value frozen at import would make both lie.
    """
    return _flag("SEMANTIC_PASSAGES", default=True)


def min_similarity() -> float:
    try:
        return float(os.getenv("SEMANTIC_MIN_SIMILARITY", DEFAULT_MIN_SIMILARITY))
    except ValueError:
        return DEFAULT_MIN_SIMILARITY


# ── Singleton holder ─────────────────────────────────────────────────
_instance: "PassageRanker | None" = None
_instance_lock = threading.Lock()


def get_passage_ranker() -> "PassageRanker":
    """Return the application-wide ranker (created on first call)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = PassageRanker()
    return _instance


class PassageRanker:
    """Lazy-loaded, thread-safe sentence embedder with observable status.

    Mirrors ``NLIService`` deliberately: ``/api/health`` reports both the same
    way, and anyone who has read one knows how the other fails.

    Parameters
    ----------
    model_factory : callable, optional
        Injected for tests; replaces ``sentence_transformers.SentenceTransformer``.
    model_name : str, optional
        Defaults to ``PASSAGE_EMBED_MODEL`` or all-MiniLM-L6-v2.
    """

    DISABLED = "disabled"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"

    def __init__(
        self,
        model_factory: Callable | None = None,
        model_name: str | None = None,
    ):
        self.model_name: str = model_name or os.getenv(
            "PASSAGE_EMBED_MODEL", DEFAULT_MODEL
        )
        self._model_factory = model_factory
        self._model = None
        self._lock = threading.Lock()
        # A LOAD failure is sticky: retrying a failed download inside every
        # request would add its timeout to each one. An ENCODE failure is not:
        # one bad batch says nothing about the next, and switching dense
        # ranking off for the rest of the process over it would be a silent,
        # permanent downgrade. It is recorded here for /api/health instead.
        self._failed_error: str | None = None
        self._last_encode_error: str | None = None

    # ── Public status API ────────────────────────────────────────────
    @property
    def _status(self) -> str:
        # Derived on every read: the enable flag is read at call time, so a
        # status cached at construction would contradict it.
        if not semantic_passages_enabled():
            return self.DISABLED
        if self._failed_error is not None:
            return self.FAILED
        return self.READY if self._model is not None else self.LOADING

    @property
    def status(self) -> dict:
        """Authoritative dense-ranking status — used by ``/api/health``."""
        status = self._status
        error = None
        if status == self.DISABLED:
            error = "dense passage ranking disabled via SEMANTIC_PASSAGES"
        elif status == self.FAILED:
            error = self._failed_error
        elif self._last_encode_error:
            error = f"last request fell back to lexical: {self._last_encode_error}"
        return {
            "enabled": status != self.DISABLED,
            "model": self.model_name,
            "status": status,
            "error": error,
        }

    @property
    def is_available(self) -> bool:
        return self._status in {self.LOADING, self.READY}

    def warm_up(self) -> dict:
        """Load the model now rather than inside the first request.

        Same reasoning as ``NLIService.warm_up``: a first-use download inside
        an HTTP request is unbounded by the evidence budget and invisible.
        """
        if semantic_passages_enabled():
            self._ensure_loaded()
        return self.status

    # ── Lazy model loading ───────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._model is not None or self._failed_error is not None:
            return
        with self._lock:
            if self._model is not None or self._failed_error is not None:
                return
            try:
                factory = self._model_factory
                if factory is None:
                    from sentence_transformers import SentenceTransformer
                    factory = SentenceTransformer
                self._model = factory(self.model_name, device="cpu")
                print(f"Passage embedding model loaded: {self.model_name}")
            except Exception as exc:  # noqa: BLE001 - any load failure degrades
                self._failed_error = str(exc) or exc.__class__.__name__
                print(f"Passage embedding model failed to load: {self._failed_error}")

    # ── Ranking ──────────────────────────────────────────────────────
    def semantic_order(
        self, claim: str, sentences: Sequence[str]
    ) -> list[int] | None:
        """Indices of ``sentences`` semantically close to ``claim``, best first.

        Returns
        -------
        None
            Dense ranking is disabled, the model could not load, or encoding
            failed. The caller must behave exactly as if this module did not
            exist.
        []
            The model ran and no sentence reached ``SEMANTIC_MIN_SIMILARITY``.
        list[int]
            Indices into ``sentences`` at or above the floor, by descending
            cosine similarity; equal similarities keep article order.

        Only the first ``MAX_SENTENCES`` sentences are considered; the rest
        are never nominated, and so keep their lexical position.
        """
        if not semantic_passages_enabled():
            return None
        if not claim or not claim.strip() or not sentences:
            return []

        self._ensure_loaded()
        if self._model is None:
            return None

        window = list(sentences[:MAX_SENTENCES])
        try:
            vectors = self._model.encode(
                [claim, *window],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
                batch_size=32,
            )
        except Exception as exc:  # noqa: BLE001
            # Reported, not swallowed: /api/health shows it, and this article
            # falls back to lexical ranking. The ranker stays usable.
            self._last_encode_error = str(exc) or exc.__class__.__name__
            print(f"Passage embedding failed: {exc}")
            return None
        self._last_encode_error = None

        claim_vector, sentence_vectors = vectors[0], vectors[1:]
        # Normalised embeddings: the dot product IS the cosine similarity.
        similarities = sentence_vectors @ claim_vector
        floor = min_similarity()
        ranked = sorted(
            (index for index in range(len(window)) if float(similarities[index]) >= floor),
            key=lambda index: (-float(similarities[index]), index),
        )
        return ranked


def semantic_order(claim: str, sentences: Sequence[str]) -> list[int] | None:
    """Module-level convenience for the application-wide ranker.

    This is the seam tests monkeypatch, so the pipeline never needs a model.
    """
    return get_passage_ranker().semantic_order(claim, sentences)


def reciprocal_rank_fusion(*orders: Sequence[int], k: int = RRF_K) -> list[int]:
    """Merge several rankings of the same items into one.

    Each item scores ``sum(1 / (k + rank))`` over the lists it appears in,
    with ``rank`` starting at 1. An item absent from a list simply gets
    nothing from it — so a sentence found only by the embedding model still
    competes, and one found by both lists beats either alone.

    Ties are broken by the item itself, i.e. original article order. That is
    the same tie rule the lexical ranker uses, so fusion never reshuffles
    sentences that neither signal can tell apart.
    """
    scores: dict[int, float] = {}
    for order in orders:
        for rank, item in enumerate(order, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))
