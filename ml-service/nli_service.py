"""Single, authoritative NLI (Natural Language Inference) service.

Every part of the application — the health endpoint, the evidence pipeline,
and the API response — queries this one service for NLI state.  There is no
second copy of the model status anywhere.

State machine
─────────────
    disabled  ──►  (NLI_ENABLED is falsy; model is never loaded)
    loading   ──►  (model download / initialization in progress)
    ready     ──►  (model loaded, inference available)
    failed    ──►  (model load or inference error; stores error message)

Thread safety
─────────────
A threading lock prevents concurrent duplicate initialization when the first
few requests arrive simultaneously.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Iterable


# ── Singleton holder ─────────────────────────────────────────────────
_instance: "NLIService | None" = None
_instance_lock = threading.Lock()


def get_nli_service() -> "NLIService":
    """Return the application-wide NLI service (created on first call)."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = NLIService()
    return _instance


# ── Label normalisation ──────────────────────────────────────────────
# Some HuggingFace model configs don't define semantic id2label names, so
# the pipeline emits generic "LABEL_0"/"LABEL_1"/"LABEL_2" strings instead
# of "entailment"/"contradiction"/"neutral". Label ORDER is not standardized
# across NLI models — guessing wrong silently inverts every verdict — so we
# only resolve indexed labels for models explicitly verified here. Never add
# an entry without confirming the model's actual id2label order first.
# Order confirmed from the model's own config at load time:
#   id2label={0: 'contradiction', 1: 'entailment', 2: 'neutral'}
# An earlier version of this table had 1 and 2 swapped. It never produced a
# wrong verdict, because these models ship real label names and the
# substring path below wins — but a guessed order that silently inverts
# entailment and neutral is exactly the failure this table exists to
# prevent, so it is now the observed order, not an assumed one.
_KNOWN_INDEXED_LABEL_ORDERS: dict[str, dict[str, str]] = {
    "cross-encoder/nli-deberta-v3-small": {
        "label_0": "contradiction", "label_1": "entailment", "label_2": "neutral",
    },
    "cross-encoder/nli-deberta-v3-xsmall": {
        "label_0": "contradiction", "label_1": "entailment", "label_2": "neutral",
    },
    "cross-encoder/nli-deberta-v3-base": {
        "label_0": "contradiction", "label_1": "entailment", "label_2": "neutral",
    },
}


class UnresolvableLabelError(ValueError):
    """Raised when a model emits a label we cannot safely map to a class."""


def _normalise_label(label: str, model_name: str) -> str:
    """Map one model output label to a canonical NLI class.

    Named labels ("entailment"/"contradiction"/"neutral") are safe to match
    by substring regardless of index order. Indexed "LABEL_N" labels are
    only resolved via the explicit, per-model table above — an unrecognized
    model emitting indexed labels raises rather than guessing.
    """
    normalised = label.lower().strip()
    if "entail" in normalised:
        return "entailment"
    if "contradict" in normalised:
        return "contradiction"
    if "neutral" in normalised:
        return "neutral"

    indexed_key = normalised.replace(" ", "_")
    order = _KNOWN_INDEXED_LABEL_ORDERS.get(model_name)
    if order and indexed_key in order:
        return order[indexed_key]

    raise UnresolvableLabelError(
        f"Model '{model_name}' emitted unrecognized label '{label}' with no "
        "known named or indexed mapping — refusing to guess."
    )


# ── Model loading ────────────────────────────────────────────────────
def _hf_pipeline():
    """``transformers.pipeline``, imported lazily (and replaceable in tests)."""
    from transformers import pipeline
    return pipeline


def _cpu_pipeline(task: str, **kwargs):
    """The real Hugging Face pipeline, always in float32.

    The pipeline otherwise honours the checkpoint's own ``torch_dtype``, and
    some checkpoints ship "float16" — MoritzLaurer/DeBERTa-v3-base-mnli-fever-
    anli does. Half precision on a CPU has no fast kernels: one 19-token pair
    took 13.7 s instead of 0.18 s, eight pairs 153 s instead of 1.5 s, and
    every check ran silently past the proxy timeout. This service only ever
    runs on CPU, so float32 is always right.
    """
    import torch
    return _hf_pipeline()(task, dtype=torch.float32, **kwargs)


# ── Core service ─────────────────────────────────────────────────────
class NLIService:
    """Lazy-loaded, thread-safe NLI scorer with observable status.

    Parameters
    ----------
    pipeline_factory : callable, optional
        Injected factory for testing (replaces ``transformers.pipeline``).
    model_name : str, optional
        HuggingFace model identifier.  Defaults to the ``NLI_MODEL``
        environment variable or ``cross-encoder/nli-deberta-v3-base``.
    """

    # Valid states
    DISABLED = "disabled"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"

    def __init__(
        self,
        pipeline_factory: Callable | None = None,
        model_name: str | None = None,
    ):
        # Defaults to an MNLI + FEVER + ANLI checkpoint, not the SNLI-trained
        # cross-encoder family. SNLI labels unrelated pairs "contradiction",
        # and cross-encoder/nli-deberta-v3-small scored an unrelated sentence
        # as a 1.00 contradiction of the claim. On stance_sweep.py's labelled
        # corpus this checkpoint scores 0.91 accuracy against 0.61 and invents
        # 2 positions against 6. It emits named labels, so it needs no entry in
        # the label-order table; the cross-encoder models remain one env var
        # away and are pre-verified there.
        self.model_name: str = model_name or os.getenv(
            "NLI_MODEL", "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
        )
        self._pipeline_factory = pipeline_factory
        self._pipeline = None
        self._lock = threading.Lock()

        # Determine initial state from environment
        enabled_raw = os.getenv("NLI_ENABLED", "true")
        self._enabled: bool = enabled_raw.lower() in {"1", "true", "yes", "on"}
        self._status: str = self.DISABLED if not self._enabled else self.LOADING
        self._error: str | None = (
            None if self._enabled
            else "NLI disabled via NLI_ENABLED environment variable"
        )

        # If explicitly enabled, we still defer the actual model load to the
        # first call to ``score_many()``.  This keeps startup fast while
        # letting the health endpoint report "loading" rather than the stale
        # "loaded lazily" message the old code used.
        if self._enabled:
            self._status = self.LOADING

    # ── Public status API ────────────────────────────────────────────
    @property
    def status(self) -> dict:
        """Authoritative NLI status — used by ``/api/health``."""
        return {
            "enabled": self._enabled,
            "model": self.model_name,
            "status": self._status,
            "error": self._error,
        }

    @property
    def is_ready(self) -> bool:
        return self._status == self.READY

    @property
    def is_available(self) -> bool:
        """True when the model *could* score — either already loaded or
        still waiting for the first call to trigger loading."""
        return self._status in {self.LOADING, self.READY}

    def warm_up(self) -> dict:
        """Load the model now rather than on the first scoring call.

        Called at service startup. Without this the first evidence-requiring
        request pays for a multi-hundred-megabyte model download *inside the
        HTTP request*, which no upstream timeout can absorb — and it happens
        invisibly, since nothing is logged until the load finishes. Doing it
        at boot makes the cost visible in the startup log and leaves
        ``/api/health`` honest about readiness before any traffic arrives.
        """
        self._ensure_loaded()
        return self.status

    # ── Lazy model loading ───────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        """Load the model exactly once, under a lock."""
        if self._pipeline is not None or self._status in {self.DISABLED, self.FAILED}:
            return

        with self._lock:
            # Double-check after acquiring the lock
            if self._pipeline is not None or self._status in {self.DISABLED, self.FAILED}:
                return

            self._status = self.LOADING
            try:
                factory = self._pipeline_factory or _cpu_pipeline
                self._pipeline = factory(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=-1,  # CPU
                )
                self._status = self.READY
                self._error = None

                # Log the model's actual id2label mapping so the real label
                # scheme is observable in deployment logs — this is the
                # verification step that must happen whenever NLI_MODEL
                # changes, since label order is not standardized across
                # models (see _KNOWN_INDEXED_LABEL_ORDERS above).
                try:
                    id2label = self._pipeline.model.config.id2label
                    print(f"NLI model loaded: {self.model_name} — id2label={id2label}")
                except Exception:
                    print(f"NLI model loaded: {self.model_name}")
            except Exception as exc:
                self._status = self.FAILED
                self._error = str(exc)
                print(f"NLI model failed to load: {exc}")

    # ── Inference ────────────────────────────────────────────────────
    def score_many(
        self, claim: str, passages: Iterable[str]
    ) -> list[dict]:
        """Score claim against each passage.

        Returns a list of dicts, one per passage::

            {"entailment": float, "contradiction": float,
             "neutral": float, "available": bool}

        When the model is unavailable, every entry has ``available=False``
        and the caller **must** treat this as abstention, not as evidence.
        """
        passages = list(passages)
        if not passages:
            return []

        self._ensure_loaded()

        if self._pipeline is None:
            return [
                {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "available": False,
                }
                for _ in passages
            ]

        pairs = [{"text": passage, "text_pair": claim} for passage in passages]
        try:
            results = self._pipeline(
                pairs, top_k=None, truncation=True, max_length=512
            )
        except TypeError:
            # Fallback for injected test doubles or older pipeline versions
            results = [self._pipeline(pair) for pair in pairs]
        except Exception as exc:
            self._error = str(exc)
            self._status = self.FAILED
            return [
                {
                    "entailment": 0.0,
                    "contradiction": 0.0,
                    "neutral": 1.0,
                    "available": False,
                }
                for _ in passages
            ]

        scores: list[dict] = []
        for output in results:
            if isinstance(output, dict):
                output = [output]
            values = {"entailment": 0.0, "contradiction": 0.0, "neutral": 0.0}
            try:
                for item in output:
                    values[_normalise_label(str(item.get("label", "")), self.model_name)] = float(
                        item.get("score", 0.0)
                    )
                values["available"] = True
            except UnresolvableLabelError as exc:
                # An unrecognized label scheme means we cannot trust any
                # score from this model — abstain entirely rather than risk
                # an inverted (silently wrong) verdict.
                self._status = self.FAILED
                self._error = str(exc)
                print(f"NLI label mapping failed: {exc}")
                values = {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0, "available": False}
            scores.append(values)
        return scores
