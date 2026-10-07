"""Test-session defaults: no network models, no LLM calls.

WHY THIS EXISTS:
Tests must never download a model or call a hosted API — a suite that passes
only on a machine with a warm cache and a valid key is measuring the machine.
Both features added in the RAG upgrade default ON at runtime, so they are
switched OFF here, before any test module imports the service. A test that
exercises one of them turns it back on explicitly and mocks the model.

``setdefault`` rather than assignment: a developer who deliberately exports
SEMANTIC_PASSAGES=true to try something locally gets what they asked for.

The Hugging Face offline flags make "never download" a guarantee rather than
a convention. Existing API tests enter FastAPI's lifespan, which preloads the
NLI model; offline, that load comes from the local cache or fails cleanly
into the service's `failed` state — which the pipeline already treats as
abstention — instead of fetching hundreds of megabytes mid-test.
"""

import os

os.environ.setdefault("SEMANTIC_PASSAGES", "false")
os.environ.setdefault("EXPLANATIONS_ENABLED", "false")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
