"""The NLI model always runs in float32 on CPU.

WHY THIS EXISTS:
Some NLI checkpoints declare ``"torch_dtype": "float16"`` in their config, and
the Hugging Face pipeline honours it. On a CPU half precision has no fast
kernels: with MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli one 19-token pair
took 13.7 s instead of 0.18 s, so every check ran past the proxy timeout while
looking merely "slow". Nothing errored; it was found by timing. This pins the
dtype the service asks for, without loading a model.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parent.parent
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

import nli_service  # noqa: E402
from nli_service import NLIService  # noqa: E402

try:
    import torch
    _AVAILABLE = True
except Exception:  # noqa: BLE001 - a broken torch install skips, not fails
    _AVAILABLE = False


@unittest.skipUnless(_AVAILABLE, "torch/transformers not importable")
class TestNLIDtype(unittest.TestCase):

    def test_the_real_pipeline_is_built_in_float32(self):
        captured = {}

        class _Pipe:
            model = None

        def fake_pipeline(task, **kwargs):
            captured.update(kwargs)
            return _Pipe()

        # nli_service._hf_pipeline, not transformers.pipeline: transformers is
        # a lazy module that ignores attribute patches on `from ... import`.
        with patch.object(nli_service, "_hf_pipeline", lambda: fake_pipeline):
            NLIService(model_name="some/checkpoint").warm_up()
        self.assertIs(captured.get("dtype"), torch.float32)
        self.assertEqual(captured.get("device"), -1)

    def test_an_injected_factory_is_not_handed_a_dtype(self):
        """Test doubles keep their old signature."""
        captured = {}

        def factory(task, **kwargs):
            captured.update(kwargs)
            return object()

        NLIService(pipeline_factory=factory, model_name="x").warm_up()
        self.assertNotIn("dtype", captured)


if __name__ == "__main__":
    unittest.main()
