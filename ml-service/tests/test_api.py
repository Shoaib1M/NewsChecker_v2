"""End-to-end tests for the FastAPI app: /api/health and /api/check.

Requires fastapi/numpy/pandas (ml-service/requirements.txt) to be installed —
skipped automatically if they aren't, so this doesn't break environments
that only run the pure-stdlib unit tests.
"""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

try:
    from fastapi.testclient import TestClient
    _FASTAPI_AVAILABLE = True
except ImportError:
    _FASTAPI_AVAILABLE = False


@unittest.skipUnless(_FASTAPI_AVAILABLE, "fastapi not installed")
class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import main
        cls.main = main
        # Entering as a context manager is required for FastAPI's lifespan
        # (model loading) to actually run — a bare TestClient(app) skips it.
        cls._client_cm = TestClient(main.app)
        cls.client = cls._client_cm.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls._client_cm.__exit__(None, None, None)

    def test_health_reports_nli_and_provider_state(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("status", body["nli"])
        self.assertIn("status", body["passage_ranking"])
        self.assertIn("status", body["explanations"])
        # The legacy LIAR classifier was removed; health no longer reports it.
        self.assertNotIn("model_loaded", body)
        self.assertIn(body["nli"]["status"], {"disabled", "loading", "ready", "failed"})
        for provider in ("gnews", "guardian", "newsapi", "duckduckgo"):
            self.assertIn(provider, body["search_providers"])

    def test_deterministic_claim_short_circuits_pipeline(self):
        """A claim knowledge_verifier can answer deterministically must never
        touch the network-dependent evidence pipeline."""
        with patch.object(self.main, "run_pipeline") as mock_pipeline:
            response = self.client.post(
                "/api/check",
                json={"statement": "Water freezes at 0°C at sea level."},
            )
            mock_pipeline.assert_not_called()

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["verdict"], "true")
        self.assertEqual(body["verification"]["status"], "supported")

    def test_search_failed_is_not_reported_as_verified(self):
        """When retrieval fails outright, the response must say SEARCH_FAILED,
        never quietly report insufficient_evidence indistinguishable from a
        real 'nothing relevant found' outcome."""
        from evidence_pipeline import PipelineOutcome

        failed_outcome = PipelineOutcome(
            stance={
                "support": 0.0, "contradiction": 0.0, "net": 0.0,
                "verdict": "insufficient evidence", "status": "insufficient_evidence",
                "nli_available": False, "evidence_count": 0,
            },
            evidence=[],
            retrieval_status="SEARCH_FAILED",
            diagnostics=[{"provider": "gnews", "status": "failed", "error": "timeout"}],
            candidate_count=0,
            relevant_count=0,
        )
        with patch.object(self.main, "run_pipeline", return_value=failed_outcome):
            response = self.client.post(
                "/api/check",
                json={"statement": "Some claim with no deterministic answer occurred yesterday."},
            )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["retrieval"]["status"], "SEARCH_FAILED")
        self.assertEqual(body["evidence"]["independent_groups"], 0)

    def test_short_statement_is_rejected(self):
        response = self.client.post("/api/check", json={"statement": "hi"})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
