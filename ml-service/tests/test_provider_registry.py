"""Tests for providers/registry.py: dedup, and that a provider failure is
reported as a diagnostic (provider_failure), never silently treated as
zero_results or allowed to crash the whole search."""

import sys
from pathlib import Path
import unittest
from unittest.mock import patch

SERVICE_DIR = Path(__file__).resolve().parents[1]
if str(SERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(SERVICE_DIR))

from providers import SearchResult
import providers.registry as registry
from providers.registry import deduplicate, search_all_providers


class DeduplicateTests(unittest.TestCase):
    def test_removes_exact_url_duplicates(self):
        results = [
            SearchResult(url="https://a.example/x", title="Same story"),
            SearchResult(url="https://a.example/x", title="Same story"),
        ]
        self.assertEqual(len(deduplicate(results)), 1)

    def test_removes_near_duplicate_syndicated_titles(self):
        results = [
            SearchResult(url="https://a.example/x", title="Reuters: Global markets rally on rate cut news today"),
            SearchResult(url="https://b.example/y", title="Global markets rally on rate cut news today"),
        ]
        self.assertEqual(len(deduplicate(results)), 1)

    def test_distinct_stories_both_kept(self):
        results = [
            SearchResult(url="https://a.example/x", title="Government announces new tax policy"),
            SearchResult(url="https://b.example/y", title="Local team wins championship game"),
        ]
        self.assertEqual(len(deduplicate(results)), 2)


class SearchAllProvidersTests(unittest.TestCase):
    def setUp(self):
        # These tests predate the keyless providers. Patching PROVIDERS alone
        # left Google News and Wikipedia enabled by default, so every test in
        # this class sent real queries to the live internet — and, online,
        # their real results were counted against assertions written for one
        # fake provider. Offline they "passed" because the live calls failed.
        patcher = patch.object(registry, "KEYLESS_PROVIDERS", [])
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_provider_exception_becomes_failed_diagnostic_not_silent_empty(self):
        """A provider that raises must surface as status='failed' with the
        error recorded — never collapse into an indistinguishable empty
        result set (which would look identical to a real zero-results query)."""
        def _boom(query, max_results=5):
            raise RuntimeError("boom")

        fake_providers = [("gnews", "GNEWS_API_KEY", _boom)]
        with patch.dict("os.environ", {"GNEWS_API_KEY": "test-key"}, clear=False), \
             patch.object(registry, "PROVIDERS", fake_providers), \
             patch.object(registry, "ddg_search", return_value=[]):
            results, diagnostics = search_all_providers(["some query"], use_duckduckgo=False)

        self.assertEqual(results, [])
        gnews_diags = [d for d in diagnostics if d.provider == "gnews"]
        self.assertTrue(gnews_diags)
        self.assertTrue(all(d.status == "failed" for d in gnews_diags))
        self.assertTrue(all(d.error == "boom" for d in gnews_diags))

    def test_disabled_provider_is_reported_disabled_not_failed(self):
        fake_providers = [("newsapi", "NEWSAPI_KEY", lambda query, max_results=5: [])]
        with patch.dict("os.environ", {}, clear=True), \
             patch.object(registry, "PROVIDERS", fake_providers), \
             patch.object(registry, "ddg_search", return_value=[]):
            _, diagnostics = search_all_providers(["some query"], use_duckduckgo=False)

        newsapi_diags = [d for d in diagnostics if d.provider == "newsapi"]
        self.assertTrue(newsapi_diags)
        self.assertTrue(all(not d.enabled for d in newsapi_diags))
        self.assertTrue(all(d.status == "disabled" for d in newsapi_diags))

    def test_successful_results_flow_through_with_diagnostics(self):
        fake_result = SearchResult(url="https://reuters.com/story", title="A real story", provider="gnews")
        fake_providers = [("gnews", "GNEWS_API_KEY", lambda query, max_results=5: [fake_result])]
        with patch.dict("os.environ", {"GNEWS_API_KEY": "test-key"}, clear=False), \
             patch.object(registry, "PROVIDERS", fake_providers), \
             patch.object(registry, "ddg_search", return_value=[]):
            results, diagnostics = search_all_providers(["some query"], use_duckduckgo=False)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].url, "https://reuters.com/story")
        gnews_diags = [d for d in diagnostics if d.provider == "gnews"]
        self.assertTrue(any(d.status == "success" for d in gnews_diags))


if __name__ == "__main__":
    unittest.main()
