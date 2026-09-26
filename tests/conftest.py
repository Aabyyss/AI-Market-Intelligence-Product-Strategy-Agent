"""Shared fixtures for the service-level tests.

An empty conftest.py at the project root already puts the repo root on
sys.path (so ``import market_intel`` resolves). This one holds the
fixtures both test_api.py and test_console_and_launcher.py drive the app
through: a session corpus seeded from the tracked CI fixture, and a
TestClient pinned to it.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import run_corpus
from market_intel import api

FIXTURE = "tests/fixtures/ci_corpus.json"


@pytest.fixture(scope="session")
def corpus_db(tmp_path_factory):
    """A DB + vector index built from the tracked fixture (no network)."""
    path = tmp_path_factory.mktemp("corpus") / "market_intel.db"
    run_corpus.seed(FIXTURE, str(path))
    return str(path)


@pytest.fixture
def client(corpus_db, tmp_path, monkeypatch):
    """A TestClient wired to the fixture corpus and a throwaway report dir."""
    monkeypatch.setattr(api, "DB_PATH", corpus_db)
    monkeypatch.setattr(api, "REPORT_DIR", str(tmp_path / "reports"))
    # Pin the provider. /ask and /reports call get_provider() per request,
    # which auto-detects by probing localhost:11434 — so without this they
    # 503 on a machine with nothing running (CI) and quietly pass on a box
    # that happens to have Ollama up. No LLM is ever reached either way:
    # the tests that get that far stub answer_question / build_report.
    monkeypatch.setattr(api, "resolve_provider", lambda: "ollama")
    with api._JOBS_LOCK:
        api._JOBS.clear()
    with TestClient(api.app) as test_client:
        yield test_client
