"""Tests for the Phase 6 API service.

Everything here is exercised through the real HTTP layer (FastAPI's
TestClient), because the interesting bugs in a service live in the parts
a unit test skips: status codes, validation, the job state machine, the
request-id plumbing, and whether an error leaks a traceback.

Two deliberate choices keep this fast and hermetic:

  * the corpus is seeded once per session from the tracked fixture
    (``run_corpus.py``'s exact CI path), so search/eval run against real
    embeddings without any network call;
  * no LLM is ever called — the one place that would, ``/ask``, has the
    answer layer stubbed. The suite must pass on a machine with nothing
    running, same rule the rest of the repo follows. That includes the
    *provider check*: ``/ask`` and ``/reports`` resolve the LLM per request
    by probing localhost:11434, so the ``client`` fixture pins the provider
    rather than trusting whatever happens to be listening.

Run with:
    pip install -r requirements-dev.txt
    python -m pytest tests/test_api.py -v
"""

import time

import pytest
from fastapi.testclient import TestClient

import run_corpus
from market_intel import api, config, fetch
from market_intel.llm import LLMError

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


def fake_summary(**overrides) -> dict:
    """A build_report-shaped summary, so job tests never touch the agents."""
    summary = {
        "brief": "fees and developer payouts",
        "competitors": list(config.COMPETITORS),
        "provider": "ollama",
        "model": "llama3.2",
        "queries": ['"shopify" fees'],
        "evidence": 7,
        "per_competitor": {c: 2 for c in config.COMPETITORS},
        "verdicts": {"SUPPORTED": 3, "PARTIAL": 1, "UNSUPPORTED": 0},
        "citation_audit": {"used": [1, 2, 3], "out_of_range": [],
                           "uncited_ids": [], "uncited_competitor": 0},
        "unstructured_sections": [],
        "generated_at": "2026-01-02T03:04:05+00:00",
        "report_path": None,
        "markdown": "# Market report\n\nEvidence [1].\n",
    }
    summary.update(overrides)
    return summary


def wait_for_job(client, job_id: str, timeout: float = 20.0) -> dict:
    """Poll a job until it leaves queued/running (the worker is a thread)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/jobs/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish within {timeout}s")


# --- liveness / introspection -----------------------------------------


def test_root_lists_endpoints(client):
    body = client.get("/").json()
    assert "/health" in body["endpoints"]


def test_health_reports_corpus_and_version(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["version"] == api.__version__
    assert body["posts"] > 0
    assert body["chunks"] >= body["posts"]
    assert body["index_built_at"]
    assert body["embeddings"]


def test_health_stays_ok_without_an_llm(client, monkeypatch):
    """A health check that 500s when a dependency is down is unreadable."""

    def no_provider():
        raise LLMError("no LLM provider found")

    monkeypatch.setattr(api, "resolve_provider", no_provider)
    body = client.get("/health")
    assert body.status_code == 200
    payload = body.json()
    assert payload["llm_provider"] is None
    assert payload["llm_detail"] == "no LLM provider found"


def test_health_without_a_corpus_reports_nulls(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "nope.db"))
    body = client.get("/health").json()
    assert body["posts"] is None
    assert body["chunks"] is None


# --- request context ---------------------------------------------------


def test_request_id_is_echoed_and_honoured(client):
    resp = client.get("/health", headers={"X-Request-ID": "abc123"})
    assert resp.headers["X-Request-ID"] == "abc123"


def test_request_id_is_generated_when_absent(client):
    resp = client.get("/health")
    assert len(resp.headers["X-Request-ID"]) == 8


def test_error_body_carries_the_request_id(client):
    resp = client.get("/jobs/does-not-exist", headers={"X-Request-ID": "req-42"})
    assert resp.status_code == 404
    body = resp.json()
    assert body["request_id"] == "req-42"
    assert "does-not-exist" in body["error"]


def test_unhandled_error_is_logged_not_leaked(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(api, "search", boom)
    # The default TestClient re-raises server exceptions, which would
    # bypass the handler under test — so drive this one as a real client:
    # a 500 response, not a traceback.
    with TestClient(api.app, raise_server_exceptions=False) as raw:
        resp = raw.post("/search", json={"query": "anything"})
    assert resp.status_code == 500
    body = resp.json()
    assert body["error"] == "internal error: RuntimeError"
    assert "secret internal detail" not in resp.text


# --- search -----------------------------------------------------------


def test_search_returns_ranked_hits(client):
    resp = client.post("/search", json={"query": "checkout fees", "top_k": 5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["count"] == 5
    assert [h["score"] for h in body["hits"]] == sorted(
        (h["score"] for h in body["hits"]), reverse=True
    )
    assert all(h["post_id"] and h["title"] for h in body["hits"])
    assert all(len(h["excerpt"]) <= 401 for h in body["hits"])


def test_search_competitor_filter(client):
    resp = client.post(
        "/search", json={"query": "pricing", "top_k": 3, "competitor": "SHOPIFY"}
    )
    assert resp.status_code == 200
    assert resp.json()["count"] == 3
    assert {h["competitor"] for h in resp.json()["hits"]} == {"shopify"}


def test_search_rejects_unknown_competitor(client):
    resp = client.post("/search", json={"query": "x", "competitor": "myspace"})
    assert resp.status_code == 422
    assert "unknown competitor" in resp.text


@pytest.mark.parametrize("payload", [
    {"query": ""},                 # too short
    {"query": "x", "top_k": 0},    # below the floor
    {"query": "x", "top_k": 99},   # above the ceiling
])
def test_search_validates_input(client, payload):
    assert client.post("/search", json=payload).status_code == 422


def test_search_without_a_corpus_is_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "DB_PATH", str(tmp_path / "missing.db"))
    resp = client.post("/search", json={"query": "fees"})
    assert resp.status_code == 503
    assert "corpus not found" in resp.json()["error"]


# --- ask ---------------------------------------------------------------


def stub_answer(monkeypatch, sources, cited, answer="Grounded answer [1]."):
    monkeypatch.setattr(
        api, "answer_question",
        lambda conn, q, top_k, competitor, provider: {
            "question": q, "answer": answer, "sources": sources, "cited": cited,
        },
    )


def make_sources(n=3):
    return [
        {"post_id": f"p{i}", "title": f"Post {i}", "url": f"https://x/{i}"}
        for i in range(1, n + 1)
    ]


def test_ask_maps_citations_to_the_retrieved_sources(client, monkeypatch):
    stub_answer(monkeypatch, make_sources(3), cited=[1, 3])
    body = client.post("/ask", json={"query": "why switch?", "top_k": 3}).json()
    assert body["grounded"] is True
    assert body["sources_considered"] == 3
    assert [c["n"] for c in body["citations"]] == [1, 3]
    # [n] indexes the evidence list the model saw — off-by-one here would
    # silently attribute every claim to the wrong post.
    assert [c["post_id"] for c in body["citations"]] == ["p1", "p3"]
    assert body["citations"][1]["url"] == "https://x/3"


def test_ask_reports_an_ungrounded_answer(client, monkeypatch):
    stub_answer(monkeypatch, make_sources(2), cited=[], answer="I don't know.")
    body = client.post("/ask", json={"query": "unanswerable"}).json()
    assert body["grounded"] is False
    assert body["citations"] == []


def test_ask_without_a_provider_is_503(client, monkeypatch):
    def no_provider(explicit=None):
        raise api.HTTPException(503, detail="no LLM provider found")

    monkeypatch.setattr(api, "get_provider", no_provider)
    resp = client.post("/ask", json={"query": "hello"})
    assert resp.status_code == 503


def test_llm_endpoints_do_not_depend_on_an_ambient_provider(client, monkeypatch):
    """Hide every real provider and check the LLM endpoints still respond.

    This is the regression guard for the first CI run: it failed here while
    passing on a developer box that was running Ollama, because the provider
    was auto-detected per request. The ``client`` fixture pins it; if that
    pin is ever dropped, the real resolver now finds nothing and this fails.
    """
    monkeypatch.setattr("market_intel.llm._reachable", lambda *a, **k: False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    stub_answer(monkeypatch, make_sources(2), cited=[1])
    assert client.post("/ask", json={"query": "fees"}).status_code == 200

    monkeypatch.setattr(api, "build_report", lambda *a, **k: fake_summary())
    assert client.post("/reports", json={"brief": "fees"}).status_code == 202


# --- report jobs -------------------------------------------------------


def test_report_job_runs_and_exposes_the_markdown(client, monkeypatch):
    monkeypatch.setattr(api, "build_report",
                        lambda *a, **k: fake_summary())
    resp = client.post("/reports", json={"brief": "fees and payouts"})
    assert resp.status_code == 202
    job = resp.json()
    assert job["status"] in ("queued", "running")
    assert job["kind"] == "report"

    done = wait_for_job(client, job["job_id"])
    assert done["status"] == "succeeded"
    assert done["duration_seconds"] is not None
    assert done["result"]["verdicts"]["SUPPORTED"] == 3
    # The bulk stays out of the JSON result: evidence text would make the
    # poll response megabytes.
    assert "markdown" not in done["result"]
    assert "evidence_items" not in done["result"]
    assert done["markdown_url"] == f"/jobs/{job['job_id']}/markdown"

    md = client.get(done["markdown_url"])
    assert md.status_code == 200
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith("# Market report")


def test_report_job_records_a_failure(client, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(api, "build_report", boom)
    job = client.post("/reports", json={"brief": "boom"}).json()
    done = wait_for_job(client, job["job_id"])
    assert done["status"] == "failed"
    assert "model exploded" in done["error"]
    # A failed job has no markdown, and must say so rather than 500.
    assert client.get(f"/jobs/{job['job_id']}/markdown").status_code == 409


def test_report_job_writes_the_markdown_to_disk(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, "build_report", lambda *a, **k: fake_summary())
    job = client.post("/reports", json={"brief": "fees"}).json()
    done = wait_for_job(client, job["job_id"])
    written = list((tmp_path / "reports").glob("market_report_*.md"))
    assert len(written) == 1
    assert written[0].read_text(encoding="utf-8").startswith("# Market report")


def test_report_validates_and_defaults_competitors(client, monkeypatch):
    seen = {}

    def capture(conn, brief, competitors, provider, per_query):
        seen["competitors"] = competitors
        seen["per_query"] = per_query
        return fake_summary()

    monkeypatch.setattr(api, "build_report", capture)
    job = client.post("/reports", json={"brief": "everything"}).json()
    wait_for_job(client, job["job_id"])
    assert seen["competitors"] == list(config.COMPETITORS)
    assert seen["per_query"] == 4
    assert client.post("/reports", json={"brief": ""}).status_code == 422
    assert client.post(
        "/reports", json={"brief": "x", "competitors": ["nope"]}
    ).status_code == 422


def test_jobs_list_and_latest(client, monkeypatch):
    monkeypatch.setattr(api, "build_report", lambda *a, **k: fake_summary())
    first = client.post("/reports", json={"brief": "one"}).json()
    wait_for_job(client, first["job_id"])
    assert client.get("/jobs").json()[0]["job_id"] == first["job_id"]
    assert client.get("/jobs?kind=report").json()[0]["job_id"] == first["job_id"]
    assert client.get("/jobs?kind=refresh").json() == []
    assert client.get("/jobs/latest?kind=report").json()["job_id"] == first["job_id"]
    assert client.get("/jobs/latest?kind=refresh").status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/jobs/nope").status_code == 404


# --- corpus refresh ----------------------------------------------------


def test_refresh_job_runs_the_pipeline(client, monkeypatch):
    """The refresh worker must wire fetch -> clean -> store -> index."""
    calls = {}

    def fake_fetch(limit, tolerate_failures=False):
        calls["limit"] = limit
        calls["tolerate"] = tolerate_failures
        return {("shopify", '"shopify"'): [{"objectID": "1"}]}

    monkeypatch.setattr(api, "fetch_all", fake_fetch)
    monkeypatch.setattr(api, "clean_posts", lambda raw: [
        {"id": "hackernews_1", "competitor": "shopify", "source": "hackernews",
         "community": "hackernews", "title": "New post", "body": "body",
         "url": "https://x/1", "author": "a", "created_utc": 1, "score": 1,
         "num_comments": 0},
    ])
    monkeypatch.setattr(api, "build_index", lambda conn, verbose=True: 115)

    job = client.post("/pipeline/refresh", json={"limit": 7}).json()
    assert job["kind"] == "refresh"
    done = wait_for_job(client, job["job_id"])
    assert done["status"] == "succeeded"
    assert calls["limit"] == 7
    # The scheduled path must tolerate a single failing query.
    assert calls["tolerate"] is True
    assert done["result"]["inserted"] == 1
    assert done["result"]["chunks"] == 115
    assert done["result"]["query_failures"] == fetch.planned_queries() - 1
    assert done["markdown_url"] is None
    # A refresh job has no report: asking for markdown is a conflict, not a 500.
    assert client.get(f"/jobs/{job['job_id']}/markdown").status_code == 409


def test_refresh_can_skip_reindexing(client, monkeypatch):
    monkeypatch.setattr(
        api, "fetch_all", lambda limit, tolerate_failures=False: {}
    )
    monkeypatch.setattr(api, "clean_posts", lambda raw: [])

    def should_not_run(*args, **kwargs):
        raise AssertionError("build_index called despite reindex=false")

    monkeypatch.setattr(api, "build_index", should_not_run)
    job = client.post(
        "/pipeline/refresh", json={"reindex": False}
    ).json()
    done = wait_for_job(client, job["job_id"])
    assert done["status"] == "succeeded"
    assert done["result"]["chunks"] is None


# --- evaluation --------------------------------------------------------


def test_eval_endpoint_scores_retrieval_without_an_llm(client):
    resp = client.post("/eval", json={"top_k": 5, "limit": 4})
    assert resp.status_code == 200
    body = resp.json()
    assert body["questions"] == 4
    assert body["corpus"] > 0
    averages = body["averages"]
    assert 0.0 <= averages["mrr"] <= 1.0
    assert 0.0 <= averages["recall"] <= 1.0
    assert "citation_precision" not in averages  # no LLM => no citation layer
    assert len(body["rows"]) == 4
    assert body["rows"][0]["retrieved"]
