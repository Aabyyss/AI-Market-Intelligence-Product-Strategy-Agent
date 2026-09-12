"""Tests for the fetch layer's resilience (no network).

The refresh endpoint runs nightly, so the failure modes that matter are
the boring ones: a reset socket, a 503, a query that keeps failing. These
tests drive those with a fake ``requests.get`` — no Hacker News, no
sleeping for real (``time.sleep`` is patched out, so the backoff
schedule is asserted by argument rather than waited out).
"""

import requests

from market_intel import config, fetch


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {"hits": [{"objectID": "1"}]}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(
                f"{self.status_code} error", response=self
            )

    def json(self):
        return self._payload


def http_error(status):
    """A RequestException carrying a response, as requests raises them."""
    return requests.HTTPError(f"{status} error", response=FakeResponse(status))


def test_search_returns_hits(monkeypatch):
    monkeypatch.setattr(fetch.requests, "get", lambda *a, **k: FakeResponse())
    assert fetch.search_hn("x") == [{"objectID": "1"}]


def test_transient_failure_is_retried_then_succeeds(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    calls = []

    def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("connection reset")
        return FakeResponse()

    monkeypatch.setattr(fetch.requests, "get", flaky)
    assert fetch.search_hn("flaky") == [{"objectID": "1"}]
    assert len(calls) == 2  # one failure, one success — no third attempt


def test_backoff_grows_exponentially(monkeypatch):
    slept = []

    def unreachable(*args, **kwargs):
        raise requests.ConnectionError("nope")

    monkeypatch.setattr(fetch.time, "sleep", slept.append)
    monkeypatch.setattr(fetch.requests, "get", unreachable)

    try:
        fetch.search_hn("always-fails")
    except requests.ConnectionError:
        pass
    # attempts-1 waits: 2s, then 4s (not 2s twice).
    assert slept[: fetch.MAX_ATTEMPTS - 1] == [2.0, 4.0]
    assert len(slept) == fetch.MAX_ATTEMPTS - 1


def test_retries_stop_after_max_attempts(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    calls = []

    def always_500(*args, **kwargs):
        calls.append(1)
        return FakeResponse(503)

    monkeypatch.setattr(fetch.requests, "get", always_500)
    try:
        fetch.search_hn("down")
        raise AssertionError("expected the 503 to propagate")
    except requests.HTTPError:
        pass
    assert len(calls) == fetch.MAX_ATTEMPTS


def test_client_errors_are_not_retried(monkeypatch):
    """A 400 means we sent something wrong — retrying cannot help."""
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    calls = []

    def bad_request(*args, **kwargs):
        calls.append(1)
        return FakeResponse(400)

    monkeypatch.setattr(fetch.requests, "get", bad_request)
    try:
        fetch.search_hn("bad")
        raise AssertionError("expected the 400 to propagate")
    except requests.HTTPError:
        pass
    assert len(calls) == 1


def test_fetch_all_is_strict_by_default(monkeypatch):
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.requests, "get",
                        lambda *a, **k: FakeResponse(400))
    try:
        fetch.fetch_all(limit=1)
        raise AssertionError("expected fetch_all to raise")
    except requests.HTTPError:
        pass


def test_fetch_all_tolerates_a_failing_query(monkeypatch):
    """One permanently-failing angle must not throw away the other four."""
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)

    def sometimes(url, params=None, **kwargs):
        # Every attempt for the "migrate" angle fails; the rest succeed.
        return FakeResponse(500 if "migrate" in params["query"] else 200)

    monkeypatch.setattr(fetch.requests, "get", sometimes)
    results = fetch.fetch_all(limit=1, tolerate_failures=True)

    planned = fetch.planned_queries()
    assert planned == len(config.COMPETITORS) * len(config.INTEL_ANGLES)
    # 15 queries, one angle (x3 competitors) permanently down -> 12 kept.
    assert len(results) == planned - len(config.COMPETITORS)
    assert all(v for v in results.values())    # every kept query has hits


def test_fetch_all_raises_when_nothing_was_collected(monkeypatch):
    """Success with zero data would report on a stale corpus silently."""
    monkeypatch.setattr(fetch.time, "sleep", lambda s: None)
    monkeypatch.setattr(fetch.requests, "get",
                        lambda *a, **k: FakeResponse(503))
    try:
        fetch.fetch_all(limit=1, tolerate_failures=True)
        raise AssertionError("expected a RuntimeError")
    except RuntimeError as exc:
        assert "every query failed" in str(exc)
