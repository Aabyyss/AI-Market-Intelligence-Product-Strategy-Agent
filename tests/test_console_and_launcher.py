"""The /console endpoint and the desktop launcher."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from market_intel import api

CONSOLE = Path(__file__).resolve().parents[1] / "market_intel" / "console.html"


def test_console_serves_html(client):
    r = client.get("/console")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert b"market-intel console" in r.content


def test_console_matches_the_packaged_file(client):
    """The endpoint serves the packaged file byte-for-byte — no drift."""
    r = client.get("/console")
    assert r.content == CONSOLE.read_bytes()


def test_console_is_self_contained():
    """No external assets: the console must work fully offline."""
    html = CONSOLE.read_text(encoding="utf-8")
    for tag in re.findall(r"<(?:script|img|link)\b[^>]*>", html):
        assert "src=" not in tag.replace("src=", "", 0) or "http" not in tag, tag
    assert "@import" not in html
    assert "fonts.googleapis" not in html
    # it must only talk to same-origin endpoints
    for url in re.findall(r"""["'](https?://[^"']+)["']""", html):
        assert "127.0.0.1" not in url and "localhost" not in url


def test_console_calls_only_real_endpoints(client, monkeypatch):
    """Every fetch() target exists on the app — catches typos and drift."""
    from unittest.mock import patch

    # The POSTs below start background jobs. Without stubbing, the worker
    # would run a real multi-agent LLM report (minutes) inside pytest's
    # non-daemon thread pool — and a real refresh would re-embed the whole
    # corpus. Both stubs return the exact summary shapes the API trims.
    monkeypatch.setattr(api, "build_report", lambda *a, **kw: {
        "brief": kw.get("brief", "t"), "competitors": [], "provider": "ollama",
        "model": "llama3.1", "queries": [], "evidence": 0,
        "per_competitor": {}, "verdicts": {"SUPPORTED": 0, "PARTIAL": 0,
        "UNSUPPORTED": 0}, "citation_audit": {"used": [], "out_of_range": [],
        "uncited_competitor": 0}, "unstructured_sections": [],
        "markdown": "# t", "generated_at": "2026-01-01T00:00:00",
    })
    monkeypatch.setattr(api, "fetch_all", lambda **kw: {})

    html = CONSOLE.read_text(encoding="utf-8")
    # startJob() is the console's job-starting helper (it wraps api()).
    found = re.findall(r"""(?:api|fetch|startJob)\(\s*["'`](/[^"'`?]+)""", html)
    paths = {p.rstrip("/") for p in found}
    assert {"/health", "/ask", "/reports", "/pipeline/refresh", "/jobs"} <= paths

    with patch("market_intel.api.get_provider", return_value="ollama"):
        assert client.get("/health").status_code == 200
        assert client.get("/jobs").status_code == 200
        r = client.post("/reports", json={"brief": "test brief"})
        assert r.status_code == 202, r.json()
        assert client.post("/pipeline/refresh", json={"reindex": False}).status_code == 202
        job = client.get(f"/jobs/{r.json()['job_id']}").json()
        assert job["status"] in {"running", "queued", "succeeded"}


def test_free_port_returns_a_bindable_port():
    import socket

    from app import free_port

    port = free_port()
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))  # must succeed — that is the point


def test_wait_healthy_reads_health_json():
    """wait_healthy parses a real /health payload served on a scratch port."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from app import wait_healthy

    payload = {"status": "ok", "version": "test", "posts": 83, "chunks": 115}

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):  # silence
            pass

    server = HTTPServer(("127.0.0.1", 0), H)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        assert wait_healthy(port, timeout=5) == payload
    finally:
        server.shutdown()
