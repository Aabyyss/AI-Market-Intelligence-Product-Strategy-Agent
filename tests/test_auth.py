"""Tests for multiuser auth and per-user jobs (no LLM, no network).

The auth store lives in its own SQLite file (auth.AUTH_DB), so every
test points it at a throwaway path first — the service is born in open
mode, and the first registered account locks it down. These tests pin
that lifecycle end to end through the HTTP layer.
"""

import pytest

from market_intel import auth as auth_store
from market_intel import api


@pytest.fixture
def auth_db(tmp_path, monkeypatch):
    """A throwaway auth store + open-mode state for one test."""
    path = str(tmp_path / "auth.db")
    monkeypatch.setattr(auth_store, "AUTH_DB", path)
    monkeypatch.setattr(auth_store, "SERVICE_TOKEN", "")
    return path


def summary_stub():
    """Minimal build_report-shaped summary (jobs must not touch the LLM)."""
    return {
        "brief": "fees", "competitors": ["shopify"], "provider": "ollama",
        "model": "llama3.2", "queries": [], "learned_queries": [],
        "evidence": 1, "per_competitor": {"shopify": 1},
        "verdicts": {"SUPPORTED": 0, "PARTIAL": 0, "UNSUPPORTED": 0},
        "citation_audit": {"used": [], "out_of_range": [],
                           "uncited_competitor": 0},
        "unstructured_sections": [], "markdown": "# t",
        "generated_at": "2026-01-01T00:00:00",
    }


def register(client, username, password="password123"):
    return client.post("/auth/register",
                       json={"username": username, "password": password})


def authed(client, token):
    return {"Authorization": f"Bearer {token}"}


# --- open mode ----------------------------------------------------------

def test_open_mode_until_first_user(client, auth_db, monkeypatch):
    """No accounts: everything behaves like the pre-auth service."""
    monkeypatch.setattr(api, "build_report", lambda *a, **k: summary_stub())
    assert client.get("/auth/me").json()["username"] == "anonymous"
    r = client.post("/reports", json={"brief": "fees"})
    assert r.status_code == 202
    assert r.json()["user"] == "anonymous"
    assert r.json()["source"] == "api"


def test_register_validates_input(client, auth_db):
    assert client.post("/auth/register", json={
        "username": "ab", "password": "password123"}).status_code == 422
    assert client.post("/auth/register", json={
        "username": "bad name!", "password": "password123"}).status_code == 422
    assert client.post("/auth/register", json={
        "username": "alice", "password": "short"}).status_code == 422


# --- the lockdown transition -------------------------------------------

def test_first_user_becomes_admin_and_locks_the_service(client, auth_db, monkeypatch):
    monkeypatch.setattr(api, "build_report", lambda *a, **k: summary_stub())

    r = register(client, "alice")
    assert r.status_code == 201
    body = r.json()
    assert body["role"] == "admin" and body["token"]

    # the service is now authenticated mode: anonymous writes are 401
    assert client.post("/reports", json={"brief": "fees"}).status_code == 401
    assert client.get("/jobs").status_code == 401
    assert client.get("/auth/me").status_code == 401

    # ...but the token from registration works
    me = client.get("/auth/me", headers=authed(client, body["token"])).json()
    assert me == {"username": "alice", "role": "admin"}


def test_second_user_is_a_plain_user(client, auth_db):
    register(client, "alice")
    r = register(client, "bob")
    assert r.json()["role"] == "user"


def test_duplicate_username_is_409(client, auth_db):
    register(client, "alice")
    assert register(client, "alice").status_code == 409


# --- login / logout -----------------------------------------------------

def test_login_is_uniform_about_failures(client, auth_db):
    register(client, "alice")
    wrong_pw = client.post("/auth/login",
                           json={"username": "alice", "password": "wrong-pass"})
    no_user = client.post("/auth/login",
                          json={"username": "nobody", "password": "wrong-pass"})
    assert wrong_pw.status_code == no_user.status_code == 401
    # the service's uniform error handler may key the message "error" or
    # FastAPI's "detail"; either way unknown-user == wrong-password
    def msg(resp):
        body = resp.json()
        return body.get("detail", body.get("error"))
    assert msg(wrong_pw) == msg(no_user)


def test_login_logout_round_trip(client, auth_db):
    register(client, "alice")
    r = client.post("/auth/login",
                    json={"username": "alice", "password": "password123"})
    token = r.json()["token"]
    assert client.get("/auth/me",
                      headers=authed(client, token)).json()["username"] == "alice"
    assert client.post("/auth/logout",
                       headers=authed(client, token)).status_code == 200
    assert client.get("/auth/me", headers=authed(client, token)).status_code == 401


# --- per-user jobs ------------------------------------------------------

def test_jobs_are_scoped_per_user_and_hidden_from_others(client, auth_db, monkeypatch):
    monkeypatch.setattr(api, "build_report", lambda *a, **k: summary_stub())

    alice = register(client, "alice").json()
    job_a = client.post("/reports", json={"brief": "fees"},
                        headers=authed(client, alice["token"])).json()
    assert job_a["user"] == "alice"

    bob = register(client, "bob").json()
    job_b = client.post("/reports", json={"brief": "payouts"},
                        headers=authed(client, bob["token"])).json()

    # bob cannot see alice's job: 404, not 403 (ids must not be enumerable)
    assert client.get(f"/jobs/{job_a['job_id']}",
                      headers=authed(client, bob["token"])).status_code == 404
    assert client.get(f"/jobs/{job_a['job_id']}/markdown",
                      headers=authed(client, bob["token"])).status_code == 404

    # each list is scoped: bob (plain user) sees only his own jobs;
    # alice (first account = admin) sees everything
    alice_jobs = client.get("/jobs", headers=authed(client, alice["token"])).json()
    bob_jobs = client.get("/jobs", headers=authed(client, bob["token"])).json()
    assert {j["job_id"] for j in alice_jobs} == {job_a["job_id"], job_b["job_id"]}
    assert {j["job_id"] for j in bob_jobs} == {job_b["job_id"]}

    # the admin (first account) sees everything
    admin_jobs = client.get("/jobs", headers=authed(client, alice["token"]))
    assert admin_jobs.status_code == 200

    # and open-mode style access is gone for anonymous callers
    assert client.get(f"/jobs/{job_a['job_id']}").status_code == 401


def test_users_list_is_admin_only(client, auth_db):
    register(client, "alice")
    register(client, "bob")
    alice = client.post("/auth/login",
                        json={"username": "alice", "password": "password123"}).json()
    bob = client.post("/auth/login",
                      json={"username": "bob", "password": "password123"}).json()
    assert client.get("/auth/users", headers=authed(client, bob["token"])).status_code == 403
    users = client.get("/auth/users", headers=authed(client, alice["token"])).json()
    assert [u["username"] for u in users] == ["alice", "bob"]


# --- the service token (scheduled flows) --------------------------------

def test_service_token_authenticates_as_n8n(client, auth_db, monkeypatch):
    monkeypatch.setattr(api, "build_report", lambda *a, **k: summary_stub())
    register(client, "alice")  # lock the service down
    monkeypatch.setattr(auth_store, "SERVICE_TOKEN", "svc-secret-token")

    r = client.post("/pipeline/refresh", json={"reindex": False},
                    headers=authed(client, "svc-secret-token"))
    assert r.status_code == 202
    assert r.json()["user"] == "n8n"
    assert r.json()["source"] == "n8n" or r.json()["source"] == "api"

    # an arbitrary long token must not accidentally match the service token
    assert client.get("/auth/me",
                      headers=authed(client, "svc-secret-token-x")).status_code == 401


def test_auth_store_hashing_and_expiry(auth_db, monkeypatch):
    """Unit-level: passwords are salted+PBKDF2, expired tokens are dead."""
    user = auth_store.create_user("carol", "password123")
    assert user["role"] in {"admin", "user"}
    assert auth_store.authenticate("carol", "password123") is not None
    assert auth_store.authenticate("carol", "wrong-pass") is None

    session = auth_store.create_session("carol")
    assert auth_store.resolve_token(session["token"])["username"] == "carol"
    assert auth_store.drop_session(session["token"]) is True
    assert auth_store.resolve_token(session["token"]) is None

    # an expired session resolves to nothing
    import sqlite3 as s3
    conn = s3.connect(auth_db)
    token = "expired-token"
    conn.execute(
        "INSERT INTO sessions (token_hash, username, created_at, expires_at)"
        " VALUES (?, ?, ?, ?)",
        (__import__("hashlib").sha256(token.encode()).hexdigest(),
         "carol", "2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()
    assert auth_store.resolve_token(token) is None
