"""Multiuser auth: users + sessions in their own small SQLite store.

Design constraints come from the repo, not from enterprise auth:

  * stdlib only — passwords are PBKDF2-HMAC-SHA256 (200k iterations,
    per-user salt), sessions are opaque random tokens stored hashed;
  * the store is a *separate* file from the corpus DB
    (``MARKET_INTEL_AUTH_DB``, default ``data/auth.db``) so wiping the
    corpus never touches accounts and tests stay hermetic;
  * the service runs in **open mode** until the first user exists —
    zero-config locally, and the n8n workflows keep working; the moment
    an account is registered, protected endpoints demand a token;
  * a static service token (``MARKET_INTEL_SERVICE_TOKEN``) authenticates
    as user ``n8n`` so scheduled workflows survive locked-down mode.

Session tokens travel as ``Authorization: Bearer <token>`` headers (no
cookies, no CSRF surface). Only the SHA-256 of a token is stored, so a
copy of the database cannot be replayed as a login.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

from market_intel.store import connect

# PBKDF2 parameters (OWASP-recommended ballpark for SHA-256, 2023+).
ITERATIONS = 200_000
SESSION_TTL_DAYS = 7

AUTH_DB = os.environ.get("MARKET_INTEL_AUTH_DB", "data/auth.db")
SERVICE_TOKEN = os.environ.get("MARKET_INTEL_SERVICE_TOKEN", "")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def ensure_tables(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS users (
            username     TEXT PRIMARY KEY,
            pw_salt      TEXT NOT NULL,
            pw_hash      TEXT NOT NULL,
            role         TEXT NOT NULL DEFAULT 'user',
            created_at   TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sessions (
            token_hash  TEXT PRIMARY KEY,
            username    TEXT NOT NULL REFERENCES users(username),
            created_at  TEXT NOT NULL,
            expires_at  TEXT NOT NULL
        );
        """
    )
    conn.commit()


def _hash_password(password: str, salt_hex: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode(), bytes.fromhex(salt_hex), ITERATIONS
    ).hex()


def create_user(username: str, password: str, role: str = "user") -> dict:
    """Register a user; the first user ever becomes the admin."""
    if not (3 <= len(username) <= 32) or not username.replace("_", "").isalnum():
        raise ValueError("username must be 3-32 chars: letters, digits, _")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    conn = connect(AUTH_DB)
    try:
        ensure_tables(conn)
        first = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
        salt = secrets.token_hex(16)
        conn.execute(
            "INSERT INTO users (username, pw_salt, pw_hash, role, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (username, salt, _hash_password(password, salt),
             "admin" if first else role, _now().isoformat()),
        )
        conn.commit()
        row = conn.execute(
            "SELECT username, role, created_at FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return {"username": row[0], "role": row[1], "created_at": row[2]}
    finally:
        conn.close()


def user_count() -> int:
    """Accounts on file; 0 when the store is unavailable.

    The connect() sits inside the try on purpose: on a fresh runner the
    data/ directory may not exist yet, and sqlite3 cannot create the
    file through a missing directory. That must read as "no accounts"
    (open mode), never as a 500 - auth availability is an operational
    detail, not a request-breaking failure.
    """
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error:
        return 0
    try:
        ensure_tables(conn)
        return conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    except sqlite3.Error:
        return 0
    finally:
        conn.close()


def auth_mode() -> bool:
    """True once any account exists: protected endpoints demand tokens."""
    return user_count() > 0


def authenticate(username: str, password: str) -> dict | None:
    """Verify credentials; returns the user dict or None."""
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error:
        _hash_password(password, "00" * 16)  # constant-ish time even when down
        return None
    try:
        ensure_tables(conn)
        row = conn.execute(
            "SELECT username, pw_salt, pw_hash, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        # Same error shape for unknown user and bad password, and a
        # dummy verify so timing does not reveal which one it was.
        _hash_password(password, "00" * 16)
        return None
    salt, stored, role = row[1], row[2], row[3]
    if not hmac.compare_digest(_hash_password(password, salt), stored):
        return None
    return {"username": row[0], "role": role}


def create_session(username: str) -> dict:
    """Mint a session token; only its SHA-256 is stored."""
    token = secrets.token_urlsafe(32)
    conn = connect(AUTH_DB)
    try:
        ensure_tables(conn)
        conn.execute(
            "INSERT INTO sessions (token_hash, username, created_at, expires_at)"
            " VALUES (?, ?, ?, ?)",
            (hashlib.sha256(token.encode()).hexdigest(), username,
             _now().isoformat(),
             (_now() + timedelta(days=SESSION_TTL_DAYS)).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()
    return {
        "token": token,
        "username": username,
        "expires_at": (_now() + timedelta(days=SESSION_TTL_DAYS)).isoformat(),
    }


def resolve_token(token: str) -> dict | None:
    """The user behind a bearer token, or None (unknown/expired/unavailable)."""
    if not token:
        return None
    if SERVICE_TOKEN and hmac.compare_digest(token, SERVICE_TOKEN):
        return {"username": "n8n", "role": "service"}
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error:
        return None
    try:
        ensure_tables(conn)
        row = conn.execute(
            """SELECT u.username, u.role, s.expires_at
               FROM sessions s JOIN users u ON u.username = s.username
               WHERE s.token_hash = ?""",
            (hashlib.sha256(token.encode()).hexdigest(),),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    if row[2] < _now().isoformat():
        return None  # expired; sessions are swept lazily on login below
    return {"username": row[0], "role": row[1]}


def drop_session(token: str) -> bool:
    conn = connect(AUTH_DB)
    try:
        ensure_tables(conn)
        cur = conn.execute(
            "DELETE FROM sessions WHERE token_hash = ?",
            (hashlib.sha256(token.encode()).hexdigest(),),
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def sweep_sessions() -> int:
    """Delete expired sessions; returns how many went."""
    conn = connect(AUTH_DB)
    try:
        ensure_tables(conn)
        cur = conn.execute("DELETE FROM sessions WHERE expires_at < ?",
                           (_now().isoformat(),))
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def list_users() -> list[dict]:
    conn = connect(AUTH_DB)
    try:
        ensure_tables(conn)
        rows = conn.execute(
            "SELECT username, role, created_at FROM users ORDER BY created_at"
        ).fetchall()
        return [{"username": r[0], "role": r[1], "created_at": r[2]}
                for r in rows]
    finally:
        conn.close()


class LastAdminError(Exception):
    """Refused: the operation would remove the only admin."""


def _count_admins(conn: sqlite3.Connection) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM users WHERE role = 'admin'"
    ).fetchone()[0]


def user_exists(username: str) -> bool:
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error:
        return False
    try:
        ensure_tables(conn)
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        return row is not None
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def delete_user(username: str) -> bool:
    """Remove an account and its sessions. Raises LastAdminError when the
    account is the only admin; returns False when it does not exist."""
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error:
        return False
    try:
        ensure_tables(conn)
        row = conn.execute(
            "SELECT role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            return False
        if row[0] == "admin" and _count_admins(conn) <= 1:
            raise LastAdminError("cannot delete the only admin")
        conn.execute("DELETE FROM sessions WHERE username = ?", (username,))
        conn.execute("DELETE FROM users WHERE username = ?", (username,))
        conn.commit()
        return True
    finally:
        conn.close()


def reset_password(username: str, new_password: str) -> None:
    """Replace an account's password; ValueError for unknown users or a
    too-short password. Callers revoke sessions afterwards."""
    if len(new_password) < 8:
        raise ValueError("password must be at least 8 characters")
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error as exc:
        raise ValueError(f"auth store unavailable: {exc}") from exc
    try:
        ensure_tables(conn)
        row = conn.execute(
            "SELECT 1 FROM users WHERE username = ?", (username,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown user {username}")
        salt = secrets.token_hex(16)
        conn.execute(
            "UPDATE users SET pw_salt = ?, pw_hash = ? WHERE username = ?",
            (salt, _hash_password(new_password, salt), username),
        )
        conn.commit()
    finally:
        conn.close()


def revoke_user_sessions(username: str) -> int:
    """Delete every session for an account; returns how many died."""
    try:
        conn = connect(AUTH_DB)
    except sqlite3.Error:
        return 0
    try:
        ensure_tables(conn)
        cur = conn.execute(
            "DELETE FROM sessions WHERE username = ?", (username,)
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()
