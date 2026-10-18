"""Live check of the admin auth endpoints, against a throwaway service.

The suite covers the admin endpoints with `tests/test_auth.py`, but a
test client never shows the effects that make them real: a session
token actually dying mid-life after a password reset, a deletion
refusing on self and on the last admin, and the documented status
codes coming out of a real listener. This script boots an isolated
API - temp databases, free port, normal startup path - and walks the
whole flow. Nothing here touches a real data directory.

    python verify_admin.py

Exit 0 prints one PASS line per check; any failure prints FAIL and
exits 1, so CI or a manual run can both gate on it.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent

failures: list[str] = []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def api(base: str, method: str, path: str,
        token: str | None = None, payload: dict | None = None) -> tuple[int, dict]:
    """One HTTP call that returns (status, body as dict); errors stay data."""
    req = urllib.request.Request(base + path, method=method)
    if payload is not None:
        req.data = json.dumps(payload).encode()
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            return r.status, (json.loads(body) if body else {})
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read() or b"{}")
        except Exception:  # noqa: BLE001 - a malformed error body is still data
            return exc.code, {}


def check(name: str, ok: bool, detail: str = "") -> None:
    mark = "PASS" if ok else "FAIL"
    line = f"{mark}  {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    if not ok:
        failures.append(name)


def server_python() -> str:
    """Interpreter that can import the project: the repo venv, or ours.

    The script itself only needs the stdlib, but the server it spawns
    imports fastembed/openai etc., which live in the project venv - so
    when run with a bare system python, prefer the venv's interpreter
    for the child process.
    """
    for candidate in (ROOT / ".venv" / "Scripts" / "python.exe",
                      ROOT / ".venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def wait_healthy(base: str, timeout: float = 90.0, log: Path | None = None) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status, _ = api(base, "GET", "/health")
            if status == 200:
                return
        except Exception:  # noqa: BLE001 - connection refused = keep waiting
            pass
        time.sleep(0.5)
    detail = ""
    if log is not None:
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-8:])
        detail = f"\nserver log:\n{tail}\n"
    raise SystemExit(
        f"throwaway API never became healthy on {base}{detail}"
    )


def last_admin_is_protected(auth_db: Path) -> None:
    """Reach the LastAdminError that HTTP cannot (self-deletion intercepts).

    The check runs in-process against the isolated auth database the
    server uses; the store is sqlite, so a one-shot connection is safe.
    """
    sys.path.insert(0, str(ROOT))
    os.environ["MARKET_INTEL_AUTH_DB"] = str(auth_db)
    from market_intel.auth import LastAdminError, delete_user

    try:
        delete_user("principal")
        check("last admin cannot be removed", False, "deletion unexpectedly succeeded")
    except LastAdminError as exc:
        check("last admin cannot be removed", True, str(exc))
    finally:
        sys.path.remove(str(ROOT))


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="auth-live-"))
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    env = dict(os.environ,
               MARKET_INTEL_DB=str(tmp / "main.db"),
               MARKET_INTEL_AUTH_DB=str(tmp / "auth.db"),
               MARKET_INTEL_REPORT_DIR=str(tmp / "reports"))
    server_log = tmp / "server.log"
    with server_log.open("wb") as handle:
        proc = subprocess.Popen(
            [server_python(), str(ROOT / "app.py"),
             "--port", str(port), "--no-browser"],
            env=env, stdout=handle, stderr=subprocess.STDOUT,
        )
    try:
        wait_healthy(base, log=server_log)

        code, principal = api(base, "POST", "/auth/register",
                              payload={"username": "principal",
                                       "password": "AdminPass-2026!"})
        check("first registration becomes admin",
              code == 201 and principal.get("role") == "admin",
              f"HTTP {code}, role={principal.get('role')}")
        at = principal.get("token", "")

        code, intern = api(base, "POST", "/auth/register",
                           payload={"username": "intern",
                                    "password": "InternPass-2026!"})
        check("second registration is a normal user",
              code == 201 and intern.get("role") == "user",
              f"HTTP {code}, role={intern.get('role')}")
        old_it = intern.get("token", "")

        code, users = api(base, "GET", "/auth/users", token=at)
        names = [u["username"] for u in users] if isinstance(users, list) else []
        check("admin lists accounts", code == 200 and sorted(names) == ["intern", "principal"],
              f"HTTP {code}, users={names}")

        code, body = api(base, "GET", "/auth/me", token=old_it)
        check("new session token works", code == 200, f"HTTP {code}")

        code, body = api(base, "POST", "/auth/users/intern/password", token=at,
                         payload={"new_password": "NewIntern-2026!"})
        check("admin resets a password", code == 200 and body.get("sessions_revoked", 0) >= 1,
              f"HTTP {code}, {body}")

        code, body = api(base, "GET", "/auth/me", token=old_it)
        check("password reset revokes the old session",
              code == 401, f"old token answered HTTP {code}")

        code, body = api(base, "POST", "/auth/login",
                         payload={"username": "intern", "password": "NewIntern-2026!"})
        it = body.get("token", "")
        check("login works with the new password",
              code == 200 and body.get("username") == "intern", f"HTTP {code}")

        code, body = api(base, "DELETE", "/auth/users/principal", token=it)
        check("non-admin cannot delete an account",
              code == 403, f"HTTP {code}, {body}")

        code, body = api(base, "DELETE", "/auth/users/intern", token=at)
        check("admin deletes another account", code == 200, f"HTTP {code}, {body}")

        code, body = api(base, "DELETE", "/auth/users/intern", token=at)
        check("deleting an unknown account is 404", code == 404, f"HTTP {code}")

        code, body = api(base, "DELETE", "/auth/users/principal", token=at)
        check("an admin cannot delete their own account",
              code == 422, f"HTTP {code}, {body}")

        last_admin_is_protected(tmp / "auth.db")

        code, intern2 = api(base, "POST", "/auth/register",
                            payload={"username": "intern",
                                     "password": "InternPass-2026!"})
        code, body = api(base, "POST", "/auth/users/intern/revoke", token=intern2.get("token", ""))
        check("session revoke endpoint works",
              code == 200 and body.get("sessions_revoked", 0) >= 1,
              f"HTTP {code}, {body}")
        code, body = api(base, "GET", "/auth/me", token=intern2.get("token", ""))
        check("revoked session no longer authenticates",
              code == 401, f"revoked token answered HTTP {code}")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:  # pragma: no cover - rare
            proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print("\nall admin auth checks passed against a live service")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
