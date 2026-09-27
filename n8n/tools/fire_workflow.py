"""Fire a market_intel n8n workflow from its schedule trigger and poll it.

Wraps n8n's internal REST API (cookie session - the public API key path
is blocked on some plans). Reads credentials from the environment or
flags; never hardcodes them.

Examples:
    python n8n/tools/fire_workflow.py --trigger "Daily 07:00" --wait
    python n8n/tools/fire_workflow.py --trigger "Daily 06:00" --workflow-id abc123
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request

N8N_BASE = os.environ.get("N8N_BASE_URL", "http://127.0.0.1:5678")


def login(email: str, password: str) -> str:
    req = urllib.request.Request(
        f"{N8N_BASE}/rest/login",
        data=json.dumps({"emailOrLdapLoginId": email, "password": password}).encode(),
        method="POST",
    )
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as resp:
        set_cookie = resp.headers.get("set-cookie", "")
    match = re.search(r"n8n-auth=[^;]+", set_cookie)
    if not match:
        raise SystemExit("login succeeded but no n8n-auth cookie came back")
    return match.group(0)


def http_json(url: str, payload: dict | None = None, cookie: str | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, method="POST" if payload is not None else "GET")
    req.add_header("content-type", "application/json")
    if cookie:
        req.add_header("cookie", cookie)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def fire(cookie: str, workflow_id: str, trigger_name: str) -> str:
    out = http_json(
        f"{N8N_BASE}/rest/workflows/{workflow_id}/run",
        {"mode": "trigger", "triggerToStartFrom": {"name": trigger_name}},
        cookie=cookie,
    )
    return out["data"]["executionId"]


def execution_status(cookie: str, execution_id: str) -> str:
    out = http_json(f"{N8N_BASE}/rest/executions/{execution_id}", cookie=cookie)
    return out["data"]["status"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trigger", required=True, help='trigger node name, e.g. "Daily 07:00"')
    parser.add_argument(
        "--workflow-id",
        default=os.environ.get("MI_N8N_WORKFLOW_ID", ""),
        help="workflow id from the editor URL (default: MI_N8N_WORKFLOW_ID env)",
    )
    parser.add_argument("--email", default=os.environ.get("N8N_EMAIL", ""))
    parser.add_argument("--password", default=os.environ.get("N8N_PASSWORD", ""))
    parser.add_argument("--wait", action="store_true", help="poll until the execution finishes")
    parser.add_argument(
        "--timeout", type=int, default=1800, help="max seconds to poll with --wait"
    )
    args = parser.parse_args()

    if not args.workflow_id:
        raise SystemExit("no workflow id: pass --workflow-id or set MI_N8N_WORKFLOW_ID")
    if not (args.email and args.password):
        raise SystemExit("no credentials: pass --email/--password or set N8N_EMAIL/N8N_PASSWORD")

    cookie = login(args.email, args.password)
    execution_id = fire(cookie, args.workflow_id, args.trigger)
    print(f"fired {args.trigger!r}: execution {execution_id}")

    if not args.wait:
        return 0

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        status = execution_status(cookie, execution_id)
        if status in ("success", "error", "canceled", "crashed", "timeout"):
            print(f"execution {execution_id}: {status}")
            return 0 if status == "success" else 1
        time.sleep(15)
    print(f"execution {execution_id}: still running after {args.timeout}s")
    return 2


if __name__ == "__main__":
    sys.exit(main())
