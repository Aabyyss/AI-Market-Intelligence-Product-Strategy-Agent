"""Inspect an n8n execution blob without full pointer resolution.

n8n's flatted storage stores shared values as numeric-string pointers
into the top-level array. Node-run data lives under resultData.runData,
whose KEYS are node names; resolve only a few hops per value, with a
cycle guard, instead of walking the whole graph.

Usage:
    python n8n/tools/inspect_exec.py <execution_id> [node_name_substring]

Prints the nodes that ran (with per-node run counts and error marks);
with a substring, dumps that node's first-run entry including its
output items - e.g. the Notify Slack node's output holds the webhook
response body ("ok") as the delivery receipt.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

DB = os.environ.get(
    "N8N_DB_PATH",
    os.path.expanduser(r"~/.mi-n8n/data/.n8n/database.sqlite"),
)


def deref(root, value, max_hops=10):
    """Follow numeric-string pointers until a real value appears."""
    hops = 0
    while isinstance(value, str) and value.isdigit() and int(value) < len(root):
        if hops >= max_hops:
            return value
        value = root[int(value)]
        hops += 1
    return value


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    exec_id = sys.argv[1]
    want = sys.argv[2].lower() if len(sys.argv) > 2 else None

    conn = sqlite3.connect(DB)
    row = conn.execute(
        "SELECT d.data FROM execution_data d WHERE d.executionId = ?", (exec_id,)
    ).fetchone()
    if not row:
        print(f"execution {exec_id}: no stored data")
        return 1
    root = json.loads(row[0])

    rd = deref(root, root[2])
    if isinstance(rd, str):
        rd = json.loads(rd)
    run = deref(root, rd.get("runData"))
    if isinstance(run, str):
        run = json.loads(run)
    if not isinstance(run, dict):
        print(f"execution {exec_id}: no runData stored")
        return 1

    print(f"execution {exec_id} nodes ({len(run)}):")
    for name in sorted(run):
        entries = deref(root, run[name])
        if isinstance(entries, str):
            entries = json.loads(entries)
        if not isinstance(entries, list) or not entries:
            print(f"  - {name} (no runs stored)")
            continue
        first = deref(root, entries[0])
        error = deref(root, first.get("error")) if isinstance(first, dict) else None
        marker = " ERROR" if error else ""
        print(f"  - {name} x{len(entries)}{marker}")
        if want and want in name.lower():
            print(json.dumps(first, default=str, indent=1)[:2500])
    return 0


if __name__ == "__main__":
    sys.exit(main())
