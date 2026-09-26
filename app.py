"""Desktop launcher: start the API and open the console in the browser.

Double-clicking ``Market Intelligence.bat`` (or running this file) starts
uvicorn on a free port, waits for /health to answer, then opens
http://127.0.0.1:<port>/console in the default browser. The server keeps
running so the console keeps working; close the window to stop it.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PORT_RANGE = range(8790, 8840)


def free_port() -> int:
    """Pick a port nobody is listening on (the API's default first)."""
    for port in (8000, *PORT_RANGE):
        with socket.socket() as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise RuntimeError("no free port in 8000, 8790-8839")


def wait_healthy(port: int, timeout: float = 90.0) -> dict:
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=2
            ) as r:
                return json.loads(r.read())
        except Exception as exc:  # noqa: BLE001 - any failure = keep waiting
            last = exc
            time.sleep(0.5)
    raise RuntimeError(f"API did not become healthy on port {port}: {last}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    port = args.port or free_port()
    os.environ.setdefault("PORT", str(port))
    os.environ.setdefault("HOST", "127.0.0.1")

    import uvicorn

    config = uvicorn.Config(
        "market_intel.api:app", host="127.0.0.1", port=port, log_level="info"
    )
    server = uvicorn.Server(config)

    import threading

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    print(f"starting market-intel on http://127.0.0.1:{port} ...")
    try:
        health = wait_healthy(port)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(
        f"ready: {health.get('posts')} posts / {health.get('chunks')} chunks, "
        f"llm: {health.get('llm_provider') or 'none configured'}"
    )
    url = f"http://127.0.0.1:{port}/console"
    if not args.no_browser:
        webbrowser.open(url)
    print(f"console: {url}")
    print("close this window to stop the app.")
    try:
        while thread.is_alive():
            thread.join(timeout=1)
    except KeyboardInterrupt:
        server.should_exit = True
    return 0


if __name__ == "__main__":
    sys.exit(main())
