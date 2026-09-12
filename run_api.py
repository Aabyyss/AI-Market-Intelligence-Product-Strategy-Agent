"""Phase 6: run the API service.

    python run_api.py                     # http://127.0.0.1:8000
    python run_api.py --port 9000 --reload
    python run_api.py --host 0.0.0.0      # in a container

Then:  http://127.0.0.1:8000/docs  (interactive OpenAPI docs)

For production use a process manager (or the Dockerfile) and more than
one uvicorn worker only if the report worker count is raised to match —
report jobs are held in memory per process, so a job started on one
worker is not visible to another.
"""
import argparse
import os
import sys

import uvicorn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def main(host: str, port: int, reload: bool, log_level: str) -> None:
    print(f"market-intel API on http://{host}:{port}  (docs at /docs)")
    uvicorn.run(
        "market_intel.api:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    parser.add_argument("--reload", action="store_true", help="auto-reload on edits")
    parser.add_argument("--log-level", default=os.environ.get("LOG_LEVEL", "info").lower())
    args = parser.parse_args()
    main(args.host, args.port, args.reload, args.log_level)
