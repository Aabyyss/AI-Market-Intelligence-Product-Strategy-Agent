"""Logging setup for the service layer (Phase 6).

Two things matter in a deployed service that do not matter in a CLI:

  1. **One log format, machine-readable.** A CLI can print anything; a
     service's logs get shipped somewhere and grepped, so every line
     carries a level, a timestamp, and — for anything handling a
     request — the same request id the caller was handed back.

  2. **A request id that spans the whole request.** When a report job
     fails three layers down in agents.py, the only way to tie that
     traceback to the HTTP call that caused it is to stamp the id on
     every record. That is what the contextvar below does: the
     middleware sets it once, and a logging.Filter on the handler copies
     it onto every record, no matter which module logged it.

Request ids also come *in* from the caller (``X-Request-ID``): n8n and
load balancers usually send one, and honouring it means our logs line up
with theirs instead of starting a second, unrelated trace.
"""

import logging
import os
import uuid
from contextvars import ContextVar

# The current request's id. A contextvar (not a module global) because
# each request runs in its own task/thread — a global would let
# concurrent requests overwrite each other's ids.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# %(request_id)s is filled in by RequestIdFilter below.
LOG_FORMAT = (
    "%(asctime)s %(levelname)-7s [%(request_id)s] %(name)s: %(message)s"
)


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record that lacks one."""

    def filter(self, record: logging.LogRecord) -> bool:
        if not hasattr(record, "request_id"):
            record.request_id = request_id_var.get()
        return True


QUIET_LOGGERS = {"httpx"}


def _noisy_library(name: str) -> bool:
    """Libraries that log every outbound call at INFO.

    The LLM calls are the whole point of the agents, so their HTTP
    chatter would bury the request-scoped lines that matter.

    Matching is on the *stem* of the logger name (dots and trailing
    digits stripped) because the same library shows up under several
    names in the wild — ``httpx``, ``httpx._client``, and even a
    renamed ``httpx2`` bundled by another dependency. A plain
    ``getLogger("httpx").setLevel(...)`` misses all but the first.
    """
    stem = name.split(".")[0].rstrip("0123456789")
    return stem in QUIET_LOGGERS


class QuietLibrariesFilter(logging.Filter):
    """Drop sub-WARNING records from known-noisy libraries."""

    def filter(self, record: logging.LogRecord) -> bool:
        if _noisy_library(record.name):
            return record.levelno >= logging.WARNING
        return True


def new_request_id() -> str:
    """A short, sortable-enough id for one request (first 8 hex chars)."""
    return uuid.uuid4().hex[:8]


def setup_logging(level: str | None = None) -> None:
    """Configure root logging once, idempotently.

    ``force=True`` matters for uvicorn: importing the app can happen
    after uvicorn has already installed its own handlers, and without
    force the two formats fight and the request id never shows up.
    """
    logging.basicConfig(
        level=(level or LOG_LEVEL),
        format=LOG_FORMAT,
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(RequestIdFilter())
        handler.addFilter(QuietLibrariesFilter())
    logging.getLogger("httpx").setLevel(logging.WARNING)
