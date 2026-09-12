"""Phase 6: the service layer — FastAPI over the RAG + agent pipeline.

Same code as the CLIs, exposed over HTTP so other systems can drive it.
That is what turns this from a script collection into something an n8n
workflow (or a dashboard, or a Slack bot) can orchestrate.

    GET  /                     service banner + links
    GET  /health               liveness + corpus/provider state
    POST /search               semantic search over the vector index
    POST /ask                  evidence-backed answer with citations
    POST /reports              start a market report job  -> 202 + job id
    POST /pipeline/refresh     fetch new posts + rebuild the index -> 202
    GET  /jobs                 recent jobs (filter with ?kind=)
    GET  /jobs/latest          the newest job (?kind=report)
    GET  /jobs/{job_id}        job status + typed result
    GET  /jobs/{job_id}/markdown   the rendered report
    POST /eval                 retrieval-quality metrics vs labels

Design decisions worth naming:

**Long work is a job, not a request.** A full five-agent report takes
minutes on a CPU, and a corpus refresh waits on Hacker News. Holding an
HTTP connection open for either is how you get gateway timeouts and
retried work that runs twice. So both POSTs return 202 with an id
immediately and the caller polls — exactly the shape n8n wants.

**One worker by default.** Both jobs are CPU-bound on a single box; two
at once just thrash (and a refresh rebuilding the index while a report
reads it is a race worth serializing). ``MARKET_INTEL_WORKERS`` raises it.

**Jobs are in memory.** Honest for this stage: it keeps the service a
single process with no broker, and a restart loses history, not data
(reports are already on disk). A database-backed queue is the natural
next step if this ever runs on more than one machine.

**A request id on every log line.** The job worker runs in a thread, so
it copies the submitting request's context (``contextvars.copy_context``)
— otherwise the report you started at 09:00 logs under no id at all.

**Health never fails because a dependency is down.** ``/health`` reports
that no LLM provider is reachable instead of returning an error itself:
a health check that 500s is a health check you cannot read.

Environment:
    MARKET_INTEL_DB              SQLite path (default data/market_intel.db)
    MARKET_INTEL_REPORT_DIR      where report markdown is written
    MARKET_INTEL_WORKERS         job workers (default 1)
    MARKET_INTEL_EVAL_QUESTIONS  labeled question set for POST /eval
    LOG_LEVEL                    INFO by default
"""

import contextvars
import logging
import os
import sqlite3
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field, field_validator

from market_intel import __version__, config
from market_intel.agents import build_report
from market_intel.answer import answer_question
from market_intel.clean import clean_posts
from market_intel.fetch import fetch_all, planned_queries
from market_intel.llm import LLMError, model_for, resolve_provider
from market_intel.logging_config import (
    new_request_id,
    request_id_var,
    setup_logging,
)
from market_intel.store import connect, insert_posts
from market_intel.vector import build_index, search

setup_logging()
log = logging.getLogger("market_intel.api")

DB_PATH = os.environ.get("MARKET_INTEL_DB", "data/market_intel.db")
REPORT_DIR = os.environ.get("MARKET_INTEL_REPORT_DIR", config.REPORT_DIR)
EVAL_QUESTIONS = os.environ.get(
    "MARKET_INTEL_EVAL_QUESTIONS", "tests/fixtures/eval_questions.json"
)
WORKERS = max(1, int(os.environ.get("MARKET_INTEL_WORKERS", "1")))

app = FastAPI(
    title="AI Market Intelligence & Product Strategy Agent",
    version=__version__,
    summary="RAG + multi-agent market research over a local vector index.",
)


# --- dependencies ------------------------------------------------------


def get_conn():
    """One SQLite connection per request (connections are not thread-safe)."""
    if not Path(DB_PATH).exists():
        raise HTTPException(
            status_code=503,
            detail=(
                f"corpus not found at {DB_PATH} — build it with "
                "`python run_pipeline.py && python run_index.py`, or seed the "
                "tracked fixture with `python run_corpus.py seed`"
            ),
        )
    conn = connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def get_provider(explicit: str | None = None) -> str:
    """Resolve the LLM provider, or fail with a 503 the caller can act on."""
    try:
        return explicit or resolve_provider()
    except LLMError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _valid_competitor(value: str | None) -> str | None:
    if value is None:
        return None
    normalised = value.strip().lower()
    if normalised not in config.COMPETITORS:
        raise ValueError(
            f"unknown competitor {value!r}; known: {', '.join(config.COMPETITORS)}"
        )
    return normalised


# --- schemas -----------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=500,
                       examples=["shopify checkout fees"])
    top_k: int = Field(5, ge=1, le=25)
    competitor: str | None = Field(
        None, description=f"optional filter: {', '.join(config.COMPETITORS)}"
    )

    @field_validator("competitor")
    @classmethod
    def check_competitor(cls, v):
        return _valid_competitor(v)


class SearchHit(BaseModel):
    post_id: str
    competitor: str
    title: str
    url: str | None = None
    score: float
    chunk_index: int
    excerpt: str


class SearchResponse(BaseModel):
    query: str
    count: int
    hits: list[SearchHit]


class AskRequest(SearchRequest):
    pass


class Citation(BaseModel):
    n: int = Field(description="the [n] marker used in the answer")
    post_id: str
    title: str
    url: str | None = None


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool = Field(
        description="True when the answer cites at least one retrieved source"
    )
    citations: list[Citation]
    sources_considered: int


class ReportRequest(BaseModel):
    brief: str = Field(
        min_length=3, max_length=300,
        examples=["fees and developer payouts"],
    )
    competitors: list[str] | None = None
    per_query: int = Field(4, ge=1, le=8,
                           description="evidence chunks retrieved per query")
    provider: str | None = None

    @field_validator("competitors")
    @classmethod
    def check_competitors(cls, v):
        if not v:
            return None
        return [_valid_competitor(c) for c in v]


class RefreshRequest(BaseModel):
    limit: int = Field(25, ge=1, le=100,
                       description="Hacker News hits per query")
    reindex: bool = Field(True, description="rebuild the vector index afterwards")


class JobOut(BaseModel):
    job_id: str
    kind: str = Field(description="report | refresh")
    status: str = Field(description="queued | running | succeeded | failed")
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = None
    error: str | None = None
    result: dict | None = Field(
        None, description="kind-specific output (report summary / refresh counts)"
    )
    markdown_url: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    db: str
    posts: int | None = None
    chunks: int | None = None
    index_built_at: str | None = None
    embeddings: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_detail: str | None = None


class EvalRequest(BaseModel):
    top_k: int = Field(5, ge=1, le=25)
    limit: int | None = Field(None, ge=1,
                              description="score only the first N questions")
    with_answers: bool = Field(
        False, description="also run the LLM and score citation quality (slow)"
    )


# --- helpers -----------------------------------------------------------


def _hit(row: dict, limit: int = 400) -> SearchHit:
    text = (row.get("text") or "").strip()
    return SearchHit(
        post_id=row["post_id"],
        competitor=row["competitor"],
        title=row["title"],
        url=row.get("url"),
        score=round(float(row["score"]), 4),
        chunk_index=row["chunk_index"],
        excerpt=text if len(text) <= limit else text[:limit] + "…",
    )


def _corpus_stats() -> dict:
    """posts / chunks / index timestamp, or {} when there is no DB yet."""
    if not Path(DB_PATH).exists():
        return {}
    conn = connect(DB_PATH)
    try:
        conn.row_factory = sqlite3.Row
        stats = {"posts": conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]}
        try:
            stats["chunks"] = conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0]
            stats["index_built_at"] = conn.execute(
                "SELECT MAX(built_at) FROM chunks"
            ).fetchone()[0]
            row = conn.execute("SELECT model FROM chunks LIMIT 1").fetchone()
            stats["embeddings"] = row[0] if row else None
        except sqlite3.OperationalError:
            stats["chunks"] = 0  # DB built, index never built
        return stats
    finally:
        conn.close()


# --- request context ---------------------------------------------------


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Stamp a request id, and log method/path/status/duration once.

    The id is echoed back in ``X-Request-ID``, so a caller (n8n, curl)
    can quote it when reporting a problem and we can grep the exact line.
    """
    incoming = request.headers.get("X-Request-ID")
    rid = (incoming or new_request_id())[:64]
    token = request_id_var.set(rid)
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = rid
        return response
    finally:
        log.info(
            "%s %s -> %s (%.0f ms)",
            request.method,
            request.url.path,
            status,
            (time.perf_counter() - started) * 1000,
            extra={"request_id": rid},
        )
        request_id_var.reset(token)


# --- routes ------------------------------------------------------------


@app.get("/", include_in_schema=False)
def root() -> dict:
    return {
        "service": "AI Market Intelligence & Product Strategy Agent",
        "version": __version__,
        "docs": "/docs",
        "endpoints": ["/health", "/search", "/ask", "/reports",
                      "/pipeline/refresh", "/jobs", "/eval"],
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness + what the service actually has to work with."""
    provider = model = detail = None
    try:
        provider = resolve_provider()
        model = model_for(provider)
    except LLMError as exc:
        # Not an error response: the service is up, the LLM just is not
        # configured/reachable. Retrieval-only endpoints still work.
        detail = str(exc).splitlines()[0]
    return HealthResponse(
        status="ok",
        version=__version__,
        db=DB_PATH,
        llm_provider=provider,
        llm_model=model,
        llm_detail=detail,
        **_corpus_stats(),
    )


@app.post("/search", response_model=SearchResponse)
def search_index(req: SearchRequest, conn=Depends(get_conn)) -> SearchResponse:
    """Semantic search: top_k chunks, ranked by cosine similarity."""
    rows = search(conn, req.query, top_k=req.top_k, competitor=req.competitor)
    return SearchResponse(
        query=req.query, count=len(rows), hits=[_hit(r) for r in rows]
    )


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, conn=Depends(get_conn)) -> AskResponse:
    """Retrieve evidence, answer from it, and cite the posts used."""
    provider = get_provider()
    result = answer_question(
        conn, req.query, top_k=req.top_k, competitor=req.competitor,
        provider=provider,
    )
    sources = result["sources"]
    return AskResponse(
        question=req.query,
        answer=result["answer"],
        grounded=bool(result["cited"]),
        citations=[
            Citation(
                n=n,
                post_id=sources[n - 1]["post_id"],
                title=sources[n - 1]["title"],
                url=sources[n - 1].get("url"),
            )
            for n in result["cited"]
        ],
        sources_considered=len(sources),
    )


# --- background jobs ---------------------------------------------------

_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=WORKERS, thread_name_prefix="job")


def _job_public(job: dict) -> JobOut:
    return JobOut(
        job_id=job["job_id"],
        kind=job["kind"],
        status=job["status"],
        created_at=job["created_at"],
        started_at=job["started_at"],
        finished_at=job["finished_at"],
        duration_seconds=job["duration_seconds"],
        error=job["error"],
        result=job["result"],
        markdown_url=(
            f"/jobs/{job['job_id']}/markdown"
            if job["kind"] == "report" and job["status"] == "succeeded"
            else None
        ),
    )


def _start_job(kind: str, **fields) -> dict:
    """Register a job and hand it to the worker pool. Returns the record.

    The submitting request's context is copied in explicitly: a
    contextvar set in the request does not travel into the worker
    thread on its own, and without this the job's log lines would lose
    the request id that ties them to the HTTP call.
    """
    job_id = uuid.uuid4().hex[:12]
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "kind": kind,
            "status": "queued",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "started_at": None,
            "finished_at": None,
            "duration_seconds": None,
            "error": None,
            "result": None,
            "markdown": None,
            **fields,
        }
        job = _JOBS[job_id]
    worker = _run_report_job if kind == "report" else _run_refresh_job
    _executor.submit(contextvars.copy_context().run, worker, job_id)
    log.info("%s job %s queued", kind, job_id)
    return job


def _finish(job_id: str, status: str, result: dict | None,
            started: float, markdown: str | None = None) -> None:
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job["status"] = status
        job["result"] = result
        if markdown is not None:
            job["markdown"] = markdown
        job["duration_seconds"] = round(time.perf_counter() - started, 1)
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


def _trimmed_summary(summary: dict) -> dict:
    """The report summary without the bulk (evidence text, full markdown)."""
    keep = (
        "brief", "competitors", "provider", "model", "queries", "evidence",
        "per_competitor", "verdicts", "citation_audit",
        "unstructured_sections", "generated_at", "report_path",
    )
    return {k: summary[k] for k in keep if k in summary}


def _run_report_job(job_id: str) -> None:
    """Worker: run the five agents, write the markdown, record the outcome."""
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        brief, competitors = job["brief"], job["competitors"]
        per_query, provider = job["per_query"], job["provider"]

    started = time.perf_counter()
    log.info("report job %s started: %r", job_id, brief)
    try:
        conn = connect(DB_PATH)
        try:
            summary = build_report(
                conn, brief, competitors, provider, per_query=per_query
            )
        finally:
            conn.close()

        report_dir = Path(REPORT_DIR)
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = summary["generated_at"][:19].replace(":", "").replace("T", "_")
        out_path = report_dir / f"market_report_{stamp}.md"
        out_path.write_text(summary["markdown"], encoding="utf-8")
        summary["report_path"] = str(out_path)

        _finish(job_id, "succeeded", _trimmed_summary(summary), started,
                markdown=summary["markdown"])
        log.info("report job %s -> %s", job_id, out_path)
    except Exception as exc:  # noqa: BLE001 - the job records, it never raises
        _finish(job_id, "failed", None, started)
        with _JOBS_LOCK:
            _JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("report job %s failed", job_id)


def _run_refresh_job(job_id: str) -> None:
    """Worker: fetch new posts, clean + store them, rebuild the index.

    This is the "data collection" half of the scheduled workflow — the
    pipeline step that keeps the knowledge base current, so reports are
    never generated from a stale corpus. Inserts are idempotent
    (``INSERT OR IGNORE``), so re-running it is safe.

    One query failing does not fail the job (the fetch retries first,
    then skips it and reports the count) — only collecting *nothing* is
    an error, because a job that succeeds while fetching no data would
    silently report on a stale corpus.
    """
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        limit, reindex = job["limit"], job["reindex"]

    started = time.perf_counter()
    log.info("refresh job %s started (limit=%s)", job_id, limit)
    try:
        raw = fetch_all(limit=limit, tolerate_failures=True)
        records = clean_posts(raw)
        conn = connect(DB_PATH)
        try:
            inserted = insert_posts(conn, records)
            total = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
            chunks = build_index(conn, verbose=False) if reindex else None
        finally:
            conn.close()

        result = {
            "queries": len(raw),
            "queries_planned": planned_queries(),
            "query_failures": planned_queries() - len(raw),
            "fetched": sum(len(v) for v in raw.values()),
            "kept": len(records),
            "inserted": inserted,
            "posts_total": total,
            "chunks": chunks,
        }
        _finish(job_id, "succeeded", result, started)
        log.info("refresh job %s -> %s new posts (%s total)",
                 job_id, inserted, total)
    except Exception as exc:  # noqa: BLE001 - the job records, it never raises
        _finish(job_id, "failed", None, started)
        with _JOBS_LOCK:
            _JOBS[job_id]["error"] = f"{type(exc).__name__}: {exc}"
        log.exception("refresh job %s failed", job_id)


@app.post("/reports", response_model=JobOut, status_code=202)
def start_report(req: ReportRequest) -> JobOut:
    """Queue a market report; poll GET /jobs/{job_id} for the outcome."""
    if not Path(DB_PATH).exists():
        raise HTTPException(503, detail=f"corpus not found at {DB_PATH}")
    provider = get_provider(req.provider)
    job = _start_job(
        "report",
        brief=req.brief,
        competitors=req.competitors or list(config.COMPETITORS),
        per_query=req.per_query,
        provider=provider,
    )
    return _job_public(job)


@app.post("/pipeline/refresh", response_model=JobOut, status_code=202)
def start_refresh(req: RefreshRequest) -> JobOut:
    """Queue a corpus refresh (fetch + clean + store + reindex).

    Idempotent: posts already in the DB are skipped, so the schedule can
    run it daily without duplicating anything.
    """
    job = _start_job("refresh", limit=req.limit, reindex=req.reindex)
    return _job_public(job)


@app.get("/jobs", response_model=list[JobOut])
def list_jobs(kind: str | None = None, limit: int = 20) -> list[JobOut]:
    """Recent jobs, newest first. ``?kind=report`` filters by kind."""
    with _JOBS_LOCK:
        jobs = [j for j in _JOBS.values() if kind is None or j["kind"] == kind]
    jobs.sort(key=lambda j: j["created_at"], reverse=True)
    return [_job_public(j) for j in jobs[: max(1, min(limit, 100))]]


@app.get("/jobs/latest", response_model=JobOut)
def latest_job(kind: str | None = None) -> JobOut:
    """The newest job (404 until one has been started)."""
    jobs = list_jobs(kind=kind, limit=1)
    if not jobs:
        raise HTTPException(
            404, detail=f"no {kind or ''} jobs yet".replace("  ", " ").strip()
        )
    return jobs[0]


@app.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str) -> JobOut:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"unknown job {job_id}")
    return _job_public(job)


@app.get("/jobs/{job_id}/markdown", response_class=PlainTextResponse)
def get_job_markdown(job_id: str) -> PlainTextResponse:
    """The rendered report markdown (report jobs only, once succeeded)."""
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, detail=f"unknown job {job_id}")
    if job["kind"] != "report":
        raise HTTPException(
            409, detail=f"job {job_id} is a {job['kind']} job — no markdown"
        )
    if job["status"] != "succeeded":
        raise HTTPException(
            409, detail=f"job {job_id} is {job['status']}, not succeeded yet"
        )
    return PlainTextResponse(job["markdown"], media_type="text/markdown")


# --- evaluation --------------------------------------------------------


@app.post("/eval")
def run_eval_endpoint(req: EvalRequest, conn=Depends(get_conn)) -> dict:
    """Retrieval quality against the labeled question set.

    Retrieval-only by default: no LLM, so it is fast and safe to call
    from CI or a dashboard. ``with_answers`` adds the LLM layer and the
    citation metrics (requires a provider).
    """
    # Imported here, not at module scope: run_eval is a CLI and the
    # service should stay importable even if the harness moves.
    import run_eval as harness

    try:
        questions = harness.load_questions(EVAL_QUESTIONS)
    except (OSError, SystemExit) as exc:
        raise HTTPException(503, detail=f"labels unavailable: {exc}") from exc
    if req.limit:
        questions = questions[: req.limit]

    provider = get_provider() if req.with_answers else None
    rows = []
    for question in questions:
        row = harness.score_retrieval(conn, question, req.top_k)
        if provider:
            row = harness.score_answer(conn, row, req.top_k, provider)
        rows.append(row)

    averages = {
        "precision": round(harness.mean([r["precision"] for r in rows]), 4),
        "recall": round(harness.mean([r["recall"] for r in rows]), 4),
        "mrr": round(harness.mean([r["mrr"] for r in rows]), 4),
        "ndcg": round(harness.mean([r["ndcg"] for r in rows]), 4),
    }
    if provider:
        averages["citation_precision"] = round(
            harness.mean([r["cit_precision"] for r in rows]), 4
        )
        averages["citation_recall"] = round(
            harness.mean([r["cit_recall"] for r in rows]), 4
        )
    return {
        "questions": len(rows),
        "top_k": req.top_k,
        "corpus": _corpus_stats().get("posts"),
        "averages": averages,
        "rows": rows,
    }


# --- error handling ----------------------------------------------------


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Uniform error body, carrying the request id for log correlation."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "request_id": request_id_var.get(),
        },
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """Never leak a traceback to the caller; log it with the request id."""
    log.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": f"internal error: {type(exc).__name__}",
            "status_code": 500,
            "request_id": request_id_var.get(),
        },
    )
