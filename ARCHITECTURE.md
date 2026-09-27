# Architecture

How the system is put together, why each piece looks the way it does, and
what breaks when it breaks. For the build story and the roadmap see
[PROJECT.md](PROJECT.md); for day-to-day usage see the [README](README.md).

## The system in one diagram

```
 Hacker News (Algolia API, public)
        |  fetch.py   retry + backoff, per-query tolerance
        v
    clean.py  ->  store.py  ->  SQLite (data/market_intel.db)
                                 posts | chunks | vec0 virtual table
        ^                                    |
        | refresh job                        | cosine k-NN (sqlite-vec)
        |                                    v
  +----------------+      chunk.py -> embed.py (bge-small, local)
  | FastAPI service|<--- answer.py    grounded Q&A with [n] citations
  |  api.py        |<--- agents.py   5-agent report pipeline
  +----------------+
     |  ^                 ^                ^
     |  | 202 + job id    |                |
     v  |                 |                |
  n8n workflows      console.html      curl / scripts / tests
  (2 schedules,      (self-contained   (everything speaks the
   poll loop,         browser UI)       same HTTP API)
   Slack receipts)
```

Everything above the dotted line runs locally: the embedding model, the
LLM (Ollama by default), the database, the scheduler. The only external
dependency at runtime is the public HN API - and the test suite never
touches even that (see "Failure modes").

## Design principles

1. **Evidence first.** Nothing reaches the user without a citation back
   to a real post. Retrieval happens before generation, the model sees
   only numbered evidence, and every `[n]` it emits is mechanically
   validated afterwards. This is layered the same way at Q&A scale
   (`answer.py`) and report scale (`agents.py` + the citation audit).
2. **Local and free.** bge-small embeddings via fastembed, Ollama for the
   LLM, SQLite for storage, self-hosted n8n, a Slack webhook for
   notification. No paid API anywhere; the whole system runs on a laptop.
3. **Jobs, not open connections.** Anything that takes minutes (a report,
   a refresh) returns `202` with a job id and is polled. This is what
   makes n8n orchestration possible without HTTP timeouts and retries,
   and it makes runs observable: a job is a real object with
   `started_at`, `duration_seconds` and a typed result.
4. **One place to edit.** Competitors, queries, chunk and embed knobs all
   live in `config.py`. Adding a competitor is a one-line change.
5. **Defensive LLM plumbing.** Model replies are parsed with a tolerant
   JSON extractor, validated, and never trusted for arithmetic. A section
   that fails to parse degrades to flagged free text instead of crashing
   the pipeline.

## Layers

### 1. Collection - `fetch.py`, `clean.py`, `store.py`

`fetch.py` turns the competitor list into planned HN Algolia queries and
calls them with retry and exponential backoff. A single failing query is
tolerated (and counted as `query_failures`) so one 429 cannot kill a
scheduled refresh; `planned_queries()` is exposed so the API can report
"13 of 15 queries succeeded" honestly. `clean.py` normalizes titles and
bodies, drops junk, and dedupes; `store.py` owns the SQLite schema and
idempotent inserts (a re-fetched post updates, never duplicates).

### 2. RAG foundation - `chunk.py`, `embed.py`, `vector.py`

Posts are split into overlapping chunks sized for the embedding model,
embedded locally (bge-small-en-v1.5 through fastembed, ONNX runtime), and
stored in the same SQLite file next to the metadata. The index is
sqlite-vec's `vec0` virtual table with cosine distance.

Why SQLite and not a vector database: the corpus is thousands of chunks,
not millions. A second database service would add a moving part, an auth
surface, and a backup story to solve a problem this scale does not have.
A regression suite pins sqlite-vec's results to a reference numpy cosine
ranking, so the index cannot silently drift from the math.

One deliberate quirk: `vec0` cannot filter on non-vector columns, so the
`--competitor` filter is applied as a SQL join after the scan. A scan
covers the whole corpus anyway, and this guarantees results match a
global cosine ranking exactly (the tie-breaking between byte-identical
chunks is tested).

### 3. Grounded Q&A - `answer.py`, `llm.py`

`llm.py` resolves a provider per call (Ollama probe, then OpenAI if a key
exists) and is the only place that knows about providers. `answer.py`
retrieves top-k chunks, prompts the model with numbered evidence only,
and post-validates every `[n]` against what was actually retrieved: an
answer citing nothing is reported as not grounded, not passed through.

### 4. The five-agent pipeline - `agents.py`

One job per role, all reading the same evidence pool:

| agent | job | output contract |
|---|---|---|
| research | turn the brief into queries, gather numbered evidence | query list + evidence items |
| competitor | positioning, pricing, strengths, weaknesses per competitor | claims with `[n]` citations |
| customer | community complaints, pain points, praise | claims with `[n]` citations |
| strategy | ranked product opportunities grounded in evidence | claims with `[n]` citations |
| critic | fact-check the draft against the evidence, flag unsupported claims and tone misreads (sarcasm) | per-claim verdicts |

The analysts and the critic speak **typed JSON**, not markdown: the
report renders from validated structures (claims, verdicts keyed by
claim id), so the model never does arithmetic and the parser never
scrapes prose. `build_report()` assembles the final markdown, runs the
citation audit (every citation must point at evidence that was actually
retrieved), writes the report file, and returns the summary the API
serves - the markdown itself is streamed from disk, never embedded in a
poll response.

### 5. Service - `api.py`, `logging_config.py`

FastAPI in front of everything. Long operations run on a one-worker
thread pool (the pipeline is CPU-bound; a second concurrent report only
slows the first) with an in-memory job registry and lock:

```
queued -> running -> succeeded | failed
```

- `POST /reports` and `POST /pipeline/refresh` accept an optional
  `source` tag (default `"api"`); the n8n workflows send
  `source: "n8n"`, so scheduled runs are distinguishable from manual
  ones end to end - in the job record, the API responses, and the
  console UI.
- Every request gets a request id (honoring an incoming
  `X-Request-ID`), logged on every line, echoed in the response, copied
  into the worker thread so a background report still logs under the id
  its submitter was given.
- `/health` reports corpus size, embedding model, and LLM reachability;
  an unreachable LLM is a detail field, not a 500. `/smoke` proves the
  two native dependencies (sqlite-vec query, embedder) at request time.
- Honest limits: jobs are in memory, so a restart loses job history
  (reports themselves are on disk). Multi-worker deployments need
  `MARKET_INTEL_WORKERS` awareness; a DB-backed queue is the next step
  if this ever leaves one machine.

### 6. Orchestration - `n8n/`

Two exported workflows, importable into any n8n (Community edition is
enough): **06:00 corpus refresh** and **07:00 report + Slack**.

The interesting part is the poll loop. Every poll output funnels into a
single Code node (`Route`) that maps the job state to
`wait / done / failed / timeout` and carries an attempt counter; two IF
nodes branch on that one field. Bounded attempts (`max_polls x
poll_seconds`, 60 x 45 s by default - a measured report run took 1375 s,
so the budget is ~2x) turn a wedged API into a `timeout` with a Slack
alert instead of an execution that polls forever. Slack messages carry
the verdict counts, duration, and the job id; refresh notifies only when
something was actually found. The workflows read configuration from
`$env` (loaded from the repo `.env` by `n8n/start_n8n.cmd`), never from
hardcoded values, which is why the exported JSON is safe to commit.

Live verification, receipts, and the in-place re-import recipe live in
`n8n/README.md` and `n8n/LIVE_RUN_2026-09-27.md`.

### 7. UI - `console.html`

A single self-contained HTML file served at `/console` - no build step,
no CDN, no external fonts. It speaks the same API as n8n: ask, search
stats, start report/refresh jobs, watch them poll to completion, open
the rendered report. n8n-started jobs wear a source badge. The
constraints are enforced by tests: byte-equality with the packaged file,
no external assets, and every `fetch()` target must exist on the app.

## Data model

```
posts(id TEXT PRIMARY KEY, competitor, source, community, title, body,
      url, author, created_utc, score, num_comments)
chunks(id, post_id -> posts, chunk_index, text, model)
vec0 virtual table(chunk_id, embedding)   -- sqlite-vec, cosine
```

Job records live in memory (typed pydantic models); finished reports are
markdown files under `data/reports/`. Everything durable survives in
SQLite plus those files.

## Quality gates

| gate | where | what it pins |
|---|---|---|
| vector correctness | `tests/test_vector_search.py` | sqlite-vec == numpy cosine, filters, tie-breaks |
| agent plumbing | `tests/test_agents.py` | tolerant parsing, audits, typed verdicts |
| service | `tests/test_api.py` | HTTP contract, job state machine, no leaked tracebacks |
| console | `tests/test_console_and_launcher.py` | offline, real endpoints, byte-equality |
| n8n exports | `tests/test_n8n_workflows.py` | structure, budgets, ASCII Slack text |
| retrieval quality | `run_eval.py --min-mrr --min-recall` | CI gate against labeled questions |

CI runs the whole suite plus the eval gate on every push, with a
concurrency group that cancels superseded runs. Two guards matter more
than they look:

- The **network guard** (autouse fixture in `tests/conftest.py`) points
  `fetch_all` / `build_report` at raising stubs by default. A test that
  forgets its stub fails in milliseconds with a message naming the fix,
  instead of a real multi-minute HN fetch inside CI - which happened
  exactly once, and is why the guard exists.
- The **provider pin** in the `client` fixture keeps the suite passing
  identically on a machine with Ollama running and on a bare runner.

## Failure modes

| failure | behavior | where handled |
|---|---|---|
| HN query fails mid-refresh | tolerated, counted in `query_failures`, job still succeeds honestly | `fetch.py`, refresh worker |
| LLM unreachable | `/ask` -> 503 with detail; `/reports` job fails with a clean error, no traceback leak | `api.py`, `llm.py` |
| model reply is not JSON | section degrades to flagged free text; report still renders | `agents.py` |
| citation points at unretrieved evidence | flagged by the audit, shown in report + console | `agents.py` |
| report runs longer than the n8n budget | workflow returns `timeout` + Slack alert | poll loop |
| job worker restarts mid-run | job history lost (reports persist on disk); documented limit | `api.py` |
| memory pressure (API + Ollama + tests at once) | ONNX can fail to allocate; serialize heavy local runs | ops note in README |
| a test hits the network | fails immediately with a "stub me" message | `tests/conftest.py` |

## Extension points

- **More sources:** the pipeline is source-agnostic; add a fetcher that
  yields raw items and a `clean` rule. Reddit needs OAuth, which is the
  only reason it is not already wired.
- **More competitors:** one line in `config.py` (queries scale with it).
- **Different LLM:** implement the provider in `llm.py`; everything else
  is provider-blind.
- **Multi-machine:** replace the in-memory job registry with the
  DB-backed queue the API already anticipates.

## Repository layout

See the annotated tree at the bottom of the [README](README.md#layout).
