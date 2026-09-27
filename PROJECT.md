# Project

The project-management view: why this exists, how it was built phase by
phase, the decisions worth remembering, and a log of the problems that
actually bit — with their fixes. Architecture lives in
[ARCHITECTURE.md](ARCHITECTURE.md); usage in the [README](README.md).

Repo: https://github.com/Aabyyss/AI-Market-Intelligence-Product-Strategy-Agent

## Why

Product teams drown in community signal. Hacker News threads are full of
complaints about platform fees, payout delays, missing integrations, and
migration stories — real market research buried in real developer
discussions. Reading it by hand does not scale; keyword alerts drown you;
a generic LLM summary hallucinates because it never saw the evidence.

The idea: a system that collects that signal continuously, grounds every
claim in a retrievable post, and produces evidence-backed product
strategy — scoped deliberately so it runs entirely on a free stack, on a
laptop, with no paid API anywhere. The target users are indie hackers,
product managers, and developer-tool teams who need competitor
intelligence but do not have a research budget.

## What it does

- Collects public HN discussions about **Shopify, WooCommerce,
  BigCommerce** on a schedule (retrying, deduplicating, normalizing).
- Builds a local RAG index: chunks embedded with bge-small-en-v1.5 via
  fastembed, searchable with sqlite-vec — no cloud, no keys.
- Answers questions **with citations back to real posts**, rejecting
  anything the evidence does not support.
- Runs a **five-agent LLM pipeline** (research → competitor → customer →
  strategy → critic) to produce a full market report whose every claim
  is cited and mechanically audited.
- Exposes everything over a **FastAPI** service with background jobs,
  request ids, and typed results.
- Orchestrates the schedule with **n8n**: 06:00 refresh, 07:00 report,
  Slack receipts on success and failure.
- Ships a **self-contained browser console**, Docker packaging, an
  evaluation harness that gates CI, and a 115-test regression suite.

## The phased build story

Each phase shipped something usable before the next began — the
repo's commit history is the phase log.

1. **Data pipeline.** Fetch HN via the public Algolia API, clean,
   store in SQLite. Deliberately source-agnostic: Reddit's API went
   OAuth-only mid-build, which validated the decision to isolate
   fetching behind one interface.
2. **RAG foundation.** Chunk → embed locally → vector search inside
   SQLite via sqlite-vec. Regression tests pin the index's results to
   a reference numpy cosine ranking before anything consumed it.
3. **Grounded Q&A.** Retrieval before generation; numbered evidence
   only; every `[n]` validated against what was actually retrieved.
4. **Agents.** Five roles over one evidence pool, replying in typed
   JSON (claims + verdicts), a critic that fact-checks the draft, and a
   mechanical citation audit over the assembled report.
5. **Orchestration.** n8n workflows with a poll loop over the API's job
   system, bounded budgets, and Slack receipts. Verified live
   end-to-end on 2026-09-27 (evidence in `n8n/LIVE_RUN_2026-09-27.md`).
6. **Production habits.** FastAPI service (jobs, request ids, uniform
   errors), Docker + compose with build-time smoke tests, env-var
   config, and an evaluation harness that gates CI on retrieval
   quality.

## Decision log

| # | decision | why | what it cost |
|---|---|---|---|
| 1 | SQLite + sqlite-vec instead of a vector DB | corpus is thousands of chunks, not millions; one less service to run, back up, and auth | vec0 cannot filter on non-vector columns, so the competitor filter is a post-scan join (tested to match global ranking exactly) |
| 2 | Local embedding model (fastembed/bge-small) | no API key, no per-call cost, works offline; CI downloads and caches it once | ONNX memory pressure when stacked with Ollama + pytest on one laptop — serialize heavy runs |
| 3 | Provider-agnostic LLM layer, resolved per request | Ollama locally, OpenAI if a key exists; nothing else knows about providers | an ambient provider made tests pass on a dev box and fail in CI — the resolver is now pinned in tests |
| 4 | 202 + job id for anything slow, poll from n8n | HTTP nodes time out and retry — a retried report runs the whole pipeline twice | in-memory job registry; restart loses job history (reports persist on disk) |
| 5 | One Code node owns the poll-loop state machine | `wait/done/failed/timeout` decided in exactly one place instead of duplicated across IF nodes | none observed; it made the loop auditable |
| 6 | Refresh Slack-notifies only on new posts | a daily "0 new posts" message trains people to mute the channel | the "silent when healthy" failure mode needs the alert path tested — done live (exec #7) |
| 7 | Source tag on every job (`api` / `n8n`) | scheduled runs must be distinguishable from manual ones when auditing history | none; one field end to end |
| 8 | Tests stub the network, and a guard enforces it | CI must never touch HN; an unstubbed test once fetched Hacker News for real inside a CI run | none; raising stubs fail in milliseconds with the fix in the message |
| 9 | Exported n8n JSONs stay secret-free via `$env` | the workflows are committed; only the `.env` (gitignored) knows the webhook | n8n denies `$env` reads by default — the launcher sets `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` |
| 10 | Console is one offline HTML file, enforced by tests | zero build step, works on an airgapped laptop; drift is caught by byte-equality test | no shared component library; acceptable at this size |

## Problem → fix log

Real failures, kept as a record because each one changed how the code
looks today.

1. **CI fetched Hacker News for real.** A refresh test posted
   `/pipeline/refresh` without stubbing `fetch_all`; a CI runner spent
   minutes fetching live data and failed. *Fix:* the missing stub —
   then an autouse network guard in `tests/conftest.py` so any future
   unstubbed test fails instantly with a message naming the fix.
2. **Tests passed locally, failed in CI.** The provider resolver
   auto-detected per request by probing `localhost:11434`; a developer
   box with Ollama running silently took a different path than the
   bare CI runner. *Fix:* the `client` fixture pins the provider; a
   regression test hides all providers and expects the endpoints to
   still respond.
3. **A retried report would run twice.** Driving a minutes-long report
   over one HTTP call means the n8n HTTP node times out and retries.
   *Fix:* the job handshake — `202` + job id, the workflow polls.
4. **The measured report took 1375 s.** The workflow's original poll
   budget (30 × 30 s) would have declared a healthy run a timeout.
   *Fix:* 60 × 45 s (≈2x margin), chosen from the measured run, not a
   guess.
5. **`PUT /rest/workflows/{id}` returned 404.** The internal REST API
   does not serve that route; the editor-UI import seemed like the
   only path. *Fix:* the public API (`PUT /api/v1/workflows/{id}`)
   updates in place and keeps the workflow id — `id`, `active` and
   `versionId` are read-only body fields that must be omitted.
   Verified by firing a refresh afterwards: the job came back with
   `source: "n8n"`.
6. **n8n 2.x nests `.n8n` under `N8N_USER_FOLDER`.** Pointing the
   variable at a folder already named `.n8n` silently created a fresh
   empty instance — the login "stopped working". *Fix:* pin the user
   folder at the install's `data` directory; documented in
   `n8n/README.md`.
7. **A Windows `cmd` launcher dropped the path at the first `&`.** The
   project directory contains `&`, and `cmd /c "path with &"` splits
   there. *Fix:* launch the `.cmd` via the shell's association instead
   of an intermediate `cmd /c` string.
8. **Memory starvation OOM'd the embedder.** API server + Ollama +
   pytest's corpus seeding together exhausted RAM and ONNX could not
   allocate. *Fix:* serialize heavy local runs (kill :8000 before a
   full test pass); the failure path itself was then proven useful —
   the resulting failed refresh produced a correct Slack alert.
9. **Citations pointing at nothing.** A model can emit `[7]` having
   been shown six sources. *Fix:* layered validation — inline `[n]`
   checks in `answer.py`, the full audit in `agents.py`, and the
   verdict surfaced in the console report view.
10. **Console job rows stopped being clickable.** A refactor set
    `onclick` through an unattached node; the throw vanished inside a
    silent `catch`. *Fix:* attach handlers to the row after it is
    built; the lesson (never assign through a node that is not in the
    DOM yet) is why the pattern is now explicit.

## Status

- 115 tests green locally and in CI; retrieval-quality gate on every push.
- Both n8n workflows verified live end-to-end, including the failure
  path (Slack alert on a failed refresh).
- All six phases shipped; v2 candidates live in the roadmap below.

## Roadmap

1. ✅ Phase 1 — data pipeline (fetch → clean → store)
2. ✅ Phase 2 — RAG foundation (chunk → embed → vector search)
3. ✅ Phase 3 — grounded Q&A with citations
4. ✅ Phase 4 — five-agent report pipeline with critic + audit
5. ✅ Phase 5 — n8n orchestration with Slack receipts (verified live)
6. ✅ Phase 6 — FastAPI service, Docker, eval-gated CI
7. ⬜ v2 — DB-backed job queue for multi-process service
8. ⬜ v2 — second source (Reddit via OAuth) through the same pipeline
9. ⬜ v2 — strategy section → PRD generator (the deliverable the market
   map exists to produce)
10. ⬜ v2 — hosted demo: read-only console against a small public corpus

## Working agreements

- Free stack only: fastembed, Ollama, SQLite, self-hosted n8n Community,
  Slack webhook, GitHub Free — no paid service anywhere.
- Secrets live in the gitignored `.env` (`.env.example` documents every
  knob); Slack text in committed files is ASCII-only and tested.
- Every claim in the docs is either measured, verified live, or marked
  as a limit.
