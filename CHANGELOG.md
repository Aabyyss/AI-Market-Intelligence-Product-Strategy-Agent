# Changelog

## 1.2.0 - 2026-09-27

The service becomes multi-user and the query plan becomes adaptive;
the console gets light mode; the project site gets a one-click way to
try the UI.

### Added
- **Multi-user auth** (`market_intel/auth.py`, `/auth/*`): register /
  login / logout / me / users. PBKDF2-HMAC-SHA256 passwords, opaque
  7-day bearer tokens stored hashed, in a separate `data/auth.db`.
  Open mode until the first account exists; that account becomes admin
  and protected endpoints lock down. Jobs record who started them,
  lists are scoped per user (admins see all), foreign job ids 404.
  `MARKET_INTEL_SERVICE_TOKEN` authenticates the n8n flows as user
  `n8n`.
- **Self-learning query loop** (`market_intel/learning.py`,
  `GET /learning`): report queries that retrieve nothing are remembered
  per brief, previously-successful queries are seeded into the next
  report's plan (strongest first, capped), and every query's hit count
  is fed back after each run. Additive and best-effort by design;
  `MARKET_INTEL_LEARNING=0` freezes it; report summaries carry
  `learned_queries`.
- Console: light mode with a persisted toggle (respects
  `prefers-color-scheme` on first visit; shift+T), sign-in card, per-user
  job tags; `Ctrl/Cmd+K` focuses the ask box.
- Demo videos in **both themes** (`docs/demo.mp4`, `docs/demo-light.mp4`)
  via `render_demo.py --theme light`.
- `docs/preview.html`: a static, interactive preview of the console
  (canned data from the verified live runs) on the project site - the
  one-click "try it" answer for the README and social posts.
- **Admin user management**: `DELETE /auth/users/{name}` (self-deletion
  and last-admin refused), password reset with automatic session
  revocation, and bulk session revoke - scoped admin-or-self.
- The n8n workflow HTTP nodes send the service token as a bearer header,
  so scheduled runs survive authenticated mode; documented in
  `.env.example` (`MARKET_INTEL_SERVICE_TOKEN`, `MARKET_INTEL_AUTH_DB`,
  `MARKET_INTEL_LEARNING`).
- Console: sign-in card and a "What it learned" panel fed by
  `GET /learning`.
- `tests/test_learning.py` (9), `tests/test_auth.py` (12) and new
  workflow/auth-header regression tests; suite now at 142 tests.

### Changed
- `tests/conftest.py` clears the in-memory job registry per test, so
  job history cannot leak between tests now that jobs carry users.
- Project site links the preview first and embeds both theme videos.

## 1.1.1 - 2026-09-27

The automation layer ran for real: both n8n workflows executed
end-to-end against the local API through their schedule triggers and
delivered Slack notifications over a free incoming webhook.

### Added
- `n8n/start_n8n.cmd`: one-command launcher that pins the n8n user
  folder and loads the repo `.env` into the n8n process environment.
- Job lineage: `POST /reports` and `POST /pipeline/refresh` accept a
  `source` tag (workflows send `n8n`), returned on every job object and
  badged in the console - scheduled runs are now distinguishable from
  manual ones.
- `n8n/tools/fire_workflow.py` + `n8n/tools/inspect_exec.py` and
  `n8n/RUNBOOK.md`: fire a scheduled workflow on demand and decode the
  n8n execution blob to verify what actually ran.
- `tests/test_n8n_workflows.py`: regression tests pinning the workflow
  contracts (trigger names, env wiring, the Route config-carry fix,
  the Slack guard, launcher flags).
- Live-run evidence log (`n8n/LIVE_RUN_2026-09-27.md`) and a
  "verified live" section on the project page.

### Changed
- Report poll budget raised to 60x45 s: the measured live report took
  1375 s, which left too thin a margin under the old 40x45 s budget.

### Fixed
- n8n env access: `N8N_BLOCK_ENV_ACCESS_IN_NODE=false` is required for
  the Config node's `$env.*` reads ("access to env vars denied").
- `N8N_USER_FOLDER` gotcha documented: n8n 2.x nests `.n8n` under the
  folder you give it, so pointing it at a `.n8n` directory silently
  creates a fresh, empty instance.

## 1.1.0

- Console UI, desktop launcher, Docker CI job, demo video and project
  page (see the repository README).
