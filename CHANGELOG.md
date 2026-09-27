# Changelog

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
