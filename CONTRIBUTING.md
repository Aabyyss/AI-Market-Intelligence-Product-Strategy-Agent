# Contributing

The conventions below are the ones the CI suite actually enforces, so
reading this once saves you a red run.

## Dev environment

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # POSIX: .venv/bin/pip
.venv/Scripts/python -m pytest tests -q         # or run_*.py with .venv activated
```

Wait for the corpus seed on the first run (~165 s); afterwards the
whole suite is quick. Nothing in the suite needs the network or a
running service: an autouse fixture in `tests/conftest.py` stubs the
HN fetch, and a `-p ci_sim_plugin` flag exists for running the suite
with no LLM reachable.

One practical warning: the API server embeds a local model, so leaving
it listening on port 8000 during a pytest run can starve the
embedding model's ONNX session. Stop :8000 to run the suite, restart it
afterwards (`python app.py --no-browser`).

## What the guards expect

- **Demo assets** (`tests/test_demo_assets.py`): every edit to
  `docs/demo.html` stales the committed videos through the reel stamp -
  re-render both themes (`python render_demo.py` and
  `--theme light`) before committing. The same file guards the light
  palette of the reel, `market_intel/console.html`, and
  `docs/preview.html`: no dark-only hardcoded background may ship
  without its light repaint or a named exemption.
- **Workflows** (`.github/workflows/ci.yml`, `reel-smoke.yml`): CI runs
  the full suite plus the retrieval eval gate, then smoke-renders one
  demo frame per theme. If you add a renderer flag the workflows
  depend on, pin it in `test_demo_assets.py` so a rename fails with a
  message, not a blank CI error.
- **Secrets**: Slack webhooks, service tokens, and personal drafts
  (e.g. `LINKEDIN_POST.md`) stay out of git - `.env` and the draft
  files are gitignored for a reason.

## Commit style

Lowercase imperative subject, body explaining the *why*, footers:

```
make the intro badges readable in the light demo reel

The .badge chips kept their dark-theme background ... (why, not a
diff narration)

🤖 Generated with Codebuff
Co-Authored-By: Codebuff <noreply@codebuff.com>
```

## Verifying live behaviour

`python verify_admin.py` boots an isolated API and walks the admin
auth endpoints, printing one PASS/FAIL line per step. Similar
one-command receipts live for n8n (`n8n/tools/`). Receipts from real
runs are logged under `docs/` - add yours there rather than
describing them from memory.
