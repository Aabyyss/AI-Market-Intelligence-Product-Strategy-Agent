# Runbook - firing the scheduled demo by hand

The workflows carry `scheduleTrigger` nodes ("Daily 06:00" / "Daily
07:00"), so on any day they fire themselves. This runbook is for
**triggering a run on demand** - a demo, a rehearsal, or a retry after a
failure - without clicking through the editor.

Everything here talks to n8n's *internal* REST API with the owner
session cookie. The public API would be cleaner, but creating an API
key from the web UI is blocked on some plans ("Invalid scopes for user
role"); the cookie route works everywhere and costs nothing.

## 0. One-time setup

1. Install n8n under Node 22 (newer Node fails to build n8n's sqlite
   native module without Visual Studio Build Tools):

   ```bat
   md "%USERPROFILE%\.mi-n8n"
   cd /d "%USERPROFILE%\.mi-n8n"
   "<path-to-node-22>\node.exe" "<path-to-node-22>\npm.cmd" install n8n
   ```

2. Start it (loads the repo `.env`, sets the two required flags):

   ```bat
   n8n\start_n8n.cmd
   ```

   First boot asks you to create the owner account in the browser at
   `http://localhost:5678` - do that once, then keep the credentials in
   your `.env`-style secrets store, not in git.

3. Start the market_intel API:

   ```bash
   .venv/Scripts/python.exe app.py --no-browser     # serves 127.0.0.1:8000
   curl http://127.0.0.1:8000/health
   ```

4. Import both workflow JSONs (n8n editor: Workflows -> ... -> Import
   from File) and note their IDs from the editor URL.

## 1. Log a session in

```bash
curl -s -i -X POST http://127.0.0.1:5678/rest/login \
  -H "content-type: application/json" \
  -d '{"emailOrLdapLoginId":"you@example.local","password":"..."}' \
  | grep -o "n8n-auth=[^;]*" > /tmp/n8n_cookie
```

The cookie is a JWT; it expires after a few days - just log in again.

## 2. Fire a workflow from its schedule trigger

`triggerToStartFrom` executes the trigger node's path exactly as the
scheduler would, so the run in the history looks like the real thing:

```bash
WF=<workflow id from the editor URL>
curl -s -X POST "http://127.0.0.1:5678/rest/workflows/$WF/run" \
  -H "content-type: application/json" \
  -H "cookie: $(cat /tmp/n8n_cookie)" \
  -d '{"mode":"trigger","triggerToStartFrom":{"name":"Daily 07:00"}}'
# -> {"data":{"executionId":"42"}}
```

Or use the helper, which wraps login, fire and poll:

```bash
python n8n/tools/fire_workflow.py --trigger "Daily 07:00" --wait
```

## 3. Poll the execution

```bash
curl -s "http://127.0.0.1:5678/rest/executions/42" \
  -H "cookie: $(cat /tmp/n8n_cookie)"
# status: new -> running -> success | error
```

A *report* takes ~20 minutes (five agents over llama3.2 on CPU); the
workflow's own Wait/poll loop is what takes that long, so be patient.
When it finishes, check the last nodes ran - the success path ends in
`Format success -> Notify Slack`:

```bash
python n8n/tools/inspect_exec.py 42 notify
```

Slack's webhook answering `ok` is the delivery receipt.

## 4. When something fails

- `Invalid wait amount` in `Wait` - you are on an old export that drops
  `poll_seconds` on the Route loop-back; update from git (fixed in
  `corpus_refresh.json` / `market_report.json`).
- `access to env vars denied` in `Config` - n8n is blocking `$env`
  reads; start it with `n8n/start_n8n.cmd`, which sets
  `N8N_BLOCK_ENV_ACCESS_IN_NODE=false`.
- Login suddenly rejected and migrations run "from scratch" on boot -
  `N8N_USER_FOLDER` points one level too deep; n8n 2.x nests `.n8n`
  itself. See the note in `n8n/README.md`.
- The API job failed inside the pipeline - that is n8n doing its job;
  the workflow should have sent the failure Slack message. Debug the
  pipeline with `GET /jobs/{id}` (the `error` field has the traceback).
