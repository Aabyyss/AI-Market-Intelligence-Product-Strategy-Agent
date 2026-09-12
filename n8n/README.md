# Phase 5 — n8n orchestration

n8n owns the *business* workflow: when to run, what to do when a run
fails, and who gets told. Python owns the intelligence. The two meet at
the HTTP API (`run_api.py`) — n8n never imports our code, and our code
never knows n8n exists.

```text
06:00  corpus_refresh.json   POST /pipeline/refresh   -> poll /jobs/{id}
                               fetch HN -> clean -> store -> rebuild index
07:00  market_report.json    POST /reports            -> poll /jobs/{id}
                               research -> competitor x3 -> customer
                               -> strategy -> critic -> markdown report
                             then: Slack summary (or a no-op)
```

Two workflows, not one, because they have different failure meanings: a
failed refresh means the corpus is stale and the report should probably
not run; a failed report means the data is fine and the analysis broke.
Keeping them separate also means the hour between them is slack for a
slow fetch.

## Files

| file | what it does |
|---|---|
| `corpus_refresh.json` | 06:00 daily — refresh the corpus and rebuild the vector index |
| `market_report.json` | 07:00 daily — generate the market report, notify Slack |

Both are importable: **n8n → Workflows → ⋯ → Import from File**.

## Setup

1. **Start the API** (it must be reachable from the n8n container/host):

   ```bash
   python run_api.py --port 8000            # from the project root
   # or: docker compose up --build
   ```

   Confirm: `curl http://127.0.0.1:8000/health`

2. **Import** both JSON files into n8n.

3. **Set the environment** on the n8n process (or edit the `Config` node
   — every value there is an expression with a default, so it also runs
   with none of these set):

   | variable | default | meaning |
   |---|---|---|
   | `MARKET_INTEL_API_BASE` | `http://127.0.0.1:8000` | where the API lives |
   | `MARKET_INTEL_BRIEF` | `fees, payouts and platform switching` | the research question |
   | `MARKET_INTEL_PER_QUERY` | `4` | evidence chunks retrieved per query |
   | `MARKET_INTEL_FETCH_LIMIT` | `25` | HN hits per query on refresh |
   | `MARKET_INTEL_MAX_POLLS` | `40` / `60` | give up after N polls |
   | `MARKET_INTEL_POLL_SECONDS` | `45` | seconds between report polls |
   | `SLACK_WEBHOOK_URL` | *(empty)* | incoming-webhook URL; empty = skip notification |

   Notes: `$env` access must not be blocked in n8n
   (`N8N_BLOCK_ENV_ACCESS_IN_NODE=false`, which is the default), and in
   Docker the API is not on `localhost` — set
   `MARKET_INTEL_API_BASE=http://api:8000` (compose service name) or
   `http://host.docker.internal:8000` (API on the host).

4. **Dry run**: open a workflow and click *Execute Workflow*. Or skip the
   UI entirely and drive the API by hand:

   ```bash
   curl -s -X POST localhost:8000/reports \
     -H 'content-type: application/json' \
     -d '{"brief":"fees and developer payouts","per_query":4}'
   # -> {"job_id":"...","status":"queued",...}
   curl -s localhost:8000/jobs/<job_id>
   ```

## Why the workflows look like this

**Poll, don't wait.** A five-agent report is minutes of LLM work. n8n's
HTTP node would time out and, worse, *retry* — running the whole pipeline
twice. So the API returns `202` with a job id immediately and the
workflow polls `GET /jobs/{id}` on a `Wait` node. This is the standard
async-job handshake and it is also what makes runs observable: the job's
status is a real object with `started_at` and `duration_seconds`.

**The route decision is one Code node.** Every poll output funnels into
`Route`, which maps the job state to `wait` / `done` / `failed` /
`timeout` and carries an attempt counter. Two IF nodes then branch on
that single field, so the loop-back condition exists in exactly one
place instead of being duplicated across IFs.

**Bounded retries.** `max_attempts` turns "the API is wedged" into a
`timeout` failure with a Slack message, instead of an execution that
polls until the heat death of the universe. A report on a CPU-only box
genuinely takes ~20 minutes, which is why the default is 40 polls × 45 s.

**Silence is a feature.** The refresh posts to Slack only when it
actually found something (`changed`); a daily "0 new posts" message just
teaches people to mute the channel. Failures always notify.

**No credentials in the workflow.** The Slack webhook comes from an env
var, so the exported JSON is safe to commit — which is why these files
are in git and your `.env` is not.

## Extending it

- **Notion instead of Slack**: add an HTTP Request node to
  `POST https://api.notion.com/v1/pages` with a Notion credential, and
  send `$json.result` (the typed summary) as page properties.
- **Email digest**: add an `n8n-nodes-base.emailSend` node on the success
  branch and attach the report from `GET /jobs/{id}/markdown`.
- **Event-driven**: give the report endpoint an n8n Webhook trigger
  instead of a schedule, then call it from Slack with a slash command.
- **Quality gate in the loop**: call `POST /eval` after a refresh and
  fail the workflow if `averages.recall` drops — the retrieval gate from
  CI, applied to the live corpus.
