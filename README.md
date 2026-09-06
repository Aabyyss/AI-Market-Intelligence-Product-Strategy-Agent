# AI Market Intelligence & Product Strategy Agent

Monitors e-commerce platform competitors (**Shopify, WooCommerce,
BigCommerce**) and will generate evidence-backed product opportunities
and PRDs. Built incrementally, phase by phase.

## Phase 1 (current): data pipeline — fetch → clean → store

Collects public discussions about each competitor into a local SQLite
database, ready for the RAG / analysis stages.

**Data source:** Hacker News via the [Algolia API](https://hn.algolia.com/api)
— public, no auth required. (Reddit's public JSON API now requires OAuth
app credentials, so the pipeline was kept source-agnostic to add it later.)

## Run

```bash
python -m venv .venv
# Windows:  .venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
python run_pipeline.py [--limit 25]
```

## Layout

```
market_intel/
  config.py     # competitors + search queries (one place to edit)
  fetch.py      # API calls -> raw JSON items
  clean.py      # normalize, drop junk, dedupe -> records
  store.py      # SQLite schema + idempotent inserts
run_pipeline.py # CLI: fetch -> clean -> store
data/           # raw JSON dumps + market_intel.db (gitignored)
```

## Roadmap

1. ✅ Phase 1 — Python foundation: APIs, JSON, cleaning, SQLite storage
2. ⬜ Phase 2 — RAG: chunking, embeddings, vector search, citations
3. ⬜ Phase 3 — LLM: structured outputs, prompts, tool calling
4. ⬜ Phase 4 — Agents: research → competitor → customer → strategy → critic
5. ⬜ Phase 5 — n8n: scheduled workflow + notifications
6. ⬜ Phase 6 — Production: FastAPI, Docker, env vars, eval, deploy