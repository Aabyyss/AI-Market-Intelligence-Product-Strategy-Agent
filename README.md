# AI Market Intelligence & Product Strategy Agent

Monitors e-commerce platform competitors (**Shopify, WooCommerce,
BigCommerce**) and will generate evidence-backed product opportunities
and PRDs. Built incrementally, phase by phase.

## Phase 1: data pipeline — fetch → clean → store

Collects public discussions about each competitor into a local SQLite
database, ready for the RAG / analysis stages.

**Data source:** Hacker News via the [Algolia API](https://hn.algolia.com/api)
— public, no auth required. (Reddit's public JSON API now requires OAuth
app credentials, so the pipeline was kept source-agnostic to add it later.)

## Phase 2: RAG foundation — chunk → embed → vector search

Reads the stored posts, splits them into overlapping chunks, embeds
each chunk with a **local** model (bge-small-en-v1.5 via fastembed — no
API key, no cost), and stores the vectors in SQLite next to the posts.
`run_search.py` ranks chunks by cosine similarity and prints the source
URL with each hit (the citation chain).

```bash
python run_index.py                        # build/refresh the vector index
python run_search.py "shopify checkout fees"
python run_search.py "migrating away from shopify" --competitor shopify
python run_search.py                       # interactive mode
```

The embeddings live in the same SQLite file for now; cosine similarity
runs in numpy. When the corpus grows, the index can move to a real
vector DB (sqlite-vec / qdrant / chroma) behind the same interface.

## Phase 3: grounded Q&A — retrieve → answer → cite

Closes the RAG loop: `run_ask.py` retrieves the top chunks for a
question, asks an LLM to answer **using only that evidence**, and prints
the sources it cited. Anti-hallucination is layered — the model sees
numbered evidence only, is told to cite inline as [n], and every [n] is
post-validated against the sources actually retrieved.

```bash
python run_ask.py "Why do app developers complain about Shopify?"
python run_ask.py "What should I consider before switching platforms?" --top 4
python run_ask.py                                   # interactive mode
```

The model provider is pluggable over any OpenAI-compatible endpoint and
auto-detected: a local **Ollama** server (free, no key) if reachable,
otherwise the **OpenAI API** when `OPENAI_API_KEY` is set. Override with
`LLM_PROVIDER=openai|ollama|custom` or `--provider`. Credentials live in
the environment, never in the code. On machines where Ollama's GPU
(Vulkan) discovery hangs, start it with `OLLAMA_VULKAN=false`.

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
  config.py     # competitors + queries + chunk/embed knobs (one place to edit)
  fetch.py      # API calls -> raw JSON items
  clean.py      # normalize, drop junk, dedupe -> records
  store.py      # SQLite schema + idempotent inserts
  chunk.py      # split posts into overlapping chunks
  embed.py      # local embedding model wrapper (fastembed/bge)
  vector.py     # chunk embeddings in SQLite + cosine search
  llm.py        # pluggable LLM client (ollama / openai / custom)
  answer.py     # grounded Q&A: evidence prompt + citation validation
run_pipeline.py # CLI: fetch -> clean -> store
run_index.py    # CLI: chunk + embed -> vector index
run_search.py   # CLI: semantic search over the index
run_ask.py      # CLI: evidence-backed Q&A with citations
data/           # raw JSON dumps + market_intel.db (gitignored)
```

## Roadmap

1. ✅ Phase 1 — Python foundation: APIs, JSON, cleaning, SQLite storage
2. ✅ Phase 2 — RAG: chunking, embeddings, vector search, citations
3. ✅ Phase 3 — LLM: grounded Q&A with evidence and citations
4. ⬜ Phase 4 — Agents: research → competitor → customer → strategy → critic
5. ⬜ Phase 5 — n8n: scheduled workflow + notifications
6. ⬜ Phase 6 — Production: FastAPI, Docker, env vars, eval, deploy