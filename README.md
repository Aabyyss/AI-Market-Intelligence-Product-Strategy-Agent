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

The embeddings live in the same SQLite file, indexed by a real vector
index — the **sqlite-vec** extension's `vec0` virtual table (k-NN search,
`distance_metric=cosine`), alongside the `chunks` metadata table. The
`--competitor` filter is applied as a SQL join after the scan, since
`vec0` cannot filter on non-vector columns; a scan covers the whole
corpus so results match a global cosine ranking exactly.

The vector index has a **regression test suite** (`tests/`) that pins
sqlite-vec's results to a reference numpy cosine ranking — the exact
algorithm the index replaced. It covers ranking, the competitor filter
(including the post-filter bug scenario), tie-breaking between
byte-identical chunks (reposted articles), and an empty corpus.

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -v
```

(First run downloads the embedding model once; it is cached afterwards,
same as `python run_index.py`.)

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

## Phase 4: multi-agent layer — research → analyze → strategize → fact-check

Turns the knowledge base into a market report. Five agents, one job
each, all working from the vector index (never raw text):

```text
research   -> plans search queries from the brief, gathers numbered evidence
competitor -> analyzes each competitor: positioning/pricing, strengths, weaknesses
customer   -> complaints, pain points, and praise from the community
strategy   -> ranked product opportunities, risks, next step — all cited
critic     -> fact-checks the draft against the evidence, flags unsupported claims
```

```bash
python run_report.py "fees and developer payouts"
python run_report.py "why merchants switch platforms" --competitors shopify woocommerce
python run_report.py "checkout friction" --out my_report.md
```

The report lands in `data/reports/` (or `--out`) with a sources section
mapping every citation number back to a URL. Anti-hallucination is the
same layered defense as Phase 3, at report scale: every agent must cite
numbered evidence, the critic judges *support* claim by claim, and a
mechanical audit (code, not LLM) verifies every [n] points at evidence
that was actually retrieved — including a warning when a draft cites
nothing at all.

Honest limits to know: a small local model (e.g. llama3.2:3b) produces
generic analyst prose and a critic that over-flags subjective-but-
accurate claims, so read reports critically — and the same model on a
CPU-only machine makes a full report take several minutes. Both are
better with a stronger model and faster hardware; the architecture is
provider-agnostic.

## Run

```bash
python -m venv .venv
# Windows:  .venv\Scripts\python -m pip install -r requirements.txt
# macOS/Linux: .venv/bin/python -m pip install -r requirements.txt
python run_pipeline.py [--limit 25]
```

## Layout

```
tests/
  test_vector_search.py  # sqlite-vec vs numpy cosine regression suite
conftest.py              # pytest sys.path bootstrap (empty)
market_intel/
  config.py     # competitors + queries + chunk/embed knobs (one place to edit)
  fetch.py      # API calls -> raw JSON items
  clean.py      # normalize, drop junk, dedupe -> records
  store.py      # SQLite schema + idempotent inserts
  chunk.py      # split posts into overlapping chunks
  embed.py      # local embedding model wrapper (fastembed/bge)
  vector.py     # sqlite-vec vec0 index over chunks + cosine search
  llm.py        # pluggable LLM client (ollama / openai / custom)
  answer.py     # grounded Q&A: evidence prompt + citation validation
  agents.py     # Phase 4: five agents + report assembly (research..critic)
run_pipeline.py # CLI: fetch -> clean -> store
run_index.py    # CLI: chunk + embed -> vector index
run_search.py   # CLI: semantic search over the index
run_ask.py      # CLI: evidence-backed Q&A with citations
run_report.py   # CLI: multi-agent market report
data/           # raw JSON dumps + market_intel.db + reports/ (gitignored)
```

## Roadmap

1. ✅ Phase 1 — Python foundation: APIs, JSON, cleaning, SQLite storage
2. ✅ Phase 2 — RAG: chunking, embeddings, vector search, citations
3. ✅ Phase 3 — LLM: grounded Q&A with evidence and citations
4. ✅ Phase 4 — Agents: research → competitor → customer → strategy → critic
5. ⬜ Phase 5 — n8n: scheduled workflow + notifications
6. ⬜ Phase 6 — Production: FastAPI, Docker, env vars, eval, deploy