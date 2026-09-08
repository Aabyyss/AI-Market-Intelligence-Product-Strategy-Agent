"""Central configuration: which competitors we monitor and how we query sources.

Keeping this in one place means the rest of the code never hard-codes
a brand name. Adding a new competitor later is a one-line change here.
"""

import os

# The market we chose: e-commerce platforms.
COMPETITORS = ["shopify", "woocommerce", "bigcommerce"]

# Proper display names for reports (brands have specific capitalization).
COMPETITOR_LABELS = {
    "shopify": "Shopify",
    "woocommerce": "WooCommerce",
    "bigcommerce": "BigCommerce",
}

# Intel angles we monitor per competitor. For every angle the fetcher
# runs a query made of the quoted brand name PLUS the angle keyword,
# e.g. '"shopify" checkout'. "" = any mention of the brand at all.
INTEL_ANGLES = ["", "checkout", "fees", "migrate", "downtime"]

# Identify ourselves when calling external APIs.
USER_AGENT = "market-intel-agent/0.1 (portfolio project; learning data pipelines)"

# Seconds to wait between API calls, to stay within rate limits.
REQUEST_DELAY_SECONDS = 1.0

# --- Phase 2: chunking + embeddings ---
# Chunk size in words; overlapping neighbours so no idea is lost at a
# chunk boundary. Most HN posts are short, so most get a single chunk.
CHUNK_MAX_WORDS = 250
CHUNK_OVERLAP_WORDS = 40

# Local embedding model (fastembed / ONNX Runtime). bge-small-en-v1.5 is
# small (384 dims) and strong for English retrieval. Downloaded once on
# first use, then cached on disk.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# --- Phase 3: LLM (evidence-backed answering) ---
# "auto" picks OpenAI if OPENAI_API_KEY is set, else Ollama if it
# responds on localhost:11434. Override with LLM_PROVIDER=openai|ollama|custom.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "auto")

# Empty = use the per-provider default below. Override with LLM_MODEL=...
LLM_MODEL = os.environ.get("LLM_MODEL", "")
LLM_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "ollama": "llama3.2",
    "custom": "gpt-4o-mini",
}

# Low temperature: grounded answers should be deterministic, not creative.
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 800

# Ollama (and any OpenAI-compatible server) expose /v1 chat completions.
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
CUSTOM_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")

# --- Phase 4: multi-agent layer ---
# Where run_report.py writes market reports.
REPORT_DIR = "data/reports"

# Research agent: how many topical queries per competitor it may propose
# (the quoted brand name itself is always included as a baseline).
QUERIES_PER_COMPETITOR = 2
RESEARCH_MAX_TOKENS = 300

# Analysts write longer outputs than a single answer — let them breathe.
# (Smaller is also faster on CPU-only machines like a local Ollama box.)
AGENT_MAX_TOKENS = 600

# The critic has the most to write (one verdict line per claim + summary).
CRITIC_MAX_TOKENS = 900


def queries_for(competitor: str) -> list[str]:
    """The exact search queries to run for one competitor.

    The brand name is always quoted: that forces an exact-phrase match
    in HN's search, which stops fuzzy matches like "spotify" leaking
    into shopify results. Each non-empty angle narrows to posts that
    also discuss that topic.
    """
    queries = []
    for angle in INTEL_ANGLES:
        phrase = f'"{competitor}"'
        if angle:
            phrase += f" {angle}"
        queries.append(phrase)
    return queries