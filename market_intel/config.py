"""Central configuration: which competitors we monitor and how we query sources.

Keeping this in one place means the rest of the code never hard-codes
a brand name. Adding a new competitor later is a one-line change here.
"""

# The market we chose: e-commerce platforms.
COMPETITORS = ["shopify", "woocommerce", "bigcommerce"]

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