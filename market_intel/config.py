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