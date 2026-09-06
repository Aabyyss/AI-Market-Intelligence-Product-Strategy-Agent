"""Fetch competitor discussions from the Hacker News Algolia API.

Algolia hosts an official read-only API over HN data — no auth, no key,
generous rate limits (10k requests/hour), and real JSON:

    https://hn.algolia.com/api/v1/search_by_date?query=<q>&tags=story

``search_by_date`` returns the newest stories first. ``tags=story`` keeps
only discussion stories (not individual comments). The response shape::

    {"hits": [ {objectID, title, url, author, points,
                num_comments, created_at_i, ...} ]}

Why not Reddit? We originally targeted Reddit's public JSON API, but
Reddit now blocks anonymous JSON access from many networks (403 on
www.reddit.com, HTML instead of JSON on old.reddit.com) and requires
OAuth app credentials. The fetch/clean/store pipeline is source-agnostic,
so Reddit (or any other source) can be added later — Phase 6 covers
credentials and environment variables.
"""

import time

import requests

from market_intel import config

SEARCH_URL = "https://hn.algolia.com/api/v1/search_by_date"


def search_hn(query: str, limit: int = 25) -> list[dict]:
    """Return the newest story dicts matching ``query``."""
    params = {
        "query": query,
        "tags": "story",     # only stories, not comments
        "hitsPerPage": limit,
    }
    headers = {"User-Agent": config.USER_AGENT}
    response = requests.get(SEARCH_URL, params=params, headers=headers, timeout=30)
    response.raise_for_status()  # raises HTTPError on 4xx/5xx
    return response.json()["hits"]


def fetch_all(limit: int = 25) -> dict[tuple[str, str], list[dict]]:
    """Fetch stories for every competitor × angle query.

    Returns a dict keyed by (competitor, query) so callers know which
    query produced which results. The same story can legitimately match
    several queries (e.g. both "checkout" and "migrate"); dedup happens
    later in the clean step.
    """
    results: dict[tuple[str, str], list[dict]] = {}
    for competitor in config.COMPETITORS:
        for query in config.queries_for(competitor):
            print(f"  [fetch] hackernews q={query}")
            results[(competitor, query)] = search_hn(query, limit)
            time.sleep(config.REQUEST_DELAY_SECONDS)  # stay well under rate limits
    return results