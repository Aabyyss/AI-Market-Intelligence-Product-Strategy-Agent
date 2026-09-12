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

# Statuses worth retrying: rate limiting and server-side trouble. A 400
# or 404 means *we* sent something wrong, so retrying only wastes time.
RETRY_STATUS = {429, 500, 502, 503, 504}
MAX_ATTEMPTS = 3        # one try + two retries
BACKOFF_SECONDS = 2.0   # 2s, then 4s


def _retryable(exc: requests.RequestException) -> bool:
    """Is this failure worth another attempt?

    No response at all (DNS, TLS handshake, timeout, connection reset)
    is transient by definition; an HTTP status is only transient if it is
    in RETRY_STATUS.
    """
    status = getattr(exc.response, "status_code", None)
    return status is None or status in RETRY_STATUS


def search_hn(query: str, limit: int = 25,
              attempts: int = MAX_ATTEMPTS) -> list[dict]:
    """Return the newest story dicts matching ``query``.

    Transient failures are retried with exponential backoff. This one
    matters more than it looks: the refresh runs on a nightly schedule,
    and a single reset socket should not turn into a failed job (and a
    6am notification) when two seconds of patience would have fixed it.
    """
    params = {
        "query": query,
        "tags": "story",     # only stories, not comments
        "hitsPerPage": limit,
    }
    headers = {"User-Agent": config.USER_AGENT}

    for attempt in range(1, attempts + 1):
        try:
            response = requests.get(
                SEARCH_URL, params=params, headers=headers, timeout=30
            )
            response.raise_for_status()  # raises HTTPError on 4xx/5xx
            return response.json()["hits"]
        except requests.RequestException as exc:
            if attempt == attempts or not _retryable(exc):
                raise
            wait = BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  [fetch] {type(exc).__name__} on q={query} — "
                  f"retrying in {wait:.0f}s ({attempt}/{attempts - 1})")
            time.sleep(wait)

    raise AssertionError("unreachable: the loop either returns or raises")


def planned_queries() -> int:
    """How many requests a full fetch makes (competitors × angles)."""
    return sum(len(config.queries_for(c)) for c in config.COMPETITORS)


def fetch_all(limit: int = 25,
              tolerate_failures: bool = False) -> dict[tuple[str, str], list[dict]]:
    """Fetch stories for every competitor × angle query.

    Returns a dict keyed by (competitor, query) so callers know which
    query produced which results. The same story can legitimately match
    several queries (e.g. both "checkout" and "migrate"); dedup happens
    later in the clean step.

    ``tolerate_failures`` is for the scheduled path: one query failing
    after its retries should not throw away the other fourteen, so it is
    skipped (and logged) instead. If *every* query fails the corpus is
    not refreshed at all, which is an error worth raising — a job that
    reports success while collecting nothing is worse than a failed one.
    """
    results: dict[tuple[str, str], list[dict]] = {}
    failures: list[str] = []
    for competitor in config.COMPETITORS:
        for query in config.queries_for(competitor):
            print(f"  [fetch] hackernews q={query}")
            try:
                results[(competitor, query)] = search_hn(query, limit)
            except requests.RequestException as exc:
                if not tolerate_failures:
                    raise
                failures.append(query)
                print(f"  [fetch] SKIPPED q={query} after retries "
                      f"({type(exc).__name__})")
            time.sleep(config.REQUEST_DELAY_SECONDS)  # stay under rate limits

    if results == {} and failures:
        raise RuntimeError(
            f"every query failed ({len(failures)}): no data collected; "
            f"first error was for q={failures[0]!r}"
        )
    if failures:
        print(f"  [fetch] {len(failures)}/{planned_queries()} queries failed")
    return results