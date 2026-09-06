"""Turn raw API items into clean, normalized records for the database.

"Cleaning" here means the boring-but-essential data engineering:
  - collapse whitespace so titles are single-line text
  - drop items that don't actually mention their target brand
  - drop items with missing/empty titles
  - dedupe by a stable id (the pipeline can be re-run safely)

Precision backstop: HN search is fuzzy — ``query=shopify`` can return
Spotify stories. Quoted-phrase queries (see config.queries_for) fix most
of that server-side; ``mentions_brand`` below is the client-side safety
net, keeping a post only if the brand name appears verbatim in its
title, URL, or body.
"""

import re

from market_intel import config

# Prefix ids by source so ids can never collide between sources.
ID_PREFIX = "hackernews"

# One word-boundary pattern per competitor, e.g. \bshopify\b (i).
# Word boundaries stop "shopify" matching "shopifyyy" or "spotify".
_BRAND_PATTERNS = {
    competitor: re.compile(rf"\b{re.escape(competitor)}\b", re.IGNORECASE)
    for competitor in config.COMPETITORS
}


def normalize_text(text: str | None) -> str:
    """Collapse newlines/tabs/multiple spaces into single spaces."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text.replace("\r", " ")).strip()


def _brand_matches(competitor: str, *texts: str) -> bool:
    """True if the brand name appears in any of the given texts."""
    return bool(_BRAND_PATTERNS[competitor].search(" ".join(texts)))


def mentions_brand(raw_item: dict, competitor: str) -> bool:
    """True if a raw API item names the competitor (title/url/body)."""
    return _brand_matches(
        competitor,
        raw_item.get("title", ""),
        raw_item.get("url", ""),
        raw_item.get("story_text", ""),
    )


def record_mentions_brand(record: dict, competitor: str) -> bool:
    """Same check for stored records (different key names: body not story_text).

    Used by store.prune_rows so the brand rule also holds for rows that
    were inserted before the filter existed.
    """
    return _brand_matches(
        competitor,
        record.get("title", ""),
        record.get("body", ""),
        record.get("url", ""),
    )


def is_usable(raw_item: dict, competitor: str) -> bool:
    """True if the item is real content for this competitor."""
    return (
        bool(normalize_text(raw_item.get("title", "")))
        and mentions_brand(raw_item, competitor)
    )


def clean_item(raw_item: dict, competitor: str) -> dict:
    """Map one raw HN hit to a clean record for the DB."""
    object_id = raw_item["objectID"]
    return {
        "id": f"{ID_PREFIX}_{object_id}",
        "competitor": competitor,
        "source": "hackernews",
        "community": "news.ycombinator.com",
        "title": normalize_text(raw_item.get("title", "")),
        "body": normalize_text(raw_item.get("story_text", "")),
        # Link posts point at the article; Ask HN posts have no external URL.
        "url": raw_item.get("url")
        or f"https://news.ycombinator.com/item?id={object_id}",
        "author": raw_item.get("author", ""),
        "created_utc": raw_item.get("created_at_i"),  # unix seconds
        "score": raw_item.get("points", 0),
        "num_comments": raw_item.get("num_comments", 0),
    }


def clean_posts(raw_items: dict[tuple[str, str], list[dict]]) -> list[dict]:
    """Clean + dedupe all fetched items into one list of records.

    The same story often matches several angle queries (e.g. both
    "checkout" and "migrate"), so we dedupe by story id across queries.
    """
    cleaned: list[dict] = []
    seen: set[str] = set()

    for (competitor, _query), items in raw_items.items():
        for raw in items:
            if not is_usable(raw, competitor):
                continue
            record = clean_item(raw, competitor)
            if record["id"] in seen:
                continue
            seen.add(record["id"])
            cleaned.append(record)

    return cleaned