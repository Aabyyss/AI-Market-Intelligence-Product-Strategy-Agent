"""Phase 1 pipeline: fetch competitor discussions -> clean -> store in SQLite.

Usage:
    python run_pipeline.py [--limit N]

The raw API responses are dumped to data/raw/ for provenance, and the
cleaned records land in data/market_intel.db. Re-running never
duplicates rows.
"""

import argparse
import json
import re
import time
from pathlib import Path

from market_intel.clean import clean_posts
from market_intel.fetch import fetch_all
from market_intel.store import connect, insert_posts, prune_rows


def main(limit: int) -> None:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    db_path = Path("data/market_intel.db")

    print("== Step 1: fetch raw posts from Hacker News ==")
    raw_items = fetch_all(limit=limit)

    # Keep the raw JSON on disk: provenance, re-processing, debugging.
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for (competitor, query), items in raw_items.items():
        tag = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
        dump = raw_dir / f"hackernews_{competitor}_{tag}_{stamp}.json"
        dump.write_text(json.dumps(items, indent=2))

    print("== Step 2: clean + dedupe (incl. brand filter) ==")
    records = clean_posts(raw_items)

    print("== Step 3: store in SQLite ==")
    conn = connect(str(db_path))
    inserted = insert_posts(conn, records)
    pruned = prune_rows(conn)  # drop old rows that violate the brand filter
    conn.close()

    n_raw = sum(len(items) for items in raw_items.values())
    print(f"\n  fetched : {n_raw} raw items (across all angle queries)")
    print(f"  cleaned : {len(records)} unique, usable items after brand filter")
    print(f"  stored  : {inserted} new rows (duplicates skipped)")
    print(f"  pruned  : {pruned} stale rows removed (brand filter at rest)")
    print(f"  db      : {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=25,
                        help="max items to fetch per query (default: 25)")
    args = parser.parse_args()
    main(args.limit)