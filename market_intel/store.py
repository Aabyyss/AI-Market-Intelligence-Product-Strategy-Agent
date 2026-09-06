"""SQLite storage for cleaned records.

SQLite is the right store for this phase: zero setup, single file,
real SQL. If the project outgrows it (concurrent writers, cloud),
the same schema ports cleanly to Postgres later.
"""

import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts (
    id            TEXT PRIMARY KEY,        -- "<source>_<source_id>"
    competitor    TEXT NOT NULL,
    source        TEXT NOT NULL,           -- e.g. "hackernews" / "reddit"
    community     TEXT,                    -- where the item appeared
    title         TEXT NOT NULL,
    body          TEXT,
    url           TEXT,
    author        TEXT,
    created_utc   INTEGER,                 -- unix seconds
    score         INTEGER DEFAULT 0,
    num_comments  INTEGER DEFAULT 0,
    fetched_at    TEXT NOT NULL            -- ISO timestamp of this run
);

CREATE INDEX IF NOT EXISTS idx_posts_competitor ON posts (competitor);
CREATE INDEX IF NOT EXISTS idx_posts_created   ON posts (created_utc);
"""


def connect(db_path: str) -> sqlite3.Connection:
    """Open the DB and make sure the schema exists."""
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def insert_posts(conn: sqlite3.Connection, records: list[dict]) -> int:
    """Insert records, skipping ids already present. Returns rows inserted.

    ``INSERT OR IGNORE`` makes the pipeline idempotent: re-running it
    never duplicates data, so it is safe to schedule daily later.
    """
    sql = """
        INSERT OR IGNORE INTO posts
        (id, competitor, source, community, title, body, url, author,
         created_utc, score, num_comments, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    fetched_at = datetime.now(timezone.utc).isoformat()
    rows = [
        (
            r["id"], r["competitor"], r["source"], r["community"], r["title"],
            r["body"], r["url"], r["author"], r["created_utc"], r["score"],
            r["num_comments"], fetched_at,
        )
        for r in records
    ]
    cursor = conn.executemany(sql, rows)
    conn.commit()
    return cursor.rowcount  # only counts rows actually inserted


def prune_rows(conn: sqlite3.Connection) -> int:
    """Delete rows that no longer mention their competitor's brand.

    Rows inserted before the brand filter existed (or by older query
    configs) can pollute the corpus. Re-running the pipeline now enforces
    the same rule over everything stored. Returns rows deleted.
    """
    from market_intel.clean import record_mentions_brand

    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, competitor, title, body, url FROM posts"
    ).fetchall()
    stale = [
        row["id"]
        for row in rows
        if not record_mentions_brand(
            {"title": row["title"], "body": row["body"], "url": row["url"]},
            row["competitor"],
        )
    ]
    if stale:
        placeholders = ",".join("?" * len(stale))
        conn.execute(
            f"DELETE FROM posts WHERE id IN ({placeholders})", stale
        )
        conn.commit()
    return len(stale)