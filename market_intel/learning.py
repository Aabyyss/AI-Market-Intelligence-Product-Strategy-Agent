"""The self-learning loop: the corpus grows, the query plan adapts.

Every report runs a set of research queries over the vector index. Some
retrieve nothing — the brief asked about a topic the fixed query list
does not cover. Those misses are signal: this module remembers them,
resurfaces them as *seeded queries* on the next report for the same
brief, and reinforces the seeds that did retrieve evidence. Queries that
keep working rank higher; the plan therefore drifts toward what the
corpus can actually answer without anyone editing a prompt.

Deliberately small and honest about what it is:

  * it is a per-topic query feedback store (SQLite, same DB as the
    corpus), not an online model — nothing about the agents' weights
    changes, only the queries feeding them;
  * it never *removes* a planned query — LLM-proposed and baseline
    queries always run; seeds are additive and capped, so a bad memory
    costs at most MAX_SEEDS searches;
  * every failure is swallowed at the call site — learning must never
    be the reason a report fails.

Schema (created lazily, so old databases keep working):

    learning(topic TEXT, query TEXT, strength INTEGER,
             first_seen TEXT, last_seen TEXT,
             PRIMARY KEY (topic, query))

``topic`` is the report brief; ``strength`` starts at 1 when a gap is
first seen and grows by 1 each time the query later retrieves evidence
for that topic.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from datetime import datetime, timezone

log = logging.getLogger("market_intel.learning")

# Seeds added to a report's query plan, per brief. Small on purpose: the
# loop explores, it does not take over the plan.
MAX_SEEDS = 2

# Set MARKET_INTEL_LEARNING=0 to freeze the loop (tests, debugging).
enabled = os.environ.get("MARKET_INTEL_LEARNING", "1") != "0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """CREATE TABLE IF NOT EXISTS learning (
               topic      TEXT NOT NULL,
               query      TEXT NOT NULL,
               strength   INTEGER NOT NULL DEFAULT 1,
               first_seen TEXT NOT NULL,
               last_seen  TEXT NOT NULL,
               PRIMARY KEY (topic, query)
           )"""
    )
    conn.commit()


def _norm(topic: str) -> str:
    """Topics are briefs; normalize so 'Fees!' and 'fees' share a key."""
    return " ".join(topic.lower().split())


def record_gap(conn: sqlite3.Connection, topic: str, query: str) -> None:
    """Remember a planned query that retrieved nothing for this topic."""
    ensure_table(conn)
    conn.execute(
        """INSERT INTO learning (topic, query, strength, first_seen, last_seen)
           VALUES (?, ?, 1, ?, ?)
           ON CONFLICT(topic, query) DO UPDATE SET last_seen=excluded.last_seen""",
        (_norm(topic), query.strip(), _now(), _now()),
    )
    conn.commit()


def record_reinforcement(conn: sqlite3.Connection, topic: str, query: str) -> None:
    """A seeded (or planned) query retrieved evidence — make it stronger."""
    ensure_table(conn)
    conn.execute(
        """INSERT INTO learning (topic, query, strength, first_seen, last_seen)
           VALUES (?, ?, 2, ?, ?)
           ON CONFLICT(topic, query) DO UPDATE
               SET strength = strength + 1, last_seen=excluded.last_seen""",
        (_norm(topic), query.strip(), _now(), _now()),
    )
    conn.commit()


def seed_queries(conn: sqlite3.Connection, topic: str,
                 limit: int = MAX_SEEDS) -> list[str]:
    """Learned queries for this brief, strongest and freshest first.

    Only queries that ever worked (strength >= 2) or were missed at most
    recently come back; the cap keeps the addition bounded. Never raises:
    a broken or foreign database must not break report planning.
    """
    try:
        ensure_table(conn)
        rows = conn.execute(
            """SELECT query FROM learning
               WHERE topic = ?
               ORDER BY strength DESC, last_seen DESC
               LIMIT ?""",
            (_norm(topic), limit),
        ).fetchall()
        return [r[0] for r in rows]
    except sqlite3.Error:
        log.exception("learning: seed lookup failed; continuing without")
        return []


def merge_seeds(queries: list[str], seeds: list[str],
                limit: int = MAX_SEEDS) -> list[str]:
    """Append seeds to a query plan without duplicates or reordering.

    Pure helper so the merge rule is testable without a database: seeds
    go last (they are the adaptive tail of the plan), case-insensitively
    deduped against the planned queries and each other.
    """
    seen = {q.strip().lower() for q in queries}
    out = list(queries)
    for seed in seeds:
        key = seed.strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(seed.strip())
        if len(out) - len(queries) >= limit:
            break
    return out


def feedback(conn: sqlite3.Connection, topic: str,
             queries: list[str], hit_counts: dict[str, int]) -> None:
    """Close the loop for one report: reinforce hits, remember misses.

    ``hit_counts`` maps query -> number of raw search hits (before
    dedupe). Never raises; one bad row must not poison the report.
    """
    if not enabled:
        return
    try:
        for q in queries:
            hits = hit_counts.get(q, 0)
            if hits > 0:
                record_reinforcement(conn, topic, q)
            else:
                record_gap(conn, topic, q)
    except sqlite3.Error:
        log.exception("learning: feedback failed; continuing")


def stats(conn: sqlite3.Connection) -> dict:
    """Totals for /learning: topics tracked and the strongest seeds."""
    ensure_table(conn)
    topics = conn.execute("SELECT COUNT(DISTINCT topic) FROM learning").fetchone()[0]
    tracked = conn.execute("SELECT COUNT(*) FROM learning").fetchone()[0]
    top = conn.execute(
        """SELECT topic, query, strength, last_seen FROM learning
           ORDER BY strength DESC, last_seen DESC LIMIT 10"""
    ).fetchall()
    return {
        "topics": topics,
        "queries_tracked": tracked,
        "top": [
            {"topic": t, "query": q, "strength": s, "last_seen": ls}
            for t, q, s, ls in top
        ],
    }
