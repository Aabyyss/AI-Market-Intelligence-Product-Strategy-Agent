"""Tests for the self-learning loop (no LLM, no network).

The loop touches the same database the reports run on, so these tests
pin its contract: seeds come back strongest-first and capped, feedback
separates hits from gaps, the merge rule is pure, and a broken store
degrades to "no seeds" instead of breaking a report.
"""

import sqlite3

from market_intel import learning


def fresh_db(tmp_path):
    conn = sqlite3.connect(tmp_path / "learn.db")
    learning.ensure_table(conn)
    return conn


def test_gap_then_reinforce_grows_strength(tmp_path):
    conn = fresh_db(tmp_path)
    learning.record_gap(conn, "fees", '"shopify" refund windows')
    assert learning.seed_queries(conn, "fees") == ['"shopify" refund windows']

    learning.record_reinforcement(conn, "fees", '"shopify" refund windows')
    learning.record_reinforcement(conn, "fees", '"shopify" refund windows')
    stats = learning.stats(conn)
    assert stats["topics"] == 1
    assert stats["top"][0]["strength"] == 3  # 1 (gap) + 2 (reinforcements)


def test_seed_queries_cap_and_isolation_by_topic(tmp_path):
    conn = fresh_db(tmp_path)
    for q in ["q1", "q2", "q3", "q4", "q5"]:
        learning.record_reinforcement(conn, "fees", q)
    # MAX_SEEDS bounds the adaptive tail of any query plan
    assert len(learning.seed_queries(conn, "fees")) == learning.MAX_SEEDS
    # a different brief gets nothing
    assert learning.seed_queries(conn, "payouts") == []


def test_stronger_seeds_rank_first(tmp_path):
    conn = fresh_db(tmp_path)
    learning.record_gap(conn, "t", "weak")
    learning.record_reinforcement(conn, "t", "strong")
    learning.record_reinforcement(conn, "t", "strong")
    assert learning.seed_queries(conn, "t", limit=2) == ["strong", "weak"]


def test_topics_are_normalized(tmp_path):
    conn = fresh_db(tmp_path)
    learning.record_gap(conn, "  Fees   and payouts ", "q")
    assert learning.seed_queries(conn, "fees and payouts") == ["q"]


def test_merge_seeds_dedupes_and_is_pure():
    planned = ['"shopify"', "shopify checkout fees"]
    merged = learning.merge_seeds(
        planned, ["Shopify Checkout Fees", "shopify refunds", "  ", "s1", "s2"]
    )
    # case-insensitive dedupe against the plan, blanks dropped, capped
    assert merged == planned + ["shopify refunds", "s1"][: learning.MAX_SEEDS]
    assert planned == ['"shopify"', "shopify checkout fees"]  # untouched


def test_broken_store_degrades_to_no_seeds():
    conn = sqlite3.connect(":memory:")
    conn.close()  # every use now raises sqlite3.ProgrammingError
    assert learning.seed_queries(conn, "fees") == []


def test_feedback_splits_hits_from_gaps(tmp_path):
    conn = fresh_db(tmp_path)
    learning.feedback(
        conn, "t",
        ["hitquery", "missquery"],
        {"hitquery": 4, "missquery": 0},
    )
    rows = dict(
        conn.execute("SELECT query, strength FROM learning").fetchall()
    )
    assert rows["hitquery"] >= 2   # born reinforced
    assert rows["missquery"] == 1  # remembered as a gap


def test_disabled_flag_skips_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(learning, "enabled", False)
    conn = fresh_db(tmp_path)
    learning.feedback(conn, "t", ["q"], {"q": 0})
    assert learning.stats(conn)["queries_tracked"] == 0


def test_learning_endpoint_exposes_stats(client):
    r = client.get("/learning")
    assert r.status_code == 200
    body = r.json()
    assert {"topics", "queries_tracked", "top"} <= set(body)
    assert isinstance(body["top"], list)
