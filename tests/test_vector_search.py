"""Regression tests: sqlite-vec search must reproduce the numpy cosine ranking.

The vector index used to be a numpy scan — L2-normalize every chunk
embedding, dot with the query, sort descending. The sqlite-vec migration
(commit 7230430) replaced that backend behind the same ``search``
interface. This suite pins the new backend to the old behavior by
comparing every result against a reference implementation of the numpy
algorithm, so a future change to the index (different metric, wrong k,
a broken competitor filter, a reverted distance sign) fails loudly.

Reference data is synthetic and self-contained: a handful of posts with
deliberately distinct vocabularies, embedded with the same local bge
model the pipeline uses. The first run downloads the model (~130 MB,
cached afterwards) — the same one-time cost as ``python run_index.py``.

Run with:
    pip install -r requirements-dev.txt
    python -m pytest tests/ -v
"""

import sqlite3

import numpy as np
import pytest

from market_intel import store, vector
from market_intel.embed import get_embedder

# vec0 reports cosine distance and L2-normalizes internally; its scores
# agree with numpy's normalized dot product to ~1e-7 (measured over 50
# random 384-dim vectors). 1e-5 gives ~60x headroom while still failing
# on any genuine regression (wrong metric, missing normalization, ...).
SCORE_ATOL = 1e-5

# Synthetic corpus: two posts per competitor, each with a distinct topic
# so the semantic ranking is meaningful (checkout fees vs migration vs
# plugins vs enterprise pricing ...). Fields mirror store.insert_posts.
POSTS = [
    # --- shopify ---
    {
        "id": "shopify_1", "competitor": "shopify", "source": "hackernews",
        "community": "hackernews", "title": "Shopify raises checkout fees again",
        "body": ("Shopify is changing its checkout pricing model again. "
                 "Merchants complain that the new checkout fees raise costs for "
                 "small stores, and payment processing fees are eating into "
                 "already thin margins. Several sellers said they will look at "
                 "alternatives if fees keep climbing."),
        "url": "https://news.ycombinator.com/item?id=1", "author": "alice",
        "created_utc": 1700000000, "score": 120, "num_comments": 45,
    },
    {
        "id": "shopify_2", "competitor": "shopify", "source": "hackernews",
        "community": "hackernews", "title": "Shopify cuts app developer payouts",
        "body": ("Shopify told app developers to expect a smaller revenue share "
                 "this quarter. Developers are unhappy with the new terms and "
                 "some are considering building for other platforms entirely. "
                 "The app store has been a major source of income for many "
                 "small studios."),
        "url": "https://news.ycombinator.com/item?id=2", "author": "bob",
        "created_utc": 1700000100, "score": 90, "num_comments": 30,
    },
    # --- woocommerce ---
    {
        "id": "woo_1", "competitor": "woocommerce", "source": "hackernews",
        "community": "hackernews", "title": "Migrating from Shopify to WooCommerce",
        "body": ("Migrating from Shopify to WooCommerce was easier than expected. "
                 "We moved 12000 products with a migration plugin, and our "
                 "monthly costs dropped dramatically because WooCommerce is open "
                 "source and we host it ourselves."),
        "url": "https://news.ycombinator.com/item?id=3", "author": "carol",
        "created_utc": 1700000200, "score": 200, "num_comments": 80,
    },
    {
        "id": "woo_2", "competitor": "woocommerce", "source": "hackernews",
        "community": "hackernews", "title": "Why we chose WooCommerce for plugins",
        "body": ("The WooCommerce plugin ecosystem is the main reason we chose "
                 "it over BigCommerce. Every feature we need already exists as "
                 "a plugin, from subscriptions to shipping labels to custom "
                 "checkout flows."),
        "url": "https://news.ycombinator.com/item?id=4", "author": "dave",
        "created_utc": 1700000300, "score": 75, "num_comments": 20,
    },
    # --- bigcommerce ---
    {
        "id": "bc_1", "competitor": "bigcommerce", "source": "hackernews",
        "community": "hackernews", "title": "BigCommerce enterprise pricing tiers",
        "body": ("BigCommerce positions itself at enterprise sellers with "
                 "high-volume pricing tiers. The API is solid and the uptime "
                 "has been great, but the monthly cost is higher than "
                 "WooCommerce for a small store."),
        "url": "https://news.ycombinator.com/item?id=5", "author": "erin",
        "created_utc": 1700000400, "score": 60, "num_comments": 15,
    },
    {
        "id": "bc_2", "competitor": "bigcommerce", "source": "hackernews",
        "community": "hackernews", "title": "Evaluating BigCommerce for our store",
        "body": ("We evaluated BigCommerce for our store. The built-in features "
                 "are strong out of the box, but customizing them means working "
                 "with their own theming system, which felt limiting compared "
                 "to open source options."),
        "url": "https://news.ycombinator.com/item?id=6", "author": "frank",
        "created_utc": 1700000500, "score": 40, "num_comments": 10,
    },
]


def _assert_matches_reference(
    got: list[dict], ref_ranking: list[dict], top_k: int
) -> None:
    """Assert vec0's top-k reproduces the reference's top-k up to ties.

    ``ref_ranking`` is the FULL reference ranking (every chunk, desc).
    The contract has three parts:

    1. Scores: every reported score matches the reference within
       SCORE_ATOL.
    2. Order: the result never contradicts the reference ranking beyond
       float noise (monotonic within tolerance).
    3. Cutoff: the result is exactly the top tier. Near-ties at the
       boundary — chunks whose scores differ by less than SCORE_ATOL,
       e.g. the same article reposted under different story IDs — may
       legitimately break either way, so the *set* of returned chunks is
       allowed to swap only within a tied tier: nothing below the cutoff
       by more than tolerance got in, nothing above it was left out.
    """
    n = len(got)
    assert n == min(top_k, len(ref_ranking))
    if n == 0:
        return

    ref_by_id = {r["chunk_id"]: r["score"] for r in ref_ranking}
    got_ids = [r["chunk_id"] for r in got]
    got_scores = np.asarray([r["score"] for r in got], dtype=np.float64)
    # KeyError here means vec0 returned a chunk outside the reference
    # subset (e.g. a different competitor) — that is a real discrepancy.
    ref_scores = np.asarray(
        [ref_by_id[cid] for cid in got_ids], dtype=np.float64
    )

    assert np.allclose(got_scores, ref_scores, atol=SCORE_ATOL)
    assert np.all(np.diff(got_scores) <= SCORE_ATOL)

    subset_scores = np.asarray(
        [r["score"] for r in ref_ranking], dtype=np.float64
    )
    cutoff = float(subset_scores[n - 1])  # n-th largest = lowest tier kept
    assert float(np.min(ref_scores)) >= cutoff - SCORE_ATOL
    excluded = [
        r["score"] for r in ref_ranking if r["chunk_id"] not in got_ids
    ]
    if excluded:
        assert float(np.max(excluded)) <= cutoff + SCORE_ATOL


def numpy_cosine_ranking(
    conn: sqlite3.Connection,
    query_vec: np.ndarray,
    competitor: str | None = None,
) -> list[dict]:
    """Reference ranking — the pre-sqlite-vec algorithm, exactly.

    L2-normalize every stored chunk embedding, dot with the normalized
    query (cosine similarity), sort descending, filter by competitor if
    requested. Returns the FULL ranking (not truncated to top_k), so the
    caller can check both the top tier and the cutoff boundary. This is
    the oracle the vec0 backend must reproduce.

    NOTE: the competitor filter is applied *after* scoring every chunk
    in the corpus. That is deliberately NOT "top-k globally, then
    filter" — the latter returns nothing when a competitor's nearest
    chunks all belong to other brands, which was the real bug this
    suite guards against.
    """
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, embedding, competitor FROM chunks"
    ).fetchall()
    if not rows:
        return []

    matrix = np.vstack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    q = query_vec.astype(np.float32)
    q /= np.linalg.norm(q)
    scores = matrix @ q  # (n_chunks,) float32

    order = np.argsort(-scores)
    return [
        {"chunk_id": rows[i]["id"], "score": float(scores[i])}
        for i in order
        if competitor is None or rows[i]["competitor"] == competitor
    ]


@pytest.fixture()
def db() -> sqlite3.Connection:
    """Fresh in-memory DB: schema + synthetic posts + built index."""
    conn = store.connect(":memory:")
    store.insert_posts(conn, POSTS)
    vector.build_index(conn, verbose=False)
    yield conn
    conn.close()


RANKING_CASES = [
    ("shopify checkout fees", 3),
    ("migrating away from shopify", 2),
    ("plugin marketplace for online stores", 3),
    ("enterprise pricing for large merchants", 2),
    # top_k larger than the corpus: both sides return every chunk, ranked.
    ("shopify checkout fees", 10),
]


@pytest.mark.parametrize(
    "query,top_k",
    RANKING_CASES,
    ids=[f"top{top_k}:{query}" for query, top_k in RANKING_CASES],
)
def test_vec0_matches_numpy_reference(db, query, top_k):
    embedder = get_embedder()
    q_vec = embedder.embed_query(query)

    ref = numpy_cosine_ranking(db, q_vec)
    got = vector.search(db, query, top_k=top_k)

    _assert_matches_reference(got, ref, top_k)
    # Citation contract: every hit joins back to its source post.
    assert got and all(r["title"] and r["url"] for r in got)


@pytest.mark.parametrize(
    "query,competitor,top_k",
    [
        # Regression: the query's globally-nearest chunks are all shopify,
        # so a post-filter with k = bigcommerce's chunk count would return
        # nothing. The fix (scan the whole corpus, then filter) must return
        # bigcommerce's best local match, exactly like numpy.
        ("shopify checkout fees", "bigcommerce", 2),
        ("shopify checkout fees", "bigcommerce", 5),
        ("plugin marketplace", "woocommerce", 1),
        ("migrating away from shopify", "shopify", 3),
    ],
    ids=[
        "bigcommerce-top2",
        "bigcommerce-top5",
        "woocommerce-top1",
        "shopify-top3",
    ],
)
def test_competitor_filter_matches_numpy(db, query, competitor, top_k):
    embedder = get_embedder()
    q_vec = embedder.embed_query(query)

    ref = numpy_cosine_ranking(db, q_vec, competitor=competitor)
    got = vector.search(db, query, top_k=top_k, competitor=competitor)

    _assert_matches_reference(got, ref, top_k)
    # The regression scenario: a competitor with no globally-near chunks
    # must still return its best local match (never silently empty).
    assert got


def test_tied_chunks_allow_either_order():
    """Byte-identical chunks (the same article reposted under different
    story IDs, as in the real corpus) tie exactly at float32 precision.
    vec0 may break the tie either way at the cutoff, and the reference
    contract must accept either outcome — this guards the tie-tolerance
    path itself, so a future over-tightening of the assertions fails.
    """
    dup_a = dict(POSTS[0], id="dup_a")  # identical text, different id
    dup_b = dict(POSTS[0], id="dup_b")
    conn = store.connect(":memory:")
    store.insert_posts(conn, [dup_a, dup_b, POSTS[4]])  # + a distractor
    vector.build_index(conn, verbose=False)

    embedder = get_embedder()
    ref = numpy_cosine_ranking(conn, embedder.embed_query("checkout fees"))
    got = vector.search(conn, "checkout fees", top_k=1)

    assert len(got) == 1
    assert got[0]["chunk_id"] in {"dup_a_0", "dup_b_0"}
    # The tie is exact: both duplicates share one score to float precision.
    assert abs(ref[0]["score"] - ref[1]["score"]) < 1e-6
    _assert_matches_reference(got, ref, top_k=1)
    conn.close()


def test_empty_corpus_returns_no_results():
    conn = store.connect(":memory:")
    assert vector.search(conn, "shopify checkout fees") == []
    conn.close()