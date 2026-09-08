"""Vector index over the posts, stored in SQLite via the sqlite-vec extension.

The chunk *metadata* (post_id, competitor, text) lives in the ``chunks``
table. The embeddings themselves are indexed by a ``vec0`` virtual table
(``vec_chunks``) from the sqlite-vec extension — a real vector index with
k-NN search, instead of the earlier numpy scan over the whole corpus.

vec0's cosine metric reports *distance* = 1 - cos(a, b) and L2-normalizes
both sides internally, so ``score = 1 - distance`` reproduces the exact
cosine-similarity scores the numpy implementation produced.

vec0 tables hold only (chunk_id, embedding) — arbitrary column filters
are not supported inside the MATCH. The ``--competitor`` filter is applied
by scanning k = all chunks of that competitor and then joining to the
chunks table, which returns the same top-k as the old in-memory ranking.
(If the corpus grows, move competitor into the vec0 table itself.)

If EMBED_MODEL ever changes dimension, drop the vec_chunks table once so
it gets recreated with the new shape:
    DROP TABLE vec_chunks;
"""

import sqlite3
from datetime import datetime, timezone

import numpy as np
import sqlite_vec

from market_intel.chunk import chunks_for_record
from market_intel.embed import Embedder, get_embedder

CHUNKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,      -- "<post_id>_<chunk_index>"
    post_id      TEXT NOT NULL REFERENCES posts(id) ON DELETE CASCADE,
    competitor   TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    text         TEXT NOT NULL,
    embedding    BLOB,                  -- float32 little-endian vector
    model        TEXT NOT NULL,         -- which embedding model produced it
    built_at     TEXT NOT NULL          -- ISO timestamp of the index build
);

CREATE INDEX IF NOT EXISTS idx_chunks_post        ON chunks (post_id);
CREATE INDEX IF NOT EXISTS idx_chunks_competitor  ON chunks (competitor);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(CHUNKS_SCHEMA)


def _load_vec(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension onto a connection (idempotent)."""
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    """Create the vec0 virtual table if missing. ``dim`` must match the model."""
    _load_vec(conn)
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0("
        f" chunk_id TEXT PRIMARY KEY, embedding float[{dim}] distance_metric=cosine)"
    )


def build_index(conn: sqlite3.Connection, verbose: bool = True) -> int:
    """Chunk + embed every post and (re)build the index.

    Strategy: wipe and rebuild from scratch. The corpus is small, so a
    full rebuild is cheap and guarantees the index always matches the
    posts table. An incremental mode (embed only new/changed posts) is
    a later optimization. Returns number of chunks stored.
    """
    ensure_schema(conn)
    embedder = get_embedder()
    ensure_vec_table(conn, embedder.dim)

    conn.row_factory = sqlite3.Row
    posts = conn.execute(
        "SELECT id, competitor, title, body FROM posts ORDER BY id"
    ).fetchall()
    if not posts:
        print("  no posts to index — run run_pipeline.py first")
        return 0

    all_chunks: list[dict] = []
    for row in posts:
        all_chunks.extend(chunks_for_record(dict(row)))

    matrix = embedder.embed_documents([c["text"] for c in all_chunks])
    model = embedder.model_name
    built_at = datetime.now(timezone.utc).isoformat()

    conn.execute("DELETE FROM chunks")
    conn.execute("DELETE FROM vec_chunks")
    rows = [
        (
            c["id"], c["post_id"], c["competitor"], c["chunk_index"],
            c["text"], matrix[i].tobytes(), model, built_at,
        )
        for i, c in enumerate(all_chunks)
    ]
    conn.executemany(
        "INSERT INTO chunks (id, post_id, competitor, chunk_index,"
        " text, embedding, model, built_at) VALUES (?,?,?,?,?,?,?,?)",
        rows,
    )
    conn.executemany(
        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?, ?)",
        [(c["id"], matrix[i].tobytes()) for i, c in enumerate(all_chunks)],
    )
    conn.commit()

    if verbose:
        print(f"  posts  : {len(posts)}")
        print(f"  chunks : {len(all_chunks)} (index rebuilt)")
        print(f"  model  : {model} ({matrix.shape[1]} dims)")
    return len(rows)


def search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 5,
    competitor: str | None = None,
) -> list[dict]:
    """Semantic search: k-NN over the vec0 index, ranked by cosine similarity.

    Returns the top_k chunks joined with their post (title, url) so
    results carry a citation back to the source discussion.
    """
    # A fresh DB (pipeline run, index never built) has no ``chunks``
    # table yet — treat it as an empty index rather than crash.
    ensure_schema(conn)
    embedder = get_embedder()
    ensure_vec_table(conn, embedder.dim)

    conn.row_factory = sqlite3.Row
    # vec0 cannot filter on non-vector columns, so the scan must cover
    # the whole corpus and the competitor filter is applied in SQL after
    # the MATCH. (Scanning k = one competitor's chunk count would not be
    # equivalent: its globally-nearest chunks might all belong to another
    # competitor, leaving nothing after the WHERE.)
    k = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    if not k:
        return []

    q = embedder.embed_query(query)
    sql = (
        "SELECT c.id AS chunk_id, c.post_id, c.competitor, c.chunk_index,"
        " c.text, p.title, p.url, p.created_utc, v.distance"
        " FROM vec_chunks v"
        " JOIN chunks c ON c.id = v.chunk_id"
        " JOIN posts p ON p.id = c.post_id"
        " WHERE v.embedding MATCH ? AND k = ?"
    )
    params: tuple = (q.tobytes(), k)
    if competitor:
        sql += " AND c.competitor = ?"
        params += (competitor,)
    rows = conn.execute(sql, params).fetchall()

    rows.sort(key=lambda r: r["distance"])
    rows = rows[:top_k]
    return [
        {
            "score": float(1 - r["distance"]),  # vec0 cosine distance -> similarity
            "chunk_id": r["chunk_id"],
            "post_id": r["post_id"],
            "competitor": r["competitor"],
            "chunk_index": r["chunk_index"],
            "text": r["text"],
            "title": r["title"],
            "url": r["url"],
            "created_utc": r["created_utc"],
        }
        for r in rows
    ]