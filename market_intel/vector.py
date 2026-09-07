"""Vector index over the posts, stored in SQLite.

The chunk embeddings live right beside the posts in the same SQLite
file — no separate vector database yet. Cosine similarity is computed
in numpy over the whole corpus, which is instant at this scale
(hundreds of chunks). When the corpus grows, the index can move to a
real vector DB (sqlite-vec / qdrant / chroma) behind the same
``build_index`` / ``search`` interface.

Cosine similarity: cos(a, b) = a·b / (|a|·|b|). After L2-normalizing
every vector, cosine similarity is just the dot product, which numpy
computes for the whole matrix in one shot.
"""

import sqlite3
from datetime import datetime, timezone

import numpy as np

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


def build_index(conn: sqlite3.Connection, verbose: bool = True) -> int:
    """Chunk + embed every post and (re)build the chunks table.

    Strategy: wipe and rebuild from scratch. The corpus is small, so a
    full rebuild is cheap and guarantees the index always matches the
    posts table. An incremental mode (embed only new/changed posts) is
    a later optimization. Returns number of chunks stored.
    """
    ensure_schema(conn)
    embedder = get_embedder()

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

    conn.execute("DELETE FROM chunks")  # full rebuild
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
    """Semantic search: embed the query, rank all chunks by cosine similarity.

    Returns the top_k chunks joined with their post (title, url) so
    results carry a citation back to the source discussion.
    """
    embedder = get_embedder()
    q = embedder.embed_query(query)
    q = q / np.linalg.norm(q)

    conn.row_factory = sqlite3.Row
    sql = (
        "SELECT c.id AS chunk_id, c.post_id, c.competitor, c.chunk_index,"
        " c.text, c.embedding, p.title, p.url, p.created_utc"
        " FROM chunks c JOIN posts p ON p.id = c.post_id"
    )
    params: tuple = ()
    if competitor:
        sql += " WHERE c.competitor = ?"
        params = (competitor,)
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    matrix = np.vstack(
        [np.frombuffer(r["embedding"], dtype=np.float32) for r in rows]
    )
    # L2-normalize rows -> cosine similarity becomes a dot product.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms == 0, 1, norms)
    scores = matrix @ q

    order = np.argsort(-scores)[:top_k]
    return [
        {
            "score": float(scores[i]),
            "chunk_id": rows[i]["chunk_id"],
            "post_id": rows[i]["post_id"],
            "competitor": rows[i]["competitor"],
            "chunk_index": rows[i]["chunk_index"],
            "text": rows[i]["text"],
            "title": rows[i]["title"],
            "url": rows[i]["url"],
            "created_utc": rows[i]["created_utc"],
        }
        for i in order
    ]