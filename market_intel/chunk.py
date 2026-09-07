"""Split post text into retrieval-ready chunks.

Why chunk at all? Embedding a whole post makes the vector generic and
can exceed model context windows. Chunking first means retrieval can
return the *specific passage* that answers a question, not just "some
post mentioned this".

Strategy (v1, deliberately simple):
  - sliding word window with overlap between neighbours, so no idea is
    lost at a chunk boundary
  - every chunk is prefixed with the post title, so even a chunk from
    deep inside a long body still carries document context

Refinements for later: sentence-boundary snapping (avoid cutting mid-
sentence), heading-aware chunking, and windows measured in tokens
rather than words.
"""

from market_intel import config


def chunk_words(words: list[str], max_words: int, overlap_words: int) -> list[str]:
    """Sliding word windows: window i spans words[i*stride : i*stride + max_words].

    ``overlap_words`` is the amount of text shared between neighbouring
    chunks. It costs a little extra storage/embedding time, but it
    prevents a chunk boundary from splitting an idea in half.
    """
    if not words:
        return []
    if len(words) <= max_words:
        return [" ".join(words)]

    stride = max(1, max_words - overlap_words)
    chunks: list[str] = []
    start = 0
    while start < len(words):
        chunks.append(" ".join(words[start : start + max_words]))
        start += stride
    return chunks


def chunks_for_record(record: dict) -> list[dict]:
    """Produce chunk dicts for one stored post record.

    Every chunk keeps the post id, competitor, and an index so search
    results can be traced straight back to the source post (that is
    the citation chain). A post with no body yields one chunk: its title.
    """
    post_id = record["id"]
    title = (record.get("title") or "").strip()
    body = (record.get("body") or "").strip()

    body_parts = chunk_words(
        body.split(), config.CHUNK_MAX_WORDS, config.CHUNK_OVERLAP_WORDS
    ) if body else []

    # Prefix the title to every chunk for document-level context.
    texts = [f"{title}. {part}" for part in body_parts]
    if not texts and title:
        texts = [title]

    return [
        {
            "id": f"{post_id}_{i}",
            "post_id": post_id,
            "competitor": record["competitor"],
            "chunk_index": i,
            "text": text,
        }
        for i, text in enumerate(texts)
    ]