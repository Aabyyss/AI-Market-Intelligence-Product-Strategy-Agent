"""AI Market Intelligence & Product Strategy Agent.

Phase 1: data pipeline — fetch, clean, store competitor discussions.
Phase 2: RAG foundation — chunk the posts, embed them locally, and
search them with cosine similarity over SQLite.
Phase 3: grounded Q&A — retrieve evidence, answer from it with an LLM,
and cite the source posts.
"""

__version__ = "0.3.0"