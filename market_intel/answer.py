"""Evidence-backed Q&A over the vector index (Phase 3).

The RAG loop: retrieve the top-k chunks for a question, hand them to an
LLM as numbered evidence, and return a grounded answer with citations.

Anti-hallucination controls, in order of defense:
  1. the model sees ONLY the retrieved chunks, numbered [1]..[k]
  2. the system prompt forbids outside knowledge and demands inline [n]
     citations matching those numbers
  3. we post-validate every [n] the model emits against the sources we
     actually retrieved — so a citation can never point at something
     that was not in the evidence
"""

import html
import re

from market_intel.llm import chat
from market_intel.vector import search

CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = """\
You are a market intelligence analyst tracking the e-commerce platform
market (Shopify, WooCommerce, BigCommerce) through public discussions.

Answer the user's question using ONLY the numbered evidence provided.
Rules:
- Base every claim on the evidence. Do not use outside knowledge.
- Cite your sources inline as [n] right after each claim; n must match
  a number in the evidence list.
- If the evidence does not answer the question, say so plainly instead
  of guessing.
- Synthesize across sources; note where they disagree.
- Be concise, specific, and business-focused."""


def _clean(text: str) -> str:
    """Strip HTML left over from HN story bodies (<p>, &#x27;, ...)."""
    text = html.unescape(text)           # &#x27; -> '
    text = re.sub(r"<[^>]+>", " ", text)  # <p> etc -> space
    return re.sub(r"\s+", " ", text).strip()


def dedupe_by_post(results: list[dict]) -> list[dict]:
    """Collapse retrieved chunks to one per source post (keep best score).

    Search returns chunks; citations point at posts. Multiple chunks of
    the same discussion (title + several body windows) would otherwise
    crowd the context and produce duplicate citations like [1] and [4]
    for the same URL.
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for r in results:  # already ordered by descending score
        if r["post_id"] not in seen:
            seen.add(r["post_id"])
            deduped.append(r)
    return deduped


def _format_evidence(results: list[dict]) -> str:
    """Render retrieved chunks as a numbered evidence list for the prompt."""
    lines = []
    for i, r in enumerate(results, 1):
        excerpt = _clean(r["text"])
        if len(excerpt) > 600:
            excerpt = excerpt[:600] + "…"
        lines.append(f"[{i}] title: {r['title']}\n    url: {r['url']}\n    {excerpt}")
    return "\n\n".join(lines)


def extract_citations(text: str, n_sources: int) -> list[int]:
    """The [n] markers the model used, kept only if 1 <= n <= n_sources.

    Anything out of range is a hallucinated citation and is dropped.
    """
    return sorted(
        {int(m) for m in CITATION_RE.findall(text) if 1 <= int(m) <= n_sources}
    )


def answer_question(
    conn,
    question: str,
    top_k: int = 5,
    competitor: str | None = None,
    provider: str | None = None,
) -> dict:
    """Retrieve evidence, ask the LLM, return answer + sources + citations.

    Over-fetch 3x so that dropping duplicate chunks of the same post
    still leaves up to ``top_k`` distinct sources.
    """
    results = dedupe_by_post(
        search(conn, question, top_k=top_k * 3, competitor=competitor)
    )[:top_k]
    if not results:
        return {
            "question": question,
            "answer": "(no evidence found in the index — run run_index.py or fetch more posts)",
            "sources": [],
            "cited": [],
        }

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question: {question}\n\n"
                f"Evidence:\n{_format_evidence(results)}\n\n"
                "Answer the question, citing the evidence as [n]."
            ),
        },
    ]
    text = chat(messages, provider=provider)
    return {
        "question": question,
        "answer": text,
        "sources": results,
        "cited": extract_citations(text, len(results)),
    }