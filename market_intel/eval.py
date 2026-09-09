"""Evaluation metrics (Phase 6 preview): score retrieval and answer quality.

Pure functions — no DB, no LLM, no embeddings — so they are trivially
testable and can run in CI on synthetic data. run_eval.py wires them to
the real index and the labeled question set in data/eval_questions.json.

Two layers are scored:

  retrieval    — does the vector index surface the posts a human
                 labeled as relevant?  precision@k, recall@k, MRR, nDCG@k
  answer       — do the citations in an LLM answer point at relevant
                 posts?  citation precision (of the posts the answer
                 cites, how many are relevant) and citation recall
                 (of the relevant posts, how many got cited)

Conventions: retrieved/cited are ordered id lists; relevant is an
id collection. Any metric on an empty or irrelevant result set is 0.0 —
a conservative default that keeps aggregates honest.
"""

import math


def _dedupe(ids: list) -> list:
    seen: set = set()
    out = []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


def precision_at_k(retrieved: list, relevant, k: int | None = None) -> float:
    """Fraction of the top-k retrieved items that are relevant."""
    top = _dedupe(retrieved)[:k]
    if not top:
        return 0.0
    return sum(1 for r in top if r in relevant) / len(top)


def recall_at_k(retrieved: list, relevant, k: int | None = None) -> float:
    """Fraction of all relevant items that appear in the top-k."""
    if not relevant:
        return 0.0
    top = set(_dedupe(retrieved)[:k])
    return sum(1 for r in relevant if r in top) / len(relevant)


def mrr(retrieved: list, relevant) -> float:
    """Reciprocal rank of the first relevant hit (0 if none)."""
    for rank, r in enumerate(_dedupe(retrieved), 1):
        if r in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list, relevant, k: int | None = None) -> float:
    """nDCG@k with binary relevance over the top-k retrieved list.

    DCG is the sum of gain / log2(rank + 1); IDCG is DCG of the ideal
    ordering (all relevant retrieved items first). Equal to 1.0 when the
    ranking is perfect.
    """
    if not relevant:
        return 0.0
    top = _dedupe(retrieved)[:k]
    if not top:
        return 0.0

    def dcg(order: list) -> float:
        return sum(
            (1.0 if item in relevant else 0.0) / math.log2(rank + 1)
            for rank, item in enumerate(order, 1)
        )

    ideal = sorted(top, key=lambda item: item not in relevant)
    ideal_dcg = dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return dcg(top) / ideal_dcg


def citation_precision(cited: list, relevant) -> float:
    """Of the distinct posts an answer cites, how many are relevant?"""
    distinct = _dedupe(cited)
    if not distinct:
        return 0.0
    return sum(1 for c in distinct if c in relevant) / len(distinct)


def citation_recall(cited: list, relevant) -> float:
    """Of the labeled-relevant posts, how many did the answer cite?"""
    if not relevant:
        return 0.0
    distinct = set(_dedupe(cited))
    return sum(1 for r in relevant if r in distinct) / len(relevant)