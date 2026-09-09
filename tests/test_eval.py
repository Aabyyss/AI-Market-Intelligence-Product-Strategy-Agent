"""Unit tests for the evaluation metrics (no DB, no LLM).

Each expected value is hand-computed from the metric definitions, so a
refactor of eval.py cannot silently change what the eval report means.

Run with:
    python -m pytest tests/test_eval.py -v
"""

import math

from market_intel.eval import (
    citation_precision,
    citation_recall,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


# --- precision / recall ---

def test_precision_at_k_basic():
    assert precision_at_k(["a", "b", "c"], {"a"}, k=3) == 1 / 3
    assert precision_at_k(["a", "b", "c"], {"a"}, k=1) == 1.0
    assert precision_at_k(["b", "c"], {"a"}, k=2) == 0.0


def test_precision_dedupes_and_truncates():
    # Duplicate retrieval entries must not inflate precision.
    assert precision_at_k(["a", "a", "b"], {"a", "b"}, k=3) == 1.0
    assert precision_at_k(["a", "b"], {"a"}, k=1) == 1.0


def test_recall_at_k_basic():
    # 2 of the 3 relevant posts are in the top-2.
    assert recall_at_k(["a", "c"], {"a", "b", "c"}, k=2) == 2 / 3
    assert recall_at_k(["x", "y"], {"a", "b"}, k=2) == 0.0


def test_empty_result_sets_score_zero():
    assert precision_at_k([], {"a"}) == 0.0
    assert recall_at_k([], {"a"}) == 0.0
    assert precision_at_k(["a"], set()) == 0.0
    assert recall_at_k(["a"], set()) == 0.0
    assert mrr([], {"a"}) == 0.0


# --- MRR ---

def test_mrr_is_inverse_first_hit_rank():
    assert mrr(["a", "b", "c"], {"b"}) == 0.5
    assert mrr(["a", "b", "c"], {"c", "a"}) == 1.0  # first hit at rank 1
    assert mrr(["a", "b", "c"], {"z"}) == 0.0


def test_mrr_ignores_duplicates():
    assert mrr(["a", "a", "b"], {"b"}) == 0.5


# --- nDCG@k ---

def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, k=3) == 1.0


def test_ndcg_hand_computed():
    # Gains [1, 0, 1]: DCG = 1/log2(2) + 0 + 1/log2(4) = 1.5
    # Ideal [1, 1, 0]:  IDCG = 1 + 1/log2(3) = 1.63093...
    expected = 1.5 / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k(["a", "c", "b"], {"a", "b"}, k=3),
                        expected, rel_tol=1e-9)


def test_ndcg_zero_when_nothing_relevant():
    assert ndcg_at_k(["a", "b"], {"z"}, k=2) == 0.0
    assert ndcg_at_k([], {"a"}, k=2) == 0.0


def test_ndcg_respects_k():
    # With k=1 the top hit is irrelevant, so the score is 0 even though
    # a relevant post sits at rank 2.
    assert ndcg_at_k(["a", "b"], {"b"}, k=1) == 0.0
    # With k=2 the ranking [a, b] is not perfect: the relevant post sits
    # at rank 2, so DCG = 1/log2(3) against an ideal of 1.0.
    assert math.isclose(ndcg_at_k(["a", "b"], {"b"}, k=2), 1 / math.log2(3))


# --- answer citations ---

def test_citation_precision_and_recall():
    cited = ["a", "b"]          # the answer cites posts a and b
    relevant = {"a", "c"}       # only a is labeled relevant
    assert citation_precision(cited, relevant) == 0.5
    assert citation_recall(cited, relevant) == 0.5


def test_citation_metrics_edge_cases():
    assert citation_precision([], {"a"}) == 0.0
    assert citation_recall(["a"], set()) == 0.0
    assert citation_precision(["a", "a"], {"a"}) == 1.0  # deduped