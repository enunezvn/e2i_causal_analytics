"""Unit tests for retrieval-quality metric primitives (Recall@k, MRR, nDCG@k).

These tests are pure-Python: no external services, no async. They cover both
happy-path and boundary cases so a regression in the metric implementation
itself surfaces before the harness ever runs against a real retriever.

Falsifiability anchors per test are documented inline. Per memory
``feedback_overclaiming_during_planning``: the formulas below quote the
textbook definitions verbatim; no thresholds invented here.
"""

from __future__ import annotations

import math

import pytest

from tests.benchmarks._metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

# ============================================================================
# Recall@k
# ============================================================================
#
# Definition (Manning, Raghavan, Schütze, "Introduction to Information
# Retrieval" §8.1, equation 8.1):
#
#     Recall@k(q) = |relevant ∩ top_k(q)| / |relevant(q)|
#
# Edge cases:
#   - relevant=∅ → Recall is undefined; we return 0.0 and log a warning so
#     it neither boosts nor tanks the aggregate. Document explicitly so
#     the harness aggregator can treat these queries as ineligible.
#   - top_k longer than k → only the first k retrieved items count.
#   - duplicate ids in top_k → de-duplicated in the intersection.


class TestRecallAtK:
    """Recall@k coverage with happy + boundary cases."""

    def test_full_recall_3_of_3(self) -> None:
        """relevant={A,B,C}, top10 contains all three → Recall=1.0."""
        relevant = {"A", "B", "C"}
        top_k = ["A", "B", "X", "Y", "Z", "W", "V", "U", "T", "C"]
        assert recall_at_k(top_k, relevant, k=10) == 1.0

    def test_partial_recall_2_of_3(self) -> None:
        """relevant={A,B,C}, top10 misses C → Recall=2/3."""
        relevant = {"A", "B", "C"}
        top_k = ["A", "B", "X", "Y", "Z", "W", "V", "U", "T", "S"]
        assert recall_at_k(top_k, relevant, k=10) == pytest.approx(2.0 / 3.0)

    def test_zero_recall(self) -> None:
        """relevant={A,B,C}, top10 has none → Recall=0.0 (boundary)."""
        relevant = {"A", "B", "C"}
        top_k = ["X", "Y", "Z", "W", "V", "U", "T", "S", "R", "Q"]
        assert recall_at_k(top_k, relevant, k=10) == 0.0

    def test_k_truncates_top_list(self) -> None:
        """top list longer than k: only the first k items count."""
        relevant = {"A"}
        # A is at rank 11; with k=10, A is OUT of top_k.
        top_k_long = ["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8", "X9", "X10", "A"]
        assert recall_at_k(top_k_long, relevant, k=10) == 0.0
        # With k=11 it's in.
        assert recall_at_k(top_k_long, relevant, k=11) == 1.0

    def test_duplicate_top_ids_collapsed_in_intersection(self) -> None:
        """Same id appearing twice in top_k counts once toward intersection."""
        relevant = {"A", "B"}
        top_k = ["A", "A", "X", "Y", "Z", "W", "V", "U", "T", "S"]
        # Intersection {A,B} ∩ {A,A,X,...} = {A} → 1/2.
        assert recall_at_k(top_k, relevant, k=10) == 0.5

    def test_empty_relevant_returns_zero(self) -> None:
        """Undefined case (no relevant docs): contract returns 0.0."""
        assert recall_at_k(["A", "B"], set(), k=10) == 0.0

    def test_empty_top_k_returns_zero(self) -> None:
        """Empty top_k with non-empty relevant: Recall=0/N=0.0."""
        assert recall_at_k([], {"A", "B"}, k=10) == 0.0

    def test_k_must_be_positive(self) -> None:
        """k=0 is meaningless; k<0 is meaningless."""
        with pytest.raises(ValueError):
            recall_at_k(["A"], {"A"}, k=0)
        with pytest.raises(ValueError):
            recall_at_k(["A"], {"A"}, k=-1)


# ============================================================================
# Reciprocal Rank + MRR
# ============================================================================
#
# Definition (Voorhees 1999, TREC-8 QA Track, eq. 1):
#
#     RR(q) = 1 / rank_of_first_relevant_result(q)    (1-indexed)
#     RR(q) = 0    if no relevant result appears
#
#     MRR = mean(RR(q)) over q in Q
#
# Edge cases:
#   - First hit at rank 1 → 1.0 (boundary upper)
#   - No hit (rank=∞) → 0.0 (boundary lower)
#   - First hit at rank 3 → 1/3 (non-trivial)


class TestReciprocalRank:
    """Per-query reciprocal-rank coverage."""

    def test_rr_first_hit_at_rank_1(self) -> None:
        """Best case: first item is relevant → RR=1.0."""
        assert reciprocal_rank(["A", "B", "C"], {"A"}) == 1.0

    def test_rr_first_hit_at_rank_3(self) -> None:
        """First relevant at rank 3 → 1/3."""
        top_k = ["X", "Y", "A", "Z", "B"]
        # First match is A at rank 3 (1-indexed); B at rank 5 ignored
        # because we want rank of FIRST relevant.
        assert reciprocal_rank(top_k, {"A", "B"}) == pytest.approx(1.0 / 3.0)

    def test_rr_no_hit_returns_zero(self) -> None:
        """No relevant in top_k → RR=0.0 (boundary)."""
        assert reciprocal_rank(["X", "Y", "Z"], {"A"}) == 0.0

    def test_rr_empty_top_k_returns_zero(self) -> None:
        """Empty top_k → RR=0.0."""
        assert reciprocal_rank([], {"A"}) == 0.0

    def test_rr_empty_relevant_returns_zero(self) -> None:
        """Empty relevant set → RR=0.0 (undefined → contract zero)."""
        assert reciprocal_rank(["A", "B"], set()) == 0.0


class TestMeanReciprocalRank:
    """Aggregate MRR coverage."""

    def test_mrr_mean_of_known_values(self) -> None:
        """MRR over [1.0, 1/3, 0.0] = (1.0 + 1/3 + 0) / 3 = 4/9."""
        # Q1: hit at rank 1 → 1.0
        # Q2: hit at rank 3 → 1/3
        # Q3: no hit → 0.0
        per_query_rrs = [1.0, 1.0 / 3.0, 0.0]
        expected = (1.0 + 1.0 / 3.0 + 0.0) / 3.0
        assert mean_reciprocal_rank(per_query_rrs) == pytest.approx(expected)

    def test_mrr_all_zero_returns_zero(self) -> None:
        """Boundary: all queries miss → MRR=0.0."""
        assert mean_reciprocal_rank([0.0, 0.0, 0.0]) == 0.0

    def test_mrr_all_one_returns_one(self) -> None:
        """Boundary: every query hit at rank 1 → MRR=1.0."""
        assert mean_reciprocal_rank([1.0, 1.0, 1.0]) == 1.0

    def test_mrr_empty_list_raises(self) -> None:
        """Empty input is meaningless; raise rather than silently 0.0."""
        with pytest.raises(ValueError):
            mean_reciprocal_rank([])


# ============================================================================
# nDCG@k (optional — implemented if relevance grades present in query set)
# ============================================================================
#
# Definition (Järvelin & Kekäläinen 2002, "Cumulated gain-based evaluation
# of IR techniques", §3, eq. 1+2; standard log2 discount):
#
#     DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
#     iDCG@k = DCG@k of ideal (sorted-descending-by-relevance) ranking
#     nDCG@k = DCG@k / iDCG@k    (0.0 if iDCG=0)
#
# Coverage:
#   - Perfect ranking → nDCG=1.0 (boundary)
#   - Reversed worst-rank → nDCG < 1.0 (non-trivial)
#   - No relevant in top_k → nDCG=0.0 (boundary)
#   - iDCG=0 (no relevant grades) → nDCG=0.0 (contract)


class TestNDCGAtK:
    """nDCG@k coverage."""

    def test_ndcg_perfect_ranking_equals_one(self) -> None:
        """Sorted-descending-by-grade ranking → nDCG=1.0."""
        # top_k: ids ordered A,B,C; grades A=3, B=2, C=1
        # iDCG matches DCG ⇒ ratio 1.0
        top_k = ["A", "B", "C"]
        grades = {"A": 3, "B": 2, "C": 1}
        assert ndcg_at_k(top_k, grades, k=10) == pytest.approx(1.0)

    def test_ndcg_no_relevant_in_top_k_equals_zero(self) -> None:
        """top_k has none of the relevant ids → nDCG=0.0."""
        top_k = ["X", "Y", "Z"]
        grades = {"A": 3, "B": 2, "C": 1}
        assert ndcg_at_k(top_k, grades, k=10) == 0.0

    def test_ndcg_empty_grades_returns_zero(self) -> None:
        """No relevance grades → iDCG=0 → nDCG=0.0 by contract."""
        assert ndcg_at_k(["A", "B"], {}, k=10) == 0.0

    def test_ndcg_reversed_ranking_lt_perfect(self) -> None:
        """Worst-case ordering of relevant items: nDCG < 1.0 (non-trivial)."""
        # Same grades, but ranked C,B,A (worst → best). iDCG ranks A,B,C.
        top_k = ["C", "B", "A"]
        grades = {"A": 3, "B": 2, "C": 1}
        score = ndcg_at_k(top_k, grades, k=10)
        # Compute expected by hand for falsifiability.
        # DCG = 1/log2(2) + 2/log2(3) + 3/log2(4)
        #     = 1.0 + 2/1.5849... + 3/2 = 1.0 + 1.2618... + 1.5 = 3.7618...
        # iDCG = 3/log2(2) + 2/log2(3) + 1/log2(4) = 3.0 + 1.2618 + 0.5 = 4.7618
        dcg = 1.0 / math.log2(2) + 2.0 / math.log2(3) + 3.0 / math.log2(4)
        idcg = 3.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        expected = dcg / idcg
        assert score == pytest.approx(expected)
        # Sanity: must be strictly less than perfect.
        assert score < 1.0

    def test_ndcg_k_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            ndcg_at_k(["A"], {"A": 1}, k=0)
