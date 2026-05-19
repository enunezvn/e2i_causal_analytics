"""Retrieval-quality metric primitives: Recall@k, MRR, optional nDCG@k.

Formulas quoted from textbook sources (Manning IIR §8.1, Voorhees 1999,
Järvelin & Kekäläinen 2002). Per memory ``feedback_overclaiming_during_planning``
this module invents no thresholds — all comparisons are pure functions of
inputs.

Conventions:
- Ranks are 1-indexed (rank 1 = best retrieved item) to match the textbook
  definitions and the convention used in MTEB / BEIR.
- ``relevant`` is a ``set[str]`` of expected-relevant document IDs.
- ``top_k`` is a ``list[str]`` of retrieved document IDs in rank order.
- ``relevance_grades`` is a ``dict[str, int]`` of doc_id → integer grade
  (typical scale: 3=highly-relevant, 2=relevant, 1=marginally-relevant,
  absent=irrelevant). Used only by nDCG@k.

All functions are sync (the harness async-awaits the retriever before
computing metrics on the materialized list of ids).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Set


def _validate_k(k: int) -> None:
    """Guard against degenerate ``k``.

    ``bool`` is excluded explicitly because ``isinstance(True, int)`` is True
    in Python and would otherwise pass through this validator silently (per
    PR #374 codex-finding pattern on the ``max_staleness`` filter).
    """
    if isinstance(k, bool) or not isinstance(k, int):
        raise ValueError(f"k must be int, got {type(k).__name__}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")


def recall_at_k(
    top_k: Sequence[str],
    relevant: Set[str],
    k: int,
) -> float:
    """Compute Recall@k for a single query.

    Definition (Manning IIR §8.1, eq. 8.1):
        Recall@k = |relevant ∩ top_k| / |relevant|

    Args:
        top_k: Retrieved document IDs in rank order. May be longer than k;
            we truncate.
        relevant: Set of expected-relevant document IDs for this query.
        k: Cutoff rank (positive int).

    Returns:
        Recall@k as float in [0.0, 1.0].

        Contract for boundary inputs:
        - ``len(relevant) == 0`` → 0.0 (Recall is mathematically undefined;
          we return zero so an empty-relevant query is a no-op for the
          aggregator. The harness logs + flags these queries separately
          so they don't silently dilute the mean.)
        - ``len(top_k) == 0`` → 0.0.

    Raises:
        ValueError: if ``k <= 0`` (caller used a degenerate cutoff).
    """
    _validate_k(k)

    if not relevant:
        return 0.0

    top_k_truncated = set(top_k[:k])
    hits = top_k_truncated & relevant
    return len(hits) / len(relevant)


def reciprocal_rank(
    top_k: Sequence[str],
    relevant: Set[str],
) -> float:
    """Compute the reciprocal rank (1 / rank-of-first-relevant) for one query.

    Definition (Voorhees 1999, TREC-8 QA Track, eq. 1):
        RR(q) = 1 / rank_of_first_relevant(q)    (1-indexed)
        RR(q) = 0   if no relevant result appears

    Args:
        top_k: Retrieved document IDs in rank order.
        relevant: Set of expected-relevant document IDs.

    Returns:
        Reciprocal rank as float in [0.0, 1.0].
    """
    if not relevant:
        return 0.0
    for rank, doc_id in enumerate(top_k, start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(per_query_rrs: Iterable[float]) -> float:
    """Aggregate per-query reciprocal ranks into Mean Reciprocal Rank.

    Args:
        per_query_rrs: Iterable of reciprocal ranks (one per query).

    Returns:
        Mean of inputs as float in [0.0, 1.0].

    Raises:
        ValueError: if the iterable is empty (MRR is mathematically
            undefined over zero queries; raising is louder than silently
            returning 0.0 — caller should not call MRR on an empty
            evaluation set).
    """
    rrs: List[float] = list(per_query_rrs)
    if not rrs:
        raise ValueError("mean_reciprocal_rank requires at least one query")
    return sum(rrs) / len(rrs)


def ndcg_at_k(
    top_k: Sequence[str],
    relevance_grades: dict,
    k: int,
) -> float:
    """Compute normalised Discounted Cumulative Gain at k.

    Definition (Järvelin & Kekäläinen 2002, "Cumulated gain-based evaluation
    of IR techniques", §3, eq. 1+2; standard log2 discount):

        DCG@k = sum_{i=1..k} rel_i / log2(i + 1)
        iDCG@k = DCG@k of ideal (sorted-descending-by-relevance) ranking
        nDCG@k = DCG@k / iDCG@k

    Args:
        top_k: Retrieved document IDs in rank order.
        relevance_grades: Mapping of doc_id → integer relevance grade.
            Absent ids are treated as grade 0 (irrelevant).
        k: Cutoff rank (positive int).

    Returns:
        nDCG@k as float in [0.0, 1.0].

        Contract for boundary inputs:
        - Empty ``relevance_grades`` (no relevant docs declared) → 0.0.
          (iDCG would be 0 → division-by-zero; contract is 0.0.)
        - No relevant doc in top_k → 0.0.

    Raises:
        ValueError: if ``k <= 0``.
    """
    _validate_k(k)

    if not relevance_grades:
        return 0.0

    # DCG of the retrieved ranking (rank 1-indexed → log2(i + 1)).
    dcg = 0.0
    for rank, doc_id in enumerate(top_k[:k], start=1):
        grade = relevance_grades.get(doc_id, 0)
        if grade > 0:
            dcg += grade / math.log2(rank + 1)

    # iDCG: ideal ranking is grades sorted descending, truncated at k.
    ideal_grades = sorted(relevance_grades.values(), reverse=True)[:k]
    idcg = 0.0
    for rank, grade in enumerate(ideal_grades, start=1):
        if grade > 0:
            idcg += grade / math.log2(rank + 1)

    if idcg == 0.0:
        return 0.0
    return dcg / idcg
