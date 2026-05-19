"""End-to-end retrieval-quality benchmark harness.

Runs `HybridRetriever.search` (and the `hybrid_search` convenience) against
the labeled query-set in `tests/benchmarks/data/retrieval_queries.jsonl`,
computes aggregate Recall@10 + MRR, and compares against the baseline in
`tests/benchmarks/baselines/retrieval_quality.json` with the documented
tolerances.

Marked ``@pytest.mark.benchmark`` so it does NOT run in the default unit-test
sweep — it runs only when invoked explicitly (locally) or via the
`.github/workflows/retrieval-benchmarks.yml` workflow (CI).

Skip semantics:
- Skips with ``requires_supabase`` if the SERVICES_AVAILABLE['supabase']
  probe in the root conftest reports False. Without Supabase, dense + sparse
  retrieval cannot run; mocking the retriever would defeat the benchmark
  (per the "no mocks of the retriever" load-bearing pattern from issue #377).
- Skips with ``requires_falkordb`` is intentionally NOT applied — the
  graph stream degrades gracefully (returns []) when FalkorDB is absent.

Per CLAUDE.md + memory ``causal_role_propagation_phases_2_7_close_20260518``:
run with ``pytest -p no:xdist -o "addopts="`` because xdist can starve the
async retriever calls under load. Workflow YAML applies this flag.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests.benchmarks._loader import LabeledQuery, load_queries
from tests.benchmarks._metrics import (
    mean_reciprocal_rank,
    recall_at_k,
    reciprocal_rank,
)


def _retrieval_env_ready() -> bool:
    """True iff the live retriever has the env it needs to run end-to-end.

    The dense stream calls OpenAI for embeddings (see
    ``src/memory/services/factories.py`` line ~103); without an
    ``OPENAI_API_KEY`` the embedding HTTP call hangs at TLS read until
    pytest-timeout kills the test. The sparse stream hits Supabase
    PostgreSQL via ``hybrid_fulltext_search``; the conftest
    ``requires_supabase`` marker already handles that.

    Per memory ``feedback_live_lm_skip_must_check_key_shape``: check the
    key SHAPE rather than just presence so CI placeholder values like
    ``'test-key'`` skip rather than 401'ing against the live API.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    return key.startswith("sk-")


# Mark every test in this module ``benchmark`` so the default unit-test sweep
# can exclude with ``-m "not benchmark"``. ``requires_supabase`` is applied
# ONLY to the live-retrieval test, not to the pure-Python smoke tests
# (query-set parsing + baseline-file shape), so the latter still run
# usefully in environments without Supabase credentials.
pytestmark = pytest.mark.benchmark

_HERE = Path(__file__).resolve().parent
_QUERY_FILE = _HERE / "data" / "retrieval_queries.jsonl"
_BASELINE_FILE = _HERE / "baselines" / "retrieval_quality.json"
_TOP_K = 10


def _load_baseline() -> Dict[str, Any]:
    """Load the baseline JSON; raise loudly if missing or schema-invalid."""
    if not _BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"baseline file missing: {_BASELINE_FILE}; seed it before running the harness"
        )
    with _BASELINE_FILE.open("r", encoding="utf-8") as fh:
        baseline: Dict[str, Any] = json.load(fh)
    # Minimal schema-validation guard. Per memory
    # ``causal_role_propagation_phases_2_7_close_20260518``: bool must be
    # excluded from numeric guards.
    for required_key in ("recall_at_10", "mrr"):
        if required_key not in baseline:
            raise KeyError(f"baseline missing required key {required_key!r}")
    return baseline


async def _run_one_query(query: LabeledQuery, k: int) -> List[str]:
    """Invoke HybridRetriever for a single labeled query.

    Lazy import inside the function body (per load-bearing pattern from PR
    #374): forward-defensive circular-import avoidance + keeps the test
    module importable even when the rag subpackage's heavy transitive
    deps (psycopg, supabase, falkordb) aren't installable.
    """
    from src.rag.retriever import hybrid_search

    results = await hybrid_search(
        query=query.query_text,
        k=k,
        filters=query.filters or None,
        max_staleness=query.max_staleness,
    )
    return [r.source_id for r in results]


def _evaluate_all(queries: List[LabeledQuery], k: int) -> Dict[str, Any]:
    """Run every query, return aggregate metrics + per-query detail.

    Returns:
        Dict with keys:
            - 'recall_at_k_mean': float
            - 'mrr_mean': float
            - 'queries_evaluated': int
            - 'queries_with_relevant': int (queries where len(relevant)>0)
            - 'per_query': list of dicts (one per query)
    """
    per_query: List[Dict[str, Any]] = []
    recalls: List[float] = []
    rrs: List[float] = []
    queries_with_relevant = 0

    loop = asyncio.new_event_loop()
    try:
        for q in queries:
            top_ids = loop.run_until_complete(_run_one_query(q, k=k))
            relevant_set = set(q.relevant_doc_ids)
            recall = recall_at_k(top_ids, relevant_set, k=k)
            rr = reciprocal_rank(top_ids, relevant_set)
            per_query.append(
                {
                    "query_id": q.query_id,
                    "category": q.category,
                    "tier3_consumer": q.tier3_consumer,
                    "top_ids": top_ids[:k],
                    "recall_at_k": recall,
                    "reciprocal_rank": rr,
                    "n_relevant": len(relevant_set),
                }
            )
            if relevant_set:
                queries_with_relevant += 1
                recalls.append(recall)
                rrs.append(rr)
    finally:
        loop.close()

    return {
        "recall_at_k_mean": (sum(recalls) / len(recalls)) if recalls else 0.0,
        "mrr_mean": mean_reciprocal_rank(rrs) if rrs else 0.0,
        "queries_evaluated": len(queries),
        "queries_with_relevant": queries_with_relevant,
        "per_query": per_query,
    }


def test_query_set_loads() -> None:
    """Smoke test: the shipped query-set parses without error.

    Falsifiability anchor: this MUST always pass on the shipped JSONL; if
    it fails, the loader or the data file is broken and no other benchmark
    assertion has a stable foundation.
    """
    queries = load_queries(_QUERY_FILE)
    assert len(queries) >= 30, f"Issue #377 DoD requires >=30 queries, got {len(queries)}"


def test_baseline_file_present_and_well_formed() -> None:
    """Baseline JSON must parse + carry the required keys + tolerances.

    Falsifiability anchor: rename baseline file → this test fails. Drop
    a required key → this test fails.
    """
    baseline = _load_baseline()
    recall = baseline["recall_at_10"]
    mrr = baseline["mrr"]
    assert "mean" in recall and "tolerance_pp" in recall
    assert "mean" in mrr and "tolerance_abs" in mrr
    # Sanity: tolerances must be positive numbers (bool excluded).
    for spec, key in ((recall, "tolerance_pp"), (mrr, "tolerance_abs")):
        val = spec[key]
        assert not isinstance(val, bool), f"{key} must not be bool"
        assert isinstance(val, (int, float)), f"{key} must be numeric"
        assert val > 0, f"{key} must be > 0"


@pytest.mark.requires_supabase
@pytest.mark.timeout(600)
def test_retrieval_quality_against_baseline() -> None:
    """Live-retrieval benchmark — Recall@10 + MRR vs baseline ± tolerances.

    Runs the full query-set through ``hybrid_search`` (the convenience
    function from ``src/rag/retriever.py``), computes aggregates, and
    asserts:
      - aggregate Recall@10 has NOT regressed by more than
        ``baseline.recall_at_10.tolerance_pp`` percentage points
      - aggregate MRR has NOT regressed by more than
        ``baseline.mrr.tolerance_abs`` absolute

    Improvements (current > baseline) always pass; only regressions
    beyond tolerance fail. The PR body must re-bless the baseline when
    an intentional shift lands.

    Skip semantics:
      - ``requires_supabase`` (conftest): skip when SUPABASE_URL +
        SUPABASE_KEY are unset (dense + sparse can't reach the corpus).
      - explicit ``_retrieval_env_ready`` check below: skip when
        ``OPENAI_API_KEY`` is missing or not in the ``sk-*`` shape —
        without it, the dense-stream embedding call would hang at TLS
        read for the full pytest-timeout window.

    Per the load-bearing pattern from PR #374: the graph stream degrades
    gracefully (returns []) when FalkorDB is absent, so this test does
    NOT require FalkorDB.

    Per CLAUDE.md timeouts: pytest's per-test default is 30s but a 36-
    query benchmark with N round-trips to Supabase + OpenAI per query
    can easily exceed that; the local marker raises this test's ceiling
    to 600s while keeping the rest of the suite at the global 30s.
    """
    if not _retrieval_env_ready():
        pytest.skip(
            "OPENAI_API_KEY missing or not in sk-* shape; dense-stream "
            "embedding service would hang. Set a real key to run the "
            "live benchmark end-to-end."
        )
    queries = load_queries(_QUERY_FILE)
    baseline = _load_baseline()

    summary = _evaluate_all(queries, k=_TOP_K)

    # Emit a terminal-summary line so CI logs show the numbers even when
    # the test passes. Helpful for "blessing" a new baseline.
    print(
        f"\n[issue-#377] Retrieval benchmark summary:"
        f"\n  queries_evaluated={summary['queries_evaluated']}"
        f"\n  queries_with_relevant={summary['queries_with_relevant']}"
        f"\n  Recall@{_TOP_K} (mean): {summary['recall_at_k_mean']:.4f}"
        f"\n  MRR (mean):             {summary['mrr_mean']:.4f}"
        f"\n  Baseline Recall@{_TOP_K}: {baseline['recall_at_10']['mean']:.4f}"
        f"  (tolerance: {baseline['recall_at_10']['tolerance_pp']}pp)"
        f"\n  Baseline MRR:           {baseline['mrr']['mean']:.4f}"
        f"  (tolerance: {baseline['mrr']['tolerance_abs']} abs)",
        file=sys.stderr,
        flush=True,
    )

    # Regression gates. Per issue #377 §D: "Recall@10 must not regress by
    # more than -5pp; MRR no more than -0.05".
    recall_baseline = float(baseline["recall_at_10"]["mean"])
    recall_tolerance_pp = float(baseline["recall_at_10"]["tolerance_pp"])
    recall_observed = float(summary["recall_at_k_mean"])
    # Convert tolerance pp → fraction (5pp == 0.05 absolute on a [0,1] scale).
    recall_drop = recall_baseline - recall_observed
    assert recall_drop <= (recall_tolerance_pp / 100.0), (
        f"Recall@{_TOP_K} regressed by {recall_drop * 100.0:.2f}pp "
        f"(observed={recall_observed:.4f}, baseline={recall_baseline:.4f}, "
        f"tolerance={recall_tolerance_pp}pp). "
        f"If this drop is intentional, re-bless the baseline file in this PR."
    )

    mrr_baseline = float(baseline["mrr"]["mean"])
    mrr_tolerance_abs = float(baseline["mrr"]["tolerance_abs"])
    mrr_observed = float(summary["mrr_mean"])
    mrr_drop = mrr_baseline - mrr_observed
    assert mrr_drop <= mrr_tolerance_abs, (
        f"MRR regressed by {mrr_drop:.4f} "
        f"(observed={mrr_observed:.4f}, baseline={mrr_baseline:.4f}, "
        f"tolerance={mrr_tolerance_abs} abs). "
        f"If this drop is intentional, re-bless the baseline file in this PR."
    )
