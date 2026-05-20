"""HybridRetriever fused-search latency benchmark.

Box 2 of issue #391 PERFORMANCE slice: benchmark
``src.rag.retriever.HybridRetriever.search`` (and the ``hybrid_search``
convenience function) against the existing issue #377 query set.

**Target**: < 200ms for fused search (issue #391, box 2).

**Companion**: this benchmark is the latency-shaped sibling of PR #379's
``test_retrieval_quality.py`` (Recall@10 + MRR). Both share
``tests/benchmarks/data/retrieval_queries.jsonl`` and the labeled-query
loader from ``tests/benchmarks/_loader.py``.

**Baseline strategy (placeholder-first-run-blesses, per PR #379 +
[[feat-377-phase2-benchmark-close-20260519]])**: the first run on a given
environment BLESSES the measured p50/p95 as the baseline (re-write
``tests/benchmarks/baselines/performance.json`` in that PR). Subsequent
runs compare against the blessed value within the documented tolerance
bands.

**Skip semantics**:
* Skips with ``requires_supabase`` if the SERVICES_AVAILABLE['supabase']
  probe in the root conftest reports False — without Supabase, the dense
  + sparse retrieval streams cannot run.
* Skips when ``OPENAI_API_KEY`` is missing or not in the ``sk-*`` shape —
  the dense stream's embedding HTTP call would hang at TLS read until
  pytest-timeout kills the test (per
  [[feedback-live-lm-skip-must-check-key-shape]]).
* Does NOT require FalkorDB — the graph stream degrades gracefully to []
  when absent (per PR #374 load-bearing pattern).

**xdist disabled** (per
[[causal-role-propagation-phases-2-7-close-20260518]]): xdist can starve
async retriever calls; the workflow YAML applies ``-p no:xdist`` + clears
``addopts`` so the pyproject default doesn't sneak xdist back in.

Marked ``@pytest.mark.benchmark`` so it does NOT run in the default unit-
test sweep.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from tests.benchmarks._loader import LabeledQuery, load_queries

pytestmark = pytest.mark.benchmark

_HERE = Path(__file__).resolve().parent
_QUERY_FILE = _HERE / "data" / "retrieval_queries.jsonl"
_BASELINE_FILE = _HERE / "baselines" / "performance.json"
_TOP_K = 10


def _retrieval_env_ready() -> bool:
    """True iff the live retriever has the env it needs to run end-to-end.

    Mirrors ``tests/benchmarks/test_retrieval_quality.py::_retrieval_env_ready``.
    Per [[feedback-live-lm-skip-must-check-key-shape]]: check the key SHAPE
    rather than just presence so CI placeholder values (e.g. ``'test-key'``)
    skip rather than 401'ing against the live API.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    return key.startswith("sk-")


def _load_baseline() -> Dict[str, Any]:
    if not _BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"performance baseline file missing: {_BASELINE_FILE}; "
            "seed it before running the harness"
        )
    with _BASELINE_FILE.open("r", encoding="utf-8") as fh:
        baseline: Dict[str, Any] = json.load(fh)
    return baseline


async def _run_one_query_timed(query: LabeledQuery, k: int) -> float:
    """Invoke HybridRetriever for one labeled query, returning wall-clock ms.

    Lazy import (per PR #374 load-bearing pattern): keeps the module
    importable when src/rag's heavy transitive deps aren't installed.
    """
    from src.rag.retriever import hybrid_search

    start = time.perf_counter()
    await hybrid_search(
        query=query.query_text,
        k=k,
        filters=query.filters or None,
        max_staleness=query.max_staleness,
    )
    return (time.perf_counter() - start) * 1000.0


def _percentile(sorted_values: List[float], pct: float) -> float:
    """Return the ``pct`` percentile of a SORTED list.

    Uses nearest-rank (CLRS §9.3) — adequate for benchmark reporting and
    avoids a numpy dependency at this layer.
    """
    if not sorted_values:
        return 0.0
    if pct <= 0.0:
        return sorted_values[0]
    if pct >= 100.0:
        return sorted_values[-1]
    idx = int(round((pct / 100.0) * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _within_tolerance(
    observed_ms: float,
    baseline_ms: float,
    tolerance_pct: float,
    tolerance_abs_ms: float,
) -> Tuple[bool, str]:
    """Return (within_band, human_readable_reason). See test_cascade_latency
    for the rationale on max(rel, abs) banding."""
    if observed_ms <= baseline_ms:
        return True, f"improvement: observed={observed_ms:.2f}ms <= baseline={baseline_ms:.2f}ms"
    delta = observed_ms - baseline_ms
    band = max(baseline_ms * (tolerance_pct / 100.0), tolerance_abs_ms)
    if delta <= band:
        return (
            True,
            f"within band: observed={observed_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
            f"delta={delta:.2f}ms, band={band:.2f}ms ({tolerance_pct}% rel OR "
            f"{tolerance_abs_ms}ms abs)",
        )
    return (
        False,
        f"REGRESSION: observed={observed_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
        f"delta={delta:.2f}ms exceeds band={band:.2f}ms "
        f"({tolerance_pct}% rel OR {tolerance_abs_ms}ms abs)",
    )


@pytest.mark.requires_supabase
@pytest.mark.timeout(600)
def test_hybrid_retriever_latency_against_baseline() -> None:
    """Box 2 of issue #391: < 200ms target for fused search.

    Measures p50 / p95 wall-clock per query across the 36-query labeled set
    (``retrieval_queries.jsonl``), then asserts each against its blessed
    baseline within the documented tolerance band.

    **Re-blessing the baseline**: update
    ``tests/benchmarks/baselines/performance.json`` in the same PR with
    the new p50/p95 values; do NOT loosen tolerances to mask a regression.

    **Skip semantics**: see the module docstring.
    """
    if not _retrieval_env_ready():
        pytest.skip(
            "OPENAI_API_KEY missing or not in sk-* shape; dense-stream "
            "embedding service would hang. Set a real key to run the "
            "live benchmark end-to-end."
        )

    queries = load_queries(_QUERY_FILE)
    baseline = _load_baseline()

    timings_ms: List[float] = []
    loop = asyncio.new_event_loop()
    try:
        for q in queries:
            elapsed = loop.run_until_complete(_run_one_query_timed(q, k=_TOP_K))
            timings_ms.append(elapsed)
    finally:
        loop.close()

    timings_ms.sort()
    p50_ms = _percentile(timings_ms, 50.0)
    p95_ms = _percentile(timings_ms, 95.0)

    p50_spec = baseline["hybrid_retriever_search_p50"]
    p95_spec = baseline["hybrid_retriever_search_p95"]
    p50_baseline = float(p50_spec["mean_ms"])
    p95_baseline = float(p95_spec["mean_ms"])

    print(
        f"\n[issue-#391 box-2] HybridRetriever fused-search latency:"
        f"\n  queries_evaluated={len(timings_ms)}"
        f"\n  p50_ms={p50_ms:.2f}, p95_ms={p95_ms:.2f}"
        f"\n  baseline_p50_ms={p50_baseline:.2f} "
        f"(tol: {p50_spec['tolerance_pct']}% rel OR "
        f"{p50_spec['tolerance_abs_ms']}ms abs)"
        f"\n  baseline_p95_ms={p95_baseline:.2f} "
        f"(tol: {p95_spec['tolerance_pct']}% rel OR "
        f"{p95_spec['tolerance_abs_ms']}ms abs)"
        f"\n  absolute targets (issue #391 box 2): p50 < 200ms, p95 < 500ms"
        + (
            "\n  NOTE: baseline is 0.0 — this run is the placeholder-blessing "
            "first run; re-write tests/benchmarks/baselines/performance.json "
            "with the p50/p95 values in this PR."
            if p50_baseline == 0.0 or p95_baseline == 0.0
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )

    # Placeholder-first-run policy: when EITHER baseline is 0.0, we pass
    # unconditionally and emit a re-bless reminder. Mirrors PR #379.
    if p50_baseline == 0.0 or p95_baseline == 0.0:
        return

    within_p50, reason_p50 = _within_tolerance(
        p50_ms,
        p50_baseline,
        float(p50_spec["tolerance_pct"]),
        float(p50_spec["tolerance_abs_ms"]),
    )
    within_p95, reason_p95 = _within_tolerance(
        p95_ms,
        p95_baseline,
        float(p95_spec["tolerance_pct"]),
        float(p95_spec["tolerance_abs_ms"]),
    )
    assert within_p50, f"p50: {reason_p50}"
    assert within_p95, f"p95: {reason_p95}"
