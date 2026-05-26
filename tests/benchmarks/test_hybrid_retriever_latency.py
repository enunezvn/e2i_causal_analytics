"""HybridRetriever fused-search latency benchmark.

Box 2 of issue #391 PERFORMANCE slice: benchmark
``src.rag.retriever.HybridRetriever.search`` (and the ``hybrid_search``
convenience function) against the existing issue #377 query set.

**Target**: p50 < 200ms for fused search (issue #391, box 2 verbatim).
A p95 tail-latency target of < 500ms is added by this harness (2.5x the
p50 target — standard tail-latency budget for a 3-stream fused search).

**Tolerance bands** (codified in
``tests/benchmarks/baselines/performance.json``; re-stated here so the
test docstring carries the same numbers as the JSON, per codex iter-0 L1):
- p50: 20% relative OR 50ms absolute (whichever wider).
- p95: 25% relative OR 100ms absolute (whichever wider).
The wider-of-the-two policy is `max(rel, abs)` — see
``_within_tolerance`` for the rationale; absolute bands protect against
noise on near-zero baselines, relative bands catch real drift at large
baselines.

**Companion**: this benchmark is the latency-shaped sibling of PR #379's
``test_retrieval_quality.py`` (Recall@10 + MRR). Both share
``tests/benchmarks/data/retrieval_queries.jsonl`` and the labeled-query
loader from ``tests/benchmarks/_loader.py``.

**Baseline strategy (CI-blessed-median against a local pgvector
substrate — issue #414)**: box 2 runs the live HybridRetriever against a
seeded local ``pgvector`` container — the REAL ``hybrid_vector_search`` /
``hybrid_fulltext_search`` SQL functions, reached through a test-side
``DirectSQLMemoryConnector`` with a deterministic embedder — so there are
NO Supabase / OpenAI secrets and NO network. This isolates our code, so
the baseline guards CODE-latency regressions (not third-party API
latency). The baseline is CI-blessed-median like the sibling cascade +
bm25 boxes: trigger ≥3 ``workflow_dispatch`` runs, take the
median-of-medians, and write ``tests/benchmarks/baselines/performance.json``
(see ``_ci_observation`` / ``_blessed_from_ci_runs`` there). Subsequent
runs compare against the blessed value within the documented tolerance
bands.

**Run modes / skip semantics**:
* Local pgvector substrate (CI + local): set ``BENCH_SUBSTRATE=local_pg``
  + ``BENCH_PG_DSN``. The ``_substrate_connector`` fixture injects the
  test-side connector; a fail-closed direct-connector preflight asserts
  every query returns rows on BOTH streams, so a broken/unseeded substrate
  fails loudly rather than blessing a fast-but-meaningless 0.0.
* Legacy live path (no substrate configured): runs only when
  ``OPENAI_API_KEY`` is in the ``sk-*`` shape — else skips, because the
  dense stream's embedding HTTP call would otherwise hang at TLS read
  ([[feedback-live-lm-skip-must-check-key-shape]]).
* Skips entirely when neither path is available.
* Never invokes the graph stream (the benchmark passes no
  ``entities``/``kpi_name``), so FalkorDB is not required.

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
from tests.benchmarks.substrate.fixture import (
    inject as _inject_substrate,
)
from tests.benchmarks.substrate.fixture import (
    make_connector as _make_substrate_connector,
)
from tests.benchmarks.substrate.fixture import (
    reset as _reset_substrate,
)
from tests.benchmarks.substrate.fixture import (
    substrate_ready as _substrate_ready,
)

pytestmark = pytest.mark.benchmark

_HERE = Path(__file__).resolve().parent
_QUERY_FILE = _HERE / "data" / "retrieval_queries.jsonl"
_BASELINE_FILE = _HERE / "baselines" / "performance.json"
_TOP_K = 10


def _retrieval_env_ready() -> bool:
    """True iff the LEGACY live (non-substrate) retriever has the env it needs.

    Requires BOTH a real ``sk-*`` OpenAI key (dense-stream embedding) AND
    Supabase creds (dense + sparse streams go through ``get_supabase_client``).

    The Supabase check is load-bearing (codex code-review HIGH, #414): this
    function replaced the old ``@pytest.mark.requires_supabase`` marker, which
    had to be removed because it would also skip the local-substrate path. But
    a ``sk-*`` key ALONE is not enough — without Supabase the production
    connector raises, ``HybridRetriever``'s ``except: return []`` swallows it,
    and the benchmark would bless fast-but-empty 0.0 measurements with no
    failure. Mirror the ``SERVICES_AVAILABLE['supabase']`` probe
    (``conftest.py``: ``SUPABASE_URL`` + ``SUPABASE_ANON_KEY``/
    ``SUPABASE_SERVICE_KEY``). Per [[feedback-live-lm-skip-must-check-key-shape]]
    we check the key SHAPE so CI placeholder values skip rather than 401'ing.
    """
    key = os.getenv("OPENAI_API_KEY", "")
    supabase_ready = bool(
        os.getenv("SUPABASE_URL")
        and (os.getenv("SUPABASE_ANON_KEY") or os.getenv("SUPABASE_SERVICE_KEY"))
    )
    return key.startswith("sk-") and supabase_ready


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


@pytest.fixture
def _substrate_connector():
    """Substrate path (#414): inject the local-pg connector, reset + close after.

    Yields ``None`` when the local substrate isn't configured (legacy live path),
    so the test falls back to its OPENAI/Supabase skip logic.
    """
    if not _substrate_ready():
        yield None
        return
    conn = _make_substrate_connector()
    _inject_substrate(conn)
    try:
        yield conn
    finally:
        _reset_substrate()
        conn.close()


@pytest.mark.timeout(600)
def test_hybrid_retriever_latency_against_baseline(_substrate_connector) -> None:
    """Box 2 of issue #391: < 200ms target for fused search.

    Measures p50 / p95 wall-clock per query across the 36-query labeled set
    (``retrieval_queries.jsonl``), then asserts each against its blessed
    baseline within the documented tolerance band.

    **Re-blessing the baseline**: update
    ``tests/benchmarks/baselines/performance.json`` in the same PR with
    the new p50/p95 values; do NOT loosen tolerances to mask a regression.

    **Skip semantics**: see the module docstring.
    """
    use_substrate = _substrate_connector is not None
    if not use_substrate and not _retrieval_env_ready():
        pytest.skip(
            "no benchmark substrate: set BENCH_SUBSTRATE=local_pg + BENCH_PG_DSN "
            "for the local pgvector path, or provide a live sk-* OPENAI_API_KEY "
            "(+ Supabase) for the legacy end-to-end path."
        )

    queries = load_queries(_QUERY_FILE)
    baseline = _load_baseline()

    # FAIL-CLOSED preflight (codex audit HIGH-1 + HIGH-2). HybridRetriever's
    # dense/sparse paths swallow exceptions (`except: return []`,
    # src/rag/retriever.py:88,146), so a broken/unseeded substrate would
    # otherwise surface as empty results, NOT an error. Call the connector
    # DIRECTLY here (bypassing that swallow): a DB/SQL/connection error RAISES,
    # and we require EVERY query to return rows on BOTH streams so a partially
    # seeded substrate also fails loudly instead of blessing a fast 0.0.
    if use_substrate:
        pre_loop = asyncio.new_event_loop()
        try:
            empties = []
            for q in queries:
                dense = pre_loop.run_until_complete(
                    _substrate_connector.vector_search_by_text(
                        q.query_text,
                        k=_TOP_K,
                        filters=q.filters or None,
                        max_staleness=q.max_staleness,
                    )
                )
                sparse = pre_loop.run_until_complete(
                    _substrate_connector.fulltext_search(
                        q.query_text,
                        k=_TOP_K,
                        filters=q.filters or None,
                        max_staleness=q.max_staleness,
                    )
                )
                if not dense or not sparse:
                    empties.append((q.query_text[:40], len(dense), len(sparse)))
        finally:
            pre_loop.close()
        assert not empties, (
            "FAIL-CLOSED: substrate returned an empty stream for "
            f"{len(empties)}/{len(queries)} queries (query, dense_n, sparse_n): "
            f"{empties[:5]}. Broken or unseeded substrate — refusing to bless "
            "(issue #403 mode)."
        )

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

    # Persist measurements to test-results/measurements-*.json so a
    # CI-artifact-driven re-bless flow (issue #403 / follow-up GH #414)
    # can extract the raw numbers when this test eventually runs end-to-
    # end (it currently skips in CI without SUPABASE_URL + SUPABASE_KEY
    # + OPENAI_API_KEY).
    #
    # Codex iter-2 M2 closure: emit TWO records with distinct
    # `statistic` + `value_ms` so the p95 box's primary scalar IS p95
    # (not p50). Both records share the same raw `runs[]` (per-query
    # timings) because the box-split is over WHICH percentile is the
    # primary scalar for the baseline, not over WHICH queries.
    from tests.benchmarks._measurements_writer import write_measurements

    write_measurements(
        box="hybrid_retriever_search_p50",
        test="test_hybrid_retriever_latency_against_baseline",
        runs=timings_ms,
        median_ms=p50_ms,
        p95_ms=p95_ms,
        statistic="p50",
        value_ms=p50_ms,
    )
    write_measurements(
        box="hybrid_retriever_search_p95",
        test="test_hybrid_retriever_latency_against_baseline",
        runs=timings_ms,
        median_ms=p50_ms,
        p95_ms=p95_ms,
        statistic="p95",
        value_ms=p95_ms,
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
