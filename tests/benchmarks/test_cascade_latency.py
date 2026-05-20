"""Cascade-invalidation BFS latency benchmark.

Box 1 of issue #391 PERFORMANCE slice: benchmark
``src.memory.lifecycle.invalidator.cascade_invalidate`` against a synthetic
5-hop provenance DAG (~1000 nodes / ~5000 edges).

**Target**: < 500ms for 5-hop BFS (issue #391, box 1 verbatim).

**Tolerance band** (codified in
``tests/benchmarks/baselines/performance.json``; re-stated here so the
test docstring carries the same numbers as the JSON, per codex iter-0 L1):
- 10% relative OR 100ms absolute (whichever wider).
Relative-vs-absolute policy is `max(rel, abs)` — see ``_within_tolerance``
for the rationale; absolute band protects against noise on near-zero
baselines, relative band catches real drift at large baselines.

**Note (current shape)**: The shipped invalidator is BINARY (per plan
§"DECISIONS ADOPTED" 2026-05-19, Decision 3 = KEEP BINARY), not graded;
this benchmark exercises the binary cascade shape — every reachable row
gets ``invalidated_at`` set in one BFS pass.

**Baseline strategy (placeholder-first-run-blesses, per PR #379 +
[[feat-377-phase2-benchmark-close-20260519]])**: synthetic graph topology
may not match the real production DAG, so the FIRST run on a given
environment BLESSES the measured value as the baseline (re-write
``tests/benchmarks/baselines/performance.json`` in that PR). Subsequent
runs compare against the blessed value within the documented tolerance
band (10% relative OR 100ms absolute, whichever wider).

**Why we don't measure against the < 500ms absolute target**: the synthetic
graph here is a microbenchmark of the BFS topology — it uses an in-memory
``_FakeSupabaseGraph`` rather than real Postgres so we can run in CI
without Supabase. Real-Supabase latency is dominated by row-update round-
trips, which this harness does NOT exercise. We compare against the
blessed-baseline + tolerance band; the absolute target is a documentation
anchor for re-baselining once a real-Postgres benchmark surface lands.

Marked ``@pytest.mark.benchmark`` so it does NOT run in the default unit-
test sweep — it runs only when invoked explicitly (locally) or via the
``.github/workflows/benchmarks.yml`` workflow (CI).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pytest

pytestmark = pytest.mark.benchmark

_HERE = Path(__file__).resolve().parent
_GRAPH_FILE = _HERE / "data" / "synthetic_graph.jsonl"
_BASELINE_FILE = _HERE / "baselines" / "performance.json"


# ---------------------------------------------------------------------------
# Synthetic graph loader
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _SyntheticEdge:
    source_type: str
    source_id: str
    target_type: str
    target_id: str
    brand: str


def _load_synthetic_graph(path: Path) -> List[_SyntheticEdge]:
    if not path.exists():
        raise FileNotFoundError(
            f"synthetic graph file not found: {path}; re-run "
            "`python scripts/benchmarks/gen_synthetic_graph.py`"
        )
    edges: List[_SyntheticEdge] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            row = json.loads(stripped)
            edges.append(
                _SyntheticEdge(
                    source_type=row["source_type"],
                    source_id=row["source_id"],
                    target_type=row["target_type"],
                    target_id=row["target_id"],
                    brand=row["brand"],
                )
            )
    return edges


# ---------------------------------------------------------------------------
# In-memory Supabase fake — matches the surface `cascade_invalidate` uses
# ---------------------------------------------------------------------------


class _FakeQueryResult:
    def __init__(self, data: List[Dict[str, Any]]) -> None:
        self.data = data


class _FakeQuery:
    """A `.select().eq().eq().execute()` chain matching the production
    PostgREST-shaped query used in ``invalidator.cascade_invalidate``.

    The benchmark only exercises ``select`` on ``insight_edges`` (the BFS
    expansion query) and ``update`` on the invalidatable tables. We honor
    both shapes; ``update`` is a no-op (we don't care about the latency of
    writing in-memory because that is the part this harness deliberately
    skips per the docstring).
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        op: str = "select",
        update_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._rows = rows
        self._filters: List[Tuple[str, Any]] = []
        self._is_null_filters: List[str] = []
        self._op = op
        self._update_payload = update_payload

    def select(self, _cols: str) -> "_FakeQuery":
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters.append((col, val))
        return self

    def is_(self, col: str, val: str) -> "_FakeQuery":
        if val != "null":
            raise NotImplementedError(f"_FakeQuery.is_ only supports 'null' (got {val!r})")
        self._is_null_filters.append(col)
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        # Switch the chain into update mode; further .eq() calls narrow
        # the rows to update.
        self._op = "update"
        self._update_payload = payload
        return self

    def execute(self) -> _FakeQueryResult:
        # For both select and update, apply the .eq() filters to narrow
        # rows; this is enough to satisfy the BFS-expansion query.
        filtered = self._rows
        for col, val in self._filters:
            filtered = [r for r in filtered if r.get(col) == val]
        for col in self._is_null_filters:
            filtered = [r for r in filtered if r.get(col) is None]
        if self._op == "update":
            # No-op for the benchmark; we don't measure update latency
            # (synthetic in-memory write).
            return _FakeQueryResult([])
        return _FakeQueryResult(list(filtered))


class _FakeSupabaseGraph:
    """In-memory Supabase fake matching the surface used by
    ``invalidator.cascade_invalidate``.

    Carries the synthetic edges (``insight_edges`` rows) and per-target-
    table row registries (``triggers``, ``ml_predictions``,
    ``executive_insights``). The benchmark only exercises ``insight_edges``
    selects; the update path is a no-op so we measure pure BFS topology
    latency.
    """

    def __init__(self, edges: List[_SyntheticEdge]) -> None:
        self._edges: List[Dict[str, Any]] = [
            {
                "source_type": e.source_type,
                "source_id": e.source_id,
                "target_type": e.target_type,
                "target_id": e.target_id,
                "brand": e.brand,
            }
            for e in edges
        ]
        # Pre-build empty row tables for the invalidatable types so the
        # update path doesn't error out.
        self._tables: Dict[str, List[Dict[str, Any]]] = {
            "insight_edges": self._edges,
            "triggers": [],
            "ml_predictions": [],
            "executive_insights": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self._tables.get(name, []))


class _FakeRedisClient:
    """No-op redis fake for the publish_signal path."""

    async def publish(self, channel: str, payload: str) -> int:
        return 0


# ---------------------------------------------------------------------------
# Baseline + tolerance comparison
# ---------------------------------------------------------------------------


def _load_baseline() -> Dict[str, Any]:
    if not _BASELINE_FILE.exists():
        raise FileNotFoundError(
            f"performance baseline file missing: {_BASELINE_FILE}; "
            "seed it before running the harness"
        )
    with _BASELINE_FILE.open("r", encoding="utf-8") as fh:
        baseline: Dict[str, Any] = json.load(fh)
    return baseline


def _within_tolerance(
    observed_ms: float,
    baseline_ms: float,
    tolerance_pct: float,
    tolerance_abs_ms: float,
) -> Tuple[bool, str]:
    """Return (within_band, human_readable_reason).

    A measurement is "within tolerance" if its delta from baseline is
    either:
      * < tolerance_pct * baseline (relative band), OR
      * < tolerance_abs_ms (absolute band).
    The wider of the two wins (intentional — relative breaks down at
    near-zero baselines; absolute breaks down at large baselines).
    Improvements (observed < baseline) always pass — only regressions
    beyond the band fail.
    """
    if observed_ms <= baseline_ms:
        return True, f"improvement: observed={observed_ms:.2f}ms <= baseline={baseline_ms:.2f}ms"
    delta = observed_ms - baseline_ms
    rel_band = baseline_ms * (tolerance_pct / 100.0)
    abs_band = tolerance_abs_ms
    band = max(rel_band, abs_band)
    if delta <= band:
        return (
            True,
            f"within band: observed={observed_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
            f"delta={delta:.2f}ms, band={band:.2f}ms ({tolerance_pct}% rel OR {abs_band}ms abs)",
        )
    return (
        False,
        f"REGRESSION: observed={observed_ms:.2f}ms, baseline={baseline_ms:.2f}ms, "
        f"delta={delta:.2f}ms exceeds band={band:.2f}ms "
        f"({tolerance_pct}% rel OR {abs_band}ms abs)",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_synthetic_graph_loads() -> None:
    """Smoke test: shipped synthetic JSONL parses without error.

    Falsifiability anchor: this MUST always pass on the shipped JSONL; if
    it fails, the loader or the data file is broken and no other benchmark
    has a stable foundation.
    """
    edges = _load_synthetic_graph(_GRAPH_FILE)
    assert len(edges) >= 4000, f"Expected ~5000 edges per CURATION_PERF.md, got {len(edges)}"
    # First edge should be from the root.
    root_edges = [e for e in edges if e.source_id == "cp-root"]
    assert len(root_edges) >= 5, f"Expected >=5 root edges, got {len(root_edges)}"


def test_baseline_file_present_and_well_formed() -> None:
    """Performance baseline JSON must parse + carry required keys + tolerances."""
    baseline = _load_baseline()
    for key in (
        "cascade_5hop_bfs",
        "hybrid_retriever_search_p50",
        "hybrid_retriever_search_p95",
        "bm25_build_1k",
        "bm25_build_5k",
        "bm25_build_10k",
    ):
        assert key in baseline, f"baseline missing required key {key!r}"
        spec = baseline[key]
        assert "mean_ms" in spec, f"{key}: missing mean_ms"
        assert "tolerance_pct" in spec, f"{key}: missing tolerance_pct"
        assert "tolerance_abs_ms" in spec, f"{key}: missing tolerance_abs_ms"
        # bool excluded from numeric check (PR #374 codex-finding pattern).
        for fld in ("mean_ms", "tolerance_pct", "tolerance_abs_ms"):
            val = spec[fld]
            assert not isinstance(val, bool), f"{key}.{fld} must not be bool"
            assert isinstance(val, (int, float)), f"{key}.{fld} must be numeric"


def _run_cascade_once(edges: List[_SyntheticEdge]) -> float:
    """Run the real ``cascade_invalidate`` once against the in-memory fake,
    returning wall-clock ms.

    We override ``get_supabase_client`` and ``get_redis_client`` via
    monkey-patching at the invalidator module level so the production BFS
    code path runs unchanged.
    """
    # Lazy import. We `import` rather than `from … import` so the symbol
    # we monkey-patch (``get_supabase_client``) is rebound on the module
    # object itself — patching a bare ``from`` import would only rebind a
    # local reference and the production code would still call the
    # original factory.
    from src.memory.lifecycle import invalidator as _inv

    fake_db = _FakeSupabaseGraph(edges)
    fake_redis = _FakeRedisClient()

    # Save originals; restore after.
    orig_get_supabase = _inv.get_supabase_client
    orig_get_redis = _inv.get_redis_client
    _inv.get_supabase_client = lambda: fake_db  # type: ignore[assignment]
    _inv.get_redis_client = lambda: fake_redis  # type: ignore[assignment]

    try:
        loop = asyncio.new_event_loop()
        try:
            start = time.perf_counter()
            result = loop.run_until_complete(
                _inv.cascade_invalidate(
                    source_type="causal_path",
                    source_id="cp-root",
                    reason="benchmark-391-box-1",
                    scope_brand="bench",
                    publish_signal=False,  # skip the redis publish for pure-BFS measurement
                    max_depth=16,
                )
            )
            elapsed_ms = (time.perf_counter() - start) * 1000.0
        finally:
            loop.close()
    finally:
        _inv.get_supabase_client = orig_get_supabase  # type: ignore[assignment]
        _inv.get_redis_client = orig_get_redis  # type: ignore[assignment]

    # Sanity check: the shipped synthetic graph has exactly 1000 nodes
    # reachable from cp-root, distributed across 6 BFS layers [1, 10, 50,
    # 200, 500, 239] (depths 0-5). A correct cascade BFS must visit ALL
    # 1000 nodes. The codex iter-0 H1 finding showed that an earlier
    # generator allowed root-to-deep shortcuts which collapsed effective
    # BFS depth to 3 with the same `visited >= 100` guard passing — that
    # guard was too loose. We now assert the exact reachable-node count
    # so a regression that silently truncates ANY layer surfaces loudly
    # rather than producing a fast-but-wrong measurement.
    #
    # If you change the synthetic graph topology in
    # ``scripts/benchmarks/gen_synthetic_graph.py``, update this expected
    # value to match the new ``sum(_LAYER_SIZES)``.
    _EXPECTED_VISITED = 1000
    assert result.visited == _EXPECTED_VISITED, (
        f"BFS visited {result.visited} nodes — expected exactly "
        f"{_EXPECTED_VISITED} per the shipped synthetic 5-hop DAG. "
        "Either the generator drifted (regenerate via "
        "scripts/benchmarks/gen_synthetic_graph.py) or the BFS code "
        "path was truncated by a regression."
    )
    return elapsed_ms


@pytest.mark.timeout(120)
def test_cascade_5hop_bfs_latency_against_baseline() -> None:
    """Box 1 of issue #391: < 500ms target for 5-hop BFS.

    Measures wall-clock for a single ``cascade_invalidate`` run against
    the synthetic 5-hop DAG. We run 5 warm iterations to stabilize the
    measurement (cold-start JIT, dict resizing, etc. dominate the first
    run) and assert the median against the blessed baseline within the
    tolerance band.

    **Re-blessing the baseline**: if the measurement legitimately shifts
    (e.g., after a BFS refactor), update
    ``tests/benchmarks/baselines/performance.json`` in the same PR with
    the new ``mean_ms`` value. Do NOT loosen tolerances to mask a
    regression.
    """
    edges = _load_synthetic_graph(_GRAPH_FILE)
    baseline = _load_baseline()

    runs = 5
    timings: List[float] = []
    for _ in range(runs):
        timings.append(_run_cascade_once(edges))
    timings.sort()
    median_ms = timings[len(timings) // 2]
    p95_ms = timings[min(int(len(timings) * 0.95), len(timings) - 1)]

    spec = baseline["cascade_5hop_bfs"]
    baseline_ms = float(spec["mean_ms"])
    tol_pct = float(spec["tolerance_pct"])
    tol_abs = float(spec["tolerance_abs_ms"])
    target_ms = float(spec.get("_target_ms", 500.0))

    # Emit a terminal summary so CI logs show the numbers + the placeholder
    # re-bless reminder.
    print(
        f"\n[issue-#391 box-1] Cascade BFS latency:"
        f"\n  runs={runs}, timings_ms={[f'{t:.2f}' for t in timings]}"
        f"\n  median_ms={median_ms:.2f}, p95_ms={p95_ms:.2f}"
        f"\n  baseline_ms={baseline_ms:.2f}, tolerance={tol_pct}% rel OR {tol_abs}ms abs"
        f"\n  absolute target (issue #391 box 1) = < {target_ms}ms"
        + (
            "\n  NOTE: baseline is 0.0 — this run is the placeholder-blessing "
            "first run; re-write tests/benchmarks/baselines/performance.json "
            "with the median value in this PR."
            if baseline_ms == 0.0
            else ""
        ),
        file=sys.stderr,
        flush=True,
    )

    # Placeholder-first-run policy: when baseline is 0.0, we PASS the
    # benchmark unconditionally and emit a re-bless reminder. This is
    # consistent with `tests/benchmarks/baselines/retrieval_quality.json`
    # at the same baseline (per [[feat-377-phase2-benchmark-close-20260519]]).
    if baseline_ms == 0.0:
        return

    within, reason = _within_tolerance(median_ms, baseline_ms, tol_pct, tol_abs)
    assert within, reason
