"""RED-first tests for box 2: MLflow metrics on cascade / consolidation /
crystallization.

The four metrics (per issue #391 monitoring slice):

* ``e2i.cascade.frequency`` — counter, emit on each cascade complete.
* ``e2i.cascade.propagation_depth`` — gauge / histogram, per-cascade
  depth. Brand-tagged.
* ``e2i.consolidation.promotion_rate`` — gauge, brand-tagged. Computed
  as ``promotions_to_semantic / max(candidates_evaluated, 1)`` so a
  zero-candidate sweep emits 0.0 (not NaN / div-by-zero).
* ``e2i.crystal.count_by_brand`` — counter tagged with brand. Emit on
  crystallization complete.

Boundary mocking
----------------
Tests patch ``src.mlops.lifecycle_monitoring._emit_mlflow_metric``
directly. The production code path (helpers in
``src.mlops.lifecycle_monitoring``) is exercised end-to-end including
brand-tag-coercion and value-clamping logic; only the sink is replaced.
This is the [[feedback-test-must-exercise-real-catch-not-mock]] pattern:
real catches still execute.
"""

from __future__ import annotations

import math

import pytest

from src.mlops import lifecycle_monitoring as lm


@pytest.fixture
def fake_mlflow(monkeypatch: pytest.MonkeyPatch):
    """Capture (metric_name, value, tags) tuples emitted via
    ``_emit_mlflow_metric``."""
    captured: list[tuple[str, float, dict]] = []

    def _emit(metric_name: str, value: float, tags: dict | None = None) -> None:
        captured.append((metric_name, float(value), dict(tags or {})))

    monkeypatch.setattr(lm, "_emit_mlflow_metric", _emit)
    return captured


# ---------------------------------------------------------------------
# Box 2.a — cascade frequency + propagation depth
# ---------------------------------------------------------------------


def test_record_cascade_complete_emits_frequency_and_depth(fake_mlflow) -> None:
    """One cascade complete event must emit BOTH the frequency counter
    (value=1) AND the propagation-depth metric (value=depth)."""
    lm.record_cascade_complete(
        brand="kisqali",
        depth=4,
        edges_visited=12,
        duration_ms=88.0,
        invalidated_by_type={"trigger": 3},
    )

    metric_names = {m[0] for m in fake_mlflow}
    assert "e2i.cascade.frequency" in metric_names
    assert "e2i.cascade.propagation_depth" in metric_names

    # Look up each metric.
    freq = next(m for m in fake_mlflow if m[0] == "e2i.cascade.frequency")
    depth = next(m for m in fake_mlflow if m[0] == "e2i.cascade.propagation_depth")

    # Frequency is a counter — emit 1 per event.
    assert freq[1] == 1.0
    # Depth is the actual cascade depth.
    assert depth[1] == pytest.approx(4.0)
    # Both must be brand-tagged.
    assert freq[2].get("brand") == "kisqali"
    assert depth[2].get("brand") == "kisqali"


# ---------------------------------------------------------------------
# Box 2.b — consolidation promotion rate
# ---------------------------------------------------------------------


def test_record_consolidation_sweep_emits_promotion_rate(fake_mlflow) -> None:
    """The promotion rate metric MUST equal
    ``promotions_to_semantic / max(causal_paths_examined, 1)``."""
    lm.record_consolidation_sweep(
        brand="remibrutinib",
        dedup_collapses=2,
        promotions_to_semantic=3,
        promotions_to_procedural=1,
        templates_extracted=2,
        duration_ms=200.0,
        causal_paths_examined=12,  # used to compute rate
    )

    rate = next(m for m in fake_mlflow if m[0] == "e2i.consolidation.promotion_rate")
    # 3 / 12 = 0.25
    assert rate[1] == pytest.approx(0.25)
    assert rate[2].get("brand") == "remibrutinib"


def test_promotion_rate_handles_zero_candidates_without_nan(fake_mlflow) -> None:
    """Zero-candidate sweep MUST emit promotion_rate=0.0, not NaN or raise.

    Codex audit premise (a): instrumentation must never crash production.
    A consolidator pass with no candidates is normal (clean state).
    """
    lm.record_consolidation_sweep(
        brand="kisqali",
        dedup_collapses=0,
        promotions_to_semantic=0,
        promotions_to_procedural=0,
        templates_extracted=0,
        duration_ms=10.0,
        causal_paths_examined=0,
    )

    rate = next(m for m in fake_mlflow if m[0] == "e2i.consolidation.promotion_rate")
    assert rate[1] == 0.0
    assert not math.isnan(rate[1])
    assert rate[2].get("brand") == "kisqali"


# ---------------------------------------------------------------------
# Box 2.c — crystal count by brand
# ---------------------------------------------------------------------


def test_record_provenance_write_emits_brand_tagged_crystal_counter(fake_mlflow) -> None:
    """Each provenance write MUST emit a crystal-count counter tagged
    with brand. Per-brand divergence is the load-bearing observable."""
    lm.record_provenance_write(
        insight_id="ins-1",
        source_count=4,
        brand="fabhalta",
        edges_added=8,
    )

    count = next(m for m in fake_mlflow if m[0] == "e2i.crystal.count_by_brand")
    # Counter increment is 1 per crystal written.
    assert count[1] == 1.0
    assert count[2].get("brand") == "fabhalta"


def test_record_provenance_write_per_brand_tags_are_distinct(fake_mlflow) -> None:
    """Multiple brands' crystal-count events MUST land with distinct
    brand tags so downstream dashboards can chart per-brand trajectories.

    Codex audit premise (e): brand-namespacing preserved across all
    per-brand metrics.
    """
    lm.record_provenance_write(insight_id="i1", source_count=2, brand="kisqali", edges_added=4)
    lm.record_provenance_write(insight_id="i2", source_count=3, brand="remibrutinib", edges_added=5)
    lm.record_provenance_write(insight_id="i3", source_count=1, brand="fabhalta", edges_added=2)

    counts = [m for m in fake_mlflow if m[0] == "e2i.crystal.count_by_brand"]
    assert len(counts) == 3
    brands_seen = {m[2].get("brand") for m in counts}
    assert brands_seen == {"kisqali", "remibrutinib", "fabhalta"}


# ---------------------------------------------------------------------
# Box 2.d — optional-instrumentation guard
# ---------------------------------------------------------------------


def test_mlflow_helpers_noop_when_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """When MLflow is unavailable, the helpers MUST silently return.

    Codex audit premise (a): production paths run normally without
    MLflow client available.
    """
    monkeypatch.setattr(lm, "_MLFLOW_AVAILABLE", False)
    # Should not raise.
    lm.record_cascade_complete(
        brand="kisqali", depth=1, edges_visited=1, duration_ms=1.0, invalidated_by_type={}
    )
    lm.record_provenance_write(insight_id="i", source_count=1, brand="kisqali", edges_added=1)
    lm.record_consolidation_sweep(
        brand="kisqali",
        dedup_collapses=0,
        promotions_to_semantic=0,
        promotions_to_procedural=0,
        templates_extracted=0,
        duration_ms=0.0,
        causal_paths_examined=0,
    )


def test_mlflow_helpers_swallow_exceptions(monkeypatch: pytest.MonkeyPatch) -> None:
    """A broken MLflow backend MUST NOT crash the producer code path."""

    def _explode(metric_name: str, value: float, tags: dict | None = None) -> None:
        raise RuntimeError("simulated mlflow backend failure")

    monkeypatch.setattr(lm, "_emit_mlflow_metric_raw", _explode)
    lm.record_cascade_complete(
        brand="kisqali", depth=1, edges_visited=1, duration_ms=1.0, invalidated_by_type={}
    )
    lm.record_provenance_write(insight_id="i", source_count=1, brand="kisqali", edges_added=1)
    lm.record_consolidation_sweep(
        brand="kisqali",
        dedup_collapses=0,
        promotions_to_semantic=0,
        promotions_to_procedural=0,
        templates_extracted=0,
        duration_ms=0.0,
        causal_paths_examined=0,
    )
