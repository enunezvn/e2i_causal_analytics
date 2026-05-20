"""Tests for the dashboard-manifest lockstep (codex iter-0 H1 closure).

The ``config/monitoring_dashboards.yaml`` file pins every metric the
:mod:`src.mlops.lifecycle_monitoring` module emits to a downstream
dashboard panel + use-case + alert threshold. This invariant test
fails LOUDLY if a future PR drifts the producer module away from
the manifest (or vice versa).

Without this lockstep the manifest would be passive documentation
that the next code change silently invalidates.
"""

from __future__ import annotations

from src.mlops import lifecycle_monitoring as lm
from src.mlops.monitoring_dashboards import (
    get_metric_consumer,
    list_known_metric_names,
    load_dashboard_manifest,
    validate_metric_consumer_lockstep,
)


def test_dashboard_manifest_loads() -> None:
    """The YAML manifest MUST parse as a dict at the canonical path."""
    manifest = load_dashboard_manifest()
    assert isinstance(manifest, dict)
    assert "lifecycle_state" in manifest, (
        "manifest must declare lifecycle_state for the YAML guard "
        "(scripts/check_lifecycle_state.py)"
    )
    # Sanity: the four metric groups are present.
    assert "cascade_metrics" in manifest
    assert "consolidation_metrics" in manifest
    assert "crystallization_metrics" in manifest
    assert "alert_delivery_metrics" in manifest


def test_metric_consumer_lookup_returns_expected_panel() -> None:
    """Every emitted metric must resolve to a panel + use_case via
    ``get_metric_consumer``."""
    entry = get_metric_consumer(lm.METRIC_CASCADE_FREQUENCY)
    assert entry is not None
    assert entry["metric"] == lm.METRIC_CASCADE_FREQUENCY
    assert "panel_title" in entry
    assert "use_case" in entry

    entry = get_metric_consumer(lm.METRIC_ALERT_DELIVERY_LATENCY_MS)
    assert entry is not None
    assert "alert_threshold" in entry


def test_metric_consumer_lookup_unknown_metric_returns_none() -> None:
    """A metric name NOT in the manifest must return None (caller
    decides what to do — typically log + emit anyway)."""
    assert get_metric_consumer("e2i.totally.fake.metric") is None


def test_validate_metric_consumer_lockstep_holds() -> None:
    """Codex iter-0 H1 closure: every ``METRIC_*`` constant in
    ``lifecycle_monitoring`` MUST have a manifest entry, and vice
    versa.

    If this test ever fails, it means either:
    * A new metric was added to the producer module without a
      corresponding manifest entry (downstream dashboard would be
      unaware of it), OR
    * A manifest entry was added without a producer constant (the
      dashboard panel would never receive data).

    Both are bugs; fail loudly so the PR author catches it before
    merge.
    """
    validate_metric_consumer_lockstep()


def test_all_lifecycle_monitoring_metric_constants_appear_in_manifest() -> None:
    """Pin the producer side of the lockstep with an explicit
    enumeration so a future merge conflict resolution can't accidentally
    drop a metric from the loop above."""
    manifest_metrics = set(list_known_metric_names())
    producer_constants = {
        lm.METRIC_CASCADE_FREQUENCY,
        lm.METRIC_CASCADE_PROPAGATION_DEPTH,
        lm.METRIC_CONSOLIDATION_PROMOTION_RATE,
        lm.METRIC_CRYSTAL_COUNT_BY_BRAND,
        lm.METRIC_ALERT_DELIVERY_LATENCY_MS,
        lm.METRIC_ALERT_PUBLISH_COUNT,
    }
    assert producer_constants <= manifest_metrics, (
        f"metric constants not in manifest: {sorted(producer_constants - manifest_metrics)}"
    )


def test_lifecycle_state_is_advisory() -> None:
    """Codex iter-0 H1 closure: the manifest is an observability
    declaration, not a feature-drop / pipeline-halt / promotion-deny
    gate. lifecycle_state MUST be 'advisory' (per
    src/lifecycle/gate_lifecycle.py) so the
    scripts/check_lifecycle_state.py guard accepts it.
    """
    manifest = load_dashboard_manifest()
    assert manifest.get("lifecycle_state") == "advisory"
