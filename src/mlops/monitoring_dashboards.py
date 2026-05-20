"""Dashboard-manifest loader for the issue #391 monitoring slice.

Codex iter-0 H1 closure: this module makes ``config/monitoring_dashboards.yaml``
load-bearing in production. Without this loader the YAML file would
just be documentation; with it, the manifest becomes the canonical
mapping from MLflow metric name → dashboard panel + alert threshold,
and the lockstep invariant test pins that every metric emitted by
:mod:`src.mlops.lifecycle_monitoring` has a corresponding manifest
entry (and vice versa).

Public API
----------

* :func:`load_dashboard_manifest` — loads + caches the YAML.
* :func:`get_metric_consumer` — looks up the dashboard panel + use
  case for a given metric name.
* :func:`validate_metric_consumer_lockstep` — pins the invariant that
  every constant in :mod:`src.mlops.lifecycle_monitoring`'s ``METRIC_*``
  vocabulary has a manifest entry (and vice versa). Called from a
  unit test.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


# Default manifest path — repo-rooted. Override via env var if the
# operator wants to point at a different manifest (e.g. for staged
# rollout of a new metric).
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent.parent.parent / "config" / "monitoring_dashboards.yaml"
)


def _all_metric_entries(manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the top-level metric groups in the manifest into one
    list of metric-entry dicts. Excludes the ``opik_trace_spans`` and
    ``lifecycle_state`` top-level keys (Opik spans are documented
    separately, not as MLflow metrics).
    """
    out: List[Dict[str, Any]] = []
    for key, value in manifest.items():
        if key in {"lifecycle_state", "opik_trace_spans"}:
            continue
        if not isinstance(value, list):
            continue
        for entry in value:
            if isinstance(entry, dict) and "metric" in entry:
                out.append(entry)
    return out


@lru_cache(maxsize=4)
def load_dashboard_manifest(
    manifest_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load the YAML dashboard manifest with read-through caching.

    ``manifest_path`` is optional; defaults to
    :data:`DEFAULT_MANIFEST_PATH`. Cache key is the path so test
    overrides don't poison the production cache.

    Returns the parsed YAML dict. Raises FileNotFoundError if the
    manifest is missing — this is intentional: a missing manifest in
    production means the consumer-dependency-trace was broken, and
    the operator should know immediately.
    """
    path = Path(manifest_path) if manifest_path else DEFAULT_MANIFEST_PATH
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError(
            f"monitoring_dashboards.yaml at {path} did not parse as a dict; "
            "got {type(data).__name__}"
        )
    return data


def get_metric_consumer(metric_name: str) -> Optional[Dict[str, Any]]:
    """Return the dashboard-manifest entry for ``metric_name`` or None.

    Used by panel-rendering / alerting code to look up the human-
    readable panel title + use case + thresholds for a metric.
    """
    manifest = load_dashboard_manifest()
    for entry in _all_metric_entries(manifest):
        if entry.get("metric") == metric_name:
            return entry
    return None


def list_known_metric_names() -> List[str]:
    """Return the set of metric names registered in the manifest."""
    manifest = load_dashboard_manifest()
    return [e["metric"] for e in _all_metric_entries(manifest) if "metric" in e]


def validate_metric_consumer_lockstep() -> None:
    """Pin the invariant that every constant in
    :mod:`src.mlops.lifecycle_monitoring`'s ``METRIC_*`` vocabulary has
    a manifest entry (and vice versa). Raises ValueError on mismatch.

    Called from a unit test so a future PR that adds a new metric
    must ALSO update the manifest (and a future PR that adds a
    manifest entry must ALSO add the metric constant).

    Codex iter-0 H1 closure: this is the lockstep that ENFORCES the
    consumer-dependency-trace — without it the manifest could drift
    from the producer module silently.
    """
    from src.mlops import lifecycle_monitoring as lm

    producer_constants = {
        name: getattr(lm, name)
        for name in dir(lm)
        if name.startswith("METRIC_") and isinstance(getattr(lm, name), str)
    }
    producer_metric_names = set(producer_constants.values())
    manifest_metric_names = set(list_known_metric_names())

    missing_in_manifest = producer_metric_names - manifest_metric_names
    missing_in_producer = manifest_metric_names - producer_metric_names

    errors: List[str] = []
    if missing_in_manifest:
        errors.append(
            f"metrics emitted by lifecycle_monitoring without a manifest "
            f"entry: {sorted(missing_in_manifest)}"
        )
    if missing_in_producer:
        errors.append(
            f"manifest metric entries without a corresponding "
            f"lifecycle_monitoring METRIC_* constant: "
            f"{sorted(missing_in_producer)}"
        )
    if errors:
        raise ValueError(
            "monitoring_dashboards.yaml lockstep invariant broken: " + "; ".join(errors)
        )


__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "get_metric_consumer",
    "list_known_metric_names",
    "load_dashboard_manifest",
    "validate_metric_consumer_lockstep",
]
