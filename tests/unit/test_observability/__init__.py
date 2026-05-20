"""Tests for the post-implementation monitoring slice of issue #391.

This package exercises three observability hooks added by the monitoring
slice:

* **Box 1 (Opik traces)** — staleness cascade, provenance writes,
  consolidation sweeps. ``test_opik_traces.py``.
* **Box 2 (MLflow metrics)** — cascade frequency, propagation depth,
  promotion rate, crystal count by brand. ``test_mlflow_metrics.py``.
* **Box 3 (Redis pub/sub latency)** — publish→receive delta for
  ``e2i:alerts``. ``test_redis_alert_latency.py``.

Design contract (locked by the parent agent's brief, see also
:mod:`src.mlops.lifecycle_monitoring`):

1. Instrumentation is COMPLETELY OPTIONAL. Production paths must run
   normally even when Opik/MLflow client is unavailable.
2. Async code paths must NOT be contaminated by sync-only decorators.
3. MLflow metrics are emitted on EVENT, not poll.
4. Redis pub/sub latency is measured as ``publish_at → received_at``
   timestamp delta embedded in payload.
5. Brand-namespacing is preserved on per-brand metrics (no global-only
   counters that would obscure cross-brand divergence).
"""
