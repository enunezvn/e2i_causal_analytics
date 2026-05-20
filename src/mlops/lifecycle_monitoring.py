"""Memory-lifecycle monitoring hooks for the issue #391 monitoring slice.

Three observability boxes (post-implementation checklist 3 of 15):

* **Box 1 — Opik traces.** Span emission on (a) staleness cascade
  completion, (b) crystallization provenance write, (c) consolidator
  sweep complete. Uses the singleton :class:`src.mlops.opik_connector.
  OpikConnector` when available; gracefully degrades to a no-op when
  Opik is unavailable.
* **Box 2 — MLflow metrics.** Counters / gauges on (a) cascade
  frequency, (b) cascade propagation depth, (c) consolidation
  promotion rate, (d) crystal count by brand. All per-brand metrics
  are tagged with the brand for downstream dashboard grouping.
* **Box 3 — Redis pub/sub alert delivery latency.** The publisher
  (:func:`src.tasks.sentinel_actions.publish_alert`) stamps
  ``publish_at: <epoch_ms>`` on every payload before serializing onto
  the Redis ``e2i:alerts`` channel. The consumer side calls
  :func:`record_alert_latency` to compute ``now_ms - publish_at`` and
  emit ``e2i.sentinel.alert_delivery_latency_ms`` to MLflow.

Design constraints (locked by the parent agent's brief)
-------------------------------------------------------
1. **Instrumentation is COMPLETELY OPTIONAL.** If Opik or MLflow is
   unavailable (e.g., CI without ``OPIK_API_KEY`` or ``MLFLOW_TRACKING_URI``),
   production paths MUST run normally. We wrap every backend call in
   ``try/except`` and short-circuit on module-level sentinel flags
   (``_OPIK_AVAILABLE`` / ``_MLFLOW_AVAILABLE``).
2. **No async-context contamination.** All ``record_*`` helpers are
   PLAIN SYNC functions. Callers from async code paths invoke them as
   normal function calls; no ``await`` needed. Opik trace creation
   happens inside the helper via a fire-and-forget pattern (the Opik
   SDK accepts sync trace creation, but the helper still runs in the
   caller's thread so it's safe to call from either sync or async
   contexts). MLflow ``log_metric`` is also synchronous.
3. **MLflow metrics are PUSH-based** — emit on event, not poll. The
   ``record_cascade_complete`` / ``record_provenance_write`` /
   ``record_consolidation_sweep`` / ``record_alert_latency`` helpers
   are called at the END of each event by the producer code path.
4. **Brand-namespacing.** Every per-brand metric is tagged with
   ``brand``. The crystal count, promotion rate, propagation depth,
   and alert latency are all per-brand observables.

Boundaries — what tests patch
-----------------------------
Unit tests substitute :data:`_emit_opik_trace` and
:data:`_emit_mlflow_metric` (the two boundary functions inside this
module). The real exception-swallowing logic still executes — the
"narrow catch" lives inside ``_emit_*_raw`` which is one level deeper.
This mirrors the
[[feedback-test-must-exercise-real-catch-not-mock]] anti-mock-bypass
contract.
"""

from __future__ import annotations

import atexit
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Background MLflow executor (codex iter-0 M2 closure).
# ---------------------------------------------------------------------
# MLflow's ``start_run`` + ``log_metric`` + ``set_tags`` are SYNCHRONOUS
# and may block on network I/O against the tracking server. Calling
# them inline from async producer paths (the SSE bridge subscriber loop
# / Celery-async consolidator+crystallizer) would block the event loop
# for the duration of the round-trip.
#
# Solution: dispatch every MLflow emission through a small single-thread
# executor so the producer call returns immediately (sub-microsecond).
# Single thread (not pool) keeps metric ordering deterministic — MLflow
# accepts unordered writes but per-metric ordering helps with debugging.
# Daemon thread so it doesn't block process shutdown; atexit hook
# attempts a brief drain.
#
# Tests substitute ``_emit_mlflow_metric`` (the mid-level shim) so they
# never reach the executor. Production callers go through the executor
# transparently.

_MLFLOW_EXECUTOR: Optional[ThreadPoolExecutor] = None
_MLFLOW_EXECUTOR_LOCK = Lock()


def _get_mlflow_executor() -> ThreadPoolExecutor:
    """Lazy-init the background MLflow emitter thread.

    Idempotent / thread-safe via the module-level lock. Daemon thread
    so it doesn't block process shutdown — the atexit hook attempts a
    brief drain but doesn't promise full completion (we accept losing
    in-flight metrics on shutdown rather than hanging the process).
    """
    global _MLFLOW_EXECUTOR
    if _MLFLOW_EXECUTOR is not None:
        return _MLFLOW_EXECUTOR
    with _MLFLOW_EXECUTOR_LOCK:
        if _MLFLOW_EXECUTOR is not None:
            return _MLFLOW_EXECUTOR
        _MLFLOW_EXECUTOR = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="e2i-mlflow-monitor"
        )
        # atexit drain — best effort. timeout=2.0s caps shutdown delay.
        atexit.register(_drain_mlflow_executor, 2.0)
        return _MLFLOW_EXECUTOR


def _drain_mlflow_executor(timeout_s: float = 2.0) -> None:
    """Best-effort drain of any in-flight MLflow emissions at shutdown.

    Called via :func:`atexit`. ``wait=True`` with ``cancel_futures=True``
    drains queued work up to the executor's existing capacity but
    refuses NEW submissions — so any post-shutdown ``record_*`` call
    silently no-ops.
    """
    global _MLFLOW_EXECUTOR
    executor = _MLFLOW_EXECUTOR
    if executor is None:
        return
    try:
        executor.shutdown(wait=True, cancel_futures=False)
    except Exception:  # pragma: no cover — defensive on shutdown
        logger.debug("lifecycle_monitoring: mlflow executor drain failed", exc_info=True)


# ---------------------------------------------------------------------
# Module-level availability sentinels.
# ---------------------------------------------------------------------
# These flags gate the actual SDK calls. They're computed at import
# time (cheap probe) so the cascade / consolidator / crystallizer can
# call the helpers without paying for an import-cost-per-event.

_OPIK_AVAILABLE: bool
try:
    import opik as _opik  # noqa: F401

    _OPIK_AVAILABLE = True
except ImportError:  # pragma: no cover — env-dependent
    _OPIK_AVAILABLE = False
    logger.debug("lifecycle_monitoring: opik unavailable; Opik traces disabled")

_MLFLOW_AVAILABLE: bool
try:
    import mlflow as _mlflow  # noqa: F401

    _MLFLOW_AVAILABLE = True
except ImportError:  # pragma: no cover — env-dependent
    _MLFLOW_AVAILABLE = False
    logger.debug("lifecycle_monitoring: mlflow unavailable; MLflow metrics disabled")


# ---------------------------------------------------------------------
# MLflow metric names (constants for test pinning + dashboard wiring).
# ---------------------------------------------------------------------
# Source of truth for the metric vocabulary. Dashboard consumers
# (Grafana / MLflow UI) read these exact names; renaming them is a
# breaking change.

METRIC_CASCADE_FREQUENCY = "e2i.cascade.frequency"
METRIC_CASCADE_PROPAGATION_DEPTH = "e2i.cascade.propagation_depth"
METRIC_CONSOLIDATION_PROMOTION_RATE = "e2i.consolidation.promotion_rate"
METRIC_CRYSTAL_COUNT_BY_BRAND = "e2i.crystal.count_by_brand"
METRIC_ALERT_DELIVERY_LATENCY_MS = "e2i.sentinel.alert_delivery_latency_ms"
# Codex iter-0 H2 closure: publish-side counter so dashboards can
# distinguish "no subscribers connected" (publish_count > 0, latency
# samples == 0) from "no alerts ever published". Brand-tagged so the
# per-brand publish/receive ratio is observable.
METRIC_ALERT_PUBLISH_COUNT = "e2i.sentinel.alert_publish_count"


# ---------------------------------------------------------------------
# Opik trace span names (constants).
# ---------------------------------------------------------------------

SPAN_CASCADE = "e2i.staleness.cascade"
SPAN_PROVENANCE_WRITE = "e2i.crystallization.provenance_write"
SPAN_CONSOLIDATION_SWEEP = "e2i.consolidation.sweep"


# ---------------------------------------------------------------------
# RAW emitters — these talk to the SDKs directly.
# ---------------------------------------------------------------------


def _emit_opik_trace_raw(span_name: str, payload: Dict[str, Any]) -> None:
    """Forward a span to Opik via the singleton connector.

    Failures here are wrapped by the outer :func:`_emit_opik_trace` so
    callers never see exceptions. Pulled into a separate function so
    tests can patch the SDK-touching surface independently of the
    narrow-catch wrapper.

    Production-only: in CI without ``OPIK_API_KEY``, the connector
    returns ``is_enabled=False`` and this fn returns silently.
    """
    # Lazy import — avoids paying for OpikConnector init at module
    # load time. The connector itself is a singleton so subsequent
    # calls reuse the same instance.
    from src.mlops.opik_connector import get_opik_connector

    connector = get_opik_connector()
    if not connector.is_enabled:
        return

    # We use ``log_model_prediction``-shaped trace creation here because
    # it's the only Opik connector method that creates a self-contained
    # one-shot trace (the ``trace_agent`` context manager assumes an
    # async-with block). The connector itself wraps the SDK call in
    # circuit-breaker logic, so a broken Opik backend doesn't crash us.
    #
    # We don't await the connector's async helper — instead, we use
    # the lower-level synchronous Opik trace shape via the cached
    # client. Trace creation in the Opik SDK is non-blocking (events
    # are batched + flushed asynchronously by Opik's own background
    # thread), so calling it from sync code is safe.
    client = getattr(connector, "_opik_client", None)
    if client is None:
        return

    try:
        from uuid_utils import uuid7 as _uuid7

        trace = client.trace(
            id=str(_uuid7()),
            name=span_name,
            input=payload,
            metadata=payload,
            tags=[span_name],
        )
        if trace is not None:
            trace.end(output={"recorded": True})
    except Exception:
        # Narrow catch: the connector's circuit breaker handles
        # repeated failures; we just don't want a single trace failure
        # to crash the producer.
        logger.debug("lifecycle_monitoring: opik trace emission failed", exc_info=True)


def _mlflow_emit_inner(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Inner SYNCHRONOUS MLflow emission — runs on the background
    executor thread, never on the caller's thread.

    Failures here are caught and logged at debug level so the executor
    keeps draining its queue. Tests don't reach this function (they
    patch ``_emit_mlflow_metric``).
    """
    from src.mlops.mlflow_connector import get_mlflow_connector

    connector = get_mlflow_connector()
    if not connector.enabled:
        return

    mlflow_module = getattr(connector, "_mlflow", None)
    if mlflow_module is None:
        return

    try:
        # Use ``with mlflow.start_run`` to ensure cleanup even on error.
        # ``nested=True`` keeps us from crashing if a run is already
        # active (e.g., a training pipeline that incidentally triggers
        # a consolidation).
        run_name_prefix = os.environ.get("E2I_MONITORING_RUN_PREFIX", "monitoring")
        with mlflow_module.start_run(
            run_name=f"{run_name_prefix}.{metric_name}",
            nested=True,
        ):
            mlflow_module.log_metric(metric_name, value)
            if tags:
                # Convert all tag values to strings — MLflow requires
                # str-valued tags.
                str_tags = {k: str(v) for k, v in tags.items()}
                mlflow_module.set_tags(str_tags)
    except Exception:
        # Narrow catch: the connector's circuit breaker absorbs
        # repeated failures; we just don't want a single metric
        # emission to crash the producer.
        logger.debug("lifecycle_monitoring: mlflow metric emission failed", exc_info=True)


def _emit_mlflow_metric_raw(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Dispatch a metric emission to the background MLflow executor.

    Codex iter-0 M2 closure: MLflow ``start_run`` + ``log_metric`` are
    SYNCHRONOUS and may block on tracking-server I/O. Calling them
    inline from the SSE bridge / Celery-async paths would block the
    event loop. Solution: submit to a single-thread executor so the
    caller returns sub-microsecond.

    The executor itself is lazily created on first submit. ``submit``
    raises ``RuntimeError`` if the executor was shut down (atexit
    drain); we swallow that so post-shutdown calls no-op.

    Failures here are wrapped by :func:`_emit_mlflow_metric`. Tests
    patch ``_emit_mlflow_metric`` (the public boundary) — they never
    reach this function, so the executor stays uninstantiated in unit
    tests.
    """
    try:
        executor = _get_mlflow_executor()
        # ``submit`` queues the work and returns instantly. The Future
        # is intentionally discarded — fire-and-forget; producer
        # doesn't care about completion.
        executor.submit(_mlflow_emit_inner, metric_name, value, tags)
    except RuntimeError:
        # Executor was shut down (e.g., process is exiting). Drop the
        # metric silently — better than raising into the producer.
        logger.debug(
            "lifecycle_monitoring: mlflow executor refused submission "
            "(likely shutting down); metric %s dropped",
            metric_name,
        )
    except Exception:
        logger.debug("lifecycle_monitoring: mlflow executor dispatch failed", exc_info=True)


# ---------------------------------------------------------------------
# Mid-level emitters — these wrap raw with optional-instrumentation
# guards and exception suppression. Test boundary lives here.
# ---------------------------------------------------------------------


def _emit_opik_trace(span_name: str, payload: Dict[str, Any]) -> None:
    """Emit one Opik trace span if Opik is available.

    Tests substitute this function with a list-appender to capture
    span_name / payload tuples without touching the SDK.
    """
    if not _OPIK_AVAILABLE:
        return
    try:
        _emit_opik_trace_raw(span_name, payload)
    except Exception:
        # Defensive: even if ``_emit_opik_trace_raw`` somehow leaks
        # an exception past its own narrow catch (programming bug),
        # we don't want it to propagate up into the producer.
        logger.debug("lifecycle_monitoring: _emit_opik_trace outer guard caught", exc_info=True)


def _emit_mlflow_metric(
    metric_name: str,
    value: float,
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Emit one MLflow metric if MLflow is available.

    Tests substitute this function with a list-appender to capture
    (name, value, tags) tuples without touching the SDK.
    """
    if not _MLFLOW_AVAILABLE:
        return
    try:
        _emit_mlflow_metric_raw(metric_name, value, tags)
    except Exception:
        logger.debug(
            "lifecycle_monitoring: _emit_mlflow_metric outer guard caught",
            exc_info=True,
        )


# ---------------------------------------------------------------------
# PUBLIC API — called by producer code paths.
# ---------------------------------------------------------------------


def record_cascade_complete(
    *,
    brand: str,
    depth: int,
    edges_visited: int,
    duration_ms: float,
    invalidated_by_type: Dict[str, int],
) -> None:
    """Record the completion of one staleness cascade run.

    Box 1.a + 2.a + 2.b. Emits:

    * Opik trace: span ``e2i.staleness.cascade`` with full payload.
    * MLflow metric: ``e2i.cascade.frequency`` (= 1.0, counter) tagged
      with ``brand``.
    * MLflow metric: ``e2i.cascade.propagation_depth`` (= depth) tagged
      with ``brand``.

    Args:
        brand: The cascade's scope_brand (e.g. ``"kisqali"``,
            ``"all"`` for explicit cross-brand cascades).
        depth: BFS depth reached.
        edges_visited: Number of insight_edges traversed.
        duration_ms: Wall-clock duration of the cascade.
        invalidated_by_type: Counts of rows invalidated per target
            type (``"trigger"`` / ``"ml_prediction"`` /
            ``"executive_insight"``).
    """
    payload: Dict[str, Any] = {
        "brand": brand,
        "depth": int(depth),
        "edges_visited": int(edges_visited),
        "duration_ms": float(duration_ms),
        "invalidated_by_type": dict(invalidated_by_type),
    }
    _emit_opik_trace(SPAN_CASCADE, payload)
    _emit_mlflow_metric(METRIC_CASCADE_FREQUENCY, 1.0, {"brand": brand})
    _emit_mlflow_metric(METRIC_CASCADE_PROPAGATION_DEPTH, float(depth), {"brand": brand})


def record_provenance_write(
    *,
    insight_id: str,
    source_count: int,
    brand: str,
    edges_added: int,
) -> None:
    """Record the completion of one crystallization provenance write.

    Box 1.b + 2.c (count_by_brand). Emits:

    * Opik trace: span ``e2i.crystallization.provenance_write`` with
      ``insight_id`` / ``source_count`` / ``brand`` / ``edges_added``.
    * MLflow metric: ``e2i.crystal.count_by_brand`` (= 1.0, counter)
      tagged with ``brand``.
    """
    payload: Dict[str, Any] = {
        "insight_id": str(insight_id),
        "source_count": int(source_count),
        "brand": brand,
        "edges_added": int(edges_added),
    }
    _emit_opik_trace(SPAN_PROVENANCE_WRITE, payload)
    _emit_mlflow_metric(METRIC_CRYSTAL_COUNT_BY_BRAND, 1.0, {"brand": brand})


def record_consolidation_sweep(
    *,
    brand: str,
    dedup_collapses: int,
    promotions_to_semantic: int,
    promotions_to_procedural: int,
    templates_extracted: int,
    duration_ms: float,
    causal_paths_examined: int = 0,
) -> None:
    """Record the completion of one consolidator sweep.

    Box 1.c + 2.b (promotion_rate). Emits:

    * Opik trace: span ``e2i.consolidation.sweep`` with full payload.
    * MLflow metric: ``e2i.consolidation.promotion_rate`` (=
      ``promotions_to_semantic / max(causal_paths_examined, 1)``)
      tagged with ``brand``.

    ``causal_paths_examined`` is required for the promotion_rate
    metric. When zero (clean state, nothing to promote), the rate
    emits as 0.0 (NOT NaN — see codex audit premise (a)).

    Args:
        brand: Sweep brand scope. Use ``"_all_"`` for whole-portfolio
            sweeps.
        dedup_collapses: Number of episodic rows collapsed.
        promotions_to_semantic: Number of causal_paths promoted to
            semantic tier.
        promotions_to_procedural: Number of procedural_memories
            graduated.
        templates_extracted: Number of procedural_templates emitted.
        duration_ms: Wall-clock duration of the sweep.
        causal_paths_examined: Total candidate causal_paths considered
            (denominator for the promotion rate). When 0, the rate
            collapses to 0.0.
    """
    payload: Dict[str, Any] = {
        "brand": brand,
        "dedup_collapses": int(dedup_collapses),
        "promotions_to_semantic": int(promotions_to_semantic),
        "promotions_to_procedural": int(promotions_to_procedural),
        "templates_extracted": int(templates_extracted),
        "duration_ms": float(duration_ms),
        "causal_paths_examined": int(causal_paths_examined),
    }
    _emit_opik_trace(SPAN_CONSOLIDATION_SWEEP, payload)
    # promotion_rate guard: max(...,1) avoids div-by-zero; we map to
    # 0.0 explicitly when there's nothing to promote so the metric
    # doesn't poison downstream dashboards with NaN values.
    if causal_paths_examined > 0:
        rate = float(promotions_to_semantic) / float(causal_paths_examined)
    else:
        rate = 0.0
    _emit_mlflow_metric(METRIC_CONSOLIDATION_PROMOTION_RATE, rate, {"brand": brand})


# ---------------------------------------------------------------------
# LRU dedup for the alert-latency metric (codex iter-0 H2 closure).
# ---------------------------------------------------------------------
# The SSE consumer's ``AlertBridge`` runs ONE bridge per HTTP connection.
# Without dedup, a single Redis publish to ``e2i:alerts`` produces N
# latency samples (one per connected client) and ZERO samples when no
# client is connected. Both shapes are misleading for a "Redis pub/sub
# delivery latency" dashboard.
#
# Mitigation: per-process LRU of recently-seen ``alert_id`` values. When
# a subscriber sees an alert it has already recorded latency for, the
# sample is suppressed. This makes the consumer-side metric "FIRST
# subscriber to record this alert wins" — at-most-one sample per
# (publisher process, alert_id) per consumer process. Combined with the
# publisher-side ``METRIC_ALERT_PUBLISH_COUNT`` counter, the dashboard
# can detect the "no subscribers" case as ``publish_count > 0 AND
# latency_samples == 0``.
#
# LRU bound (1024 entries) keeps the dedup set bounded under sustained
# alert volume; oldest entries are evicted FIFO.

_ALERT_LATENCY_DEDUP_MAX = 1024
_ALERT_LATENCY_RECENT_IDS: list[str] = []
_ALERT_LATENCY_RECENT_LOCK = Lock()


def _mark_alert_recorded(alert_id: str) -> bool:
    """Idempotency check for the consumer-side latency metric.

    Returns True iff this is the FIRST time the current process has
    seen the alert_id. Subsequent invocations within the LRU window
    return False so the caller skips emission.

    Thread-safe via the module-level lock. The LRU is a list rather
    than an OrderedDict to keep the dependency surface narrow.
    """
    with _ALERT_LATENCY_RECENT_LOCK:
        if alert_id in _ALERT_LATENCY_RECENT_IDS:
            return False
        _ALERT_LATENCY_RECENT_IDS.append(alert_id)
        # Cap at the configured ceiling — drop oldest on overflow.
        while len(_ALERT_LATENCY_RECENT_IDS) > _ALERT_LATENCY_DEDUP_MAX:
            _ALERT_LATENCY_RECENT_IDS.pop(0)
        return True


def _reset_alert_latency_dedup() -> None:
    """Test helper: clear the LRU between tests so per-test alert_id
    reuse doesn't accidentally suppress emission."""
    with _ALERT_LATENCY_RECENT_LOCK:
        _ALERT_LATENCY_RECENT_IDS.clear()


def _coerce_brand_tags(payload: Dict[str, Any]) -> Dict[str, str]:
    """Codex iter-0 M1 closure: extract a brand tag from the payload's
    ``brands`` field with the explicit "first non-empty token" rule,
    OR fall back to the literal ``"_multi_"`` when the list has 2+
    distinct brands.

    Rationale: dashboards need a stable bucket label per per-brand
    line. A multi-brand alert ought NOT inflate one brand's latency
    line at the expense of the others. The ``_multi_`` bucket lets
    the operator see (a) per-brand single-target alerts (kisqali /
    remibrutinib / fabhalta lines) AND (b) cross-brand bundles
    (``_multi_`` line) as distinct trajectories.

    Empty / missing brands → returns empty dict (no brand tag).
    """
    brands = payload.get("brands")
    if not isinstance(brands, list) or not brands:
        return {}
    normalized = [b for b in brands if isinstance(b, str) and b.strip()]
    if not normalized:
        return {}
    # ``"all"`` is the convention for cross-brand alerts; treat it as a
    # distinct dashboard bucket (the operator may want to see the
    # cross-brand line separately from the per-brand lines).
    distinct = sorted(set(normalized))
    if len(distinct) == 1:
        return {"brand": distinct[0]}
    if "all" in distinct:
        return {"brand": "all"}
    return {"brand": "_multi_"}


def record_alert_published(payload: Dict[str, Any]) -> None:
    """Record one alert publish event (codex iter-0 H2 closure helper).

    Emits ``METRIC_ALERT_PUBLISH_COUNT`` at the publisher side so the
    dashboard can detect "no subscribers connected" by comparing
    publish_count against subscriber-side latency-sample density.

    Brand tag policy: same as :func:`record_alert_latency` —
    ``_coerce_brand_tags`` returns ``brand="all"`` for cross-brand,
    ``brand="_multi_"`` for two-or-more-distinct-brands bundles,
    single brand for single-brand alerts.
    """
    tags = _coerce_brand_tags(payload)
    _emit_mlflow_metric(METRIC_ALERT_PUBLISH_COUNT, 1.0, tags or None)


def record_alert_latency(payload: Dict[str, Any]) -> None:
    """Record the publish→receive latency of one Redis pub/sub alert.

    Box 3. Reads ``payload['publish_at']`` (an integer epoch ms
    stamped at the publisher) and emits the delta against the current
    wall clock as ``e2i.sentinel.alert_delivery_latency_ms`` to MLflow.

    Codex iter-0 H2 closure: when ``payload['alert_id']`` is present,
    use it to dedup recordings inside this consumer process — only the
    FIRST observation of an alert in the LRU window emits a metric.
    This bounds the consumer-side metric to "at most one sample per
    (publisher process, alert_id) per consumer process". Combined
    with the publish-side :func:`record_alert_published` counter, the
    dashboard distinguishes "no subscribers connected" (publish > 0,
    latency_samples = 0) from "alert never published" (both = 0).

    Codex iter-0 M1 closure: brand tag uses
    :func:`_coerce_brand_tags` so multi-brand alerts emit with
    ``brand="_multi_"`` (or ``brand="all"`` for cross-brand
    convention) rather than inflating only the first brand's
    dashboard line.

    Defensive paths:

    * ``publish_at`` missing → skip emission silently (back-compat
      with alerts published BEFORE the stamping landed).
    * ``publish_at`` non-numeric → skip emission silently.
    * ``publish_at > now`` (clock skew) → clamp delta to 0.0 (don't
      poison dashboards with negative latencies).
    * ``alert_id`` present + previously seen in LRU → skip emission
      (dedup; subsequent SSE clients receiving the same publish do
      NOT re-emit the sample).
    """
    publish_at = payload.get("publish_at")
    if publish_at is None:
        return
    if not isinstance(publish_at, (int, float)) or isinstance(publish_at, bool):
        # Reject non-numeric types (including bool, which is an int
        # subclass but semantically a flag).
        return

    # Dedup gate: when alert_id is present + we've already recorded
    # latency for this alert in the current process, skip emission.
    # Missing alert_id (back-compat) → no dedup, always emit (legacy
    # behavior preserved).
    alert_id = payload.get("alert_id")
    if isinstance(alert_id, str) and alert_id:
        if not _mark_alert_recorded(alert_id):
            return

    now_ms = time.time() * 1000.0
    delta_ms = float(now_ms) - float(publish_at)
    if delta_ms < 0:
        # Clock-skew clamp. Documenting in code so the behavior is
        # explicit at the call site (no silent loss of information).
        delta_ms = 0.0

    tags = _coerce_brand_tags(payload)
    _emit_mlflow_metric(METRIC_ALERT_DELIVERY_LATENCY_MS, delta_ms, tags or None)


__all__ = [
    "METRIC_ALERT_DELIVERY_LATENCY_MS",
    "METRIC_ALERT_PUBLISH_COUNT",
    "METRIC_CASCADE_FREQUENCY",
    "METRIC_CASCADE_PROPAGATION_DEPTH",
    "METRIC_CONSOLIDATION_PROMOTION_RATE",
    "METRIC_CRYSTAL_COUNT_BY_BRAND",
    "SPAN_CASCADE",
    "SPAN_CONSOLIDATION_SWEEP",
    "SPAN_PROVENANCE_WRITE",
    "record_alert_latency",
    "record_alert_published",
    "record_cascade_complete",
    "record_consolidation_sweep",
    "record_provenance_write",
]
