"""
Drift monitoring repository for persisting drift detection results.

This repository handles CRUD operations for:
- ml_drift_history: Individual drift detection results
- ml_monitoring_alerts: Alerts generated from drift detection
- ml_monitoring_runs: Monitoring run metadata
- ml_performance_metrics: Model performance metrics over time
- ml_retraining_history: Model retraining events

Tables: database/ml/017_model_monitoring_tables.sql

Schema realignment (#842)
-------------------------
The Pydantic record models below mirror the **live** ``ml_*`` schema column for
column. Earlier the models had drifted from the database (``model_version`` vs
the real ``model_id`` uuid FK, ``detected_at`` vs ``created_at``,
``sample_size_*`` vs ``baseline_count``/``current_count``, ``metadata`` vs
``raw_results``, ``training_config`` vs ``config``, missing NOT-NULL columns
``test_type``/``title``/``trigger_type`` …). Because inserts serialised the whole
model, a write/read 42703'd on the *first* mismatched column, so the monitoring
API and Celery tasks could not persist or read drift at all.

Key reconciliation: the application speaks in **model_version strings** (e.g.
``"propensity_v2.1.0"``) but the DB links via **``model_id`` uuid FK** to
``ml_model_registry``. The repositories resolve the handle to a uuid when the
model is registered (or when the handle already *is* a uuid); otherwise
``model_id`` is left NULL — exactly how the live seed rows are stored — and the
original handle is **preserved** inside the table's jsonb column under
``_model_version`` so it is never lost and remains queryable (PostgREST
``<jsonb>->>_model_version`` filter, verified against the live DB). No value is
fabricated: an unresolved/empty model yields NULL + a recoverable handle, and a
hard query failure surfaces honestly to the caller.
"""

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.repositories.base import BaseRepository


# ---------------------------------------------------------------------------
# Client resolution (#845)
# ---------------------------------------------------------------------------
async def get_drift_monitoring_client() -> Any:
    """Resolve the async Supabase client for the drift-monitoring repositories.

    The five drift/monitoring/retraining repositories were historically
    constructed *without* a client at every call site, so ``BaseRepository``'s
    ``if not self.client`` guards silently turned every read and write into a
    no-op: the Celery drift sweep wrote no ``ml_drift_history`` rows, the
    monitoring API read empty, performance metrics were dropped, and retraining
    history was never persisted (FAILS-OPEN-on-missing-backend, #845 — the same
    family as #820 / #821 / #829 / #840). Sibling repositories in the same
    modules already injected a client, so the bare construction was an oversight.

    Call sites resolve the client through this helper and wire it into the
    repositories so those paths actually persist. **Fail-closed:** when Supabase
    is not configured (no ``SUPABASE_URL``/key) ``get_async_supabase_client``
    raises ``ServiceConnectionError``; we propagate it (and likewise refuse a
    ``None`` client) rather than hand back a client-less repository. The caller
    then surfaces the error (HTTP 5xx / failed Celery run) instead of fabricating
    an empty success — no silent no-op.
    """
    from src.memory.services.factories import (
        ServiceConnectionError,
        get_async_supabase_client,
    )

    client = await get_async_supabase_client()
    if client is None:  # defensive: never silently degrade back to a no-op repo
        raise ServiceConnectionError(
            "Supabase",
            "async Supabase client resolved to None for drift monitoring",
        )
    return client


# ---------------------------------------------------------------------------
# Enum domains (mirrors the live Postgres enums) + pure mapping helpers
# ---------------------------------------------------------------------------
# drift_severity_enum
_VALID_SEVERITY = {"none", "low", "medium", "high", "critical"}
# Legacy/loose severities the agent code historically emitted that are NOT valid
# drift_severity_enum values (a literal insert would 22P02). Mapped to the
# nearest valid value rather than silently dropped.
_SEVERITY_ALIASES = {"warning": "high", "warn": "high", "info": "low", "ok": "none"}

# drift_type_enum
_VALID_DRIFT_TYPE = {"data", "model", "concept"}

# alert_status_enum
_VALID_ALERT_STATUS = {"active", "acknowledged", "investigating", "resolved", "dismissed"}

# statistical_test_enum
_VALID_TEST_TYPE = {
    "psi",
    "ks",
    "chi_square",
    "wasserstein",
    "js_divergence",
    "importance_correlation",
}
# Faithful default test per drift kind when the drift result does not name one.
_DEFAULT_TEST_BY_DRIFT = {"data": "psi", "model": "ks", "concept": "js_divergence"}

_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)

# Reserved jsonb key used to preserve the app-level model handle when there is
# no registered model_id to point at.
_MV_KEY = "_model_version"


def _iso(value: Any) -> Any:
    """Serialise datetimes to ISO-8601 for the PostgREST/JSON boundary."""
    return value.isoformat() if isinstance(value, datetime) else value


def _normalize_severity(severity: Optional[str]) -> str:
    """Coerce a severity string to a valid ``drift_severity_enum`` value."""
    s = (severity or "").strip().lower()
    if s in _VALID_SEVERITY:
        return s
    if s in _SEVERITY_ALIASES:
        return _SEVERITY_ALIASES[s]
    return "medium"  # unknown -> visible, valid (never an invalid enum literal)


def _normalize_test_type(test_type: Optional[str], drift_type: Optional[str] = None) -> str:
    """Coerce to a valid ``statistical_test_enum`` value (NOT NULL column)."""
    t = (test_type or "").strip().lower()
    if t in _VALID_TEST_TYPE:
        return t
    return _DEFAULT_TEST_BY_DRIFT.get((drift_type or "").lower(), "psi")


def _normalize_drift_type(
    drift_type: Optional[str], *, default: Optional[str] = "data"
) -> Optional[str]:
    """Coerce to a valid ``drift_type_enum`` value (data/model/concept).

    ``default`` is returned for an unknown/empty value: ``"data"`` for the
    NOT-NULL ml_drift_history column, ``None`` for the nullable
    ml_monitoring_alerts column. An unknown literal would 22P02 the insert.
    """
    d = (drift_type or "").strip().lower()
    return d if d in _VALID_DRIFT_TYPE else default


def _normalize_alert_status(status: Optional[str]) -> str:
    """Coerce to a valid ``alert_status_enum`` value (NOT NULL column).

    Unknown/empty -> ``"active"`` (the column + record default); an unknown
    literal would 22P02 the insert.
    """
    s = (status or "").strip().lower()
    return s if s in _VALID_ALERT_STATUS else "active"


def _derive_trigger_type(trigger_reason: Optional[str]) -> str:
    """Map a (free-text) trigger reason to the ``trigger_type`` NOT-NULL column
    domain: 'drift' / 'performance' / 'scheduled' / 'manual'."""
    r = (trigger_reason or "").strip().lower()
    if "drift" in r:
        return "drift"
    if "perf" in r:
        return "performance"
    if "sched" in r:
        return "scheduled"
    if r == "manual":
        return "manual"
    return "manual"


def _ms_to_seconds(duration_ms: Optional[int]) -> float:
    """Convert milliseconds (app vocabulary) to the DB ``duration_seconds`` unit."""
    return round((duration_ms or 0) / 1000.0, 2)


def _is_uuid(value: Optional[str]) -> bool:
    return bool(value and _UUID_RE.match(value))


async def _resolve_model_id(client: Any, model_version: Optional[str]) -> Optional[str]:
    """Resolve an app-level model handle to a ``model_id`` uuid.

    - If the handle already *is* a uuid, use it directly.
    - Else look it up in ``ml_model_registry`` (by ``model_version`` then
      ``model_name``).
    - Else return ``None`` (unregistered model -> NULL FK; the handle is
      preserved in the row's jsonb so it is never lost). Never fabricated.
    """
    if not model_version:
        return None
    if _is_uuid(model_version):
        return model_version
    if not client:
        return None
    try:
        for col in ("model_version", "model_name"):
            res = await (
                client.table("ml_model_registry")
                .select("id")
                .eq(col, model_version)
                .limit(1)
                .execute()
            )
            if res.data:
                return str(res.data[0]["id"])
    except Exception:
        # A registry lookup failure must not block recording drift — fall back
        # to the preserved-handle path rather than raising.
        return None
    return None


def _apply_model_filter(query: Any, model_id: Optional[str], model_version: str, jsonb_col: str):
    """Filter a query by model on tables with a **scalar** ``model_id`` uuid FK
    (ml_drift_history / ml_performance_metrics / ml_monitoring_alerts): by
    ``model_id`` when the handle resolved to a uuid, else by the preserved handle
    in the table's jsonb column.

    NOTE: ``ml_monitoring_runs`` does NOT have a scalar ``model_id`` column — it
    links via the ``model_ids UUID[]`` array (a run can cover several models), so
    it uses ``_apply_run_model_filter`` instead. Calling this helper on the runs
    table emits ``.eq("model_id", uuid)`` against a non-existent column and
    PostgREST 42703s (the live monitoring-runs/health 500). See migration
    database/ml/017_model_monitoring_tables.sql line 291.
    """
    if model_id is not None:
        return query.eq("model_id", model_id)
    return query.eq(f"{jsonb_col}->>{_MV_KEY}", model_version)


def _apply_run_model_filter(query: Any, model_id: Optional[str], model_version: str):
    """Filter a ``ml_monitoring_runs`` query by model.

    The runs table records the models a run covered in the ``model_ids UUID[]``
    array column (NOT a scalar ``model_id`` FK). When the app handle resolves to
    a registry uuid we filter with an array-contains (``model_ids @> {uuid}``,
    PostgREST ``cs`` via ``.contains``); otherwise we fall back to the preserved
    handle in ``config->>_model_version`` — mirroring how the row was written
    (start_run stores both). Never targets the phantom scalar ``model_id``
    column, so the filtered ``/monitoring/runs`` + ``/monitoring/health`` paths
    return the real rows instead of 42703-ing into an HTTP 500.
    """
    if model_id is not None:
        return query.contains("model_ids", [model_id])
    return query.eq(f"config->>{_MV_KEY}", model_version)


# ===========================================================================
# Record models — fields mirror the live DB columns 1:1.
# The only non-column field is ``model_version`` (the app handle, preserved in
# the row's jsonb; excluded from ``to_db_row``, reconstructed by ``from_db_row``).
# ===========================================================================
class DriftHistoryRecord(BaseModel):
    """Record for ml_drift_history table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: Optional[str] = None
    model_version: Optional[str] = None  # app handle (no column; preserved in raw_results)
    experiment_id: Optional[str] = None
    deployment_id: Optional[str] = None
    drift_type: str = "data"  # data, model, concept
    feature_name: Optional[str] = None
    test_type: str = "psi"
    test_statistic: Optional[float] = None
    p_value: Optional[float] = None
    threshold: float = 0.05
    drift_detected: bool = False
    severity: str = "none"
    # NOT NULL on ml_drift_history (CHECK baseline_end<=current_start). Always
    # supplied on write (record_drift_results) and present on read (base table),
    # so kept required — the API renders them as non-optional datetimes.
    baseline_start: datetime
    baseline_end: datetime
    current_start: datetime
    current_end: datetime
    baseline_mean: Optional[float] = None
    baseline_std: Optional[float] = None
    baseline_min: Optional[float] = None
    baseline_max: Optional[float] = None
    baseline_count: Optional[int] = None
    current_mean: Optional[float] = None
    current_std: Optional[float] = None
    current_min: Optional[float] = None
    current_max: Optional[float] = None
    current_count: Optional[int] = None
    drift_score: Optional[float] = None
    contribution_to_overall: Optional[float] = None
    raw_results: Dict[str, Any] = Field(default_factory=dict)
    detected_by: str = "drift_monitor_agent"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_db_row(self) -> Dict[str, Any]:
        raw = dict(self.raw_results or {})
        if self.model_version:
            raw[_MV_KEY] = self.model_version
        return {
            "id": self.id,
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "deployment_id": self.deployment_id,
            "drift_type": _normalize_drift_type(self.drift_type),
            "feature_name": self.feature_name,
            "test_type": _normalize_test_type(self.test_type, self.drift_type),
            "test_statistic": self.test_statistic,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "drift_detected": self.drift_detected,
            "severity": _normalize_severity(self.severity),
            "baseline_start": _iso(self.baseline_start),
            "baseline_end": _iso(self.baseline_end),
            "current_start": _iso(self.current_start),
            "current_end": _iso(self.current_end),
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "baseline_min": self.baseline_min,
            "baseline_max": self.baseline_max,
            "baseline_count": self.baseline_count,
            "current_mean": self.current_mean,
            "current_std": self.current_std,
            "current_min": self.current_min,
            "current_max": self.current_max,
            "current_count": self.current_count,
            "drift_score": self.drift_score,
            "contribution_to_overall": self.contribution_to_overall,
            "raw_results": raw,
            "detected_by": self.detected_by,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "DriftHistoryRecord":
        data = dict(row)
        raw = data.get("raw_results") or {}
        mv = raw.get(_MV_KEY) if isinstance(raw, dict) else None
        rec = cls.model_validate(data)
        rec.model_version = mv
        return rec


class MonitoringAlertRecord(BaseModel):
    """Record for ml_monitoring_alerts table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: Optional[str] = None
    model_version: Optional[str] = None  # app handle (preserved in metadata)
    alert_type: str = "drift"
    title: Optional[str] = None  # NOT NULL — derived from message when absent
    severity: str = "medium"
    status: str = "active"
    message: str = ""
    affected_features: List[str] = Field(default_factory=list)
    drift_type: Optional[str] = None
    composite_drift_score: Optional[float] = None
    recommended_action: Optional[str] = None
    recommended_priority: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def _resolved_title(self) -> str:
        if self.title:
            return self.title[:500]
        return (self.message or f"{self.alert_type} alert")[:500]

    def to_db_row(self) -> Dict[str, Any]:
        meta = dict(self.metadata or {})
        if self.model_version:
            meta[_MV_KEY] = self.model_version
        return {
            "id": self.id,
            "model_id": self.model_id,
            "alert_type": self.alert_type,
            "title": self._resolved_title(),
            "severity": _normalize_severity(self.severity),
            "status": _normalize_alert_status(self.status),
            "message": self.message,
            "affected_features": list(self.affected_features or []),
            "drift_type": _normalize_drift_type(self.drift_type, default=None),
            "composite_drift_score": self.composite_drift_score,
            "recommended_action": self.recommended_action,
            "recommended_priority": self.recommended_priority,
            "acknowledged_at": _iso(self.acknowledged_at),
            "acknowledged_by": self.acknowledged_by,
            "resolved_at": _iso(self.resolved_at),
            "resolved_by": self.resolved_by,
            "metadata": meta,
            "created_at": _iso(self.created_at),
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "MonitoringAlertRecord":
        data = dict(row)
        meta = data.get("metadata") or {}
        mv = meta.get(_MV_KEY) if isinstance(meta, dict) else None
        rec = cls.model_validate(data)
        rec.model_version = mv
        return rec


class MonitoringRunRecord(BaseModel):
    """Record for ml_monitoring_runs table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_version: Optional[str] = None  # app handle (preserved in config)
    model_ids: Optional[List[str]] = None
    run_type: str = "full"
    trigger_type: str = "scheduled"
    status: str = "running"
    total_checks: int = 0
    drift_detected_count: int = 0
    alerts_generated: int = 0
    duration_seconds: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

    def to_db_row(self) -> Dict[str, Any]:
        cfg = dict(self.config or {})
        if self.model_version:
            cfg[_MV_KEY] = self.model_version
        return {
            "id": self.id,
            "model_ids": self.model_ids,
            "run_type": self.run_type,
            "trigger_type": self.trigger_type,
            "status": self.status,
            "total_checks": self.total_checks,
            "drift_detected_count": self.drift_detected_count,
            "alerts_generated": self.alerts_generated,
            "duration_seconds": self.duration_seconds,
            "config": cfg,
            "started_at": _iso(self.started_at),
            "completed_at": _iso(self.completed_at),
            "error_message": self.error_message,
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "MonitoringRunRecord":
        data = dict(row)
        cfg = data.get("config") or {}
        mv = cfg.get(_MV_KEY) if isinstance(cfg, dict) else None
        rec = cls.model_validate(data)
        rec.model_version = mv
        return rec


class PerformanceMetricRecord(BaseModel):
    """Record for ml_performance_metrics table."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: Optional[str] = None
    model_version: Optional[str] = None  # app handle (preserved in metadata)
    metric_name: str = ""
    metric_value: float = 0.0
    sample_size: Optional[int] = None
    measurement_window_start: Optional[datetime] = None
    measurement_window_end: Optional[datetime] = None
    measured_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    # ``source`` maps to the ml_performance_metrics.source column (default 'mlflow' in DB).
    # When None the key is omitted from to_db_row() so the DB default applies; when set
    # (e.g. 'backtest_wf' / 'holdout') the value is written explicitly.
    source: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_db_row(self) -> Dict[str, Any]:
        meta = dict(self.metadata or {})
        if self.model_version:
            meta[_MV_KEY] = self.model_version
        row: Dict[str, Any] = {
            "id": self.id,
            "model_id": self.model_id,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "sample_size": self.sample_size,
            "measurement_window_start": _iso(self.measurement_window_start),
            "measurement_window_end": _iso(self.measurement_window_end),
            "measured_at": _iso(self.measured_at),
            "metadata": meta,
        }
        # Emit source only when explicitly set; absence lets the DB default ('mlflow') apply.
        if self.source is not None:
            row["source"] = self.source
        return row

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "PerformanceMetricRecord":
        data = dict(row)
        meta = data.get("metadata") or {}
        mv = meta.get(_MV_KEY) if isinstance(meta, dict) else None
        rec = cls.model_validate(data)
        rec.model_version = mv
        return rec


class RetrainingHistoryRecord(BaseModel):
    """Record for ml_retraining_history table.

    Backward-compat read accessors (``performance_before``/``performance_after``/
    ``drift_score_before``/``training_config``) derive from the real columns so
    ``src/services/retraining_trigger.py`` reads unchanged.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    model_id: Optional[str] = None
    trigger_type: str = "manual"
    old_model_version: Optional[str] = None
    new_model_version: Optional[str] = None
    trigger_reason: str = ""
    drift_severity: Optional[str] = None
    status: str = "pending"
    old_metric_value: Optional[float] = None
    new_metric_value: Optional[float] = None
    improvement: Optional[float] = None
    performance_delta: Optional[float] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    notes: Optional[str] = None
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # --- backward-compat accessors (no DB columns of their own) ---
    @property
    def performance_before(self) -> Optional[float]:
        return self.old_metric_value

    @property
    def performance_after(self) -> Optional[float]:
        return self.new_metric_value

    @property
    def drift_score_before(self) -> float:
        val = (self.config or {}).get("drift_score_before")
        return float(val) if isinstance(val, (int, float)) and not isinstance(val, bool) else 0.0

    @property
    def training_config(self) -> Dict[str, Any]:
        return dict(self.config or {})

    def to_db_row(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model_id": self.model_id,
            "trigger_type": _derive_trigger_type(self.trigger_type or self.trigger_reason),
            "old_model_version": self.old_model_version,
            "new_model_version": self.new_model_version,
            "trigger_reason": self.trigger_reason,
            "drift_severity": (
                _normalize_severity(self.drift_severity) if self.drift_severity else None
            ),
            "status": self.status,
            "old_metric_value": self.old_metric_value,
            "new_metric_value": self.new_metric_value,
            "improvement": self.improvement,
            "performance_delta": self.performance_delta,
            "config": dict(self.config or {}),
            "notes": self.notes,
            "triggered_at": _iso(self.triggered_at),
            "completed_at": _iso(self.completed_at),
        }

    @classmethod
    def from_db_row(cls, row: Dict[str, Any]) -> "RetrainingHistoryRecord":
        return cls.model_validate(dict(row))


# ===========================================================================
# Repositories
# ===========================================================================
class DriftHistoryRepository(BaseRepository[DriftHistoryRecord]):
    """Repository for drift detection history."""

    table_name = "ml_drift_history"
    model_class = DriftHistoryRecord

    def _to_model(self, data: Dict[str, Any]) -> DriftHistoryRecord:
        return DriftHistoryRecord.from_db_row(data)

    async def record_drift_results(
        self,
        model_version: str,
        drift_results: List[Dict[str, Any]],
        baseline_window: Dict[str, datetime],
        current_window: Dict[str, datetime],
    ) -> List[DriftHistoryRecord]:
        """Record multiple drift detection results.

        Args:
            model_version: Model version/handle being monitored
            drift_results: List of drift detection results
            baseline_window: Baseline period timestamps
            current_window: Current period timestamps

        Returns:
            List of created records
        """
        if not self.client or not drift_results:
            return []

        model_id = await _resolve_model_id(self.client, model_version)
        now = datetime.now(timezone.utc)

        records: List[DriftHistoryRecord] = []
        for result in drift_results:
            drift_type = result.get("drift_type", "data")
            record = DriftHistoryRecord(
                model_id=model_id,
                model_version=model_version,
                feature_name=result.get("feature", result.get("feature_name")),
                drift_type=drift_type,
                test_type=_normalize_test_type(result.get("test_type"), drift_type),
                test_statistic=result.get("test_statistic"),
                p_value=result.get("p_value"),
                drift_detected=bool(result.get("drift_detected", False)),
                severity=result.get("severity", "none"),
                baseline_start=baseline_window.get("start", now),
                baseline_end=baseline_window.get("end", now),
                current_start=current_window.get("start", now),
                current_end=current_window.get("end", now),
                baseline_count=result.get("baseline_count", result.get("sample_size_baseline")),
                current_count=result.get("current_count", result.get("sample_size_current")),
                baseline_mean=result.get("baseline_mean"),
                current_mean=result.get("current_mean"),
                drift_score=result.get("drift_score"),
                contribution_to_overall=result.get("contribution_to_overall"),
                raw_results=result.get("metadata", result.get("raw_results", {})) or {},
            )
            records.append(record)

        data = [r.to_db_row() for r in records]
        await self.client.table(self.table_name).insert(data).execute()
        return records

    async def get_latest_drift_status(
        self,
        model_version: str,
        limit: int = 50,
    ) -> List[DriftHistoryRecord]:
        """Get the latest drift status per (drift_type, feature) for a model.

        Reads the base ``ml_drift_history`` table (rather than the
        ``ml_drift_status_latest`` view, which exposes neither the period
        columns the API renders nor the preserved model handle) and keeps the
        most-recent row per (drift_type, feature_name).
        """
        if not self.client:
            return []

        model_id = await _resolve_model_id(self.client, model_version)
        query = self.client.table(self.table_name).select("*")
        query = _apply_model_filter(query, model_id, model_version, "raw_results")
        query = query.order("created_at", desc=True).limit(max(limit * 5, limit))
        result = await query.execute()

        seen: set = set()
        latest: List[DriftHistoryRecord] = []
        for row in result.data or []:
            key = (row.get("drift_type"), row.get("feature_name"))
            if key in seen:
                continue
            seen.add(key)
            latest.append(self._to_model(row))
            if len(latest) >= limit:
                break
        return latest

    async def get_drift_trend(
        self,
        model_version: str,
        feature_name: str,
        days: int = 7,
    ) -> List[DriftHistoryRecord]:
        """Get drift trend for a specific feature over the last ``days`` days."""
        if not self.client:
            return []

        model_id = await _resolve_model_id(self.client, model_version)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        query = self.client.table(self.table_name).select("*")
        query = _apply_model_filter(query, model_id, model_version, "raw_results")
        query = (
            query.eq("feature_name", feature_name)
            .gte("created_at", cutoff)
            .order("created_at", desc=True)
        )
        result = await query.execute()
        return [self._to_model(row) for row in (result.data or [])]


class MonitoringAlertRepository(BaseRepository[MonitoringAlertRecord]):
    """Repository for monitoring alerts."""

    table_name = "ml_monitoring_alerts"
    model_class = MonitoringAlertRecord

    def _to_model(self, data: Dict[str, Any]) -> MonitoringAlertRecord:
        return MonitoringAlertRecord.from_db_row(data)

    async def create_alerts_from_drift(
        self,
        model_version: str,
        drift_results: List[Dict[str, Any]],
    ) -> List[MonitoringAlertRecord]:
        """Create alerts from drift detection results.

        Generates alerts for critical and high severity drifts. Alert severity
        is the ``drift_severity_enum`` value ('critical'/'high'); the prior code
        emitted 'warning', which is not a valid enum value and would 22P02.
        """
        if not self.client:
            return []

        model_id = await _resolve_model_id(self.client, model_version)

        critical_by_type: Dict[str, List[str]] = {}
        high_by_type: Dict[str, List[str]] = {}

        for result in drift_results:
            drift_type = result.get("drift_type", "data")
            severity = result.get("severity", "none")
            feature = result.get("feature", result.get("feature_name", "unknown"))

            if severity == "critical":
                critical_by_type.setdefault(drift_type, []).append(feature)
            elif severity == "high":
                high_by_type.setdefault(drift_type, []).append(feature)

        alerts: List[MonitoringAlertRecord] = []

        for drift_type, features in critical_by_type.items():
            alerts.append(
                MonitoringAlertRecord(
                    model_id=model_id,
                    model_version=model_version,
                    alert_type=f"{drift_type}_drift",
                    severity="critical",
                    drift_type=drift_type,
                    title=f"CRITICAL {drift_type} drift",
                    message=f"CRITICAL {drift_type} drift detected in: {', '.join(features[:5])}",
                    affected_features=features,
                    recommended_action=self._get_recommendation(drift_type, "critical"),
                    recommended_priority="immediate",
                )
            )

        for drift_type, features in high_by_type.items():
            alerts.append(
                MonitoringAlertRecord(
                    model_id=model_id,
                    model_version=model_version,
                    alert_type=f"{drift_type}_drift",
                    severity="high",
                    drift_type=drift_type,
                    title=f"HIGH {drift_type} drift",
                    message=f"HIGH {drift_type} drift detected in: {', '.join(features[:5])}",
                    affected_features=features,
                    recommended_action=self._get_recommendation(drift_type, "high"),
                    recommended_priority="high",
                )
            )

        if alerts:
            data = [a.to_db_row() for a in alerts]
            await self.client.table(self.table_name).insert(data).execute()

        return alerts

    async def get_active_alerts(
        self,
        model_version: Optional[str] = None,
        limit: int = 100,
    ) -> List[MonitoringAlertRecord]:
        """Get all active (unresolved) alerts, optionally filtered by model."""
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(limit)
        )

        if model_version:
            model_id = await _resolve_model_id(self.client, model_version)
            query = _apply_model_filter(query, model_id, model_version, "metadata")

        result = await query.execute()
        return [self._to_model(row) for row in (result.data or [])]

    async def acknowledge_alert(
        self,
        alert_id: str,
        acknowledged_by: str,
    ) -> Optional[MonitoringAlertRecord]:
        """Acknowledge an alert (records who/when on the real columns)."""
        return await self.update(
            alert_id,
            {
                "status": "acknowledged",
                "acknowledged_at": datetime.now(timezone.utc).isoformat(),
                "acknowledged_by": acknowledged_by,
            },
        )

    async def resolve_alert(
        self,
        alert_id: str,
        resolved_by: str,
    ) -> Optional[MonitoringAlertRecord]:
        """Resolve an alert."""
        return await self.update(
            alert_id,
            {
                "status": "resolved",
                "resolved_at": datetime.now(timezone.utc).isoformat(),
                "resolved_by": resolved_by,
            },
        )

    def _get_recommendation(self, drift_type: str, severity: str) -> str:
        """Get recommended action based on drift type and severity."""
        recommendations = {
            ("data", "critical"): "Immediate action required: Retrain model with recent data",
            ("data", "high"): "Monitor closely: Schedule retraining if drift persists",
            ("model", "critical"): "Immediate action required: Investigate model degradation",
            ("model", "high"): "Monitor closely: Check prediction accuracy",
            (
                "concept",
                "critical",
            ): "Immediate action required: Review feature-target relationships",
            ("concept", "high"): "Monitor closely: Validate model on current data",
        }
        return recommendations.get((drift_type, severity), "Review drift detection results")


class MonitoringRunRepository(BaseRepository[MonitoringRunRecord]):
    """Repository for monitoring run metadata."""

    table_name = "ml_monitoring_runs"
    model_class = MonitoringRunRecord

    def _to_model(self, data: Dict[str, Any]) -> MonitoringRunRecord:
        return MonitoringRunRecord.from_db_row(data)

    async def start_run(
        self,
        model_version: str,
        run_type: str = "scheduled",
        config: Optional[Dict[str, Any]] = None,
    ) -> MonitoringRunRecord:
        """Start a new monitoring run.

        ``run_type`` here is the app's notion of *why* the run fired (scheduled/
        manual/triggered) — it maps to the DB ``trigger_type`` column. The DB
        ``run_type`` (what kind of monitoring) is recorded as 'full'.
        """
        model_id = await _resolve_model_id(self.client, model_version) if self.client else None
        record = MonitoringRunRecord(
            model_version=model_version,
            model_ids=[model_id] if model_id else None,
            run_type="full",
            trigger_type=run_type,
            config=config or {},
        )

        if self.client:
            await self.client.table(self.table_name).insert(record.to_db_row()).execute()

        return record

    async def complete_run(
        self,
        run_id: str,
        features_checked: int,
        drift_detected_count: int,
        alerts_generated: int,
        duration_ms: int,
        error_message: Optional[str] = None,
    ) -> Optional[MonitoringRunRecord]:
        """Complete a monitoring run (maps features_checked -> total_checks and
        duration_ms -> duration_seconds)."""
        status = "completed" if error_message is None else "failed"

        return await self.update(
            run_id,
            {
                "status": status,
                "total_checks": features_checked,
                "drift_detected_count": drift_detected_count,
                "alerts_generated": alerts_generated,
                "duration_seconds": _ms_to_seconds(duration_ms),
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "error_message": error_message,
            },
        )

    async def get_recent_runs(
        self,
        model_version: Optional[str] = None,
        limit: int = 10,
        since: Optional[datetime] = None,
    ) -> List[MonitoringRunRecord]:
        """Get recent monitoring runs."""
        if not self.client:
            return []

        query = (
            self.client.table(self.table_name)
            .select("*")
            .order("started_at", desc=True)
            .limit(limit)
        )

        if model_version:
            model_id = await _resolve_model_id(self.client, model_version)
            # ml_monitoring_runs links models via the model_ids UUID[] array, not
            # a scalar model_id column — use the array-aware run filter.
            query = _apply_run_model_filter(query, model_id, model_version)

        if since is not None:
            query = query.gte("started_at", since.isoformat())

        result = await query.execute()
        return [self._to_model(row) for row in (result.data or [])]


class PerformanceMetricRepository(BaseRepository[PerformanceMetricRecord]):
    """Repository for model performance metrics."""

    table_name = "ml_performance_metrics"
    model_class = PerformanceMetricRecord

    def _to_model(self, data: Dict[str, Any]) -> PerformanceMetricRecord:
        return PerformanceMetricRecord.from_db_row(data)

    async def record_metrics(
        self,
        model_version: str,
        metrics: Dict[str, float],
        sample_size: int,
        window_start: datetime,
        window_end: datetime,
        *,
        measured_at: Optional[datetime] = None,
        source: Optional[str] = None,
    ) -> List[PerformanceMetricRecord]:
        """Record performance metrics for a model.

        Args:
            model_version: Model version/handle being tracked.
            metrics: Mapping of metric_name → metric_value.
            sample_size: Number of samples in the evaluation window.
            window_start: Start of the measurement window.
            window_end: End of the measurement window.
            measured_at: Explicit timestamp for the measurement point (e.g. the
                eval month for a backtest sweep).  When None the record's
                ``default_factory`` (``now()``) applies — preserving the
                existing behaviour for all callers that omit this argument.
            source: Tag for the metric origin (e.g. ``'backtest_wf'`` /
                ``'holdout'``).  When None the DB column default (``'mlflow'``)
                is preserved via the ``to_db_row()`` omission rule.
        """
        if not self.client or not metrics:
            return []

        model_id = await _resolve_model_id(self.client, model_version)

        records: List[PerformanceMetricRecord] = []
        for metric_name, metric_value in metrics.items():
            kwargs: Dict[str, Any] = dict(
                model_id=model_id,
                model_version=model_version,
                metric_name=metric_name,
                metric_value=metric_value,
                sample_size=sample_size,
                measurement_window_start=window_start,
                measurement_window_end=window_end,
                source=source,
            )
            if measured_at is not None:
                kwargs["measured_at"] = measured_at
            records.append(PerformanceMetricRecord(**kwargs))

        data = [r.to_db_row() for r in records]
        await self.client.table(self.table_name).insert(data).execute()
        return records

    async def delete_metrics(
        self,
        model_id: str,
        source: str,
        split_version: Optional[str] = None,
    ) -> int:
        """Delete performance metric rows by model_id and source tag.

        Used by the MetricRecorder (T7) for idempotent re-runs: delete stale
        rows for a given (model_id, source, split_version) before re-inserting.

        Args:
            model_id: The UUID of the model in ``ml_model_registry``.
            source: The source tag to filter on (e.g. ``'backtest_wf'``).
            split_version: When supplied, further restricts to rows where
                ``metadata->>split_version`` equals this value.  The JSONB
                ``->>`` form mirrors ``_apply_model_filter`` (line 238 and 255
                of this file) which is the form the repo already uses.

        Returns:
            Number of rows deleted (0 when client is absent — safe no-op).
        """
        if not self.client:
            return 0
        q = (
            self.client.table(self.table_name)
            .delete()
            .eq("model_id", model_id)
            .eq("source", source)
        )
        if split_version is not None:
            q = q.eq("metadata->>split_version", split_version)
        res = await q.execute()
        return len(res.data or [])

    async def get_metric_trend(
        self,
        model_version: str,
        metric_name: str,
        days: int = 30,
    ) -> List[PerformanceMetricRecord]:
        """Get metric trend over time."""
        if not self.client:
            return []

        model_id = await _resolve_model_id(self.client, model_version)
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

        query = self.client.table(self.table_name).select("*")
        query = _apply_model_filter(query, model_id, model_version, "metadata")
        query = (
            query.eq("metric_name", metric_name)
            .gte("measured_at", cutoff)
            .order("measured_at", desc=True)
        )
        result = await query.execute()
        return [self._to_model(row) for row in (result.data or [])]


class RetrainingHistoryRepository(BaseRepository[RetrainingHistoryRecord]):
    """Repository for model retraining history."""

    table_name = "ml_retraining_history"
    model_class = RetrainingHistoryRecord

    def _to_model(self, data: Dict[str, Any]) -> RetrainingHistoryRecord:
        return RetrainingHistoryRecord.from_db_row(data)

    async def trigger_retraining(
        self,
        old_model_version: str,
        new_model_version: str,
        trigger_reason: str,
        drift_score_before: float,
        performance_before: float,
        training_config: Optional[Dict[str, Any]] = None,
    ) -> RetrainingHistoryRecord:
        """Record a retraining trigger.

        ``drift_score_before`` has no dedicated column on ml_retraining_history;
        it is preserved inside ``config`` (alongside the training contract).
        ``performance_before`` maps to ``old_metric_value``.
        """
        config = dict(training_config or {})
        config.setdefault("drift_score_before", drift_score_before)

        record = RetrainingHistoryRecord(
            old_model_version=old_model_version,
            new_model_version=new_model_version,
            trigger_reason=trigger_reason,
            trigger_type=_derive_trigger_type(trigger_reason),
            old_metric_value=performance_before,
            config=config,
            status="pending",
        )

        if self.client:
            await self.client.table(self.table_name).insert(record.to_db_row()).execute()

        return record

    async def complete_retraining(
        self,
        record_id: str,
        performance_after: float,
        success: bool = True,
        *,
        mlflow_run_id: Optional[str] = None,
    ) -> Optional[RetrainingHistoryRecord]:
        """Complete a retraining run.

        Args:
            record_id: Retraining record UUID
            performance_after: Performance after retraining (-> new_metric_value)
            success: Whether retraining was successful
            mlflow_run_id: Provenance pointer to the MLflow run that produced
                ``performance_after`` (#546). When supplied it is merged into the
                record's ``config`` (under ``mlflow_run_id``) so a completed
                metric remains auditable back to its run.

        Returns:
            Updated record
        """
        status = "completed" if success else "failed"

        updates: Dict[str, Any] = {
            "status": status,
            "new_metric_value": performance_after,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }

        if mlflow_run_id:
            existing = await self.get_by_id(record_id)
            if not existing:
                return None
            config = dict(existing.config or {})
            config["mlflow_run_id"] = mlflow_run_id
            updates["config"] = config

        return await self.update(record_id, updates)

    async def mark_failed(
        self,
        record_id: str,
        reason: str,
    ) -> Optional[RetrainingHistoryRecord]:
        """Mark a retraining record failed, recording the reason in ``notes``.

        ml_retraining_history has no ``error_message`` column (the failure
        reason lives in ``notes``); writing ``error_message`` would 42703.
        """
        return await self.update(
            record_id,
            {
                "status": "failed",
                "notes": reason,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    async def rollback_retraining(
        self,
        record_id: str,
    ) -> Optional[RetrainingHistoryRecord]:
        """Mark retraining as rolled back."""
        return await self.update(
            record_id,
            {
                "status": "rolled_back",
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
