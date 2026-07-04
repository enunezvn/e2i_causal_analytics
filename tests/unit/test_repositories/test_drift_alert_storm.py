"""Alert-lifecycle regression tests for the 2026-07-04 drift-alert storm.

The 6-hourly sweep created 10,080 active alerts in one morning because (a)
``create_alerts_from_drift`` re-inserted identical alerts every run with no
dedup and (b) nothing ever resolved an alert — the alert table was an
append-only event log, not a state. These tests pin the repository-side
lifecycle: dedup-on-write, auto-resolve of cleared conditions, and
database-truth counting (the API's page-derived counts said "50 active"
while 10k+ were active).

The DB-trigger writer's dedup lives in migration
093_drift_alert_dedup_lifecycle.sql (NOT EXISTS guard) and is verified live.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.repositories.drift_monitoring import MonitoringAlertRepository

_MODEL_NAME = "initiation_kisqali_goldstd_lr_v1"
_MODEL_ID = "7b3f9a10-2222-4444-8888-aaaaaaaaaaaa"


class _FakeAlertsQuery:
    """Fluent supabase-style builder over an in-memory alerts row store.

    Supports the exact chains the repository emits:
      select("title").eq("status", ...).eq("model_id", ...)
      select("id", count="exact").eq(...).limit(1)
      insert([rows])
      update(payload).eq("status", ...).eq("model_id", ...).not_.in_("title", [...])
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self._rows = rows
        self._mode = "select"
        self._filters: List[Tuple[str, str, Any]] = []
        self._count_requested = False
        self._update_payload: Optional[Dict[str, Any]] = None
        self._insert_rows: Optional[List[Dict[str, Any]]] = None
        self._negate_next = False

    # -- builder surface -----------------------------------------------------
    @property
    def not_(self) -> "_FakeAlertsQuery":
        self._negate_next = True
        return self

    def select(self, _cols: str = "*", count: Optional[str] = None) -> "_FakeAlertsQuery":
        self._mode = "select"
        self._count_requested = count == "exact"
        return self

    def insert(self, rows: List[Dict[str, Any]]) -> "_FakeAlertsQuery":
        self._mode = "insert"
        self._insert_rows = rows
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeAlertsQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def eq(self, column: str, value: Any) -> "_FakeAlertsQuery":
        self._filters.append(("eq", column, value))
        return self

    def is_(self, column: str, value: Any) -> "_FakeAlertsQuery":
        self._filters.append(("is", column, value))
        return self

    def in_(self, column: str, values: List[Any]) -> "_FakeAlertsQuery":
        op = "not.in" if self._negate_next else "in"
        self._negate_next = False
        self._filters.append((op, column, list(values)))
        return self

    def order(self, *_a: Any, **_k: Any) -> "_FakeAlertsQuery":
        return self

    def limit(self, *_a: Any, **_k: Any) -> "_FakeAlertsQuery":
        return self

    # -- execution -----------------------------------------------------------
    def _matches(self, row: Dict[str, Any]) -> bool:
        for op, column, value in self._filters:
            if op == "eq" and row.get(column) != value:
                return False
            if op == "is" and value == "null" and row.get(column) is not None:
                return False
            if op == "in" and row.get(column) not in value:
                return False
            if op == "not.in" and row.get(column) in value:
                return False
        return True

    async def execute(self) -> Any:
        if self._mode == "insert":
            assert self._insert_rows is not None
            self._rows.extend(self._insert_rows)
            data = self._insert_rows
            count = None
        elif self._mode == "update":
            assert self._update_payload is not None
            data = []
            for row in self._rows:
                if self._matches(row):
                    row.update(self._update_payload)
                    data.append(row)
            count = None
        else:
            matched = [row for row in self._rows if self._matches(row)]
            data = matched
            count = len(matched) if self._count_requested else None
        return type("Res", (), {"data": data, "count": count})()


class _FakeRegistry:
    """ml_model_registry lookup used by _resolve_model_id."""

    def __init__(self, registry: Dict[str, str]):
        self._registry = registry
        self._pending: Optional[Tuple[str, str]] = None

    def select(self, *_a: Any, **_k: Any) -> "_FakeRegistry":
        return self

    def eq(self, column: str, value: str) -> "_FakeRegistry":
        self._pending = (column, value)
        return self

    def limit(self, *_a: Any, **_k: Any) -> "_FakeRegistry":
        return self

    async def execute(self) -> Any:
        data: List[Dict[str, Any]] = []
        if self._pending is not None:
            column, value = self._pending
            if column == "model_name" and value in self._registry:
                data = [{"id": self._registry[value]}]
        return type("Res", (), {"data": data})()


class _FakeClient:
    def __init__(self, rows: List[Dict[str, Any]], registry: Dict[str, str]):
        self.rows = rows
        self._registry = registry

    def table(self, name: str) -> Any:
        if name == "ml_model_registry":
            return _FakeRegistry(self._registry)
        return _FakeAlertsQuery(self.rows)


def _repo(rows: List[Dict[str, Any]]) -> MonitoringAlertRepository:
    return MonitoringAlertRepository(_FakeClient(rows, registry={_MODEL_NAME: _MODEL_ID}))


def _active_alert(title: str, model_id: str = _MODEL_ID) -> Dict[str, Any]:
    return {"id": f"row-{title}", "model_id": model_id, "title": title, "status": "active"}


_CRITICAL_RESULT = {
    "drift_type": "data",
    "severity": "critical",
    "feature": "disease_severity",
    "drift_detected": True,
}
_HIGH_RESULT = {
    "drift_type": "data",
    "severity": "high",
    "feature": "trx_30d",
    "drift_detected": True,
}


# =============================================================================
# create_alerts_from_drift: dedup-on-write
# =============================================================================


@pytest.mark.asyncio
async def test_create_alerts_inserts_when_no_active_duplicate() -> None:
    rows: List[Dict[str, Any]] = []
    repo = _repo(rows)

    created = await repo.create_alerts_from_drift(
        model_version=_MODEL_NAME,
        drift_results=[_CRITICAL_RESULT, _HIGH_RESULT],
    )

    assert {a.title for a in created} == {"CRITICAL data drift", "HIGH data drift"}
    assert len(rows) == 2


@pytest.mark.asyncio
async def test_create_alerts_dedups_identical_active_alert() -> None:
    """An active (model, title) pair already flags the condition — a second
    sweep must not re-insert it (the storm's append-only failure mode)."""
    rows = [_active_alert("CRITICAL data drift")]
    repo = _repo(rows)

    created = await repo.create_alerts_from_drift(
        model_version=_MODEL_NAME,
        drift_results=[_CRITICAL_RESULT],
    )

    assert created == []
    assert len(rows) == 1  # nothing appended


@pytest.mark.asyncio
async def test_create_alerts_resolved_duplicate_does_not_block() -> None:
    """Only ACTIVE alerts dedup — a resolved alert is history, and the same
    condition recurring must fire a fresh alert."""
    rows = [dict(_active_alert("CRITICAL data drift"), status="resolved")]
    repo = _repo(rows)

    created = await repo.create_alerts_from_drift(
        model_version=_MODEL_NAME,
        drift_results=[_CRITICAL_RESULT],
    )

    assert [a.title for a in created] == ["CRITICAL data drift"]
    assert len(rows) == 2


# =============================================================================
# auto_resolve_cleared
# =============================================================================


@pytest.mark.asyncio
async def test_auto_resolve_clears_stale_keeps_justified() -> None:
    rows = [
        _active_alert("Data Drift Detected: disease_severity"),
        _active_alert("Data Drift Detected: trx_30d"),
        _active_alert("CRITICAL data drift"),
    ]
    repo = _repo(rows)

    resolved = await repo.auto_resolve_cleared(
        model_version=_MODEL_NAME,
        keep_titles=["Data Drift Detected: disease_severity"],
    )

    assert resolved == 2
    by_title = {row["title"]: row for row in rows}
    assert by_title["Data Drift Detected: disease_severity"]["status"] == "active"
    assert by_title["Data Drift Detected: trx_30d"]["status"] == "resolved"
    assert by_title["CRITICAL data drift"]["status"] == "resolved"
    assert by_title["CRITICAL data drift"]["resolved_by"] == "drift_monitor_auto"


@pytest.mark.asyncio
async def test_auto_resolve_all_when_run_finds_no_drift() -> None:
    rows = [
        _active_alert("Data Drift Detected: disease_severity"),
        _active_alert("HIGH data drift"),
    ]
    repo = _repo(rows)

    resolved = await repo.auto_resolve_cleared(model_version=_MODEL_NAME, keep_titles=[])

    assert resolved == 2
    assert all(row["status"] == "resolved" for row in rows)


@pytest.mark.asyncio
async def test_auto_resolve_scoped_to_model() -> None:
    """Another model's active alerts are untouched."""
    other = _active_alert("HIGH data drift", model_id="99999999-0000-0000-0000-000000000000")
    rows = [_active_alert("HIGH data drift"), other]
    repo = _repo(rows)

    resolved = await repo.auto_resolve_cleared(model_version=_MODEL_NAME, keep_titles=[])

    assert resolved == 1
    assert other["status"] == "active"


# =============================================================================
# count_alerts: database truth, not page size
# =============================================================================


@pytest.mark.asyncio
async def test_count_alerts_returns_db_count_not_page_size() -> None:
    rows = [_active_alert(f"Data Drift Detected: f{i}") for i in range(120)]
    repo = _repo(rows)

    assert await repo.count_alerts(status="active") == 120


@pytest.mark.asyncio
async def test_count_alerts_filters_status() -> None:
    rows = [
        _active_alert("A"),
        dict(_active_alert("B"), status="resolved"),
        dict(_active_alert("C"), status="resolved"),
    ]
    repo = _repo(rows)

    assert await repo.count_alerts(status="active") == 1
    assert await repo.count_alerts(status="resolved") == 2
