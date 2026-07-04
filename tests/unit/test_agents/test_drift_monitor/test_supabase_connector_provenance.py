"""Shard 07: the drift-monitor Supabase connector must default-exclude
synthetic ml_predictions rows from drift detection.

``query_predictions`` and ``query_labeled_predictions`` read ``ml_predictions``
(which carries the ``is_synthetic`` provenance column, migration 063). A
synthetic prediction must NOT register as real input/concept drift. These
tests pin the ``.eq("is_synthetic", False)`` predicate via a recording
supabase-style query builder; ``include_synthetic=True`` opts out.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.drift_monitor.connectors.base import TimeWindow
from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector


class _RecordingQuery:
    """supabase-style fluent builder recording ``.eq()`` predicates."""

    def __init__(self) -> None:
        self.eq_calls: list[tuple[Any, ...]] = []
        self.in_calls: list[tuple[Any, ...]] = []
        # ``.not_.is_(...)`` access in query_labeled_predictions.
        self.not_ = self

    def select(self, *a: Any, **k: Any) -> "_RecordingQuery":
        return self

    def eq(self, *a: Any, **k: Any) -> "_RecordingQuery":
        self.eq_calls.append(a)
        return self

    def in_(self, *a: Any, **k: Any) -> "_RecordingQuery":
        self.in_calls.append(a)
        return self

    def or_(self, *a: Any, **k: Any) -> "_RecordingQuery":
        return self

    def gte(self, *a: Any, **k: Any) -> "_RecordingQuery":
        return self

    def lte(self, *a: Any, **k: Any) -> "_RecordingQuery":
        return self

    def is_(self, *a: Any, **k: Any) -> "_RecordingQuery":
        return self

    def order(self, *a: Any, **k: Any) -> "_RecordingQuery":
        return self

    def execute(self) -> Any:
        result = MagicMock()
        result.data = []
        return result


def _connector_with_recording_query() -> tuple[SupabaseDataConnector, _RecordingQuery]:
    conn = SupabaseDataConnector(supabase_url="http://x", supabase_key="k")
    query = _RecordingQuery()
    client = MagicMock()
    client.table = MagicMock(return_value=query)
    conn._client = client
    conn._initialized = True  # bypass create_client
    return conn, query


_TW = TimeWindow(
    start=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end=datetime(2025, 2, 1, tzinfo=timezone.utc),
    label="window",
)


@pytest.mark.asyncio
async def test_query_predictions_excludes_synthetic(monkeypatch: pytest.MonkeyPatch) -> None:
    # conftest load_dotenv() imports the host .env — on showcase boxes that
    # sets E2I_INCLUDE_SYNTHETIC=true, which legitimately no-ops the filter
    # under test. Pin the strict-mode behavior deterministically.
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)
    conn, query = _connector_with_recording_query()
    await conn.query_predictions(model_id="risk_score_v1", time_window=_TW)
    assert ("is_synthetic", False) in query.eq_calls, (
        f"query_predictions did not default-exclude synthetic rows: {query.eq_calls}"
    )


@pytest.mark.asyncio
async def test_query_predictions_opt_in() -> None:
    conn, query = _connector_with_recording_query()
    await conn.query_predictions(model_id="risk_score_v1", time_window=_TW, include_synthetic=True)
    assert ("is_synthetic", False) not in query.eq_calls, (
        f"include_synthetic=True still applied the predicate: {query.eq_calls}"
    )


@pytest.mark.asyncio
async def test_query_labeled_predictions_excludes_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("E2I_INCLUDE_SYNTHETIC", raising=False)  # see strict-mode note above
    conn, query = _connector_with_recording_query()
    await conn.query_labeled_predictions(model_id="risk_score_v1", time_window=_TW)
    assert ("is_synthetic", False) in query.eq_calls, (
        f"query_labeled_predictions did not default-exclude synthetic rows: {query.eq_calls}"
    )


@pytest.mark.asyncio
async def test_query_labeled_predictions_opt_in() -> None:
    conn, query = _connector_with_recording_query()
    await conn.query_labeled_predictions(
        model_id="risk_score_v1", time_window=_TW, include_synthetic=True
    )
    assert ("is_synthetic", False) not in query.eq_calls, (
        f"include_synthetic=True still applied the predicate: {query.eq_calls}"
    )


@pytest.mark.asyncio
async def test_get_available_models_hard_excludes_synthetic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storm regression (2026-07-04): the sweep's model enumeration must
    exclude planted registry rows EVEN on showcase instances where
    E2I_INCLUDE_SYNTHETIC=true turns the shared provenance filter into a
    no-op. That no-op put 360 planted stage='production' models into the
    6-hourly sweep — 10,080 alerts in one morning."""
    monkeypatch.setenv("E2I_INCLUDE_SYNTHETIC", "true")
    conn, query = _connector_with_recording_query()
    await conn.get_available_models(stage="production")
    assert ("is_synthetic", False) in query.eq_calls, (
        "get_available_models must hard-exclude is_synthetic rows regardless "
        f"of E2I_INCLUDE_SYNTHETIC: {query.eq_calls}"
    )


@pytest.mark.asyncio
async def test_get_available_models_stages_filter() -> None:
    """The sweep enumerates production AND staging (goldstd models live at
    'staging' by design — cohort_deployer.py refuses 'production')."""
    conn, query = _connector_with_recording_query()
    await conn.get_available_models(stages=["production", "staging"])
    assert ("stage", ["production", "staging"]) in query.in_calls, (
        f"stages filter not applied via .in_: {query.in_calls}"
    )
    assert ("is_synthetic", False) in query.eq_calls
