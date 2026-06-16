"""Regression: the drift connector must resolve feature NAMES to their
``feature_id`` (uuid) via the ``features`` registry before querying
``feature_values``.

Root cause of the hollow monitoring history (all 1453 ml_monitoring_runs read
"15 checks / 0 drift", ml_drift_history ~empty): ``_get_feature_id_subquery``
returned the feature NAME, so ``.eq("feature_id", "years_experience")`` raised
22P02 (invalid input syntax for type uuid) for every feature. Every drift query
silently produced no data -> a fabricated "0 drift / healthy" for every model.
Resolve name -> uuid first; skip names absent from the registry (honest empty).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.drift_monitor.connectors.base import TimeWindow
from src.agents.drift_monitor.connectors.supabase_connector import SupabaseDataConnector

# Only this name is registered in the `features` table; the other is unknown.
_FEATURES = {"years_experience": "11111111-1111-1111-1111-111111111111"}


class _TableQuery:
    """supabase-style fluent builder, table-aware so execute() can return the
    registry mapping for `features` and value rows for `feature_values`."""

    def __init__(self, table: str, sink: dict[str, Any]) -> None:
        self._table = table
        self._sink = sink
        self.not_ = self

    def select(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def in_(self, col: str, vals: Any) -> "_TableQuery":
        self._sink.setdefault("in_calls", []).append((self._table, col, list(vals)))
        return self

    def eq(self, col: str, val: Any) -> "_TableQuery":
        self._sink.setdefault("eq_calls", []).append((self._table, col, val))
        return self

    def gte(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def lte(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def contains(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def or_(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def is_(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def order(self, *a: Any, **k: Any) -> "_TableQuery":
        return self

    def execute(self) -> Any:
        result = MagicMock()
        if self._table == "features":
            result.data = [{"id": v, "name": n} for n, v in _FEATURES.items()]
        elif self._table == "feature_values":
            result.data = [
                {
                    "value": 1.0,
                    "event_timestamp": "2025-01-15T00:00:00+00:00",
                    "entity_values": {},
                }
            ]
        else:
            result.data = []
        return result


def _connector() -> tuple[SupabaseDataConnector, dict[str, Any]]:
    conn = SupabaseDataConnector(supabase_url="http://x", supabase_key="k")
    sink: dict[str, Any] = {}
    client = MagicMock()
    client.table = MagicMock(side_effect=lambda name: _TableQuery(name, sink))
    conn._client = client
    conn._initialized = True  # bypass create_client
    return conn, sink


_TW = TimeWindow(
    start=datetime(2025, 1, 1, tzinfo=timezone.utc),
    end=datetime(2025, 2, 1, tzinfo=timezone.utc),
    label="window",
)


def _fv_feature_id_filters(sink: dict[str, Any]) -> list[Any]:
    return [v for (t, c, v) in sink.get("eq_calls", []) if t == "feature_values" and c == "feature_id"]


@pytest.mark.asyncio
async def test_query_features_filters_feature_values_by_resolved_uuid() -> None:
    conn, sink = _connector()
    result = await conn.query_features(["years_experience"], _TW)

    filters = _fv_feature_id_filters(sink)
    assert _FEATURES["years_experience"] in filters, (
        f"feature_values must be filtered by the RESOLVED uuid, got: {filters}"
    )
    assert "years_experience" not in filters, (
        f"feature_values was filtered by the raw NAME (the 22P02 bug): {filters}"
    )
    # The resolved query returns real values (non-empty FeatureData).
    assert result["years_experience"].values.size == 1


@pytest.mark.asyncio
async def test_query_features_skips_unregistered_feature() -> None:
    conn, sink = _connector()
    result = await conn.query_features(["years_experience", "not_a_real_feature"], _TW)

    # Unknown name -> honest EMPTY FeatureData, and never queried by name.
    assert result["not_a_real_feature"].values.size == 0
    assert "not_a_real_feature" not in _fv_feature_id_filters(sink)


# =============================================================================
# Fix #1: _extract_value must PRESERVE categorical labels (not force them to 0.0
# or crash on a JSONB-wrapped string). The drift node label-encodes non-numeric
# arrays, so categoricals (specialty, region…) must arrive as their raw label.
# =============================================================================


def test_extract_value_numeric_stays_float() -> None:
    conn = SupabaseDataConnector(supabase_url="http://x", supabase_key="k")
    assert conn._extract_value(7) == 7.0
    assert conn._extract_value(3.5) == 3.5
    assert conn._extract_value("3.5") == 3.5
    assert conn._extract_value({"value": 7}) == 7.0


def test_extract_value_preserves_categorical_labels() -> None:
    conn = SupabaseDataConnector(supabase_url="http://x", supabase_key="k")
    # Raw category and JSONB-wrapped category must come back as their label —
    # NOT 0.0 (silent zero) and NOT a raised ValueError (the dict-branch crash).
    assert conn._extract_value("rheumatology") == "rheumatology"
    assert conn._extract_value({"value": "low"}) == "low"


# =============================================================================
# Fix #2: query_predictions / query_labeled_predictions must NOT select the
# non-existent ml_predictions.entity_id (42703). Entities are patient_id/hcp_id.
# =============================================================================


class _SelectRecorder:
    def __init__(self) -> None:
        self.selects: list[str] = []
        self.not_ = self

    def table(self, name: str) -> "_SelectRecorder":
        return self

    def select(self, cols: str = "", *a: Any, **k: Any) -> "_SelectRecorder":
        self.selects.append(cols)
        return self

    def eq(self, *a: Any, **k: Any) -> "_SelectRecorder":
        return self

    def or_(self, *a: Any, **k: Any) -> "_SelectRecorder":
        return self

    def gte(self, *a: Any, **k: Any) -> "_SelectRecorder":
        return self

    def lte(self, *a: Any, **k: Any) -> "_SelectRecorder":
        return self

    def is_(self, *a: Any, **k: Any) -> "_SelectRecorder":
        return self

    def order(self, *a: Any, **k: Any) -> "_SelectRecorder":
        return self

    def execute(self) -> Any:
        r = MagicMock()
        r.data = []
        return r


@pytest.mark.asyncio
async def test_query_predictions_does_not_select_entity_id() -> None:
    conn = SupabaseDataConnector(supabase_url="http://x", supabase_key="k")
    rec = _SelectRecorder()
    conn._client = rec
    conn._initialized = True
    await conn.query_predictions(model_id="m", time_window=_TW)
    blob = " ".join(rec.selects)
    assert "entity_id" not in blob, f"still selects non-existent entity_id: {rec.selects}"
    assert "patient_id" in blob or "hcp_id" in blob, rec.selects


@pytest.mark.asyncio
async def test_query_labeled_predictions_does_not_select_entity_id() -> None:
    conn = SupabaseDataConnector(supabase_url="http://x", supabase_key="k")
    rec = _SelectRecorder()
    conn._client = rec
    conn._initialized = True
    await conn.query_labeled_predictions(model_id="m", time_window=_TW)
    blob = " ".join(rec.selects)
    assert "entity_id" not in blob, f"still selects non-existent entity_id: {rec.selects}"


# =============================================================================
# Fix #1 (drift side): DataDriftNode must handle CATEGORICAL (non-numeric)
# features — jointly label-encode baseline+current — instead of crashing the
# PSI/KS math on string labels (which silently failed the whole data-drift node
# once the connector started preserving categoricals).
# =============================================================================


@pytest.mark.asyncio
async def test_data_drift_handles_categorical_feature_without_crashing() -> None:
    import numpy as np

    from src.agents.drift_monitor.nodes.data_drift import DataDriftNode

    node = DataDriftNode(connector=MagicMock())
    base = np.array(["rheumatology", "dermatology", "oncology"] * 20)  # 60 categorical
    curr = np.array(["rheumatology", "dermatology", "neurology"] * 20)

    result = await node._detect_feature_drift("specialty", base, curr, 0.05, 0.2)

    # Returns a real DriftResult (encoded + scored), not a crash / None.
    assert result is not None
    assert result["feature"] == "specialty"
    assert "test_statistic" in result
