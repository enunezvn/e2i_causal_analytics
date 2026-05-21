"""
Tests for `_calculate_from_view` and `_calculate_from_tables` in
`src/kpi/calculator.py` — F-007 fix.

Closes #421 (F-007): the prior implementation returned `(None, metadata)` for
every view-backed or table-derived KPI, so the dashboard always showed "—".

After the fix:
- `_calculate_from_view(kpi, ctx)` MUST query the named Supabase view via the
  injected `self._db` client and return `(scalar_value, metadata)`.
- `_calculate_from_tables(kpi, ctx)` MUST aggregate from `kpi.tables` and
  return `(computed_value, metadata)`.
- On query failure, errors propagate via raised exception → caller surfaces
  via `KPIResult.error`. No silent fallback to `None`.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from src.kpi.calculator import KPICalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIStatus,
    KPIThreshold,
    Workstream,
)


class _FakeSupabaseResponse:
    """Stub for Supabase response objects with a `.data` attribute."""

    def __init__(self, data: List[Dict[str, Any]] | None) -> None:
        self.data = data


class _FakeSupabaseQuery:
    """Stub for fluent-chain Supabase table queries."""

    def __init__(self, data: List[Dict[str, Any]] | None) -> None:
        self._data = data

    def select(self, *args: Any, **kwargs: Any) -> "_FakeSupabaseQuery":
        return self

    def limit(self, *args: Any, **kwargs: Any) -> "_FakeSupabaseQuery":
        return self

    def execute(self) -> _FakeSupabaseResponse:
        return _FakeSupabaseResponse(self._data)


class _FakeSupabaseClient:
    """Stub Supabase client with a `.table(name)` method.

    NOT a mock — fully controlled for predictable test behavior. The
    KPICalculator should call `self._db.table(view_name).select(...).execute()`
    on it; we hand back canned responses.
    """

    def __init__(self, table_responses: Dict[str, List[Dict[str, Any]] | None]) -> None:
        self.table_responses = table_responses
        self.tables_called: List[str] = []

    def table(self, name: str) -> _FakeSupabaseQuery:
        self.tables_called.append(name)
        return _FakeSupabaseQuery(self.table_responses.get(name))


def _make_view_kpi() -> KPIMetadata:
    """KPI backed by a Supabase view (e.g., v_kpi_cross_source_match)."""
    return KPIMetadata(
        id="WS1-DQ-003",
        name="Cross-source Match Rate",
        definition="Test view KPI",
        formula="records_matched / total_records",
        calculation_type=CalculationType.DIRECT,
        workstream=Workstream.WS1_DATA_QUALITY,
        view="v_kpi_cross_source_match",
        threshold=KPIThreshold(target=0.75, warning=0.60, critical=0.40),
    )


def _make_tables_kpi() -> KPIMetadata:
    """KPI computed from raw tables (no view shortcut)."""
    return KPIMetadata(
        id="WS1-DQ-001",
        name="Source Coverage - Patients",
        definition="Test table-derived KPI",
        formula="covered_patients / reference_patients",
        calculation_type=CalculationType.DERIVED,
        workstream=Workstream.WS1_DATA_QUALITY,
        tables=["patient_journeys", "reference_universe"],
        threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.50),
    )


class TestCalculateFromView:
    """`_calculate_from_view` MUST return a real value, not None."""

    def test_returns_scalar_from_supabase_view(self) -> None:
        """
        For a view-backed KPI, the calculator queries the view via the injected
        Supabase client and returns `(scalar, metadata)` — not `(None, metadata)`.
        """
        # The view returns one row with a single column. The calculator should
        # pick the first scalar value from that row.
        fake_client = _FakeSupabaseClient(
            table_responses={
                "v_kpi_cross_source_match": [{"value": 0.78}],
            }
        )
        calc = KPICalculator(db_connection=fake_client)
        kpi = _make_view_kpi()

        value, metadata = calc._calculate_from_view(kpi, context={})

        assert value is not None, (
            "View-backed KPI returned None — F-007 placeholder behavior regressed."
        )
        assert isinstance(value, float)
        assert abs(value - 0.78) < 1e-9
        assert metadata["source"] == "view"
        assert metadata["view_name"] == "v_kpi_cross_source_match"
        # Verify it actually queried the named view.
        assert "v_kpi_cross_source_match" in fake_client.tables_called

    def test_raises_when_view_returns_empty(self) -> None:
        """
        An empty view response is a real error (not silent None). The
        calculator must raise so the caller surfaces it in KPIResult.error.
        """
        fake_client = _FakeSupabaseClient(table_responses={"v_kpi_cross_source_match": []})
        calc = KPICalculator(db_connection=fake_client)
        kpi = _make_view_kpi()

        with pytest.raises(Exception) as excinfo:
            calc._calculate_from_view(kpi, context={})
        # Error message must mention the view (so debug is fast).
        assert (
            "v_kpi_cross_source_match" in str(excinfo.value)
            or "empty" in str(excinfo.value).lower()
            or "no" in str(excinfo.value).lower()
        )

    def test_no_silent_fallback_to_none_on_query_failure(self) -> None:
        """
        If the Supabase query raises, the calculator must propagate (so
        `_default_calculate` populates `KPIResult.error`) — NOT swallow and
        return `None`.
        """

        class _FailingClient:
            def table(self, name: str) -> Any:
                raise RuntimeError("Connection refused by Supabase")

        calc = KPICalculator(db_connection=_FailingClient())
        kpi = _make_view_kpi()
        with pytest.raises(Exception):
            calc._calculate_from_view(kpi, context={})


class TestCalculateFromTables:
    """`_calculate_from_tables` MUST return a real value, not None."""

    def test_returns_aggregate_from_named_tables(self) -> None:
        """
        For a tables-derived KPI, the calculator queries the named tables and
        applies the formula. It must return a numeric value — not None.

        For the minimal MVP delegator, the calculator selects from the first
        listed table and treats the first numeric column as the source. The
        exact aggregation logic is documented in the implementation; this test
        only pins that *some real value* is returned (not None).
        """
        fake_client = _FakeSupabaseClient(
            table_responses={
                "patient_journeys": [{"covered": 850, "total": 1000}],
                "reference_universe": [{"total_count": 1000}],
            }
        )
        calc = KPICalculator(db_connection=fake_client)
        kpi = _make_tables_kpi()

        value, metadata = calc._calculate_from_tables(kpi, context={})

        assert value is not None, (
            "Tables-derived KPI returned None — F-007 placeholder behavior regressed."
        )
        assert isinstance(value, float)
        assert metadata["source"] == "tables"
        assert metadata["tables"] == ["patient_journeys", "reference_universe"]


class TestDefaultCalculateEndToEnd:
    """End-to-end: `_default_calculate` → real value with status evaluated."""

    def test_view_kpi_e2e_returns_kpi_result_with_value(self) -> None:
        """
        Top-level `_default_calculate` path: for a view-backed KPI with a real
        client, the returned `KPIResult.value` must NOT be None.
        """
        fake_client = _FakeSupabaseClient(
            table_responses={
                "v_kpi_cross_source_match": [{"match_rate": 0.82}],
            }
        )
        calc = KPICalculator(db_connection=fake_client)
        kpi = _make_view_kpi()

        result = calc._default_calculate(kpi, context={})

        assert result.value is not None, (
            f"KPIResult.value was None — view-backed KPI was not actually computed. Got: {result!r}"
        )
        # Threshold target=0.75; value 0.82 -> GOOD
        assert result.status == KPIStatus.GOOD
        assert result.error is None
        assert result.metadata.get("source") == "view"


class TestNoPlaceholderInCalculator:
    """Regression pin: the placeholder comment must not return."""

    def test_calculator_source_no_placeholder_marker(self) -> None:
        """Pin the absence of the literal placeholder comment from #421."""
        from pathlib import Path

        calc_path = Path(__file__).resolve().parents[3] / "src" / "kpi" / "calculator.py"
        source = calc_path.read_text()
        forbidden = (
            "# This is a placeholder - actual implementation will use Supabase client",
            "# Will be implemented when integrating with database",
            "# Placeholder - will be implemented for derived KPIs",
        )
        for marker in forbidden:
            assert marker not in source, (
                f"Detected re-introduction of placeholder comment {marker!r}. See #421 / F-007."
            )
