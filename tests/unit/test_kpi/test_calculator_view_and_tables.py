"""
Tests for `_calculate_from_view` and `_calculate_from_tables` in
`src/kpi/calculator.py` — F-007 fix (iter-1 codex closure).

Closes #421 (F-007). Before this PR the calculator's view + tables fallbacks
returned `(None, metadata)` placeholders, so the KPI dashboard always showed
"—" because no workstream calculators were registered.

After the fix:
- `KPICalculator.__init__` AUTO-registers the per-workstream calculators in
  `src/kpi/calculators/`. These contain the real formulas (e.g.,
  `DataQualityCalculator._calc_source_coverage_patients` runs joined SQL).
  That's the real REWIRE — no more silent placeholder.
- `_calculate_from_view` is a narrow fallback for KPIs surfaced via a
  pre-computed Supabase view (single-scalar output). It uses canonical
  column-name preference (`value`, `match_rate`, etc.) and refuses to guess
  when multiple ambiguous numeric columns are present.
- `_calculate_from_tables` raises `NotImplementedError` — generic
  table-formula evaluation requires a workstream-specific calculator. No
  silent "first numeric / first numeric" guess.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest

from src.kpi.calculator import KPICalculator
from src.kpi.models import (
    CalculationType,
    KPIMetadata,
    KPIResult,
    KPIStatus,
    KPIThreshold,
    Workstream,
)


class _FakeSupabaseResponse:
    """Stub response object with `.data` attribute (matches Supabase API)."""

    def __init__(self, data: List[Dict[str, Any]] | None) -> None:
        self.data = data


class _FakeSupabaseQuery:
    """Stub for fluent-chain `.select().limit().execute()` query."""

    def __init__(self, data: List[Dict[str, Any]] | None) -> None:
        self._data = data

    def select(self, *args: Any, **kwargs: Any) -> "_FakeSupabaseQuery":
        return self

    def limit(self, *args: Any, **kwargs: Any) -> "_FakeSupabaseQuery":
        return self

    def execute(self) -> _FakeSupabaseResponse:
        return _FakeSupabaseResponse(self._data)


class _FakeSupabaseClient:
    """Stub Supabase client with deterministic `.table(name)` responses.

    Not a MagicMock — concrete behavior, predictable assertion targets. This
    test fixture lives in test-only scope (matches existing `_FakeSupabase`
    pattern in this codebase, e.g., test_executive_insights).
    """

    def __init__(self, table_responses: Dict[str, List[Dict[str, Any]] | None]) -> None:
        self.table_responses = table_responses
        self.tables_called: List[str] = []

    def table(self, name: str) -> _FakeSupabaseQuery:
        self.tables_called.append(name)
        return _FakeSupabaseQuery(self.table_responses.get(name))


def _make_view_kpi() -> KPIMetadata:
    """KPI backed by a Supabase view that outputs `match_rate` column."""
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
    """KPI that requires multi-table formula evaluation."""
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


class TestAutoRegisterWorkstreamCalculators:
    """The REAL REWIRE: KPICalculator auto-registers per-workstream calculators.

    Before #421 fix, the workstream calculators in `src/kpi/calculators/`
    existed but were never registered, so every KPI fell through to the
    placeholder fallback. The fix registers them in `__init__`.
    """

    def test_auto_register_default(self) -> None:
        """By default, all 6 workstream calculators are auto-registered."""
        calc = KPICalculator(db_connection=_FakeSupabaseClient({}))
        assert Workstream.WS1_DATA_QUALITY in calc._calculators
        assert Workstream.WS1_MODEL_PERFORMANCE in calc._calculators
        assert Workstream.WS2_TRIGGERS in calc._calculators
        assert Workstream.WS3_BUSINESS in calc._calculators
        assert Workstream.BRAND_SPECIFIC in calc._calculators
        assert Workstream.CAUSAL_METRICS in calc._calculators

    def test_auto_register_can_be_disabled(self) -> None:
        """Tests / specialized callers can opt out via the constructor flag."""
        calc = KPICalculator(
            db_connection=_FakeSupabaseClient({}),
            auto_register_workstream_calculators=False,
        )
        assert calc._calculators == {}

    def test_workstream_calculator_receives_db_client(self) -> None:
        """Registered calculators must get the same `db_connection` as the parent."""
        fake_client = _FakeSupabaseClient({})
        calc = KPICalculator(db_connection=fake_client)
        # DataQualityCalculator stores client on `._db_client` (per its API);
        # the property returns it lazily.
        dq_calc = calc._calculators[Workstream.WS1_DATA_QUALITY]
        # Use the underscore-prefixed attribute to avoid the lazy fallback
        # to `get_supabase_client()` if no client was injected.
        assert dq_calc._db_client is fake_client  # type: ignore[attr-defined]


class TestCalculateFromView:
    """`_calculate_from_view` fallback: real query, no `return None`."""

    def test_returns_scalar_from_canonical_column(self) -> None:
        """When the view has a canonical column name (`match_rate`), return it."""
        fake_client = _FakeSupabaseClient(
            table_responses={
                "v_kpi_cross_source_match": [{"match_rate": 0.78}],
            }
        )
        calc = KPICalculator(
            db_connection=fake_client,
            auto_register_workstream_calculators=False,
        )
        kpi = _make_view_kpi()

        value, metadata = calc._calculate_from_view(kpi, context={})

        assert value == pytest.approx(0.78)
        assert metadata["source"] == "view"
        assert metadata["view_name"] == "v_kpi_cross_source_match"
        assert "v_kpi_cross_source_match" in fake_client.tables_called

    def test_returns_scalar_from_value_column(self) -> None:
        """`value` is a canonical column for generic KPI views."""
        fake_client = _FakeSupabaseClient(
            table_responses={"v_kpi_cross_source_match": [{"value": 0.91}]}
        )
        calc = KPICalculator(
            db_connection=fake_client,
            auto_register_workstream_calculators=False,
        )
        value, _ = calc._calculate_from_view(_make_view_kpi(), context={})
        assert value == pytest.approx(0.91)

    def test_raises_on_ambiguous_multi_numeric_row(self) -> None:
        """If a view returns multiple numeric columns and none match a canonical
        name, the calculator MUST raise — not silently pick one.

        Codex iter-1 critique: picking the "first numeric" without context
        mis-evaluates KPIs (e.g., row `{"a": 100, "b": 5}` could be a count
        or a rate — guessing is harmful).
        """
        fake_client = _FakeSupabaseClient(
            table_responses={
                "v_kpi_cross_source_match": [{"numer": 850, "denom": 1000}],
            }
        )
        calc = KPICalculator(
            db_connection=fake_client,
            auto_register_workstream_calculators=False,
        )
        with pytest.raises(Exception):
            calc._calculate_from_view(_make_view_kpi(), context={})

    def test_raises_when_view_returns_empty(self) -> None:
        """Empty view response is a real error (not silent None)."""
        fake_client = _FakeSupabaseClient(table_responses={"v_kpi_cross_source_match": []})
        calc = KPICalculator(
            db_connection=fake_client,
            auto_register_workstream_calculators=False,
        )
        with pytest.raises(Exception) as excinfo:
            calc._calculate_from_view(_make_view_kpi(), context={})
        assert (
            "v_kpi_cross_source_match" in str(excinfo.value)
            or "no rows" in str(excinfo.value).lower()
        )

    def test_no_silent_fallback_on_query_failure(self) -> None:
        """A network/auth error in Supabase must propagate, not return None."""

        class _FailingClient:
            def table(self, name: str) -> Any:
                raise RuntimeError("Connection refused by Supabase")

        calc = KPICalculator(
            db_connection=_FailingClient(),
            auto_register_workstream_calculators=False,
        )
        with pytest.raises(Exception):
            calc._calculate_from_view(_make_view_kpi(), context={})


class TestCalculateFromTablesRaisesNotImplemented:
    """`_calculate_from_tables` raises rather than guessing a formula.

    Codex iter-1 critique: "first numeric / first numeric" across two unrelated
    tables mis-evaluates `covered_patients / reference_patients`. The honest
    answer is: register a workstream-specific calculator. The auto-register
    in `__init__` is the production answer; the fallback raises so the error
    surfaces to the user instead of being silent.
    """

    def test_table_kpi_without_registered_calculator_raises(self) -> None:
        """Direct call to the fallback raises NotImplementedError."""
        calc = KPICalculator(
            db_connection=_FakeSupabaseClient({}),
            auto_register_workstream_calculators=False,
        )
        with pytest.raises(NotImplementedError) as excinfo:
            calc._calculate_from_tables(_make_tables_kpi(), context={})
        assert "WS1-DQ-001" in str(excinfo.value) or "workstream" in str(excinfo.value).lower()
        # Error message must point to the proper fix:
        assert (
            "register a per-workstream calculator" in str(excinfo.value).lower()
            or "calculator" in str(excinfo.value).lower()
        )


class TestDefaultCalculateEndToEnd:
    """End-to-end: `_default_calculate` produces a real `KPIResult` with value."""

    def test_view_kpi_e2e_returns_kpi_result_with_value(self) -> None:
        """View-backed KPI through `_default_calculate` returns non-None value."""
        fake_client = _FakeSupabaseClient(
            table_responses={"v_kpi_cross_source_match": [{"match_rate": 0.82}]}
        )
        calc = KPICalculator(
            db_connection=fake_client,
            auto_register_workstream_calculators=False,
        )
        result: KPIResult = calc._default_calculate(_make_view_kpi(), context={})

        assert result.value is not None, (
            f"KPIResult.value was None — view-backed KPI was not actually computed. Got: {result!r}"
        )
        assert result.value == pytest.approx(0.82)
        assert result.status == KPIStatus.GOOD
        assert result.error is None
        assert result.metadata.get("source") == "view"

    def test_table_kpi_without_calculator_surfaces_error_in_result(self) -> None:
        """When `_calculate_from_tables` raises, the caller surfaces it in
        `KPIResult.error` — no silent None, no NaN.
        """
        calc = KPICalculator(
            db_connection=_FakeSupabaseClient({}),
            auto_register_workstream_calculators=False,
        )
        result: KPIResult = calc._default_calculate(_make_tables_kpi(), context={})

        assert result.value is None
        assert result.error is not None
        assert (
            "workstream calculator" in result.error.lower()
            or "table-derived" in result.error.lower()
        )


class TestNoPlaceholderInCalculator:
    """Regression pin: the placeholder comments must not return."""

    def test_calculator_source_no_placeholder_marker(self) -> None:
        """Pin the absence of the literal placeholder comments from #421."""
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


class TestRpcFailureSurfacesAsKpiError:
    """Production-path test: RPC/Supabase failures surface as `KPIResult.error`.

    Codex iter-2 HIGH #1: prior `DataQualityCalculator._execute_query` caught
    all exceptions and returned None, which `_calc_source_coverage_patients`
    translated into `0.0`. Result: a Supabase RPC failure was indistinguishable
    from "0% coverage" on the user dashboard.

    After iter-2 fix: `_execute_query` propagates exceptions. The outer
    `DataQualityCalculator.calculate` catches and emits `KPIResult(error=...)`.
    This test pins that contract end-to-end through the auto-registered path.
    """

    def test_rpc_failure_populates_kpi_error_not_silent_zero(self) -> None:
        """When the Supabase RPC raises, `KPIResult.error` must contain a real
        message; `value` must be None (NOT silently 0.0).
        """
        from src.kpi.calculators.data_quality import DataQualityCalculator

        class _RpcFailingClient:
            """Stub that raises whenever `.rpc(...)` is called."""

            def rpc(self, fn: str, params: Dict[str, Any]) -> Any:
                raise RuntimeError("Supabase RPC connection refused")

        calc = DataQualityCalculator(db_client=_RpcFailingClient())
        kpi = KPIMetadata(
            id="WS1-DQ-001",
            name="Source Coverage - Patients",
            definition="",
            formula="",
            calculation_type=CalculationType.DERIVED,
            workstream=Workstream.WS1_DATA_QUALITY,
            threshold=KPIThreshold(target=0.85, warning=0.70, critical=0.50),
        )
        result: KPIResult = calc.calculate(kpi, context={})

        assert result.value is None, (
            f"Supabase RPC failure must NOT silently return a fabricated value. "
            f"Got value={result.value!r}, error={result.error!r}"
        )
        assert result.error is not None, (
            "KPIResult.error must be populated on RPC failure (codex iter-2 HIGH #1)"
        )
        assert (
            "connection refused" in result.error.lower()
            or "rpc" in result.error.lower()
            or "supabase" in result.error.lower()
        ), f"Expected RPC failure mention in error, got: {result.error!r}"
