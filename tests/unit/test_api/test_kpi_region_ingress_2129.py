"""Regression contract for caller-controlled KPI region scopes (#2129).

Every API ingress must resolve a supplied region to the shared ``region_type``
vocabulary before it reaches a calculator or the materialized-history
repository.  Omission or the exact empty string means the global scope; an
unknown or deliberately ambiguous label is an explicit 422 and must not touch
the downstream reader.

These tests use recording boundary objects so they assert the exact value that
would become a SQL/RPC predicate.  The PR's certification pass separately
exercises the real calculator and database path.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.routes import kpi as kpi_routes
from src.api.routes.insights_strategic import HomeKpiInsightRequest, home_kpi_insight
from src.api.schemas.kpi import BatchKPICalculationRequest, KPICalculationRequest
from src.kpi.models import KPIBatchResult, KPIResult, KPIStatus

_KPI_ID = "WS3-BI-005"
_USER = {"id": "region-ingress-test", "app_metadata": {"role": "admin"}}


@dataclass
class _RecordingCalculator:
    calculate_calls: list[dict[str, Any]] = field(default_factory=list)
    batch_calls: list[dict[str, Any]] = field(default_factory=list)

    def list_kpis(self, **_kwargs: Any) -> list[Any]:
        # The strategic-insight route enumerates metadata before calculating.
        # An empty registry result is enough to exercise its scope boundary.
        return []

    def calculate(self, kpi_id: str, **kwargs: Any) -> KPIResult:
        self.calculate_calls.append({"kpi_id": kpi_id, **kwargs})
        return KPIResult(kpi_id=kpi_id, value=1.0, status=KPIStatus.INFORMATIONAL)

    def calculate_batch(self, **kwargs: Any) -> KPIBatchResult:
        self.batch_calls.append(kwargs)
        return KPIBatchResult()


@dataclass
class _RecordingHistoryRepository:
    calls: list[dict[str, Any]] = field(default_factory=list)

    async def get_history(self, kpi_id: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append({"kpi_id": kpi_id, **kwargs})
        return []


@dataclass
class _Harness:
    calculator: _RecordingCalculator
    history: _RecordingHistoryRepository
    insight_grounding_regions: list[str | None]
    insight_cache_scopes: list[str]
    strategic_calculator_requests: list[bool]


@pytest.fixture
def harness(monkeypatch: pytest.MonkeyPatch) -> _Harness:
    """Wire recorders at the two downstream predicate boundaries."""
    from src.api.routes import insights_strategic
    from src.repositories import kpi_history

    calculator = _RecordingCalculator()
    history = _RecordingHistoryRepository()
    insight_grounding_regions: list[str | None] = []
    insight_cache_scopes: list[str] = []
    strategic_calculator_requests: list[bool] = []
    real_build_grounding = insights_strategic.home_kpi.build_grounding

    async def get_history_repository() -> _RecordingHistoryRepository:
        return history

    async def cached_insight(_key: str) -> dict[str, Any]:
        # Keep a valid strategic request deterministic and off the LLM path.
        return {
            "insight": "No KPI values were requested by this boundary test.",
            "key_takeaways": [],
            "grounding": [],
            "is_fallback": True,
        }

    async def run_inline(function: Any, /, *args: Any, **kwargs: Any) -> Any:
        # The route's thread offload is orthogonal to scope validation and can
        # outlive a fresh asyncio.run loop under the suite's watchdog.
        return function(*args, **kwargs)

    def recording_build_grounding(
        brand: str, region: str | None, metas: list[Any], results: list[Any]
    ) -> dict[str, Any]:
        insight_grounding_regions.append(region)
        return real_build_grounding(brand, region, metas, results)

    def recording_cache_key(namespace: str, scope: str, payload: dict[str, Any]) -> str:
        assert namespace == "home-kpis"
        assert payload
        insight_cache_scopes.append(scope)
        return "region-ingress-cache-key"

    def get_strategic_calculator() -> _RecordingCalculator:
        strategic_calculator_requests.append(True)
        return calculator

    monkeypatch.setattr(kpi_history, "get_kpi_history_repository", get_history_repository)
    # home_kpi_insight imports this function inside its worker closure, so the
    # module attribute (not FastAPI's captured dependency) is its seam.
    monkeypatch.setattr("src.api.routes.kpi.get_kpi_calculator", get_strategic_calculator)
    monkeypatch.setattr(insights_strategic.home_kpi, "build_grounding", recording_build_grounding)
    monkeypatch.setattr(insights_strategic, "cache_key", recording_cache_key)
    monkeypatch.setattr(insights_strategic, "cache_get", cached_insight)
    monkeypatch.setattr(insights_strategic.asyncio, "to_thread", run_inline)
    monkeypatch.setattr(
        insights_strategic.data_constraint_context,
        "build_constraint_context",
        lambda _brand, _metas: "boundary-test context",
    )
    monkeypatch.setattr(
        insights_strategic.data_constraint_context,
        "build_mitigation_playbook",
        lambda: None,
    )

    return _Harness(
        calculator,
        history,
        insight_grounding_regions,
        insight_cache_scopes,
        strategic_calculator_requests,
    )


def _assert_region_422(exc: HTTPException, supplied: object) -> None:
    text = str(exc.detail)
    assert exc.status_code == 422, text
    assert "region" in text.lower()
    # String inputs should be echoed so the client can identify the rejected
    # value; non-strings need only be identified as a region type violation.
    if isinstance(supplied, str) and supplied.strip():
        assert supplied.strip().lower() in text.lower()
        for known_region in ("northeast", "south", "midwest", "west"):
            assert known_region in text.lower()


def _get_value(harness: _Harness, region: str | None) -> Any:
    return asyncio.run(
        kpi_routes.get_kpi_value(
            kpi_id=_KPI_ID,
            use_cache=True,
            force_refresh=False,
            brand=None,
            region=region,
            segment=None,
            therapy_line=None,
            biologic=None,
            ige_tier=None,
            calculator=harness.calculator,  # type: ignore[arg-type]
        )
    )


def _get_history(region: str | None) -> Any:
    return asyncio.run(
        kpi_routes.get_kpi_history(
            kpi_id=_KPI_ID,
            brand=None,
            region=region,
            start_date=None,
            end_date=None,
        )
    )


def _calculate(harness: _Harness, context: dict[str, Any]) -> Any:
    return asyncio.run(
        kpi_routes.calculate_kpi(
            request=KPICalculationRequest(kpi_id=_KPI_ID, context=context),
            calculator=harness.calculator,  # type: ignore[arg-type]
            user=_USER,
        )
    )


def _batch(harness: _Harness, context: dict[str, Any]) -> Any:
    return asyncio.run(
        kpi_routes.calculate_batch(
            request=BatchKPICalculationRequest(kpi_ids=[_KPI_ID], context=context),
            calculator=harness.calculator,  # type: ignore[arg-type]
            user=_USER,
        )
    )


def _home(region: str | None) -> Any:
    return asyncio.run(home_kpi_insight(HomeKpiInsightRequest(brand="All", region=region), _USER))


# ---------------------------------------------------------------------------
# GET /api/kpis/{kpi_id}: live calculator/RPC predicate
# ---------------------------------------------------------------------------


def test_value_canonicalizes_a_padded_region_before_calculation(harness: _Harness) -> None:
    _get_value(harness, " West ")

    assert harness.calculator.calculate_calls[0]["context"] == {"region": "west"}


def test_value_exact_empty_region_means_global(harness: _Harness) -> None:
    _get_value(harness, "")

    assert harness.calculator.calculate_calls[0]["context"] == {}


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   "])
def test_value_rejects_unknown_or_ambiguous_region_without_calculation(
    harness: _Harness, region: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _get_value(harness, region)

    _assert_region_422(exc_info.value, region)
    assert harness.calculator.calculate_calls == []


# ---------------------------------------------------------------------------
# GET /api/kpis/{kpi_id}/history: materialized-history predicate
# ---------------------------------------------------------------------------


def test_history_canonicalizes_a_padded_alias_before_repository_read(harness: _Harness) -> None:
    response = _get_history(" Pacific ")

    assert harness.history.calls[0]["region"] == "west"
    assert response.region == "west"


def test_history_exact_empty_region_means_global(harness: _Harness) -> None:
    response = _get_history("")

    assert harness.history.calls[0]["region"] is None
    assert response.region == ""


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   "])
def test_history_rejects_unknown_or_ambiguous_region_without_repository_read(
    harness: _Harness, region: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _get_history(region)

    _assert_region_422(exc_info.value, region)
    assert harness.history.calls == []


# ---------------------------------------------------------------------------
# POST /api/kpis/calculate: typed context plus caller-controlled ``extra``
# ---------------------------------------------------------------------------


def test_calculate_canonicalizes_typed_region_after_context_merge(harness: _Harness) -> None:
    _calculate(harness, {"region": " Northeast "})

    assert harness.calculator.calculate_calls[0]["context"] == {"region": "northeast"}


def test_calculate_canonicalizes_extra_region_that_overrides_typed_region(
    harness: _Harness,
) -> None:
    _calculate(harness, {"region": "south", "extra": {"region": " Pacific "}})

    assert harness.calculator.calculate_calls[0]["context"] == {"region": "west"}


def test_calculate_final_exact_empty_region_means_global(harness: _Harness) -> None:
    _calculate(harness, {"region": "south", "extra": {"region": ""}})

    assert harness.calculator.calculate_calls[0]["context"] == {}


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   ", 42])
def test_calculate_rejects_invalid_final_extra_region_without_calculation(
    harness: _Harness, region: object
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _calculate(harness, {"extra": {"region": region}})

    _assert_region_422(exc_info.value, region)
    assert harness.calculator.calculate_calls == []


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   "])
def test_calculate_rejects_invalid_typed_region_without_calculation(
    harness: _Harness, region: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _calculate(harness, {"region": region})

    _assert_region_422(exc_info.value, region)
    assert harness.calculator.calculate_calls == []


# ---------------------------------------------------------------------------
# POST /api/kpis/batch: same final-merged-context contract as single calculate
# ---------------------------------------------------------------------------


def test_batch_canonicalizes_a_padded_region_before_calculation(harness: _Harness) -> None:
    _batch(harness, {"region": " North East "})

    assert harness.calculator.batch_calls[0]["context"] == {"region": "northeast"}


def test_batch_canonicalizes_extra_region_that_overrides_typed_region(harness: _Harness) -> None:
    _batch(harness, {"region": "south", "extra": {"region": " western "}})

    assert harness.calculator.batch_calls[0]["context"] == {"region": "west"}


def test_batch_final_exact_empty_region_means_global(harness: _Harness) -> None:
    _batch(harness, {"region": "south", "extra": {"region": ""}})

    assert harness.calculator.batch_calls[0]["context"] == {}


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   ", 42])
def test_batch_rejects_invalid_final_extra_region_without_calculation(
    harness: _Harness, region: object
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _batch(harness, {"extra": {"region": region}})

    _assert_region_422(exc_info.value, region)
    assert harness.calculator.batch_calls == []


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   "])
def test_batch_rejects_invalid_typed_region_without_calculation(
    harness: _Harness, region: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _batch(harness, {"region": region})

    _assert_region_422(exc_info.value, region)
    assert harness.calculator.batch_calls == []


# ---------------------------------------------------------------------------
# POST /api/insights/home-kpis: indirect batch-calculator ingress
# ---------------------------------------------------------------------------


def test_home_kpi_insight_canonicalizes_padded_alias_before_batch(harness: _Harness) -> None:
    _home(" Pacific ")

    assert harness.calculator.batch_calls[0]["context"] == {"region": "west"}
    assert harness.insight_grounding_regions == ["west"]
    assert harness.insight_cache_scopes == ["All:west"]
    assert harness.strategic_calculator_requests == [True]


def test_home_kpi_insight_exact_empty_region_means_global(harness: _Harness) -> None:
    _home("")

    assert harness.calculator.batch_calls[0]["context"] == {}
    assert harness.insight_grounding_regions == [None]
    assert harness.insight_cache_scopes == ["All:all-us"]
    assert harness.strategic_calculator_requests == [True]


@pytest.mark.parametrize("region", ["Atlantis", "East Coast", "   "])
def test_home_kpi_insight_rejects_invalid_region_before_batch(
    harness: _Harness, region: str
) -> None:
    with pytest.raises(HTTPException) as exc_info:
        _home(region)

    _assert_region_422(exc_info.value, region)
    assert harness.calculator.batch_calls == []
    assert harness.strategic_calculator_requests == []


def test_home_kpi_insight_schema_rejects_non_string_region_before_batch(
    harness: _Harness,
) -> None:
    with pytest.raises(ValidationError, match="region"):
        HomeKpiInsightRequest(brand="All", region=42)  # type: ignore[arg-type]

    assert harness.calculator.batch_calls == []
    assert harness.strategic_calculator_requests == []
