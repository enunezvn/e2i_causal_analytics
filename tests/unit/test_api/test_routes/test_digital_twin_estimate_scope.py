"""#2053: a stored simulation must say what its effect was estimated ON.

#2023 scoped a region-filtered simulation's ATE, CI and recommendation to the filtered regions,
but twin_simulations had no column for that scope, so a stored row read back scope-less: on the
list, history and detail reads a region-scoped row was indistinguishable from a cohort-wide one,
and the cohort-wide comparator was gone. Migration ml/042 adds the columns, save_simulation writes
them and the reads report ``estimate_scope``.

A row written before ml/042 has NULL scope and must read back as UNKNOWN: never as cohort-wide,
and never derived from ``population_filters.regions`` (a pre-#2023 row carries the same regions
filter over a cohort-wide ATE, so deriving would stamp a false "estimated on northeast").
"""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

ADMIN = {"app_metadata": {"role": "admin"}}


class _CapturingClient:
    """Async Supabase stand-in that records the row handed to insert()."""

    def __init__(self, sink):
        self._sink = sink

    def table(self, name):
        self._sink["table"] = name
        return self

    def insert(self, row):
        self._sink["row"] = row
        return self

    async def execute(self):
        return SimpleNamespace(data=[self._sink["row"]])


def _result(**scope):
    from src.digital_twin.models.simulation_models import (
        InterventionConfig,
        PopulationFilter,
        SimulationRecommendation,
        SimulationResult,
    )

    regions = scope.get("target_regions", [])
    return SimulationResult(
        model_id=uuid4(),
        intervention_config=InterventionConfig(intervention_type="email_campaign"),
        population_filters=PopulationFilter(regions=list(regions)),
        twin_count=600,
        simulated_ate=0.2585,
        simulated_ci_lower=0.1995,
        simulated_ci_upper=0.3174,
        simulated_std_error=0.03,
        recommendation=SimulationRecommendation.DEPLOY,
        recommendation_rationale="r",
        simulation_confidence=0.8,
        execution_time_ms=1,
        **scope,
    )


def _saved_row(result):
    from src.digital_twin.twin_repository import SimulationRepository

    sink = {}
    asyncio.run(
        SimulationRepository(supabase_client=_CapturingClient(sink)).save_simulation(
            result, "Kisqali"
        )
    )
    return sink["row"]


def _row(**overrides):
    """A twin_simulations row as ``select *`` returns it, before the scope columns."""
    row = {
        "simulation_id": str(uuid4()),
        "model_id": str(uuid4()),
        "intervention_type": "email_campaign",
        "intervention_config": {},
        "brand": "Kisqali",
        "twin_count": 600,
        "simulated_ate": 0.2585,
        "simulated_ci_lower": 0.1995,
        "simulated_ci_upper": 0.3174,
        "simulated_std_error": 0.03,
        "recommendation": "deploy",
        "recommendation_rationale": "r",
        "simulation_confidence": 0.8,
        "simulation_status": "completed",
        "execution_time_ms": 1,
        "created_at": datetime.now(timezone.utc),
        "population_filters": {},
        "effect_heterogeneity": {},
    }
    row.update(overrides)
    return row


def _read_detail(row):
    from src.api.routes import digital_twin as dt

    repo = SimpleNamespace(get_simulation=AsyncMock(return_value=row))
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        return asyncio.run(dt.get_simulation(row["simulation_id"], user=ADMIN))


def _read_list_item(row):
    from src.api.routes import digital_twin as dt

    repo = SimpleNamespace(
        simulations=SimpleNamespace(list_simulations=AsyncMock(return_value=[row]))
    )
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        resp = asyncio.run(
            dt.list_simulations(
                brand=None, model_id=None, status=None, page=1, page_size=20, user=ADMIN
            )
        )
    return resp.simulations[0]


def _read_history_item(row):
    from src.api.routes import digital_twin as dt

    repo = SimpleNamespace(
        simulations=SimpleNamespace(list_simulations=AsyncMock(return_value=[row]))
    )
    with patch.object(dt, "_get_twin_repo", AsyncMock(return_value=repo)):
        resp = asyncio.run(dt.get_simulation_history(brand=None, limit=20, offset=0, user=ADMIN))
    return resp.simulations[0]


# ---------------------------------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------------------------------


@pytest.mark.unit
def test_save_simulation_records_a_region_scope_and_its_cohort_comparator():
    row = _saved_row(
        _result(
            target_regions=["northeast"],
            cohort_ate=0.1352,
            cohort_ci_lower=0.0924,
            cohort_ci_upper=0.1781,
        )
    )
    assert row["effect_scope_regions"] == ["northeast"]
    assert row["cohort_ate"] == 0.1352
    assert row["cohort_ci_lower"] == 0.0924
    assert row["cohort_ci_upper"] == 0.1781


@pytest.mark.unit
def test_save_simulation_records_cohort_wide_as_empty_never_null():
    """NULL is reserved for "scope not recorded"; a new cohort-wide row must say cohort-wide."""
    row = _saved_row(_result())
    assert row["effect_scope_regions"] == []
    assert row["effect_scope_regions"] is not None
    assert row["cohort_ate"] is None
    assert row["cohort_ci_lower"] is None
    assert row["cohort_ci_upper"] is None


# ---------------------------------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------------------------------

_REGION_ROW = {
    "population_filters": {"regions": ["northeast"]},
    "effect_scope_regions": ["northeast"],
    "cohort_ate": 0.13523773,
    "cohort_ci_lower": 0.09241,
    "cohort_ci_upper": 0.17809,
}


@pytest.mark.unit
def test_detail_reports_a_stored_region_scope_and_the_cohort_comparator():
    detail = _read_detail(_row(**_REGION_ROW))
    assert detail.estimate_scope.value == "regions"
    assert detail.target_regions == ["northeast"]
    assert detail.cohort_effect == 0.1352
    assert detail.cohort_ci_lower == 0.0924
    assert detail.cohort_ci_upper == 0.1781


@pytest.mark.unit
@pytest.mark.parametrize("read", [_read_list_item, _read_history_item], ids=["list", "history"])
def test_list_views_report_a_stored_region_scope(read):
    item = read(_row(**_REGION_ROW))
    assert item.estimate_scope.value == "regions"
    assert item.target_regions == ["northeast"]


@pytest.mark.unit
def test_list_item_carries_the_cohort_comparator_next_to_the_scoped_ate():
    item = _read_list_item(_row(**_REGION_ROW))
    assert item.simulated_ate == 0.2585
    assert item.cohort_effect == 0.1352


@pytest.mark.unit
@pytest.mark.parametrize(
    "read", [_read_detail, _read_list_item, _read_history_item], ids=["detail", "list", "history"]
)
def test_a_stored_cohort_wide_row_reads_as_cohort(read):
    item = read(_row(effect_scope_regions=[]))
    assert item.estimate_scope.value == "cohort"
    assert item.target_regions == []


_LEGACY_ROWS = {
    # select * before ml/042 has no scope keys at all; after it, NULL columns.
    "no_scope_columns": _row(),
    "null_scope_columns": _row(
        effect_scope_regions=None, cohort_ate=None, cohort_ci_lower=None, cohort_ci_upper=None
    ),
    # A pre-#2023 region-filtered row: the regions filter is there, the ATE is cohort-wide.
    "regions_filter_but_no_scope": _row(population_filters={"regions": ["northeast"]}),
}


@pytest.mark.unit
@pytest.mark.parametrize("legacy", list(_LEGACY_ROWS), ids=list(_LEGACY_ROWS))
@pytest.mark.parametrize(
    "read", [_read_detail, _read_list_item, _read_history_item], ids=["detail", "list", "history"]
)
def test_a_row_without_a_recorded_scope_reads_as_unknown(read, legacy):
    item = read(dict(_LEGACY_ROWS[legacy]))
    assert item.estimate_scope.value == "unknown"
    # Not region-scoped, and not derived from population_filters.regions.
    assert item.target_regions == []
    if hasattr(item, "cohort_effect"):
        assert item.cohort_effect is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("population_filters", "expected"),
    [
        # The two live unknown-scope rows (2026-09-12) carry this shape.
        (
            {"deciles": [], "regions": ["midwest"], "specialties": [], "adoption_stages": []},
            ["midwest"],
        ),
        ({}, []),
        (None, []),
        ({"regions": None}, []),
        ({"regions": ["northeast", 7, None]}, ["northeast"]),
    ],
    ids=["regions", "no_filter", "null_filters", "null_regions", "non_string_entries"],
)
def test_history_item_carries_the_stored_regions_filter(population_filters, expected):
    """#2079: the history card needs the regions filter to tell an ambiguous region-filtered
    unknown-scope row from an unfiltered one, as the detail view does from population_filters.
    It is the stored filter, never a scope: target_regions stays empty for an unknown row."""
    item = _read_history_item(_row(population_filters=population_filters))
    assert item.filter_regions == expected
    assert item.estimate_scope.value == "unknown"
    assert item.target_regions == []
