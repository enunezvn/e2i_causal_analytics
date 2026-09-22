"""#2207 follow-up (owner decision 2026-09-22): the data-prep Feast gate measures the
freshness of the Feast views SOURCED FROM the run's data source, through the feast-free
#559 recency probe (``MAX(<raw timestamp column>)`` over PostgREST), instead of probing a
``feature_analyzer_<experiment_id>`` view that exists in no registry.

These tests pin the probe module: the inverse view map, the age/threshold arithmetic,
"unverifiable is not fresh" (#556), and that a source with no Feast view is reported as
not Feast-backed (``fresh=None``) rather than stale.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.feature_store.feast_source_freshness import (
    describe_source,
    probe_source_freshness,
    source_table_of,
)
from src.feature_store.feast_views import (
    FEAST_FEATURE_VIEW_SOURCE_TABLES,
    FEAST_SOURCE_TABLE_TIMESTAMP_COLUMNS,
    feast_views_for_source_table,
)

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# inverse map
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_inverse_map_covers_every_view_exactly_once():
    seen = []
    for table in set(FEAST_FEATURE_VIEW_SOURCE_TABLES.values()):
        seen.extend(feast_views_for_source_table(table))
    assert sorted(seen) == sorted(FEAST_FEATURE_VIEW_SOURCE_TABLES)


@pytest.mark.unit
def test_business_metrics_sources_three_views_and_unmapped_table_none():
    assert feast_views_for_source_table("business_metrics") == [
        "hcp_conversion_features",
        "hcp_engagement_features",
        "market_dynamics_features",
    ]
    assert feast_views_for_source_table("hcp_features") == []
    assert feast_views_for_source_table("") == []


@pytest.mark.unit
def test_every_source_table_has_a_timestamp_column():
    """A mapped table with no timestamp column would make its views permanently
    unverifiable — the #559 map and the view map must agree."""
    for table in set(FEAST_FEATURE_VIEW_SOURCE_TABLES.values()):
        assert table in FEAST_SOURCE_TABLE_TIMESTAMP_COLUMNS, table


# ---------------------------------------------------------------------------
# data_source shapes
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_source_table_of_and_describe_source_shapes():
    assert source_table_of("patient_journeys") == "patient_journeys"
    assert source_table_of({"type": "file_dir", "path": "x"}) is None
    assert source_table_of({"type": "files", "paths": {}}) is None
    assert source_table_of(None) is None
    assert describe_source("patient_journeys") == "table"
    assert describe_source({"type": "file_dir"}) == "file_dir"
    assert describe_source({"type": "files"}) == "files"
    assert describe_source(None) == "none"


# ---------------------------------------------------------------------------
# probe
# ---------------------------------------------------------------------------


def _recency(value):
    async def _q(table):
        return value

    return _q


@pytest.mark.unit
@pytest.mark.asyncio
async def test_fresh_when_source_table_recency_is_within_threshold():
    result = await probe_source_freshness(
        "business_metrics",
        max_staleness_hours=24.0,
        now=NOW,
        recency_query=_recency(NOW - timedelta(hours=3)),
    )
    assert result["fresh"] is True
    assert result["feast_backed"] is True
    assert result["source_table"] == "business_metrics"
    assert result["feature_views"] == feast_views_for_source_table("business_metrics")
    assert result["stale_features"] == []
    assert result["age_hours"] == pytest.approx(3.0)
    assert result["last_updated"] == (NOW - timedelta(hours=3)).isoformat()
    assert result["feature_ages"] == {v: pytest.approx(3.0) for v in result["feature_views"]}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_stale_when_recency_is_older_than_threshold():
    result = await probe_source_freshness(
        "triggers",
        max_staleness_hours=24.0,
        now=NOW,
        recency_query=_recency(NOW - timedelta(days=30)),
    )
    assert result["fresh"] is False
    assert result["feast_backed"] is True
    assert set(result["stale_features"]) == set(feast_views_for_source_table("triggers"))
    assert any("triggers" in r and "stale" in r.lower() for r in result["recommendations"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unverifiable_recency_is_not_fresh():
    """#556: no recency signal → unverifiable → not fresh (never fabricated)."""
    result = await probe_source_freshness(
        "hcp_profiles", max_staleness_hours=24.0, now=NOW, recency_query=_recency(None)
    )
    assert result["fresh"] is False
    assert result["last_updated"] is None
    assert result["age_hours"] is None
    assert any("unverifiable" in r for r in result["recommendations"])


@pytest.mark.unit
@pytest.mark.asyncio
async def test_unmapped_table_is_not_feast_backed_and_queries_nothing():
    calls = []

    async def _q(table):
        calls.append(table)
        return NOW

    result = await probe_source_freshness(
        "hcp_features", max_staleness_hours=24.0, now=NOW, recency_query=_q
    )
    assert calls == []
    assert result["fresh"] is None
    assert result["feast_backed"] is False
    assert result["feature_views"] == []
    assert any("not Feast-backed" in r for r in result["recommendations"])


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "data_source",
    [{"type": "file_dir", "path": "data/rwd/optum"}, {"type": "files", "paths": {"train": "a"}}],
)
async def test_file_sources_are_not_feast_backed(data_source):
    calls = []

    async def _q(table):
        calls.append(table)
        return NOW

    result = await probe_source_freshness(
        data_source, max_staleness_hours=24.0, now=NOW, recency_query=_q
    )
    assert calls == []
    assert result["fresh"] is None
    assert result["feast_backed"] is False
    assert result["source_kind"] == data_source["type"]
