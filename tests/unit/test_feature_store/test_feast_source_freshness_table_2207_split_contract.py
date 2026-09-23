"""#2207 split contract: the Feast freshness probe resolves a table cohort dict's table.

A run whose ``data_source`` is ``{"type": "table", "table": "patient_journeys", ...}``
loads from that table exactly as the bare string does, so it is Feast-backed by the same
views; reading it as an opaque ``dict`` reported "not Feast-backed" for the same table.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.feature_store.feast_source_freshness import (
    describe_source,
    probe_source_freshness,
    source_table_of,
)

_TABLE_DICT = {"type": "table", "table": "patient_journeys", "filters": {"brand": "Kisqali"}}


def test_table_dict_resolves_to_its_table() -> None:
    assert source_table_of(_TABLE_DICT) == "patient_journeys"
    assert describe_source(_TABLE_DICT) == "table"
    assert source_table_of({"type": "file_dir", "path": "/x"}) is None
    assert describe_source({"type": "file_dir", "path": "/x"}) == "file_dir"
    assert describe_source({"type": "s3"}) == "dict"


@pytest.mark.asyncio
async def test_table_dict_and_string_probe_identically() -> None:
    now = datetime(2026, 9, 23, tzinfo=timezone.utc)

    async def recency(table: str) -> datetime:
        assert table == "patient_journeys"
        return now - timedelta(hours=1)

    as_dict = await probe_source_freshness(_TABLE_DICT, now=now, recency_query=recency)
    as_str = await probe_source_freshness("patient_journeys", now=now, recency_query=recency)
    assert as_dict["feast_backed"] is True
    assert as_dict["source_table"] == "patient_journeys"
    for key in ("fresh", "feast_backed", "feature_views", "source_table", "stale_features"):
        assert as_dict[key] == as_str[key], key
