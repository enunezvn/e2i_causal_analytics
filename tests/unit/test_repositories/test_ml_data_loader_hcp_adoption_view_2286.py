"""#2286: the loader allowlist admits the HCP-adoption goldstd VIEW, and nothing wider.

The allowlist is deliberate (#894 fixed a wrong entry in it). The design (#2287) puts
the two-table goldstd frame behind ONE view, migration 162, so the entry added is the
view, never the raw tables: a contract naming ``hcp_brand_adoption`` would load a frame
with no covariates, and ``hcp_profiles`` carries no label. The view exposes
``is_synthetic``, so it joins the provenance SSOT too: an UNPINNED load must default-
exclude synthetic rows exactly as every other tagged relation does.

The client below is a recording transport, not a stand-in for the loader: the real
``MLDataLoader`` builds every query.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import pytest

from src.repositories.ml_data_loader import ML_TABLES, PROVENANCE_TAGGED_TABLES, MLDataLoader

VIEW = "hcp_adoption_goldstd_v"


class _Query:
    def __init__(self, calls: List[Tuple[str, Any]]):
        self.calls = calls

    def __getattr__(self, name: str):
        def _record(*args: Any, **kwargs: Any) -> "_Query":
            self.calls.append((name, args))
            return self

        return _record

    def execute(self) -> Any:
        return type("R", (), {"data": [{"adopted": 1, "data_split": "train"}], "count": 1})()


class _Client:
    def __init__(self) -> None:
        self.calls: List[Tuple[str, Any]] = []
        self.tables: List[str] = []

    def table(self, name: str) -> _Query:
        self.tables.append(name)
        return _Query(self.calls)


def _eqs(client: _Client) -> List[tuple]:
    return [args for name, args in client.calls if name == "eq"]


def test_the_view_is_allowlisted_and_the_raw_tables_are_not() -> None:
    assert VIEW in ML_TABLES
    assert "hcp_brand_adoption" not in ML_TABLES
    assert "hcp_profiles" not in ML_TABLES


@pytest.mark.asyncio
@pytest.mark.parametrize("unlisted", ["hcp_brand_adoption", "hcp_profiles", "hcp_adoption_v"])
async def test_every_loader_entry_point_still_rejects_an_unlisted_relation(unlisted: str) -> None:
    loader = MLDataLoader(supabase_client=_Client())
    with pytest.raises(ValueError, match="not supported"):
        await loader.load_table_sample(unlisted)
    with pytest.raises(ValueError, match="not supported"):
        await loader.has_column(unlisted, "data_split")
    with pytest.raises(ValueError, match="not supported"):
        await loader.load_for_training(unlisted)


@pytest.mark.asyncio
async def test_the_view_loads_through_every_entry_point() -> None:
    client = _Client()
    loader = MLDataLoader(supabase_client=client)
    df = await loader.load_table_sample(VIEW, columns=["adopted"], include_synthetic=True)
    assert list(df.columns) == ["adopted", "data_split"]
    assert await loader.has_column(VIEW, "data_split") is True
    await loader.load_for_training(VIEW, include_synthetic=True)
    assert set(client.tables) == {VIEW}


def test_the_view_is_provenance_tagged() -> None:
    assert VIEW in PROVENANCE_TAGGED_TABLES


@pytest.mark.asyncio
async def test_an_unpinned_view_load_default_excludes_synthetic_rows() -> None:
    client = _Client()
    await MLDataLoader(supabase_client=client).load_table_sample(VIEW, columns=["adopted"])
    assert ("is_synthetic", False) in _eqs(client)


@pytest.mark.asyncio
async def test_a_contract_pinned_view_load_keeps_exactly_the_contract_predicate() -> None:
    client = _Client()
    await MLDataLoader(supabase_client=client).load_table_sample(
        VIEW,
        filters={"brand": "Kisqali", "is_synthetic": True},
        columns=["adopted"],
        include_synthetic=True,
    )
    assert _eqs(client) == [("brand", "Kisqali"), ("is_synthetic", True)]
