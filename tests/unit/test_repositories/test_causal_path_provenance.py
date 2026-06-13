"""#893: CausalPathRepository must default-exclude synthetic causal_paths.

``causal_paths`` carries the ``is_synthetic`` provenance column (migration
063:18) and the synthetic loader stamps every loaded row. Until #893 the
repository never set ``HAS_PROVENANCE = True``, so ``BaseRepository.get_many``
skipped ``apply_provenance_filter`` entirely and the user-visible chat tool
``_query_causal_chains`` (src/api/routes/chatbot_tools.py) returned synthetic
causal paths as real insight (live DB at filing time: 250/250 synthetic).

These tests pin the predicate on each read path via a recording
supabase-style query builder (mirrors ``test_prediction_provenance.py``).
They do not exercise a real Postgres backend — they assert the filter chain.
The faithful live-DB proof lives in
``tests/integration/test_causal_paths_provenance_893.py``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.causal_path import CausalPathRepository


class _ChainableQuery:
    """supabase-style fluent builder that records ``.eq()`` predicates."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._execute_data: list[dict[str, Any]] = []

    def _record(self, name: str, *args: Any) -> "_ChainableQuery":
        self.calls.append((name, args))
        return self

    def select(self, *a: Any) -> "_ChainableQuery":
        return self._record("select", *a)

    def eq(self, *a: Any) -> "_ChainableQuery":
        return self._record("eq", *a)

    def limit(self, *a: Any) -> "_ChainableQuery":
        return self._record("limit", *a)

    def offset(self, *a: Any) -> "_ChainableQuery":
        return self._record("offset", *a)

    def execute(self) -> Any:
        result = MagicMock()
        result.data = list(self._execute_data)
        return AsyncMock(return_value=result)()


def _make_repo() -> tuple[CausalPathRepository, _ChainableQuery]:
    query = _ChainableQuery()
    mock_client = MagicMock()
    mock_client.table = MagicMock(return_value=query)
    return CausalPathRepository(supabase_client=mock_client), query


def _assert_excludes_synthetic(query: _ChainableQuery, method: str) -> None:
    eq_calls = [args for (name, args) in query.calls if name == "eq"]
    assert ("is_synthetic", False) in eq_calls, (
        f"{method} did not default-exclude synthetic rows. eq calls: {eq_calls}"
    )


def _assert_no_synthetic_predicate(query: _ChainableQuery, method: str) -> None:
    eq_calls = [args for (name, args) in query.calls if name == "eq"]
    assert ("is_synthetic", False) not in eq_calls, (
        f"{method} applied the provenance predicate under include_synthetic=True. "
        f"eq calls: {eq_calls}"
    )


def test_has_provenance_flag_is_set() -> None:
    assert CausalPathRepository.HAS_PROVENANCE is True


@pytest.mark.asyncio
async def test_get_many_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_many(filters={})
    _assert_excludes_synthetic(query, "get_many")


@pytest.mark.asyncio
async def test_get_many_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_many(filters={}, include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_many")


@pytest.mark.asyncio
async def test_get_by_id_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_by_id("some-id")
    _assert_excludes_synthetic(query, "get_by_id")


@pytest.mark.asyncio
async def test_get_by_id_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_by_id("some-id", include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_by_id")


@pytest.mark.asyncio
async def test_get_paths_for_cause_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_paths_for_cause(cause="hcp_engagement")
    _assert_excludes_synthetic(query, "get_paths_for_cause")


@pytest.mark.asyncio
async def test_get_paths_for_cause_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_paths_for_cause(cause="hcp_engagement", include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_paths_for_cause")


@pytest.mark.asyncio
async def test_get_paths_for_effect_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_paths_for_effect(effect="trx_growth")
    _assert_excludes_synthetic(query, "get_paths_for_effect")


@pytest.mark.asyncio
async def test_get_paths_for_effect_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_paths_for_effect(effect="trx_growth", include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_paths_for_effect")


@pytest.mark.asyncio
async def test_get_path_between_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_path_between(cause="hcp_engagement", effect="trx_growth")
    _assert_excludes_synthetic(query, "get_path_between")


@pytest.mark.asyncio
async def test_get_path_between_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_path_between(cause="hcp_engagement", effect="trx_growth", include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_path_between")


@pytest.mark.asyncio
async def test_get_by_brand_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_by_brand(brand="Kisqali")
    _assert_excludes_synthetic(query, "get_by_brand")


@pytest.mark.asyncio
async def test_get_by_brand_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_by_brand(brand="Kisqali", include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_by_brand")
