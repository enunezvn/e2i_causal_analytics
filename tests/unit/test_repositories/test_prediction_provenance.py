"""Shard 07: PredictionRepository must default-exclude synthetic ml_predictions.

``ml_predictions`` carries the ``is_synthetic`` provenance column (migration
063). Every real-mode actionable read path MUST append
``.eq("is_synthetic", False)`` and MUST NOT when ``include_synthetic=True``.

These tests pin the predicate on each read path via a recording supabase-style
query builder (mirrors ``test_prediction_gated_filter_issue_188.py``). They do
not exercise a real Postgres backend — they assert the filter chain.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.prediction import PredictionRepository


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

    def neq(self, *a: Any) -> "_ChainableQuery":
        return self._record("neq", *a)

    def or_(self, *a: Any) -> "_ChainableQuery":
        return self._record("or_", *a)

    def gte(self, *a: Any) -> "_ChainableQuery":
        return self._record("gte", *a)

    def lte(self, *a: Any) -> "_ChainableQuery":
        return self._record("lte", *a)

    def order(self, *a: Any, **k: Any) -> "_ChainableQuery":
        return self._record("order", *a, *k.items())

    def limit(self, *a: Any) -> "_ChainableQuery":
        return self._record("limit", *a)

    def offset(self, *a: Any) -> "_ChainableQuery":
        return self._record("offset", *a)

    def execute(self) -> Any:
        result = MagicMock()
        result.data = list(self._execute_data)
        return AsyncMock(return_value=result)()


def _make_repo() -> tuple[PredictionRepository, _ChainableQuery]:
    repo = PredictionRepository.__new__(PredictionRepository)
    query = _ChainableQuery()
    mock_client = MagicMock()
    mock_client.table = MagicMock(return_value=query)
    repo.client = mock_client
    repo.table_name = "ml_predictions"
    return repo, query


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
    assert PredictionRepository.HAS_PROVENANCE is True


@pytest.mark.asyncio
async def test_get_top_predictions_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_top_predictions(model_id="risk_score_v1", top_k=5)
    _assert_excludes_synthetic(query, "get_top_predictions")


@pytest.mark.asyncio
async def test_get_top_predictions_opt_in() -> None:
    repo, query = _make_repo()
    await repo.get_top_predictions(model_id="risk_score_v1", top_k=5, include_synthetic=True)
    _assert_no_synthetic_predicate(query, "get_top_predictions")


@pytest.mark.asyncio
async def test_get_model_performance_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_model_performance(model_id="risk_score_v1")
    _assert_excludes_synthetic(query, "get_model_performance")


@pytest.mark.asyncio
async def test_get_by_patient_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_by_patient(patient_id="PAT_001")
    _assert_excludes_synthetic(query, "get_by_patient")


@pytest.mark.asyncio
async def test_get_high_confidence_predictions_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_high_confidence_predictions(model_id="risk_score_v1")
    _assert_excludes_synthetic(query, "get_high_confidence_predictions")


@pytest.mark.asyncio
async def test_get_calibration_summary_excludes_synthetic() -> None:
    repo, query = _make_repo()
    await repo.get_calibration_summary(model_id="risk_score_v1")
    _assert_excludes_synthetic(query, "get_calibration_summary")


@pytest.mark.asyncio
async def test_get_by_model_excludes_synthetic_via_get_many() -> None:
    # get_by_model delegates to the inherited get_many, which gates on
    # HAS_PROVENANCE — assert the predicate still lands.
    repo, query = _make_repo()
    await repo.get_by_model(model_id="risk_score_v1")
    _assert_excludes_synthetic(query, "get_by_model")
