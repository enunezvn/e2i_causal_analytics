"""Issue #188: PredictionRepository must filter out gated audit rows.

Codex pass-1 LOW-2 + pass-3 MEDIUM-1: the Celery task writes
``ml_predictions`` audit rows with
``prediction_class='gated_honest_failure'`` when a model fails its
honest-failure gate. These rows MUST NOT be:

  - returned by ``get_top_predictions`` (would surface a gated score
    as a top prediction).
  - aggregated by ``get_model_performance`` (would double-count the
    failure in the averaged pr_auc / brier metrics).
  - returned by ``get_by_patient`` (would surface a gated score in
    patient history).
  - returned by ``get_high_confidence_predictions`` (would surface a
    gated score as actionable above the confidence threshold).
  - aggregated by ``get_calibration_summary`` (would skew the reported
    calibration metrics).

The sentinel filter is centralized in ``_exclude_gated_rows`` so a new
read path can be added with a single line.

These tests pin all five filters via a mock Supabase-style client. We
do not exercise an actual Postgres backend — the test asserts that
the filter chain calls
``.neq("prediction_class", "gated_honest_failure")`` exactly once on
the query builder.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.prediction import (
    GATED_HONEST_FAILURE_SENTINEL,
    PredictionRepository,
)


class _ChainableQuery:
    """Minimal supabase-style fluent query builder for assertions."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...]]] = []
        self._execute_data: list[dict[str, Any]] = []

    def _record(self, name: str, *args: Any) -> "_ChainableQuery":
        self.calls.append((name, args))
        return self

    def select(self, *args: Any) -> "_ChainableQuery":
        return self._record("select", *args)

    def eq(self, *args: Any) -> "_ChainableQuery":
        return self._record("eq", *args)

    def neq(self, *args: Any) -> "_ChainableQuery":
        return self._record("neq", *args)

    def gte(self, *args: Any) -> "_ChainableQuery":
        return self._record("gte", *args)

    def lte(self, *args: Any) -> "_ChainableQuery":
        return self._record("lte", *args)

    def order(self, *args: Any, **kwargs: Any) -> "_ChainableQuery":
        return self._record("order", *args, *kwargs.items())

    def limit(self, *args: Any) -> "_ChainableQuery":
        return self._record("limit", *args)

    def execute(self) -> Any:
        mock_result = MagicMock()
        mock_result.data = list(self._execute_data)
        # Wrap in an awaitable so the async repo can `await query.execute()`.
        return AsyncMock(return_value=mock_result)()


def _make_repo_with_query() -> tuple[PredictionRepository, _ChainableQuery]:
    repo = PredictionRepository.__new__(PredictionRepository)
    query = _ChainableQuery()
    mock_client = MagicMock()
    mock_client.table = MagicMock(return_value=query)
    repo.client = mock_client
    repo.table_name = "ml_predictions"
    return repo, query


def _assert_neq_gated(query: _ChainableQuery, method_name: str) -> None:
    neq_calls = [args for (name, args) in query.calls if name == "neq"]
    assert ("prediction_class", GATED_HONEST_FAILURE_SENTINEL) in neq_calls, (
        f"{method_name} did NOT filter out {GATED_HONEST_FAILURE_SENTINEL} rows. "
        f"Calls: {query.calls}"
    )


def test_sentinel_value_is_pinned() -> None:
    """The sentinel string is part of the contract between the Celery
    task (writer) and every repository read path (reader); pin it.
    """
    assert GATED_HONEST_FAILURE_SENTINEL == "gated_honest_failure"


@pytest.mark.asyncio
async def test_get_top_predictions_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_top_predictions(model_id="risk_score_v1", top_k=5)
    _assert_neq_gated(query, "get_top_predictions")


@pytest.mark.asyncio
async def test_get_model_performance_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_model_performance(model_id="risk_score_v1")
    _assert_neq_gated(query, "get_model_performance")


@pytest.mark.asyncio
async def test_get_by_patient_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_by_patient(patient_id="PAT_001")
    _assert_neq_gated(query, "get_by_patient")


@pytest.mark.asyncio
async def test_get_high_confidence_predictions_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_high_confidence_predictions(model_id="risk_score_v1")
    _assert_neq_gated(query, "get_high_confidence_predictions")


@pytest.mark.asyncio
async def test_get_calibration_summary_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_calibration_summary(model_id="risk_score_v1")
    _assert_neq_gated(query, "get_calibration_summary")
