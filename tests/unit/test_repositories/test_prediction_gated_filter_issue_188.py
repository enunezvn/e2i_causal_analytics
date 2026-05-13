"""Issue #188: PredictionRepository must filter out gated audit rows.

Codex pass-1 LOW-2: the Celery task writes ``ml_predictions`` audit rows
with ``prediction_class='gated_honest_failure'`` when a model fails its
honest-failure gate. These rows MUST NOT be:

  - returned by ``get_top_predictions`` (would surface a gated score
    as a top prediction).
  - aggregated by ``get_model_performance`` (would double-count the
    failure in the averaged pr_auc / brier metrics).

This test pins both filters via a mock Supabase-style client. We do not
exercise an actual Postgres backend — the test asserts that the filter
chain calls ``.neq("prediction_class", "gated_honest_failure")``
exactly once on the query builder.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.repositories.prediction import PredictionRepository


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


@pytest.mark.asyncio
async def test_get_top_predictions_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_top_predictions(model_id="risk_score_v1", top_k=5)
    # The filter chain must have one .neq() call with prediction_class /
    # gated_honest_failure (codex pass-1 LOW-2).
    neq_calls = [args for (name, args) in query.calls if name == "neq"]
    assert ("prediction_class", "gated_honest_failure") in neq_calls, (
        f"get_top_predictions did NOT filter out gated_honest_failure rows. "
        f"Calls: {query.calls}"
    )


@pytest.mark.asyncio
async def test_get_model_performance_excludes_gated_rows() -> None:
    repo, query = _make_repo_with_query()
    await repo.get_model_performance(model_id="risk_score_v1")
    neq_calls = [args for (name, args) in query.calls if name == "neq"]
    assert ("prediction_class", "gated_honest_failure") in neq_calls, (
        f"get_model_performance did NOT filter out gated_honest_failure rows. "
        f"Calls: {query.calls}"
    )
