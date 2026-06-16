"""Regression tests for ABResultsRepository.get_fidelity_comparisons(limit=...).

The ``GET /experiments/{id}/fidelity`` route forwards its ``limit`` query param
to the repository (``repo.get_fidelity_comparisons(id, limit=limit)``). The
repository method did NOT accept a ``limit`` kwarg, so the call raised
``TypeError: ... got an unexpected keyword argument 'limit'`` → HTTP 500, and the
Experiments page "Digital Twin" tab errored on every experiment selection (a
``# type: ignore[call-arg]`` had masked it from mypy). These tests pin that the
method accepts ``limit`` and applies it to the query.
"""

from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from src.repositories.ab_results import ABResultsRepository


def _chainable_client(rows):
    """Build a MagicMock supabase client whose query chain records calls."""
    chain = MagicMock()
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.limit.return_value = chain
    chain.execute.return_value = MagicMock(data=rows)
    client = MagicMock()
    client.table.return_value = chain
    return client, chain


@pytest.mark.asyncio
async def test_accepts_limit_kwarg_and_applies_it():
    client, chain = _chainable_client([])
    repo = ABResultsRepository(supabase_client=client)

    # Before the fix this raised TypeError (unexpected kwarg 'limit').
    result = await repo.get_fidelity_comparisons(uuid4(), limit=5)

    assert result == []
    chain.limit.assert_called_once_with(5)
    client.table.assert_called_once_with("ab_fidelity_comparisons")


@pytest.mark.asyncio
async def test_default_limit_when_omitted():
    client, chain = _chainable_client([])
    repo = ABResultsRepository(supabase_client=client)

    await repo.get_fidelity_comparisons(uuid4())

    chain.limit.assert_called_once_with(10)


@pytest.mark.asyncio
async def test_no_client_returns_empty_without_query():
    repo = ABResultsRepository(supabase_client=None)
    # Force the no-client branch deterministically (avoid env-bound get_supabase).
    repo.client = None

    assert await repo.get_fidelity_comparisons(uuid4(), limit=5) == []
