"""Proposed experiments read path on the twin repository (#2206, owner item C).

A proposed experiment is a COMPLETED twin simulation whose recommendation is
deploy or refine and which is not yet linked to an experiment
(``experiment_design_id IS NULL``). The store builds exactly that predicate; the
composite ``TwinRepository`` facade forwards it (the lane's facade-signature
trap: a store kwarg the facade does not forward is a prod TypeError that an
unconstrained AsyncMock never sees — so the facade is driven here against an
autospecced store).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, create_autospec

import pytest

from src.digital_twin.twin_repository import SimulationRepository, TwinRepository


def _chain(rows, count=None):
    """An async supabase query chain that records every filter call."""
    chain = MagicMock()
    for name in ("select", "eq", "in_", "is_", "order", "limit"):
        getattr(chain, name).return_value = chain
    chain.not_.is_.return_value = chain
    chain.execute = AsyncMock(return_value=MagicMock(data=rows, count=count))
    client = MagicMock()
    client.table.return_value = chain
    return client, chain


@pytest.mark.asyncio
async def test_list_proposed_filters_completed_deploy_or_refine_and_unlinked():
    client, chain = _chain([{"simulation_id": "s1"}])
    repo = SimulationRepository(supabase_client=client)

    rows = await repo.list_proposed(brand="Kisqali", limit=50)

    assert rows == [{"simulation_id": "s1"}]
    client.table.assert_called_once_with("twin_simulations")
    chain.eq.assert_any_call("simulation_status", "completed")
    chain.in_.assert_called_once_with("recommendation", ["deploy", "refine"])
    chain.is_.assert_called_once_with("experiment_design_id", "null")
    chain.eq.assert_any_call("brand", "Kisqali")
    chain.limit.assert_called_once_with(50)


@pytest.mark.asyncio
async def test_list_proposed_without_brand_is_unscoped_at_the_store():
    """Brand scoping is the ROUTE's job (H11 grant); the store lists every brand
    when asked to, so the admin envelope counts are whole."""
    client, chain = _chain([])
    repo = SimulationRepository(supabase_client=client)

    await repo.list_proposed(brand=None)

    assert ("brand", None) not in [c.args for c in chain.eq.call_args_list]
    assert not any(c.args[0] == "brand" for c in chain.eq.call_args_list)


@pytest.mark.asyncio
async def test_count_linked_counts_completed_simulations_with_an_experiment():
    client, chain = _chain([], count=7)
    repo = SimulationRepository(supabase_client=client)

    n = await repo.count_linked(brand="Fabhalta")

    assert n == 7
    chain.select.assert_called_once_with("simulation_id", count="exact")
    chain.eq.assert_any_call("simulation_status", "completed")
    chain.not_.is_.assert_called_once_with("experiment_design_id", "null")
    chain.eq.assert_any_call("brand", "Fabhalta")


@pytest.mark.asyncio
async def test_no_client_reads_as_empty_not_as_an_error():
    repo = SimulationRepository(supabase_client=MagicMock())
    repo.client = None
    assert await repo.list_proposed(brand=None) == []
    assert await repo.count_linked(brand=None) == 0


@pytest.mark.asyncio
async def test_facade_forwards_list_proposed_and_count_linked_to_the_store():
    """The real facade against an autospecced store: a kwarg the facade drops or
    misnames is a TypeError HERE, not in prod."""
    repo = TwinRepository(supabase_client=MagicMock())
    store = create_autospec(SimulationRepository, instance=True)
    store.list_proposed = AsyncMock(return_value=[{"simulation_id": "s1"}])
    store.count_linked = AsyncMock(return_value=3)
    repo.simulations = store

    rows = await repo.list_proposed_experiments(brand="Kisqali", limit=25)
    linked = await repo.count_linked_simulations(brand="Kisqali")

    assert rows == [{"simulation_id": "s1"}]
    assert linked == 3
    store.list_proposed.assert_awaited_once_with(brand="Kisqali", limit=25)
    store.count_linked.assert_awaited_once_with(brand="Kisqali")
