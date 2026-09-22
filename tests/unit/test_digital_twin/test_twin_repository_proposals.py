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
    # codex r4 MED: the presentation order is applied IN the database, so a bounded
    # window is the top of the whole population, not the newest rows re-sorted.
    assert [c.args[0] for c in chain.order.call_args_list] == ["recommendation", "simulated_ate"]
    assert chain.order.call_args_list[1].kwargs.get("desc") is True


@pytest.mark.asyncio
async def test_count_proposed_counts_the_whole_population():
    client, chain = _chain([], count=731)
    repo = SimulationRepository(supabase_client=client)
    assert await repo.count_proposed(brand="Kisqali") == 731
    chain.select.assert_called_once_with("simulation_id", count="exact")
    chain.in_.assert_called_once_with("recommendation", ["deploy", "refine"])
    chain.is_.assert_called_once_with("experiment_design_id", "null")
    chain.eq.assert_any_call("brand", "Kisqali")


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
    # codex r2 #4: the linked count is the linked half of the PROPOSAL population
    # (deploy/refine), so "N linked" and "0 proposed" describe one set; a linked
    # 'skip' run is not counted as a proposal that got its experiment.
    chain.in_.assert_called_once_with("recommendation", ["deploy", "refine"])


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

    store.count_proposed = AsyncMock(return_value=9)

    rows = await repo.list_proposed_experiments(brand="Kisqali", limit=25)
    linked = await repo.count_linked_simulations(brand="Kisqali")
    total = await repo.count_proposed_experiments(brand="Kisqali")

    assert rows == [{"simulation_id": "s1"}]
    assert linked == 3
    assert total == 9
    store.list_proposed.assert_awaited_once_with(brand="Kisqali", limit=25)
    store.count_linked.assert_awaited_once_with(brand="Kisqali")
    store.count_proposed.assert_awaited_once_with(brand="Kisqali")


@pytest.mark.asyncio
async def test_list_proposed_and_count_linked_propagate_a_query_failure():
    """codex r1 #3: a swallowed query failure reads as a truthful-looking empty
    portfolio (200, total_proposed=0). The failure must reach the route's 500."""
    client, chain = _chain([])
    chain.execute = AsyncMock(side_effect=RuntimeError("42703 column does not exist"))
    repo = SimulationRepository(supabase_client=client)
    with pytest.raises(RuntimeError):
        await repo.list_proposed(brand=None)
    with pytest.raises(RuntimeError):
        await repo.count_linked(brand=None)


@pytest.mark.asyncio
async def test_claim_experiment_link_is_conditional_on_an_unlinked_row():
    """codex r1 #2: the link is a CLAIM — UPDATE … WHERE experiment_design_id IS NULL —
    so two concurrent drafts cannot both link, and a zero-row update is False, not True."""
    from uuid import uuid4

    sim_id, exp_id = uuid4(), uuid4()
    client, chain = _chain([{"simulation_id": str(sim_id), "experiment_design_id": str(exp_id)}])
    chain.update.return_value = chain
    repo = SimulationRepository(supabase_client=client)

    assert await repo.claim_experiment_link(sim_id, exp_id) is True
    chain.update.assert_called_once_with({"experiment_design_id": str(exp_id)})
    chain.eq.assert_any_call("simulation_id", str(sim_id))
    chain.is_.assert_called_once_with("experiment_design_id", "null")

    chain.execute = AsyncMock(return_value=MagicMock(data=[]))
    assert await repo.claim_experiment_link(sim_id, exp_id) is False


@pytest.mark.asyncio
async def test_facade_forwards_claim_experiment_link():
    from uuid import uuid4

    repo = TwinRepository(supabase_client=MagicMock())
    store = create_autospec(SimulationRepository, instance=True)
    store.claim_experiment_link = AsyncMock(return_value=True)
    repo.simulations = store
    sim_id, exp_id = uuid4(), uuid4()
    assert await repo.claim_experiment_link(sim_id, exp_id) is True
    store.claim_experiment_link.assert_awaited_once_with(sim_id, exp_id)


@pytest.mark.asyncio
async def test_require_simulation_and_require_model_propagate_failures_and_distinguish_no_row():
    """codex r2 #2: get_simulation / get_model swallow every query failure into None,
    which the proposals routes would label 'unvalidated' or 404. The strict reads
    return None ONLY for "no such row" and let a failure propagate."""
    from uuid import uuid4

    from src.digital_twin.twin_repository import TwinModelRepository

    sim_id, model_id = uuid4(), uuid4()
    client, chain = _chain([{"simulation_id": str(sim_id)}])
    sims = SimulationRepository(supabase_client=client)
    assert await sims.require_simulation(sim_id) == {"simulation_id": str(sim_id)}
    chain.eq.assert_any_call("simulation_id", str(sim_id))

    chain.execute = AsyncMock(return_value=MagicMock(data=[]))
    assert await sims.require_simulation(sim_id) is None

    chain.execute = AsyncMock(side_effect=RuntimeError("connection reset"))
    with pytest.raises(RuntimeError):
        await sims.require_simulation(sim_id)

    mclient, mchain = _chain([{"model_id": str(model_id), "fidelity_score": None}])
    models = TwinModelRepository(supabase_client=mclient)
    assert (await models.require_model(model_id))["model_id"] == str(model_id)
    mchain.eq.assert_any_call("model_id", str(model_id))
    mchain.execute = AsyncMock(side_effect=RuntimeError("connection reset"))
    with pytest.raises(RuntimeError):
        await models.require_model(model_id)


@pytest.mark.asyncio
async def test_facade_forwards_the_strict_reads():
    from uuid import uuid4

    from src.digital_twin.twin_repository import TwinModelRepository

    repo = TwinRepository(supabase_client=MagicMock())
    sims = create_autospec(SimulationRepository, instance=True)
    models = create_autospec(TwinModelRepository, instance=True)
    sims.require_simulation = AsyncMock(return_value={"simulation_id": "s"})
    models.require_model = AsyncMock(return_value={"model_id": "m"})
    repo.simulations, repo.models = sims, models
    sim_id, model_id = uuid4(), uuid4()
    assert await repo.require_simulation(sim_id) == {"simulation_id": "s"}
    assert await repo.require_model(model_id) == {"model_id": "m"}
    sims.require_simulation.assert_awaited_once_with(sim_id)
    models.require_model.assert_awaited_once_with(model_id)
