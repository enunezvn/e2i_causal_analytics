"""Label-gater plumbing guard: `brand` must reach HeterogeneousOptimizerState.

The gater resolves a brand's FDA-label indicated-population criteria, so it needs
`brand` in state. HeterogeneousOptimizerInput already carried it (for memory /
tracking) but _build_initial_state dropped it. These guard that (1) the state
contract carries `brand`, and (2) the initial-state builder forwards it from
input_data.
"""

import pytest


@pytest.mark.unit
def test_state_contract_carries_brand():
    from src.agents.heterogeneous_optimizer.state import HeterogeneousOptimizerState

    assert "brand" in HeterogeneousOptimizerState.__annotations__, (
        "HeterogeneousOptimizerState must carry `brand` for the label-gater's "
        "indicated-population lookup"
    )


@pytest.mark.unit
def test_build_initial_state_forwards_brand():
    # _build_initial_state is a pure mapping over input_data; exercise it without
    # constructing connectors/graph (avoids heavy imports / droplet OOM).
    from src.agents.heterogeneous_optimizer.agent import HeterogeneousOptimizerAgent

    agent = HeterogeneousOptimizerAgent.__new__(HeterogeneousOptimizerAgent)
    input_data = {
        "query": "segment uplift for Remibrutinib",
        "treatment_var": "treatment",
        "outcome_var": "persistent_180d",
        "segment_vars": ["urticaria_severity_uas7"],
        "effect_modifiers": [],
        "data_source": "patient_journeys",
        "brand": "Remibrutinib",
    }
    state = agent._build_initial_state(input_data, session_id="s1", memory_context=None)
    assert state["brand"] == "Remibrutinib"
