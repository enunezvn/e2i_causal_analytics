"""#1976: role_attributions must reach the causal_impact agent.

The adjustment_set_policy node reads ``state.role_attributions``; the
dispatcher passed that key through for heterogeneous_optimizer only, so no
caller could reach the policy node on causal_impact.
"""

from src.agents.orchestrator.nodes.dispatcher import _resolve_causal_impact_input

_ATTRS = [
    {
        "feature": "adherence_rate",
        "causal_role": "mediator",
        "source": "manifest",
        "evaluator_satisfied": True,
        "evaluator_model": "n/a",
    }
]


def _dispatch(**params):
    base = {
        "treatment_var": "accepted",
        "outcome_var": "converted",
        "confounders": ["confidence_score"],
    }
    base.update(params)
    return {
        "agent_name": "causal_impact",
        "priority": "critical",
        "timeout_ms": 0,
        "parameters": base,
    }


def test_explicit_spec_passes_role_attributions_through():
    resolved = _resolve_causal_impact_input({"query": "q"}, _dispatch(role_attributions=_ATTRS))
    assert resolved["role_attributions"] == _ATTRS


def test_absent_role_attributions_are_not_fabricated():
    resolved = _resolve_causal_impact_input({"query": "q"}, _dispatch())
    assert "role_attributions" not in resolved
