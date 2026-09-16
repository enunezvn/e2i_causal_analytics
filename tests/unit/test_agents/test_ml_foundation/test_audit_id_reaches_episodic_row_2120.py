"""#2120: the audit workflow id must reach the episodic row from ``run()``.

#2099 moved the audit workflow id out of ``episodic_memories.session_id`` and
into the row's ``raw_content``; the six memory hooks add it there iff the
``state`` they are handed carries it. Three agents (scope_definer,
model_selector, model_trainer) handed the hook their public ``output`` as the
``state`` -- and ``output`` never carried the id, only the graph's
``final_state`` did. The #2099 pin exercised the hook with a state that already
had the id, so it could not see what ``run()`` actually passes.

These tests drive the real ``run()`` with a stub graph and record every kwarg
the store hook receives, so they fail exactly where the false green could not.
feature_analyzer is the positive control: it already hands ``final_state`` to
its hook and must be green before and after the fix.
"""

from __future__ import annotations

import importlib
import re
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_AUDIT = "3c9d8a2b-5e14-4f7a-8b6c-0d1e2f3a4b5c"
_EXPERIMENT = "exp-2120"

_TARGETS = [
    pytest.param(
        {
            "module": "src.agents.ml_foundation.scope_definer.agent",
            "agent_cls": "ScopeDefinerAgent",
            "hooks": "ScopeDefinerMemoryHooks",
            "store": "store_scope_definition",
            "graph_attr": "graph",
            "attrs": {},
            "patched": {
                "_persist_scope_spec": None,
                "_update_procedural_memory": None,
                "_update_semantic_memory": None,
            },
            "graph_extra": {
                "experiment_id": _EXPERIMENT,
                "scope_spec": {"experiment_id": _EXPERIMENT},
                "success_criteria": {},
            },
            "input_data": {
                "problem_description": "predict discontinuation",
                "business_objective": "retain patients",
                "target_outcome": "fewer discontinuations",
            },
        },
        id="scope_definer",
    ),
    pytest.param(
        {
            "module": "src.agents.ml_foundation.model_selector.agent",
            "agent_cls": "ModelSelectorAgent",
            "hooks": "ModelSelectorMemoryHooks",
            "store": "store_model_selection",
            "graph_attr": "_graph",  # ``graph`` is a lazy property over ``_graph``
            "attrs": {"mode": "simple"},
            "patched": {
                "_persist_model_candidate": None,
                "_update_procedural_memory": None,
                "_update_semantic_memory": None,
            },
            "graph_extra": {"algorithm_name": "logistic_regression", "selection_score": 0.8},
            "input_data": {
                "scope_spec": {
                    "experiment_id": _EXPERIMENT,
                    "problem_type": "binary_classification",
                },
                "qc_report": {"qc_passed": True},
            },
        },
        id="model_selector",
    ),
    pytest.param(
        {
            "module": "src.agents.ml_foundation.model_trainer.agent",
            "agent_cls": "ModelTrainerAgent",
            "hooks": "ModelTrainerMemoryHooks",
            "store": "store_training_result",
            "graph_attr": "graph",
            "attrs": {},
            "patched": {
                "_persist_training_run": False,
                "_update_procedural_memory": None,
                "_update_semantic_memory": None,
            },
            "graph_extra": {},
            "input_data": {
                "model_candidate": {
                    "algorithm_name": "logistic_regression",
                    "algorithm_class": "sklearn.linear_model.LogisticRegression",
                    "hyperparameter_search_space": {},
                    "default_hyperparameters": {},
                },
                "qc_report": {"qc_passed": True},
                "experiment_id": _EXPERIMENT,
                "enable_hpo": False,
                "enable_mlflow": False,
            },
        },
        id="model_trainer",
    ),
]

# Positive control: already hands final_state to its hook (design §0), so this
# case is green on the base commit and proves the harness can see the id.
_CONTROL = pytest.param(
    {
        "module": "src.agents.ml_foundation.feature_analyzer.agent",
        "agent_cls": "FeatureAnalyzerAgent",
        "hooks": "FeatureAnalyzerMemoryHooks",
        "store": "store_feature_analysis",
        "graph_attr": "_shap_graph",  # ``_get_shap_graph`` returns it when set
        "attrs": {"_full_graph": None},
        "patched": {
            "_update_semantic_memory": (False, 0),
            "_auto_register_in_feast": {},
            "_store_to_database": None,
        },
        "graph_extra": {},
        "input_data": {"experiment_id": _EXPERIMENT, "store_in_semantic_memory": False},
    },
    id="feature_analyzer_control",
)

_CASES = [*_TARGETS, _CONTROL]


class _StubGraph:
    """Stands in for the compiled LangGraph: echoes the state plus a few keys."""

    def __init__(self, extra: Dict[str, Any]):
        self._extra = extra

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {**state, **self._extra}


def _install_recorder(monkeypatch, module: Any, hooks_name: str) -> List[Dict[str, Any]]:
    """Replace the agent module's hooks class with a recorder of every store kwarg."""
    calls: List[Dict[str, Any]] = []

    class _Hooks:
        def __getattr__(self, name: str):
            async def _record(**kwargs: Any) -> Optional[str]:
                calls.append({"method": name, **kwargs})
                return "mem-1"

            return _record

    monkeypatch.setattr(module, hooks_name, _Hooks)
    return calls


def _build_agent(monkeypatch, case: Dict[str, Any]):
    """Build the agent without its heavy ``__init__`` and isolate run() from I/O."""
    module = importlib.import_module(case["module"])
    agent_cls = getattr(module, case["agent_cls"])
    agent = agent_cls.__new__(agent_cls)
    for name, value in case["attrs"].items():
        setattr(agent, name, value)
    setattr(agent, case["graph_attr"], _StubGraph(case["graph_extra"]))
    for name, return_value in case["patched"].items():
        assert hasattr(agent_cls, name), f"{case['module']} has no {name}"
        monkeypatch.setattr(agent, name, AsyncMock(return_value=return_value))
    monkeypatch.setattr(module, "_get_opik_connector", lambda: None)
    calls = _install_recorder(monkeypatch, module, case["hooks"])
    return agent, calls


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES)
async def test_run_hands_the_caller_audit_id_to_the_episodic_hook(monkeypatch, case):
    """The id the caller supplied is in the ``state`` the store hook receives."""
    agent, calls = _build_agent(monkeypatch, case)

    result = await agent.run({**case["input_data"], "audit_workflow_id": _AUDIT})

    assert "error" not in result, result
    assert len(calls) == 1, calls
    assert calls[0]["method"] == case["store"]
    assert calls[0]["session_id"] is None
    state = calls[0]["state"]
    assert state.get("audit_workflow_id") is not None, (
        f"{case['module']}: run() handed the hook a state without the audit id; "
        f"keys={sorted(state)}"
    )
    assert str(state["audit_workflow_id"]) == _AUDIT


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("case", _CASES)
async def test_run_hands_the_minted_audit_id_to_the_episodic_hook(monkeypatch, case):
    """Without a caller id the agent mints one at its boundary; it must reach the row too."""
    agent, calls = _build_agent(monkeypatch, case)

    result = await agent.run(dict(case["input_data"]))

    assert "error" not in result, result
    assert len(calls) == 1, calls
    assert calls[0]["session_id"] is None
    state = calls[0]["state"]
    got = state.get("audit_workflow_id")
    assert got is not None, f"{case['module']}: the minted audit id did not reach the hook"
    assert _UUID_RE.match(str(got)), f"not a uuid: {got!r}"
