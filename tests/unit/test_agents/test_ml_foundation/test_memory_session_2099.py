"""#2099: the six ml_foundation agents must stop persisting the audit id as a session.

Each ``_update_episodic_memory`` computed
``str(<state>.get("audit_workflow_id") or uuid4())`` and handed the result to its
``store_*`` hook as ``session_id``. Two different invented identities came out of
that one expression:

* with an audit id -- the audit chain's workflow id was written into
  ``episodic_memories.session_id``, a column that means "which conversation", so
  the row claimed a conversation that never existed;
* without one -- a fresh uuid, belonging to nothing at all.

The audit id is a real correlation handle, so it is not discarded: it moves into
the episodic ``raw_content``, where it is what it actually is. The session column
records an honest NULL. These six agents wrote nothing between 2026-07-30 and the
lane's base, so the change is zero-traffic until a pipeline is triggered.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pytest

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")
_AUDIT = "9f2a1d4e-7c3b-4a15-9e28-6b0d5a7c1e39"
_EXPERIMENT = "exp-2099"

# (agent module, agent class, hooks symbol in that module, store method, state key)
_AGENTS = [
    (
        "src.agents.ml_foundation.data_preparer.agent",
        "DataPreparerMemoryHooks",
        "store_qc_report",
    ),
    (
        "src.agents.ml_foundation.model_selector.agent",
        "ModelSelectorMemoryHooks",
        "store_model_selection",
    ),
    (
        "src.agents.ml_foundation.feature_analyzer.agent",
        "FeatureAnalyzerMemoryHooks",
        "store_feature_analysis",
    ),
    (
        "src.agents.ml_foundation.scope_definer.agent",
        "ScopeDefinerMemoryHooks",
        "store_scope_definition",
    ),
    (
        "src.agents.ml_foundation.model_deployer.agent",
        "ModelDeployerMemoryHooks",
        "store_deployment",
    ),
    (
        "src.agents.ml_foundation.model_trainer.agent",
        "ModelTrainerMemoryHooks",
        "store_training_result",
    ),
]

_HOOK_MODULES = [
    ("src.agents.ml_foundation.data_preparer.memory_hooks", "store_qc_report"),
    ("src.agents.ml_foundation.model_selector.memory_hooks", "store_model_selection"),
    ("src.agents.ml_foundation.feature_analyzer.memory_hooks", "store_feature_analysis"),
    ("src.agents.ml_foundation.scope_definer.memory_hooks", "store_scope_definition"),
    ("src.agents.ml_foundation.model_deployer.memory_hooks", "store_deployment"),
    ("src.agents.ml_foundation.model_trainer.memory_hooks", "store_training_result"),
]


def _install_recorder(monkeypatch, module_path: str, hooks_name: str, store_name: str):
    """Replace the agent's hooks class with a recorder for its store method."""
    import importlib

    module = importlib.import_module(module_path)
    calls: List[Dict[str, Any]] = []

    class _Hooks:
        def __getattr__(self, name: str):
            async def _record(**kwargs: Any) -> Optional[str]:
                calls.append({"method": name, **kwargs})
                return "mem-1"

            return _record

    monkeypatch.setattr(module, hooks_name, _Hooks)
    return calls


async def _run_update(module_path: str, state: Dict[str, Any]) -> None:
    """Drive the agent's _update_episodic_memory with a minimal final state."""
    import importlib

    module = importlib.import_module(module_path)
    agent_cls = next(
        getattr(module, n)
        for n in dir(module)
        if n.endswith("Agent") and hasattr(getattr(module, n), "_update_episodic_memory")
    )
    agent = agent_cls.__new__(agent_cls)  # no __init__: the method only reads its argument
    import inspect

    # model_deployer takes (output, state); the other five take one dict.
    arity = len(inspect.signature(agent_cls._update_episodic_memory).parameters) - 1
    if arity == 2:
        await agent._update_episodic_memory(state, state)
    else:
        await agent._update_episodic_memory(state)


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path,hooks_name,store_name", _AGENTS)
async def test_audit_workflow_id_is_not_stored_as_the_session(
    monkeypatch, module_path, hooks_name, store_name
):
    """The audit chain's id is not a conversation."""
    calls = _install_recorder(monkeypatch, module_path, hooks_name, store_name)

    await _run_update(
        module_path,
        {
            "experiment_id": _EXPERIMENT,
            "audit_workflow_id": _AUDIT,
            "scope_spec": {"experiment_id": _EXPERIMENT},
        },
    )

    assert len(calls) == 1, calls
    assert calls[0]["method"] == store_name
    assert calls[0]["session_id"] is None, (
        f"{module_path} stored the audit workflow id as the session: {calls[0]['session_id']!r}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("module_path,hooks_name,store_name", _AGENTS)
async def test_no_audit_id_mints_nothing(monkeypatch, module_path, hooks_name, store_name):
    """Without an audit id the old code minted a fresh uuid; now it is None."""
    calls = _install_recorder(monkeypatch, module_path, hooks_name, store_name)

    await _run_update(
        module_path,
        {"experiment_id": _EXPERIMENT, "scope_spec": {"experiment_id": _EXPERIMENT}},
    )

    assert len(calls) == 1, calls
    got = calls[0]["session_id"]
    assert not (isinstance(got, str) and _UUID_RE.match(got)), f"minted a session id: {got!r}"
    assert got is None


@pytest.mark.asyncio
@pytest.mark.parametrize("hooks_module,store_name", _HOOK_MODULES)
async def test_audit_workflow_id_survives_in_raw_content(monkeypatch, hooks_module, store_name):
    """Dropping it from the session column must not drop it from the row."""
    import importlib

    module = importlib.import_module(hooks_module)
    import src.memory.episodic_memory as episodic

    captured: Dict[str, Any] = {}

    async def _insert(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "mem-1"

    monkeypatch.setattr(episodic, "insert_episodic_memory", _insert)

    hooks_cls = next(getattr(module, n) for n in dir(module) if n.endswith("MemoryHooks"))
    store = getattr(hooks_cls(), store_name)
    memory_id = await store(
        session_id=None,
        result={"experiment_id": _EXPERIMENT},
        state={"experiment_id": _EXPERIMENT, "audit_workflow_id": _AUDIT},
    )

    assert memory_id == "mem-1", f"{hooks_module}: the write did not reach the inserter"
    assert captured["session_id"] is None
    assert captured["raw_content"].get("audit_workflow_id") == _AUDIT


@pytest.mark.asyncio
@pytest.mark.parametrize("hooks_module,store_name", _HOOK_MODULES)
async def test_raw_content_audit_id_is_absent_not_invented(monkeypatch, hooks_module, store_name):
    """No audit id in state -> no fabricated key."""
    import importlib

    module = importlib.import_module(hooks_module)
    import src.memory.episodic_memory as episodic

    captured: Dict[str, Any] = {}

    async def _insert(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "mem-1"

    monkeypatch.setattr(episodic, "insert_episodic_memory", _insert)

    hooks_cls = next(getattr(module, n) for n in dir(module) if n.endswith("MemoryHooks"))
    store = getattr(hooks_cls(), store_name)
    await store(
        session_id=None,
        result={"experiment_id": _EXPERIMENT},
        state={"experiment_id": _EXPERIMENT},
    )

    assert captured["raw_content"].get("audit_workflow_id") is None
