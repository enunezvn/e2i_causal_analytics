"""#2319 item 3: model_selector writes nothing to ``ml_model_registry``.

``ModelSelectorAgent.run`` used to call ``MLModelRegistryRepository.register_model_candidate``
(c4675f935, Dec 2025) to persist the selected algorithm as a ``stage='candidate'`` row
named ``candidate-<ts>``. The insert sent columns the live table does not have
(``metrics``, ``description``, ``created_at``, ``created_by``, ``tags``), was swallowed by
``except Exception: return None``, and prod holds 0 ``candidate-%`` rows (read-only census
2026-09-29). Since #2310 ``stage='candidate'`` means "retrain awaiting review", so a future
schema change must not let this writer start producing rows with a different meaning.

The structured selection is persisted by the MLflow ``model_selection_<algorithm>`` run
(``mlflow_registrar``: params ``algorithm_name`` / ``algorithm_family`` / defaults, tags
``agent=model_selector``; 812 runs in prod). The episodic ``model_selection_completed`` row
carries the rationale text only (its structured algorithm fields are NULL, a pre-existing
hook defect reported on #2319). This test does not cover either; it pins the absence of the
registry write.

Drives the real ``run()`` with a stub graph over the in-memory async supabase fake, which
accepts any column: the old writer inserts a row here, so this test is red on it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from unittest.mock import AsyncMock

import pytest

from tests.unit._fakes.async_supabase import FakeAsyncSupabase

_EXPERIMENT = "exp-2319"


class _StubGraph:
    def __init__(self, extra: Dict[str, Any]):
        self._extra = extra

    async def ainvoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {**state, **self._extra}


class _Hooks:
    def __getattr__(self, name: str):
        async def _record(**kwargs: Any) -> Optional[str]:
            return "mem-1"

        return _record


@pytest.mark.unit
@pytest.mark.asyncio
async def test_model_selector_run_writes_no_registry_row(monkeypatch):
    import src.agents.ml_foundation.model_selector.agent as module
    import src.memory.services.factories as factories

    db = FakeAsyncSupabase({"ml_model_registry": []})
    monkeypatch.setattr(factories, "get_supabase_client", lambda: db)
    monkeypatch.setattr(factories, "get_async_supabase_client", AsyncMock(return_value=db))
    monkeypatch.setattr(module, "_get_opik_connector", lambda: None)
    monkeypatch.setattr(module, "ModelSelectorMemoryHooks", _Hooks)

    agent = module.ModelSelectorAgent.__new__(module.ModelSelectorAgent)
    agent.mode = "simple"
    agent._graph = _StubGraph(
        {
            "algorithm_name": "LogisticRegression",
            "algorithm_family": "linear",
            "algorithm_class": "sklearn.linear_model.LogisticRegression",
            "selection_score": 0.76,
            "mlflow_run_id": "run-2319",
        }
    )
    monkeypatch.setattr(agent, "_update_procedural_memory", AsyncMock(return_value=None))
    monkeypatch.setattr(agent, "_update_semantic_memory", AsyncMock(return_value=None))

    result = await agent.run(
        {
            "scope_spec": {"experiment_id": _EXPERIMENT, "problem_type": "binary_classification"},
            "qc_report": {"qc_passed": True},
        }
    )

    assert "error" not in result, result
    assert result["model_candidate"]["algorithm_name"] == "LogisticRegression"
    assert db.rows("ml_model_registry") == []


def test_the_candidate_writer_is_gone_from_the_registry_repository():
    """No other caller exists (grep of src/, scripts/, tests/ on 2026-09-29)."""
    from src.repositories.ml_experiment import MLModelRegistryRepository

    assert not hasattr(MLModelRegistryRepository, "register_model_candidate")
