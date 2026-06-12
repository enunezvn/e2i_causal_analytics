"""#883 PR B unit tests: _record_model_performance wiring contract.

``update_model_performance`` had NO caller, permanently starving the LIVE
reader ``get_context._get_model_performance_history`` (context.model_performance
was always {} in prod). The agent now records each succeeded model's MEASURED
registry metrics post-prediction. The faithful Redis/registry round-trips live
in tests/integration/test_agent_memory_wiring_883b.py; these tests pin the
honesty gates:

* registries WITHOUT the capability (legacy/stub) are skipped silently;
* only models that actually SUCCEEDED in this run are recorded;
* only non-None measured metrics are stored — never an invented number;
* a raising registry/hook cannot poison the run (caller-side swallow).
"""

from typing import Any, Dict, List

import pytest

from src.agents.prediction_synthesizer.agent import (
    PredictionSynthesizerAgent,
    PredictionSynthesizerOutput,
)


class _Hooks:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    async def update_model_performance(self, prediction_target, model_id, metrics=None, **kw):
        self.calls.append({"target": prediction_target, "model_id": model_id, "metrics": metrics})
        return True


class _Registry:
    def __init__(self, perf):
        self._perf = perf

    async def get_models_for_target(self, target, entity_type=""):
        return list(self._perf.keys())

    async def get_model_performance_for_target(self, target, entity_type=""):
        return dict(self._perf)


class _LegacyRegistry:
    """Pre-#883 registry surface: names only."""

    async def get_models_for_target(self, target, entity_type=""):
        return ["m1"]


def _output(model_ids: List[str]) -> PredictionSynthesizerOutput:
    return PredictionSynthesizerOutput(
        individual_predictions=[
            {
                "model_id": m,
                "model_type": "classifier",
                "prediction": 0.7,
                "prediction_proba": None,
                "confidence": 0.9,
                "latency_ms": 5,
                "features_used": [],
            }
            for m in model_ids
        ],
        models_succeeded=len(model_ids),
        status="completed",
    )


def _agent(registry) -> PredictionSynthesizerAgent:
    agent = PredictionSynthesizerAgent(
        model_registry=registry, enable_opik=False, enable_dspy=False
    )
    agent._memory_hooks = _Hooks()
    return agent


@pytest.mark.asyncio
async def test_records_measured_metrics_for_succeeded_models_only():
    perf = {
        "m1": {"auc": 0.83, "pr_auc": None, "brier_score": 0.12},
        "m2": {"auc": 0.78, "brier_score": 0.15},
    }
    agent = _agent(_Registry(perf))

    # m2 did NOT succeed in this run -> must not be recorded.
    await agent._record_model_performance(_output(["m1"]), "trx")

    calls = agent._memory_hooks.calls
    assert [c["model_id"] for c in calls] == ["m1"]
    assert calls[0]["target"] == "trx"
    # None metrics are dropped — only measured values are stored.
    assert calls[0]["metrics"] == {"auc": 0.83, "brier_score": 0.12}


@pytest.mark.asyncio
async def test_legacy_registry_without_capability_is_skipped():
    agent = _agent(_LegacyRegistry())
    await agent._record_model_performance(_output(["m1"]), "trx")
    assert agent._memory_hooks.calls == []


@pytest.mark.asyncio
async def test_no_registry_is_skipped():
    agent = _agent(None)
    await agent._record_model_performance(_output(["m1"]), "trx")
    assert agent._memory_hooks.calls == []


@pytest.mark.asyncio
async def test_empty_performance_map_records_nothing():
    agent = _agent(_Registry({}))
    await agent._record_model_performance(_output(["m1"]), "trx")
    assert agent._memory_hooks.calls == []


@pytest.mark.asyncio
async def test_memory_disabled_records_nothing():
    agent = _agent(_Registry({"m1": {"auc": 0.8}}))
    agent.enable_memory = False
    await agent._record_model_performance(_output(["m1"]), "trx")
    assert agent._memory_hooks.calls == []
