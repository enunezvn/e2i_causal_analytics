"""Agent method dispatch and response-extraction registry.

Source of truth for how the Orchestrator's dispatcher calls each Tier 1-5 agent
and where the synthesizer finds the human-readable narrative in each agent's
output. The same data drives ``scripts/run_tier1_5_test.py`` so the harness and
the live dispatcher cannot drift apart.

Two registries:

- ``AGENT_METHOD_MAP``: per-agent dispatch spec (method name, async vs sync,
  whether to splat input as kwargs, optional Pydantic input model wrapper).
- ``AGENT_RESPONSE_FIELDS``: ordered list of output dict keys to try when
  extracting a narrative string for synthesis. Falls back to ``narrative`` and
  ``response`` if none match.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class AgentMethodSpec:
    """Dispatch specification for a single agent.

    Fields mirror ``AGENT_CONFIGS`` in ``scripts/run_tier1_5_test.py`` so the
    test harness and the production dispatcher use the same source.
    """

    method: str
    is_async: bool = True
    uses_kwargs: bool = False
    input_model: Optional[str] = None
    input_module: Optional[str] = None


# Per-agent dispatch metadata. Agents not listed here fall through to the
# legacy ``.analyze(input_data)`` contract.
AGENT_METHOD_MAP: Dict[str, AgentMethodSpec] = {
    # Tier 1: Coordination
    "orchestrator": AgentMethodSpec(method="run"),
    "tool_composer": AgentMethodSpec(method="run"),
    # Tier 2: Causal Analytics (these implement .analyze; included for clarity)
    "causal_impact": AgentMethodSpec(method="analyze"),
    "gap_analyzer": AgentMethodSpec(method="analyze"),
    "heterogeneous_optimizer": AgentMethodSpec(method="run"),
    # Tier 3: Monitoring
    "drift_monitor": AgentMethodSpec(
        method="run",
        input_model="DriftMonitorInput",
        input_module="src.agents.drift_monitor.agent",
    ),
    "experiment_designer": AgentMethodSpec(
        method="run",
        is_async=False,
        input_model="ExperimentDesignerInput",
        input_module="src.agents.experiment_designer.agent",
    ),
    "experiment_monitor": AgentMethodSpec(
        method="run_async",
        input_model="ExperimentMonitorInput",
        input_module="src.agents.experiment_monitor.agent",
    ),
    "health_score": AgentMethodSpec(method="check_health", uses_kwargs=True),
    # Tier 4: ML Predictions
    "prediction_synthesizer": AgentMethodSpec(method="synthesize", uses_kwargs=True),
    "resource_optimizer": AgentMethodSpec(method="optimize", uses_kwargs=True),
    # Tier 5: Self-Improvement
    "explainer": AgentMethodSpec(method="explain", uses_kwargs=True),
    "feedback_learner": AgentMethodSpec(method="learn", uses_kwargs=True),
}


# Per-agent ordered list of output keys to use as the narrative for synthesis.
# When multiple keys match the first non-empty one wins. Fallback path in the
# synthesizer adds ``narrative`` and ``response`` for legacy agents.
AGENT_RESPONSE_FIELDS: Dict[str, List[str]] = {
    "orchestrator": ["response_text", "synthesized_response", "narrative"],
    "tool_composer": ["response", "synthesis_response", "answer"],
    "causal_impact": ["executive_summary", "narrative", "interpretation"],
    "gap_analyzer": ["executive_summary", "narrative", "key_insights"],
    "heterogeneous_optimizer": ["executive_summary", "narrative", "cate_summary"],
    "drift_monitor": ["drift_interpretation", "narrative"],
    "experiment_designer": ["design_summary", "narrative"],
    "experiment_monitor": ["monitor_summary", "narrative"],
    "health_score": ["health_summary", "narrative"],
    "prediction_synthesizer": ["prediction_summary", "narrative"],
    "resource_optimizer": ["optimization_summary", "narrative"],
    "explainer": ["executive_summary", "narrative", "explanation_text"],
    "feedback_learner": ["learning_summary", "feedback_summary", "narrative"],
}


def get_method_spec(agent_name: str) -> AgentMethodSpec:
    """Return the dispatch spec for ``agent_name``.

    Falls back to ``analyze`` for unmapped agents to preserve backward
    compatibility with legacy tests and mock dispatch paths.
    """
    return AGENT_METHOD_MAP.get(agent_name, AgentMethodSpec(method="analyze"))


def extract_narrative(agent_name: str, output: Dict[str, object]) -> str:
    """Return the best narrative string from ``output`` for ``agent_name``.

    Tries per-agent fields first, then ``narrative``/``response`` defaults.
    Returns an empty string when nothing matches — callers decide whether to
    stringify the whole output as a last resort.
    """
    keys = AGENT_RESPONSE_FIELDS.get(agent_name, []) + ["narrative", "response"]
    for key in keys:
        value = output.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    return ""


__all__ = [
    "AgentMethodSpec",
    "AGENT_METHOD_MAP",
    "AGENT_RESPONSE_FIELDS",
    "get_method_spec",
    "extract_narrative",
]
