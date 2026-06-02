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
    """Dispatch + harness specification for a single agent.

    Fields fall in two groups:

    - **Dispatch** (used by ``orchestrator/nodes/dispatcher.py``):
      ``method``, ``is_async``, ``uses_kwargs``, ``input_model``, ``input_module``.

    - **Harness** (used by ``scripts/run_tier1_5_test.py``):
      ``tier``, ``agent_module``, ``agent_class``, ``state_module``,
      ``state_class``, ``timeout``. The harness was previously duplicating the
      dispatch fields in its own ``AGENT_CONFIGS`` literal; #252 unified them
      here so they cannot drift silently.
    """

    # Dispatch — required at construction
    method: str
    is_async: bool = True
    uses_kwargs: bool = False
    input_model: Optional[str] = None
    input_module: Optional[str] = None
    # Harness — optional (production dispatcher ignores these).
    tier: Optional[int] = None
    agent_module: Optional[str] = None
    agent_class: Optional[str] = None
    state_module: Optional[str] = None
    state_class: Optional[str] = None
    timeout: Optional[float] = None  # seconds; None ⇒ harness default


# Per-agent dispatch + harness metadata. Single source of truth — issue #252.
# Agents not listed here fall through to the legacy ``.analyze(input_data)``
# contract on the dispatcher side.
#
# NOTE on method choice: causal_impact, gap_analyzer, heterogeneous_optimizer
# all implement BOTH ``.run()`` AND ``.analyze()``. ``.run()`` is the newer
# primary entry point (returns Pydantic Output contract); ``.analyze()`` is
# a legacy alias. Per #252 unification, both the production dispatcher and
# the harness now use ``.run()`` so they exercise the same code path.
AGENT_METHOD_MAP: Dict[str, AgentMethodSpec] = {
    # Tier 1: Coordination
    "orchestrator": AgentMethodSpec(
        method="run",
        tier=1,
        agent_module="src.agents.orchestrator",
        agent_class="OrchestratorAgent",
        state_module="src.agents.orchestrator.state",
        state_class="OrchestratorState",
    ),
    "tool_composer": AgentMethodSpec(
        method="run",
        tier=1,
        agent_module="src.agents.tool_composer",
        agent_class="ToolComposerAgent",
        state_module="src.agents.tool_composer.state",
        state_class="ToolComposerState",
        timeout=90.0,  # 3 sequential LLM calls + tool execution + memory queries
    ),
    # Tier 2: Causal Analytics
    "causal_impact": AgentMethodSpec(
        method="run",
        tier=2,
        agent_module="src.agents.causal_impact",
        agent_class="CausalImpactAgent",
        state_module="src.agents.causal_impact.state",
        state_class="CausalImpactOutput",  # Output contract, not State
        # estimation + refutation + sensitivity SLA. The Tier 1-5 harness mapper
        # requests a bounded refutation suite (#606) so the real pipeline runs in
        # ~24s locally (MEASURED) — well within this SLA even at ~2x-slower CI.
        # Full-sim refutation (~10-60 min) is exercised by the slow-tests lane.
        timeout=120.0,
    ),
    "gap_analyzer": AgentMethodSpec(
        method="run",
        tier=2,
        agent_module="src.agents.gap_analyzer",
        agent_class="GapAnalyzerAgent",
        state_module="src.agents.gap_analyzer.state",
        state_class="GapAnalyzerOutput",
    ),
    "heterogeneous_optimizer": AgentMethodSpec(
        method="run",
        tier=2,
        agent_module="src.agents.heterogeneous_optimizer",
        agent_class="HeterogeneousOptimizerAgent",
        state_module="src.agents.heterogeneous_optimizer.state",
        state_class="HeterogeneousOptimizerOutput",
    ),
    # Tier 3: Monitoring
    "drift_monitor": AgentMethodSpec(
        method="run",
        input_model="DriftMonitorInput",
        input_module="src.agents.drift_monitor.agent",
        tier=3,
        agent_module="src.agents.drift_monitor",
        agent_class="DriftMonitorAgent",
        state_module="src.agents.drift_monitor.state",
        state_class="DriftMonitorState",
    ),
    "experiment_designer": AgentMethodSpec(
        method="run",
        is_async=False,
        input_model="ExperimentDesignerInput",
        input_module="src.agents.experiment_designer.agent",
        tier=3,
        agent_module="src.agents.experiment_designer",
        agent_class="ExperimentDesignerAgent",
        state_module="src.agents.experiment_designer.state",
        state_class="ExperimentDesignState",
        timeout=120.0,  # LLM-based validity audit needs more time
    ),
    "experiment_monitor": AgentMethodSpec(
        method="run_async",
        input_model="ExperimentMonitorInput",
        input_module="src.agents.experiment_monitor.agent",
        tier=3,
        agent_module="src.agents.experiment_monitor",
        agent_class="ExperimentMonitorAgent",
        # Agent returns ExperimentMonitorOutput dataclass; harness contract
        # validator flattens via __dict__, so state_class points at the
        # dataclass on the agent module itself.
        state_module="src.agents.experiment_monitor.agent",
        state_class="ExperimentMonitorOutput",
        timeout=20.0,
    ),
    "health_score": AgentMethodSpec(
        method="check_health",
        uses_kwargs=True,
        tier=3,
        agent_module="src.agents.health_score",
        agent_class="HealthScoreAgent",
        state_module="src.agents.health_score.state",
        state_class="HealthScoreState",
    ),
    # Tier 4: ML Predictions
    "prediction_synthesizer": AgentMethodSpec(
        method="synthesize",
        uses_kwargs=True,
        tier=4,
        agent_module="src.agents.prediction_synthesizer",
        agent_class="PredictionSynthesizerAgent",
        state_module="src.agents.prediction_synthesizer.state",
        state_class="PredictionSynthesizerState",
    ),
    "resource_optimizer": AgentMethodSpec(
        method="optimize",
        uses_kwargs=True,
        tier=4,
        agent_module="src.agents.resource_optimizer",
        agent_class="ResourceOptimizerAgent",
        state_module="src.agents.resource_optimizer.state",
        state_class="ResourceOptimizerState",
    ),
    # Tier 5: Self-Improvement
    "explainer": AgentMethodSpec(
        method="explain",
        uses_kwargs=True,
        tier=5,
        agent_module="src.agents.explainer",
        agent_class="ExplainerAgent",
        state_module="src.agents.explainer.state",
        state_class="ExplainerState",
    ),
    "feedback_learner": AgentMethodSpec(
        method="learn",
        uses_kwargs=True,
        tier=5,
        agent_module="src.agents.feedback_learner",
        agent_class="FeedbackLearnerAgent",
        state_module="src.agents.feedback_learner.state",
        state_class="FeedbackLearnerState",
    ),
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


def to_harness_config(spec: AgentMethodSpec) -> Dict[str, object]:
    """Project an ``AgentMethodSpec`` into the legacy ``AGENT_CONFIGS`` shape.

    The harness (``scripts/run_tier1_5_test.py``) reads this dict shape, but
    AGENT_METHOD_MAP is the source of truth. Only sets keys whose value is
    non-default so the projection matches the legacy literal field-for-field
    where the agent doesn't override.
    """
    out: Dict[str, object] = {
        "method": spec.method,
        "is_async": spec.is_async,
    }
    if spec.uses_kwargs:
        out["uses_kwargs"] = spec.uses_kwargs
    if spec.input_model is not None:
        out["input_model"] = spec.input_model
    if spec.input_module is not None:
        out["input_module"] = spec.input_module
    for field_name in (
        "tier",
        "agent_module",
        "agent_class",
        "state_module",
        "state_class",
        "timeout",
    ):
        value = getattr(spec, field_name)
        if value is not None:
            out[field_name] = value
    return out


def get_harness_configs() -> Dict[str, Dict[str, object]]:
    """Compute the harness-side AGENT_CONFIGS dict from AGENT_METHOD_MAP.

    Used by ``scripts/run_tier1_5_test.py`` to ensure the harness and the
    production dispatcher cannot drift on per-agent metadata. See #252.
    """
    return {name: to_harness_config(spec) for name, spec in AGENT_METHOD_MAP.items()}


__all__ = [
    "AgentMethodSpec",
    "AGENT_METHOD_MAP",
    "AGENT_RESPONSE_FIELDS",
    "get_method_spec",
    "extract_narrative",
    "to_harness_config",
    "get_harness_configs",
]
