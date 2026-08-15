"""Agent Factory - Central agent instantiation for orchestrator.

This module provides factory functions for creating agent instances
that can be registered with the orchestrator.

Example:
    from src.agents.factory import create_agent_registry

    # Create full registry with all agents
    registry = create_agent_registry()

    # Create orchestrator with registry
    orchestrator = OrchestratorAgent(agent_registry=registry)

    # Create subset of agents
    registry = create_agent_registry(include_tiers=[1, 2])
"""

import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Set, cast

logger = logging.getLogger(__name__)


# #1448: strict-mode switch for the registry completeness gate. When truthy, a
# registry that is missing ANY enabled + selected agent raises instead of quietly
# degrading. Off by default — see ``_require_full_default`` for the rationale.
REQUIRE_FULL_REGISTRY_ENV = "E2I_REQUIRE_FULL_AGENT_REGISTRY"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


class PartialAgentRegistryError(RuntimeError):
    """The agent registry is missing enabled agents and strict mode is armed.

    Raised by :func:`create_agent_registry` when ``require_full`` resolves True.
    Carries the machine-readable ``dropped`` / ``expected`` sets so a deploy gate
    can report exactly which agents failed to construct.
    """

    def __init__(
        self,
        dropped: Iterable[str],
        expected: Iterable[str],
        registry_size: int,
    ) -> None:
        self.dropped: List[str] = sorted(dropped)
        self.expected: List[str] = sorted(expected)
        self.registry_size = registry_size
        super().__init__(
            f"PARTIAL registry — {len(self.dropped)} of {len(self.expected)} enabled "
            f"agent(s) failed to construct: {self.dropped}. "
            f"Registry has {registry_size} agent(s); dispatches to the missing agents "
            f"would FAIL CLOSED. ({REQUIRE_FULL_REGISTRY_ENV} is armed.)"
        )


def _require_full_default() -> bool:
    """Resolve strict mode from the environment when the caller did not decide.

    Default OFF. A partial registry is a *degradation*, not a corruption: the
    dispatcher already fails closed for a missing agent (#814), so the agents that
    DID construct keep serving. Hard-failing by default would convert "18 of 21
    agents work" into "nothing works" — and, at the only production construction
    site (``src/api/routes/cognitive.get_orchestrator``), into "no orchestrator at
    all". Operators opt in per deployment / per deploy-gate invocation.
    """
    return os.environ.get(REQUIRE_FULL_REGISTRY_ENV, "").strip().lower() in _TRUTHY


# Agent metadata for lazy instantiation
#: Annotated because the value dicts mix int/str/bool, so mypy infers
#: ``dict[str, object]`` and every read has to be re-narrowed at the call
#: site -- `int(cfg["tier"])` in :func:`agent_roster_summary` was a
#: call-overload error against the ceiling (#1638). Declaring the shape
#: once is the fix; narrowing per read would have meant a fallback value,
#: and a roster that silently drops an agent is the exact defect this
#: function exists to prevent.
AGENT_REGISTRY_CONFIG: Dict[str, Dict[str, Any]] = {
    # Tier 0: ML Foundation
    "scope_definer": {
        "tier": 0,
        "module": "src.agents.ml_foundation.scope_definer",
        "class_name": "ScopeDefinerAgent",
        "enabled": True,
    },
    "data_preparer": {
        "tier": 0,
        "module": "src.agents.ml_foundation.data_preparer",
        "class_name": "DataPreparerAgent",
        "enabled": True,
    },
    "feature_analyzer": {
        "tier": 0,
        "module": "src.agents.ml_foundation.feature_analyzer",
        "class_name": "FeatureAnalyzerAgent",
        "enabled": True,
    },
    "model_selector": {
        "tier": 0,
        "module": "src.agents.ml_foundation.model_selector",
        "class_name": "ModelSelectorAgent",
        "enabled": True,
    },
    "model_trainer": {
        "tier": 0,
        "module": "src.agents.ml_foundation.model_trainer",
        "class_name": "ModelTrainerAgent",
        "enabled": True,
    },
    "model_deployer": {
        "tier": 0,
        "module": "src.agents.ml_foundation.model_deployer",
        "class_name": "ModelDeployerAgent",
        "enabled": True,
    },
    "observability_connector": {
        "tier": 0,
        "module": "src.agents.ml_foundation.observability_connector",
        "class_name": "ObservabilityConnectorAgent",
        "enabled": True,
    },
    "cohort_constructor": {
        "tier": 0,
        "module": "src.agents.cohort_constructor",
        "class_name": "CohortConstructorAgent",
        "enabled": True,
    },
    # Chat companion to cohort_constructor: profiles the eligible population by
    # clinical segment (severity + line-of-therapy) with REAL KPI counts. The
    # orchestrator routes COHORT_DEFINITION chat queries here (cohort_constructor
    # materializes patient rows for the ML pipeline and cannot run from chat).
    "cohort_profiler": {
        "tier": 0,
        "module": "src.agents.cohort_profiler",
        "class_name": "CohortProfilerAgent",
        "enabled": True,
    },
    # Tier 1: Coordination
    "orchestrator": {
        "tier": 1,
        "module": "src.agents.orchestrator",
        "class_name": "OrchestratorAgent",
        "enabled": True,
    },
    "tool_composer": {
        "tier": 1,
        "module": "src.agents.tool_composer",
        "class_name": "ToolComposerAgent",
        "enabled": True,  # Enabled in v4.2.1
    },
    # Tier 2: Causal Analytics
    "causal_impact": {
        "tier": 2,
        "module": "src.agents.causal_impact",
        "class_name": "CausalImpactAgent",
        "enabled": True,
    },
    "gap_analyzer": {
        "tier": 2,
        "module": "src.agents.gap_analyzer",
        "class_name": "GapAnalyzerAgent",
        "enabled": True,
    },
    "heterogeneous_optimizer": {
        "tier": 2,
        "module": "src.agents.heterogeneous_optimizer",
        "class_name": "HeterogeneousOptimizerAgent",
        "enabled": True,
    },
    # Tier 3: Monitoring
    "drift_monitor": {
        "tier": 3,
        "module": "src.agents.drift_monitor",
        "class_name": "DriftMonitorAgent",
        "enabled": True,
    },
    "experiment_designer": {
        "tier": 3,
        "module": "src.agents.experiment_designer",
        "class_name": "ExperimentDesignerAgent",
        "enabled": True,
    },
    "health_score": {
        "tier": 3,
        "module": "src.agents.health_score",
        "class_name": "HealthScoreAgent",
        "enabled": True,
    },
    "experiment_monitor": {
        "tier": 3,
        "module": "src.agents.experiment_monitor",
        "class_name": "ExperimentMonitorAgent",
        "enabled": True,
    },
    # Tier 4: ML Predictions
    "prediction_synthesizer": {
        "tier": 4,
        "module": "src.agents.prediction_synthesizer",
        "class_name": "PredictionSynthesizerAgent",
        "enabled": True,
    },
    "resource_optimizer": {
        "tier": 4,
        "module": "src.agents.resource_optimizer",
        "class_name": "ResourceOptimizerAgent",
        "enabled": True,
    },
    # Tier 5: Self-Improvement
    "explainer": {
        "tier": 5,
        "module": "src.agents.explainer",
        "class_name": "ExplainerAgent",
        "enabled": True,
    },
    "feedback_learner": {
        "tier": 5,
        "module": "src.agents.feedback_learner",
        "class_name": "FeedbackLearnerAgent",
        "enabled": True,
    },
}

#: Display names for the tiers in :data:`AGENT_REGISTRY_CONFIG`. The tier NUMBERS
#: are already SSOT in the registry above; only their human labels were missing,
#: which is why every surface that wanted to describe the architecture typed its
#: own prose and drifted (#1638).
#: Taken from the registry's own section comments above rather than from any
#: consumer's prose -- those had drifted apart too (the frontend called tier 1
#: "Coordination" and tier 5 "Learning"; the routing signature called them
#: "Orchestration" and "Learning"; the registry says "Coordination" and
#: "Self-Improvement"). Adjacent-to-the-data wins.
AGENT_TIER_NAMES: Dict[int, str] = {
    0: "ML Foundation",
    1: "Coordination",
    2: "Causal Analytics",
    3: "Monitoring",
    4: "ML Predictions",
    5: "Self-Improvement",
}


def build_agent_roster_block() -> str:
    """Render the agent roster as prompt-ready text, derived from the registry.

    #1638: turn 5.2 ("what agents are available") answered with TOOL names because
    the AG-UI answering prompt carried no roster at all -- only the phrase "the
    tiered architecture" with a count that had gone stale. A hand-written list
    would have fixed that turn and rotted at the next agent; this is generated, so
    adding an agent to :data:`AGENT_REGISTRY_CONFIG` updates every consumer with
    no second edit.

    Only ENABLED agents are listed: a disabled agent cannot be dispatched, so
    naming it as available would be the same class of untruth in reverse.
    """
    enabled = {n: c for n, c in AGENT_REGISTRY_CONFIG.items() if c.get("enabled")}
    by_tier: Dict[int, List[str]] = {}
    for name, cfg in enabled.items():
        by_tier.setdefault(int(cfg["tier"]), []).append(name)

    lines = [f"The E2I system has {len(enabled)} agents organized in {len(by_tier)} tiers:"]
    for tier in sorted(by_tier):
        label = AGENT_TIER_NAMES.get(tier, f"Tier {tier}")
        lines.append(f"- Tier {tier}: {label} ({', '.join(sorted(by_tier[tier]))})")
    return "\n".join(lines)


def create_agent_registry(
    include_tiers: Optional[List[int]] = None,
    include_agents: Optional[List[str]] = None,
    exclude_agents: Optional[List[str]] = None,
    fail_on_import_error: bool = False,
    require_full: Optional[bool] = None,
) -> Dict[str, Any]:
    """Create agent registry with instantiated agents.

    This factory creates all enabled agents and returns a dict
    suitable for passing to OrchestratorAgent.

    Args:
        include_tiers: Only include agents from these tiers (0-5).
                       If None, includes all tiers.
        include_agents: Explicit list of agent names to include.
                        If provided, overrides include_tiers.
        exclude_agents: Agent names to exclude from registry.
        fail_on_import_error: If True, raise on import errors.
                              If False, log warning and continue.
        require_full: #1448 completeness gate. If True, raise
                      :class:`PartialAgentRegistryError` when any selected+enabled
                      agent failed to construct. If None (default), read
                      ``E2I_REQUIRE_FULL_AGENT_REGISTRY`` from the environment.
                      Pass False to force the degrade-and-log behaviour even when
                      the env flag is armed (subset builders: benchmarks, CLIs).

    Returns:
        Dict mapping agent_name to agent instance

    Raises:
        PartialAgentRegistryError: strict mode armed and the registry is partial.
        ImportError: ``fail_on_import_error`` and an agent failed to import.

    Example:
        # All enabled agents
        registry = create_agent_registry()

        # Only Tier 2 agents
        registry = create_agent_registry(include_tiers=[2])

        # Specific agents
        registry = create_agent_registry(
            include_agents=["causal_impact", "heterogeneous_optimizer"]
        )
    """
    registry: Dict[str, Any] = {}
    exclude_set: Set[str] = set(exclude_agents or [])
    # Agents that were ENABLED + selected but failed to instantiate. Tracked so a
    # PARTIAL registry is surfaced loudly to operators (#814): a dropped agent
    # makes the dispatcher fail closed for that route, so a silent drop would look
    # like a routing bug rather than a missing-credential / import misconfig.
    dropped: List[str] = []
    # Agents that PASSED every filter and were therefore expected in the result.
    # The denominator of the completeness gate (#1448).
    expected: List[str] = []

    for agent_name, config in AGENT_REGISTRY_CONFIG.items():
        # Skip disabled agents
        if not config.get("enabled", False):
            logger.debug(f"Skipping disabled agent: {agent_name}")
            continue

        # Skip excluded agents
        if agent_name in exclude_set:
            logger.debug(f"Skipping excluded agent: {agent_name}")
            continue

        # Check tier filter
        if include_agents:
            # Explicit include list takes precedence
            if agent_name not in include_agents:
                continue
        elif include_tiers:
            # Filter by tier
            if config["tier"] not in include_tiers:
                continue

        expected.append(agent_name)

        # Try to instantiate agent
        try:
            agent_instance = _create_agent(
                module_path=cast(str, config["module"]),
                class_name=cast(str, config["class_name"]),
            )
            if agent_instance:
                registry[agent_name] = agent_instance
                logger.info(f"Registered agent: {agent_name} (Tier {config['tier']})")
            else:
                dropped.append(agent_name)
                logger.warning(f"Agent {agent_name} returned no instance; dropped from registry")

        except Exception as e:
            if fail_on_import_error:
                raise ImportError(f"Failed to import agent {agent_name}: {e}") from e
            dropped.append(agent_name)
            logger.warning(f"Failed to create agent {agent_name}: {e}")

    if dropped:
        strict = _require_full_default() if require_full is None else require_full
        # #1448: ERROR, not WARNING. A partial registry means named capabilities are
        # gone from production; at WARNING it sat in the log stream on every worker
        # boot until readers stopped seeing it. ERROR is the severity alerting rules
        # key on, and the structured ``extra`` fields make it machine-actionable
        # rather than something an operator has to regex out of prose.
        logger.error(
            "create_agent_registry: PARTIAL registry — %d of %d enabled agent(s) "
            "dropped: %s. Dispatches routed to these agents will FAIL CLOSED (no "
            "fabricated fallback); check missing credentials/imports/packaging "
            "(e.g. #1448: pyproject.toml project-root marker absent from the image). "
            "%s=%s",
            len(dropped),
            len(expected),
            sorted(dropped),
            REQUIRE_FULL_REGISTRY_ENV,
            strict,
            extra={
                "dropped_agents": sorted(dropped),
                "expected_agent_count": len(expected),
                "registry_size": len(registry),
                "require_full_agent_registry": strict,
            },
        )
        if strict:
            raise PartialAgentRegistryError(
                dropped=dropped, expected=expected, registry_size=len(registry)
            )
    logger.info(f"Created agent registry with {len(registry)} agents")
    return registry


def assert_full_agent_registry(**kwargs: Any) -> Dict[str, Any]:
    """Build the registry and RAISE unless every enabled agent constructed (#1448).

    The deploy gate. Unlike the ``E2I_REQUIRE_FULL_AGENT_REGISTRY`` env default this
    is unconditional, so an operator can probe a *running* container without
    mutating its environment or restarting it::

        docker compose -f docker/docker-compose.yml exec api \\
          python -c "from src.agents.factory import assert_full_agent_registry as a; \\
                     print(len(a(exclude_agents=['orchestrator'])))"

    Returns:
        The complete registry (never partial).

    Raises:
        PartialAgentRegistryError: any selected+enabled agent failed to construct.
    """
    kwargs["require_full"] = True
    return create_agent_registry(**kwargs)


def _create_agent(module_path: str, class_name: str) -> Optional[Any]:
    """Create a single agent instance via dynamic import.

    Args:
        module_path: Full module path (e.g., "src.agents.causal_impact")
        class_name: Class name to instantiate (e.g., "CausalImpactAgent")

    Returns:
        Agent instance or None if import fails
    """
    import importlib

    try:
        module = importlib.import_module(module_path)
        agent_class = getattr(module, class_name)
        # Phase 3 / G5 + #840: prediction_synthesizer needs BOTH production model
        # clients (loaded from a deployment manifest) AND a live model_registry
        # (so the orchestrator can resolve deployable champions for a target).
        # Both fall back to a fail-closed empty/no-op state (no manifest -> {}
        # clients; no DB client -> registry returns []), so the agent reports
        # status="failed" honestly rather than fabricating a prediction.
        if class_name == "PredictionSynthesizerAgent":
            return agent_class(**_prediction_synthesizer_kwargs())
        # #1450: health_score needs its four real backends. Without them every
        # dimension fail-closes to UNMEASURED and a chat health question answers
        # "UNKNOWN - nothing was measured" (#1447 narration) — which is honest
        # but useless. The REST route already wires the real adapters in
        # ``_execute_health_check``; this is the same construction for the CHAT
        # path (cognitive.get_orchestrator -> create_agent_registry -> here).
        if class_name == "HealthScoreAgent":
            return agent_class(**_health_score_kwargs())
        return agent_class()
    except ImportError as e:
        logger.warning(f"Import error for {module_path}.{class_name}: {e}")
        return None
    except AttributeError as e:
        logger.warning(f"Class not found: {class_name} in {module_path}: {e}")
        return None


def _health_score_kwargs() -> Dict[str, Any]:
    """Build the health_score constructor kwargs (#1450).

    Injects the SAME real backends ``src/api/routes/health_score.py``'s
    ``_execute_health_check`` uses — the component health client plus the three
    store adapters built by ``_build_real_health_stores`` — so a health check
    dispatched from CHAT measures the same four dimensions the REST endpoint
    does. The adapters are deliberately REUSED rather than reimplemented: they
    are the single bridge from the live tables (ml_model_health_dashboard /
    ml_performance_metrics / etl_pipeline_metrics / agent_registry) into the
    node Protocols.

    The route module is imported lazily HERE (not at module scope) for the same
    reason ``src/agents/cohort_profiler/agent.py`` imports ``routes.kpi``
    lazily: the routes module imports agents inside its own functions, so a
    module-level import in either direction would be a cycle. Construction is
    cheap — the adapters do no I/O until a node first calls them, and each
    fails CLOSED to an honest UNMEASURED null if its backend is unreachable.
    """
    from src.agents.health_score import SupabaseHealthClient
    from src.api.routes.health_score import _build_real_health_stores

    metrics_store, pipeline_store, agent_registry = _build_real_health_stores()
    return {
        "health_client": SupabaseHealthClient(),
        "metrics_store": metrics_store,
        "pipeline_store": pipeline_store,
        "agent_registry": agent_registry,
    }


def _prediction_synthesizer_kwargs() -> Dict[str, Any]:
    """Build the prediction_synthesizer constructor kwargs (#840).

    Injects BOTH the deployment-manifest-loaded ``model_clients`` and a
    ``LiveChampionModelRegistry`` so the orchestrator can resolve the deployable
    champion model names for a target and drive their clients. Both are
    fail-closed: an absent manifest yields ``{}`` clients and an unavailable DB
    yields an empty registry result, so the agent fails closed honestly.
    """
    from src.agents.prediction_synthesizer.registry_adapter import (
        LiveChampionModelRegistry,
    )

    return {
        "model_clients": _try_load_prod_model_clients(),
        "model_registry": LiveChampionModelRegistry(),
    }


def _try_load_prod_model_clients() -> Dict[str, Any]:
    """Best-effort loader for prediction_synthesizer model clients.

    Resolution order:
      1. ``E2I_MODEL_DEPLOYMENT_MANIFEST_PATH`` env var (JSON file path)
      2. ``data/ml_artifacts/deployment_manifest.json`` if it exists in CWD

    The default read path matches where the deploy CLI writes the manifest
    (#857): the writable ``data/ml_artifacts/`` named volume. The legacy
    ``data/`` root is read-only in the prod api container, so a manifest could
    never be written there for the factory to find.

    Any failure (file missing, parse error, bad URI) returns ``{}`` and logs
    a warning. With ``{}`` clients and a live ``model_registry`` (see
    ``_prediction_synthesizer_kwargs``), the orchestrator resolves the target's
    deployable models from the registry and then finds no matching client, so it
    FAILS CLOSED (status="failed") rather than fabricating a prediction.
    """
    import os
    from pathlib import Path

    try:
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            load_clients_from_deployment_manifest_file,
        )
    except ImportError as e:
        logger.warning(f"In-process model client unavailable: {e}")
        return {}

    manifest_path_str = os.environ.get("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH")
    if manifest_path_str:
        return load_clients_from_deployment_manifest_file(manifest_path_str)

    default_path = Path("data/ml_artifacts/deployment_manifest.json")
    if default_path.exists():
        return load_clients_from_deployment_manifest_file(default_path)

    return {}


def get_agent_config(agent_name: str) -> Optional[Dict[str, Any]]:
    """Get configuration for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        Config dict or None if not found
    """
    return AGENT_REGISTRY_CONFIG.get(agent_name)


def list_available_agents(tier: Optional[int] = None) -> List[str]:
    """List all available agent names.

    Args:
        tier: Filter by tier (0-5)

    Returns:
        List of agent names
    """
    agents = []
    for name, config in AGENT_REGISTRY_CONFIG.items():
        if config.get("enabled", False):
            if tier is None or config["tier"] == tier:
                agents.append(name)
    return agents


def get_tier0_agents() -> Dict[str, Any]:
    """Convenience function for Tier 0 (ML Foundation) agents.

    Returns:
        Dict with enabled ML foundation agents (data_preparer, etc.)
    """
    return create_agent_registry(include_tiers=[0])


def get_tier2_agents() -> Dict[str, Any]:
    """Convenience function for Tier 2 (Causal Analytics) agents.

    Returns:
        Dict with causal_impact, gap_analyzer, heterogeneous_optimizer
    """
    return create_agent_registry(include_tiers=[2])


def get_all_analytics_agents() -> Dict[str, Any]:
    """Get all analytics agents (Tiers 2-5).

    Excludes orchestrator (Tier 1) which coordinates.

    Returns:
        Dict with all analytics agents
    """
    return create_agent_registry(include_tiers=[2, 3, 4, 5])
