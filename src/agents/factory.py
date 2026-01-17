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
from typing import Any, Dict, List, Literal, Optional, Set

logger = logging.getLogger(__name__)


# Agent metadata for lazy instantiation
AGENT_REGISTRY_CONFIG = {
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


def create_agent_registry(
    include_tiers: Optional[List[int]] = None,
    include_agents: Optional[List[str]] = None,
    exclude_agents: Optional[List[str]] = None,
    fail_on_import_error: bool = False,
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

    Returns:
        Dict mapping agent_name to agent instance

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

        # Try to instantiate agent
        try:
            agent_instance = _create_agent(
                module_path=config["module"],
                class_name=config["class_name"],
            )
            if agent_instance:
                registry[agent_name] = agent_instance
                logger.info(f"Registered agent: {agent_name} (Tier {config['tier']})")

        except Exception as e:
            if fail_on_import_error:
                raise ImportError(f"Failed to import agent {agent_name}: {e}") from e
            logger.warning(f"Failed to create agent {agent_name}: {e}")

    logger.info(f"Created agent registry with {len(registry)} agents")
    return registry


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
        return agent_class()
    except ImportError as e:
        logger.warning(f"Import error for {module_path}.{class_name}: {e}")
        return None
    except AttributeError as e:
        logger.warning(f"Class not found: {class_name} in {module_path}: {e}")
        return None


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
