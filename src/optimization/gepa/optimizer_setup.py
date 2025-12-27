"""GEPA Optimizer Factory for E2I Agents.

This module provides factory functions to create configured GEPA optimizers
for different agent types in the E2I 18-agent 6-tier architecture.

GEPA (Generative Evolutionary Prompting with AI) provides:
- 10%+ performance improvement over MIPROv2
- Reflective evolution with rich textual feedback
- Joint tool optimization for DoWhy/EconML tools
- Pareto frontier for multi-objective KPI optimization
"""

from typing import TYPE_CHECKING, Optional

from dspy import Example

from src.optimization.gepa.metrics import (
    CausalImpactGEPAMetric,
    E2IGEPAMetric,
    ExperimentDesignerGEPAMetric,
    FeedbackLearnerGEPAMetric,
    StandardAgentGEPAMetric,
    get_metric_for_agent,
)

if TYPE_CHECKING:
    import dspy


# Budget presets for different agent types
BUDGET_PRESETS = {
    "light": {
        "max_metric_calls": 500,
        "description": "Quick experimentation for Standard agents",
    },
    "medium": {
        "max_metric_calls": 2000,
        "description": "Balanced optimization for Hybrid agents",
    },
    "heavy": {
        "max_metric_calls": 4000,
        "description": "Thorough optimization for Deep agents",
    },
}

# Agent type to budget mapping
AGENT_BUDGETS = {
    # Tier 2: Hybrid agents (medium budget)
    "causal_impact": "medium",
    "gap_analyzer": "medium",
    "heterogeneous_optimizer": "medium",
    # Tier 3: Hybrid agents (medium budget)
    "experiment_designer": "medium",
    # Tier 5: Deep agents (heavy budget)
    "explainer": "heavy",
    "feedback_learner": "heavy",
    # All others: Standard agents (light budget)
}


def create_gepa_optimizer(
    metric: E2IGEPAMetric,
    trainset: list[Example],
    valset: Optional[list[Example]] = None,
    reflection_model: str = "anthropic/claude-sonnet-4-20250514",
    auto: str = "medium",
    enable_tool_optimization: bool = False,
    log_dir: Optional[str] = None,
    seed: int = 42,
    **kwargs,
) -> "dspy.GEPA":
    """Factory function to create configured GEPA optimizer.

    Args:
        metric: E2I GEPA metric instance (implements E2IGEPAMetric protocol)
        trainset: Training examples for optimization
        valset: Validation examples (uses trainset if None)
        reflection_model: LM for reflection (default: Claude Sonnet)
        auto: Budget preset ("light", "medium", "heavy")
        enable_tool_optimization: Enable for Hybrid agents with DoWhy/EconML tools
        log_dir: Directory for GEPA logs
        seed: Random seed for reproducibility
        **kwargs: Additional GEPA parameters

    Returns:
        Configured GEPA optimizer ready for compile()

    Example:
        >>> metric = CausalImpactGEPAMetric()
        >>> gepa = create_gepa_optimizer(
        ...     metric=metric,
        ...     trainset=train_examples,
        ...     valset=val_examples,
        ...     auto="medium",
        ...     enable_tool_optimization=True,
        ... )
        >>> optimized_module = gepa.compile(student_module, trainset, valset)
    """
    from dspy import GEPA, LM

    # Configure reflection LM
    reflection_lm = LM(
        model=reflection_model,
        temperature=1.0,
        max_tokens=16000,
    )

    # Build GEPA optimizer
    gepa = GEPA(
        metric=metric,
        auto=auto,
        reflection_lm=reflection_lm,
        enable_tool_optimization=enable_tool_optimization,
        candidate_selection_strategy="pareto",
        use_merge=True,
        max_merge_invocations=5,
        track_stats=True,
        use_mlflow=True,
        log_dir=log_dir,
        seed=seed,
        **kwargs,
    )

    return gepa


def create_optimizer_for_agent(
    agent_name: str,
    trainset: list[Example],
    valset: Optional[list[Example]] = None,
    log_dir: Optional[str] = None,
    **kwargs,
) -> "dspy.GEPA":
    """Create a GEPA optimizer configured for a specific agent.

    Automatically selects the appropriate metric and budget based on agent type.

    Args:
        agent_name: Name of the agent (e.g., 'causal_impact', 'orchestrator')
        trainset: Training examples
        valset: Validation examples (uses trainset if None)
        log_dir: Directory for GEPA logs
        **kwargs: Additional GEPA parameters

    Returns:
        Configured GEPA optimizer for the agent

    Example:
        >>> gepa = create_optimizer_for_agent(
        ...     agent_name="causal_impact",
        ...     trainset=train_examples,
        ...     log_dir="./gepa_logs/causal_impact",
        ... )
        >>> optimized = gepa.compile(causal_impact_module, trainset)
    """
    # Get metric instance for agent
    metric = get_metric_for_agent(agent_name)

    # Get budget for agent
    budget = AGENT_BUDGETS.get(agent_name, "light")

    # Hybrid agents get tool optimization
    enable_tools = agent_name in [
        "causal_impact",
        "experiment_designer",
        "gap_analyzer",
        "heterogeneous_optimizer",
    ]

    # Set default log dir if not provided
    if log_dir is None:
        log_dir = f"./gepa_logs/{agent_name}"

    return create_gepa_optimizer(
        metric=metric,
        trainset=trainset,
        valset=valset,
        auto=budget,
        enable_tool_optimization=enable_tools,
        log_dir=log_dir,
        **kwargs,
    )


def optimize_causal_impact_agent(
    student_module,
    trainset: list[Example],
    valset: Optional[list[Example]] = None,
    log_dir: str = "./gepa_logs/causal_impact",
):
    """Optimize Causal Impact agent with GEPA.

    Convenience function for Tier 2 Hybrid agent optimization.

    Args:
        student_module: DSPy module to optimize
        trainset: Training examples
        valset: Validation examples
        log_dir: Directory for GEPA logs

    Returns:
        Optimized DSPy module
    """
    metric = CausalImpactGEPAMetric()

    gepa = create_gepa_optimizer(
        metric=metric,
        trainset=trainset,
        valset=valset,
        auto="medium",
        enable_tool_optimization=True,
        log_dir=log_dir,
    )

    return gepa.compile(
        student=student_module,
        trainset=trainset,
        valset=valset or trainset,
    )


def optimize_experiment_designer_agent(
    student_module,
    trainset: list[Example],
    valset: Optional[list[Example]] = None,
    log_dir: str = "./gepa_logs/experiment_designer",
):
    """Optimize Experiment Designer agent with GEPA.

    Convenience function for Tier 3 Hybrid agent optimization.

    Args:
        student_module: DSPy module to optimize
        trainset: Training examples
        valset: Validation examples
        log_dir: Directory for GEPA logs

    Returns:
        Optimized DSPy module
    """
    metric = ExperimentDesignerGEPAMetric()

    gepa = create_gepa_optimizer(
        metric=metric,
        trainset=trainset,
        valset=valset,
        auto="medium",
        enable_tool_optimization=True,
        log_dir=log_dir,
    )

    return gepa.compile(
        student=student_module,
        trainset=trainset,
        valset=valset or trainset,
    )


def optimize_feedback_learner_agent(
    student_module,
    trainset: list[Example],
    valset: Optional[list[Example]] = None,
    log_dir: str = "./gepa_logs/feedback_learner",
):
    """Optimize Feedback Learner agent with GEPA.

    Convenience function for Tier 5 Deep agent optimization.

    Args:
        student_module: DSPy module to optimize
        trainset: Training examples
        valset: Validation examples
        log_dir: Directory for GEPA logs

    Returns:
        Optimized DSPy module
    """
    metric = FeedbackLearnerGEPAMetric()

    gepa = create_gepa_optimizer(
        metric=metric,
        trainset=trainset,
        valset=valset,
        auto="heavy",
        enable_tool_optimization=False,
        log_dir=log_dir,
    )

    return gepa.compile(
        student=student_module,
        trainset=trainset,
        valset=valset or trainset,
    )


def optimize_standard_agent(
    student_module,
    trainset: list[Example],
    valset: Optional[list[Example]] = None,
    sla_threshold_ms: int = 2000,
    log_dir: Optional[str] = None,
):
    """Optimize Standard agent with GEPA (light budget).

    Convenience function for Standard agents in Tiers 0, 1, and 4.

    Args:
        student_module: DSPy module to optimize
        trainset: Training examples
        valset: Validation examples
        sla_threshold_ms: SLA threshold in milliseconds
        log_dir: Directory for GEPA logs

    Returns:
        Optimized DSPy module
    """
    metric = StandardAgentGEPAMetric(sla_threshold_ms=sla_threshold_ms)

    gepa = create_gepa_optimizer(
        metric=metric,
        trainset=trainset,
        valset=valset,
        auto="light",
        enable_tool_optimization=False,
        log_dir=log_dir,
    )

    return gepa.compile(
        student=student_module,
        trainset=trainset,
        valset=valset or trainset,
    )


__all__ = [
    # Main factory
    "create_gepa_optimizer",
    "create_optimizer_for_agent",
    # Convenience functions
    "optimize_causal_impact_agent",
    "optimize_experiment_designer_agent",
    "optimize_feedback_learner_agent",
    "optimize_standard_agent",
    # Constants
    "BUDGET_PRESETS",
    "AGENT_BUDGETS",
]
