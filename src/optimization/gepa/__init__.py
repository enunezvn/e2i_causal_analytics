"""GEPA (Generative Evolutionary Prompting with AI) Optimizer.

GEPA is DSPy's state-of-the-art prompt optimizer that uses reflective evolution
to improve agent instructions. It provides 10%+ improvement over MIPROv2.

Key Features:
    - Reflective evolution with rich textual feedback
    - Joint tool optimization for DoWhy/EconML tools
    - Pareto frontier for multi-objective KPI optimization
    - Fewer training examples required

Usage:
    from src.optimization.gepa import create_gepa_optimizer, get_metric_for_agent

    metric = get_metric_for_agent("causal_impact")
    optimizer = create_gepa_optimizer(metric, trainset, valset)
    optimized = optimizer.compile(agent_module)

Submodules:
    metrics: GEPA feedback metrics for each agent type
    tools: DoWhy/EconML tool definitions for optimization
    integration: MLflow, Opik, and RAGAS integrations
"""

from src.optimization.gepa.ab_test import (
    ABTestObservation,
    ABTestResults,
    ABTestVariant,
    GEPAABTest,
)
from src.optimization.gepa.metrics import (
    CausalImpactGEPAMetric,
    E2IGEPAMetric,
    ExperimentDesignerGEPAMetric,
    FeedbackLearnerGEPAMetric,
    StandardAgentGEPAMetric,
    get_metric_for_agent,
)
from src.optimization.gepa.optimizer_setup import (
    AGENT_BUDGETS,
    BUDGET_PRESETS,
    create_gepa_optimizer,
    create_optimizer_for_agent,
    optimize_causal_impact_agent,
    optimize_experiment_designer_agent,
    optimize_feedback_learner_agent,
    optimize_standard_agent,
)
from src.optimization.gepa.versioning import (
    compare_versions,
    compute_instruction_hash,
    generate_version_id,
    list_versions,
    load_optimized_module,
    rollback_to_version,
    save_optimized_module,
)

__all__ = [
    # Metrics
    "E2IGEPAMetric",
    "CausalImpactGEPAMetric",
    "ExperimentDesignerGEPAMetric",
    "FeedbackLearnerGEPAMetric",
    "StandardAgentGEPAMetric",
    "get_metric_for_agent",
    # Optimizer Factory
    "create_gepa_optimizer",
    "create_optimizer_for_agent",
    "BUDGET_PRESETS",
    "AGENT_BUDGETS",
    # Convenience Functions
    "optimize_causal_impact_agent",
    "optimize_experiment_designer_agent",
    "optimize_feedback_learner_agent",
    "optimize_standard_agent",
    # Versioning
    "generate_version_id",
    "compute_instruction_hash",
    "save_optimized_module",
    "load_optimized_module",
    "list_versions",
    "rollback_to_version",
    "compare_versions",
    # A/B Testing
    "GEPAABTest",
    "ABTestVariant",
    "ABTestObservation",
    "ABTestResults",
]
