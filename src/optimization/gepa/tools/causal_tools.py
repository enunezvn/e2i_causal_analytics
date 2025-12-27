"""DoWhy/EconML Tool Definitions for GEPA Optimization.

This module provides optimizable tool definitions for GEPA's joint tool optimization
feature. When enable_tool_optimization=True, GEPA evolves both agent instructions
and tool descriptions together.

Primary use case: Causal Impact agent (Tier 2 Hybrid) with DoWhy/EconML estimators.
"""

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class GEPATool:
    """Represents an optimizable tool for GEPA.

    Attributes:
        name: Tool identifier (e.g., 'causal_forest')
        description: Human-readable description (optimizable by GEPA)
        arguments: Dict of argument names to descriptions
        category: Tool category for grouping
        when_to_use: Guidance on when to select this tool
        when_not_to_use: Guidance on when NOT to use this tool
        prerequisites: Required conditions before using this tool
    """

    name: str
    description: str
    arguments: dict[str, str] = field(default_factory=dict)
    category: str = "causal"
    when_to_use: str = ""
    when_not_to_use: str = ""
    prerequisites: list[str] = field(default_factory=list)


# ==============================================================================
# DoWhy Estimators
# ==============================================================================

LINEAR_DML = GEPATool(
    name="linear_dml",
    description=(
        "Linear Double Machine Learning estimator from EconML. "
        "Uses two-stage ML to estimate average treatment effects while "
        "controlling for high-dimensional confounders. Best for linear relationships "
        "with continuous treatments and many potential confounders."
    ),
    arguments={
        "treatment": "The treatment variable column name",
        "outcome": "The outcome variable column name",
        "confounders": "List of confounder column names",
        "model_y": "ML model for outcome (default: LassoCV)",
        "model_t": "ML model for treatment (default: LassoCV)",
        "cv": "Number of cross-validation folds (default: 5)",
    },
    category="dml",
    when_to_use=(
        "Use when: (1) treatment effect is approximately linear, "
        "(2) many potential confounders exist, (3) need average treatment effect"
    ),
    when_not_to_use=(
        "Avoid when: (1) treatment effects vary by subgroup (use CausalForest), "
        "(2) data has strong non-linearities, (3) sample size < 1000"
    ),
    prerequisites=["DAG defined", "Confounders identified", "Treatment continuous"],
)

SPARSE_LINEAR_DML = GEPATool(
    name="sparse_linear_dml",
    description=(
        "Sparse Linear DML for high-dimensional settings with many irrelevant features. "
        "Adds L1 regularization to automatically select important confounders. "
        "Particularly useful when confounder set is large and sparse."
    ),
    arguments={
        "treatment": "The treatment variable column name",
        "outcome": "The outcome variable column name",
        "confounders": "List of confounder column names (can be many)",
        "alpha": "Regularization strength (default: 'auto')",
    },
    category="dml",
    when_to_use=(
        "Use when: (1) many potential confounders but most are irrelevant, "
        "(2) feature selection is needed, (3) interpretability important"
    ),
    when_not_to_use=(
        "Avoid when: (1) all confounders are known to be relevant, "
        "(2) sample size is very small relative to features"
    ),
    prerequisites=["DAG defined", "High-dimensional confounders"],
)

CAUSAL_FOREST = GEPATool(
    name="causal_forest",
    description=(
        "Causal Forest estimator for heterogeneous treatment effect estimation. "
        "Uses random forest methodology to estimate Conditional Average Treatment Effects (CATE) "
        "across different subgroups. Best for discovering which segments respond most to treatment."
    ),
    arguments={
        "treatment": "The treatment variable column name (binary)",
        "outcome": "The outcome variable column name",
        "confounders": "List of confounder column names",
        "effect_modifiers": "Variables that may modify treatment effect",
        "n_estimators": "Number of trees (default: 100)",
        "min_samples_leaf": "Minimum samples per leaf (default: 5)",
    },
    category="forest",
    when_to_use=(
        "Use when: (1) treatment effects likely vary by subgroup, "
        "(2) want to identify high-responder segments, "
        "(3) have binary treatment, (4) sample size > 2000"
    ),
    when_not_to_use=(
        "Avoid when: (1) only need average effect, "
        "(2) treatment is continuous, (3) sample size < 1000"
    ),
    prerequisites=["DAG defined", "Binary treatment", "Effect modifiers identified"],
)

DOUBLE_ROBUST_LEARNER = GEPATool(
    name="double_robust_learner",
    description=(
        "Doubly Robust Learner that combines propensity score weighting with outcome modeling. "
        "Provides consistent estimates if either the propensity or outcome model is correct. "
        "More robust to model misspecification than single-model approaches."
    ),
    arguments={
        "treatment": "The treatment variable column name",
        "outcome": "The outcome variable column name",
        "confounders": "List of confounder column names",
        "propensity_model": "Model for treatment propensity",
        "outcome_model": "Model for outcome regression",
    },
    category="robust",
    when_to_use=(
        "Use when: (1) unsure about correct model specification, "
        "(2) want robustness to model misspecification, "
        "(3) have selection on observables"
    ),
    when_not_to_use=(
        "Avoid when: (1) both models likely wrong in same direction, "
        "(2) extreme propensity scores present"
    ),
    prerequisites=["DAG defined", "Propensity model specified"],
)

METALEARNER_T = GEPATool(
    name="metalearner_t",
    description=(
        "T-Learner meta-learning approach for CATE estimation. "
        "Fits separate outcome models for treatment and control groups, "
        "then computes difference. Simple and interpretable approach."
    ),
    arguments={
        "treatment": "The treatment variable column name (binary)",
        "outcome": "The outcome variable column name",
        "confounders": "List of confounder column names",
        "base_model": "Base ML model to use (default: GradientBoosting)",
    },
    category="metalearner",
    when_to_use=(
        "Use when: (1) binary treatment, (2) want simple interpretation, "
        "(3) treatment and control groups have similar distributions"
    ),
    when_not_to_use=(
        "Avoid when: (1) groups have very different covariate distributions, "
        "(2) need uncertainty quantification"
    ),
    prerequisites=["Binary treatment", "Sufficient samples in both groups"],
)

METALEARNER_S = GEPATool(
    name="metalearner_s",
    description=(
        "S-Learner meta-learning approach for CATE estimation. "
        "Fits single model including treatment as feature, then computes "
        "counterfactual predictions. Can detect zero treatment effects."
    ),
    arguments={
        "treatment": "The treatment variable column name",
        "outcome": "The outcome variable column name",
        "confounders": "List of confounder column names",
        "base_model": "Base ML model to use (default: GradientBoosting)",
    },
    category="metalearner",
    when_to_use=(
        "Use when: (1) treatment might have no effect, "
        "(2) want regularization toward zero effect, "
        "(3) continuous or binary treatment"
    ),
    when_not_to_use=(
        "Avoid when: (1) treatment effect is known to be non-zero, "
        "(2) base model ignores treatment variable"
    ),
    prerequisites=["Treatment can be included as feature"],
)

INSTRUMENTAL_VARIABLE = GEPATool(
    name="instrumental_variable",
    description=(
        "Instrumental Variable estimation for handling unobserved confounding. "
        "Uses an instrument that affects outcome only through treatment. "
        "Essential when randomization not possible and confounders unmeasured."
    ),
    arguments={
        "treatment": "The treatment variable column name",
        "outcome": "The outcome variable column name",
        "instrument": "The instrumental variable column name",
        "confounders": "List of measured confounder column names (optional)",
    },
    category="iv",
    when_to_use=(
        "Use when: (1) unobserved confounders likely present, "
        "(2) valid instrument available, (3) cannot randomize treatment"
    ),
    when_not_to_use=(
        "Avoid when: (1) no valid instrument available, "
        "(2) instrument might affect outcome directly, "
        "(3) instrument is weak (F-stat < 10)"
    ),
    prerequisites=["Valid instrument identified", "Instrument exclusion justified"],
)

REGRESSION_DISCONTINUITY = GEPATool(
    name="regression_discontinuity",
    description=(
        "Regression Discontinuity Design for sharp treatment thresholds. "
        "Exploits discontinuous jumps in treatment probability at a cutoff. "
        "Strong internal validity when assumptions hold."
    ),
    arguments={
        "running_variable": "The variable determining treatment assignment",
        "cutoff": "The threshold value for treatment",
        "outcome": "The outcome variable column name",
        "bandwidth": "Bandwidth around cutoff (default: 'optimal')",
    },
    category="quasi_experimental",
    when_to_use=(
        "Use when: (1) treatment assigned by threshold rule, "
        "(2) running variable is continuous, "
        "(3) no manipulation around cutoff"
    ),
    when_not_to_use=(
        "Avoid when: (1) cutoff is not sharp, "
        "(2) subjects can manipulate running variable, "
        "(3) insufficient observations near cutoff"
    ),
    prerequisites=["Sharp cutoff exists", "Running variable identified"],
)

DIFFERENCE_IN_DIFFERENCES = GEPATool(
    name="difference_in_differences",
    description=(
        "Difference-in-Differences estimator for panel data. "
        "Compares treatment and control groups before and after intervention. "
        "Controls for time-invariant confounders and common time trends."
    ),
    arguments={
        "treatment": "The treatment indicator column",
        "post_period": "The post-treatment period indicator",
        "outcome": "The outcome variable column name",
        "entity_id": "Entity identifier for panel structure",
        "time_id": "Time period identifier",
    },
    category="quasi_experimental",
    when_to_use=(
        "Use when: (1) have pre/post data for treatment and control, "
        "(2) parallel trends assumption plausible, "
        "(3) treatment timing is known"
    ),
    when_not_to_use=(
        "Avoid when: (1) parallel trends violated, "
        "(2) treatment affects control group (spillovers), "
        "(3) only post-treatment data available"
    ),
    prerequisites=["Panel data available", "Parallel trends justified"],
)

# ==============================================================================
# Tool Registry
# ==============================================================================

CAUSAL_TOOLS = [
    LINEAR_DML,
    SPARSE_LINEAR_DML,
    CAUSAL_FOREST,
    DOUBLE_ROBUST_LEARNER,
    METALEARNER_T,
    METALEARNER_S,
    INSTRUMENTAL_VARIABLE,
    REGRESSION_DISCONTINUITY,
    DIFFERENCE_IN_DIFFERENCES,
]

# Tool lookup by name
TOOL_REGISTRY: dict[str, GEPATool] = {tool.name: tool for tool in CAUSAL_TOOLS}


def get_tools_for_agent(agent_name: str) -> list[GEPATool]:
    """Get the tools available for a specific agent.

    Args:
        agent_name: Name of the agent

    Returns:
        List of GEPATool objects available to this agent
    """
    # Causal agents get all causal tools
    if agent_name in ["causal_impact", "heterogeneous_optimizer", "gap_analyzer"]:
        return CAUSAL_TOOLS

    # Experiment designer gets quasi-experimental tools
    if agent_name == "experiment_designer":
        return [
            DIFFERENCE_IN_DIFFERENCES,
            REGRESSION_DISCONTINUITY,
        ]

    # Other agents get no special tools
    return []


def get_tool_by_name(name: str) -> Optional[GEPATool]:
    """Get a specific tool by name.

    Args:
        name: Tool name

    Returns:
        GEPATool if found, None otherwise
    """
    return TOOL_REGISTRY.get(name)


def get_tools_by_category(category: str) -> list[GEPATool]:
    """Get all tools in a category.

    Args:
        category: Tool category (e.g., 'dml', 'forest', 'metalearner')

    Returns:
        List of matching GEPATool objects
    """
    return [tool for tool in CAUSAL_TOOLS if tool.category == category]


__all__ = [
    # Data classes
    "GEPATool",
    # Tool definitions
    "LINEAR_DML",
    "SPARSE_LINEAR_DML",
    "CAUSAL_FOREST",
    "DOUBLE_ROBUST_LEARNER",
    "METALEARNER_T",
    "METALEARNER_S",
    "INSTRUMENTAL_VARIABLE",
    "REGRESSION_DISCONTINUITY",
    "DIFFERENCE_IN_DIFFERENCES",
    # Registry
    "CAUSAL_TOOLS",
    "TOOL_REGISTRY",
    # Functions
    "get_tools_for_agent",
    "get_tool_by_name",
    "get_tools_by_category",
]
