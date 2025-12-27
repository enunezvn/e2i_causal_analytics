"""GEPA Tool Definitions for E2I Agents.

This module provides optimizable tool descriptions for GEPA's joint
tool optimization feature. Primarily used for DoWhy/EconML tools
in the Causal Impact agent.

When enable_tool_optimization=True, GEPA evolves both:
1. Agent instructions (how to reason about tools)
2. Tool descriptions (how tools are presented to the LLM)

Available Tools:
    - LINEAR_DML: Linear Double Machine Learning
    - SPARSE_LINEAR_DML: Sparse Linear DML for high-dimensional settings
    - CAUSAL_FOREST: Heterogeneous treatment effect estimation
    - DOUBLE_ROBUST_LEARNER: Doubly robust estimation
    - METALEARNER_T: T-Learner meta-learning
    - METALEARNER_S: S-Learner meta-learning
    - INSTRUMENTAL_VARIABLE: IV estimation for unobserved confounding
    - REGRESSION_DISCONTINUITY: RDD for sharp thresholds
    - DIFFERENCE_IN_DIFFERENCES: DiD for panel data
"""

from src.optimization.gepa.tools.causal_tools import (
    # Data classes
    GEPATool,
    # Tool definitions
    LINEAR_DML,
    SPARSE_LINEAR_DML,
    CAUSAL_FOREST,
    DOUBLE_ROBUST_LEARNER,
    METALEARNER_T,
    METALEARNER_S,
    INSTRUMENTAL_VARIABLE,
    REGRESSION_DISCONTINUITY,
    DIFFERENCE_IN_DIFFERENCES,
    # Registry
    CAUSAL_TOOLS,
    TOOL_REGISTRY,
    # Functions
    get_tools_for_agent,
    get_tool_by_name,
    get_tools_by_category,
)

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
