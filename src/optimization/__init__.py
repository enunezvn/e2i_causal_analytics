"""E2I Optimization Module.

Provides prompt optimization infrastructure including GEPA (Generative Evolutionary
Prompting with AI) for the 18-agent 6-tier architecture.

Submodules:
    gepa: GEPA optimizer implementation with metrics, tools, and integrations
"""

from src.optimization.gepa import (
    create_gepa_optimizer,
    get_metric_for_agent,
)

__all__ = [
    "create_gepa_optimizer",
    "get_metric_for_agent",
]
