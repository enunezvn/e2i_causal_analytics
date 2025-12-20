"""Tier 0: ML Foundation agents.

This tier contains 7 agents that handle the complete ML lifecycle
from problem definition to production deployment.

Agents:
- scope_definer: Problem definition and success criteria
- data_preparer: Data quality validation and baseline metrics (IMPLEMENTED)
- model_selector: Algorithm selection
- model_trainer: Model training with HPO
- feature_analyzer: SHAP analysis and feature importance
- model_deployer: Model deployment and versioning
- observability_connector: Metrics tracking and logging
"""

from .data_preparer import DataPreparerAgent, DataPreparerState

__all__ = [
    "DataPreparerAgent",
    "DataPreparerState",
]
