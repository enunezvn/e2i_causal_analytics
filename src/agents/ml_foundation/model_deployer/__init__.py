"""Model Deployer Agent - ML Foundation Tier 0.

Manages model lifecycle from development through production:
- Model registration in MLflow
- Stage promotions (dev -> staging -> shadow -> production)
- BentoML deployments
- Health checks
- Rollback management
"""

from .agent import ModelDeployerAgent
from .state import ModelDeployerState

__all__ = [
    "ModelDeployerAgent",
    "ModelDeployerState",
]
