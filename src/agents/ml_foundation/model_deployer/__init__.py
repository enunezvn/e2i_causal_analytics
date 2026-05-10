"""Model Deployer Agent - ML Foundation Tier 0.

Manages model lifecycle from development through production:
- Model registration in MLflow
- Stage promotions (dev -> staging -> shadow -> production)
- BentoML deployments
- Health checks
- Rollback management
"""

from .agent import ModelDeployerAgent
from .regulatory_audit import (
    RegulatoryAuditMutationError,
    RegulatoryEligibilityAudit,
    is_adapted_regulatory_candidate,
    is_regulatory_eligible,
)
from .state import ModelDeployerState

__all__ = [
    "ModelDeployerAgent",
    "ModelDeployerState",
    # Gate N1 (plan v4 §2) — regulatory-eligibility primitives.
    "RegulatoryEligibilityAudit",
    "RegulatoryAuditMutationError",
    "is_regulatory_eligible",
    "is_adapted_regulatory_candidate",
]
