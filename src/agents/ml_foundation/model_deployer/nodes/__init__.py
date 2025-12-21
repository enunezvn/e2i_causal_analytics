"""Nodes for model_deployer agent.

Deployment workflow with 6 nodes:
1. Model Registration - Register in MLflow
2. Promotion Validation - Validate stage promotion criteria
3. Stage Promotion - Promote to target stage
4. Model Packaging - Package with BentoML
5. Endpoint Deployment - Deploy to BentoML endpoint
6. Health Check - Verify deployment health
"""

from .deployment_orchestrator import (
    check_rollback_availability,
    deploy_to_endpoint,
    package_model,
)
from .health_checker import check_health
from .registry_manager import (
    promote_stage,
    register_model,
    validate_promotion,
)

__all__ = [
    # Registry management
    "register_model",
    "validate_promotion",
    "promote_stage",
    # Deployment
    "package_model",
    "deploy_to_endpoint",
    "check_rollback_availability",
    # Health
    "check_health",
]
