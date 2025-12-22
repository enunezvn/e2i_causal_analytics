"""Nodes for model_deployer agent.

Deployment workflow with 8 nodes:
1. Deployment Planning - Analyze requirements, select strategy
2. Model Registration - Register in MLflow
3. Promotion Validation - Validate stage promotion criteria
4. Stage Promotion - Promote to target stage
5. Model Packaging - Package with BentoML
6. Endpoint Deployment - Deploy to BentoML endpoint
7. Health Check - Verify deployment health
8. Rollback Check - Check rollback availability
"""

from .deployment_orchestrator import (
    check_rollback_availability,
    containerize_model,
    deploy_to_endpoint,
    execute_rollback,
    package_model,
)
from .deployment_planner import (
    DeploymentPlan,
    DeploymentStrategy,
    ModelType,
    ResourceProfile,
    plan_deployment,
    validate_deployment_plan,
)
from .health_checker import check_health
from .registry_manager import (
    promote_stage,
    register_model,
    validate_promotion,
)

__all__ = [
    # Planning
    "plan_deployment",
    "validate_deployment_plan",
    "DeploymentPlan",
    "DeploymentStrategy",
    "ModelType",
    "ResourceProfile",
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
