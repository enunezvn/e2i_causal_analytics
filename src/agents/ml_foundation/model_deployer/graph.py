"""LangGraph workflow for model_deployer agent.

Deployment pipeline:
  Node 1: Model Registration (if needed)
    ↓
  Node 2: Promotion Validation
    ↓
  [Promotion allowed?]
    ↓ YES
  Node 3: Stage Promotion
    ↓
  Node 4: Model Packaging (BentoML)
    ↓
  Node 5: Endpoint Deployment
    ↓
  Node 6: Health Check
    ↓
  Node 7: Rollback Availability Check
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    check_health,
    check_rollback_availability,
    deploy_to_endpoint,
    package_model,
    promote_stage,
    register_model,
    validate_promotion,
)
from .state import ModelDeployerState


def create_model_deployer_graph() -> StateGraph:
    """Create model_deployer LangGraph workflow.

    Pipeline:
        START
          ↓
        register_model (MLflow registration)
          ↓
        validate_promotion (Check promotion criteria)
          ↓
        [Promotion allowed?]
          ↓ YES
        promote_stage (Update MLflow stage)
          ↓
        package_model (BentoML packaging)
          ↓
        deploy_to_endpoint (Deploy to BentoML)
          ↓
        check_health (Health check)
          ↓
        check_rollback_availability (Check rollback)
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ModelDeployerState)

    # Add nodes
    workflow.add_node("register_model", register_model)
    workflow.add_node("validate_promotion", validate_promotion)
    workflow.add_node("promote_stage", promote_stage)
    workflow.add_node("package_model", package_model)
    workflow.add_node("deploy_to_endpoint", deploy_to_endpoint)
    workflow.add_node("check_health", check_health)
    workflow.add_node("check_rollback_availability", check_rollback_availability)

    # Define edges
    workflow.set_entry_point("register_model")

    # Registration → Validation
    workflow.add_conditional_edges(
        "register_model",
        _should_continue_after_registration,
        {"validate_promotion": "validate_promotion", "end": END},
    )

    # Validation → Promotion (if allowed)
    workflow.add_conditional_edges(
        "validate_promotion", _should_promote, {"promote_stage": "promote_stage", "end": END}
    )

    # Promotion → Packaging
    workflow.add_conditional_edges(
        "promote_stage", _should_deploy, {"package_model": "package_model", "end": END}
    )

    # Packaging → Deployment
    workflow.add_conditional_edges(
        "package_model",
        _should_continue_after_packaging,
        {"deploy_to_endpoint": "deploy_to_endpoint", "end": END},
    )

    # Deployment → Health Check
    workflow.add_conditional_edges(
        "deploy_to_endpoint", _should_health_check, {"check_health": "check_health", "end": END}
    )

    # Health Check → Rollback Check
    workflow.add_edge("check_health", "check_rollback_availability")

    # Rollback Check → End
    workflow.add_edge("check_rollback_availability", END)

    return workflow.compile()


def _should_continue_after_registration(state: dict) -> str:
    """Determine if pipeline should continue after registration.

    Args:
        state: Current state

    Returns:
        "validate_promotion" if successful, "end" if error
    """
    if state.get("error"):
        return "end"

    if not state.get("registration_successful", False):
        return "end"

    return "validate_promotion"


def _should_promote(state: dict) -> str:
    """Determine if promotion should proceed.

    Args:
        state: Current state

    Returns:
        "promote_stage" if promotion allowed, "end" otherwise
    """
    if state.get("error"):
        return "end"

    if not state.get("promotion_allowed", False):
        return "end"

    return "promote_stage"


def _should_deploy(state: dict) -> str:
    """Determine if deployment should proceed.

    Args:
        state: Current state

    Returns:
        "package_model" if promotion successful, "end" otherwise
    """
    if state.get("error"):
        return "end"

    if not state.get("promotion_successful", False):
        return "end"

    # Check if deployment is requested
    # If only promotion was requested, end here
    deployment_action = state.get("deployment_action", "deploy")
    if deployment_action == "promote":
        # Just promotion, no deployment
        return "end"

    return "package_model"


def _should_continue_after_packaging(state: dict) -> str:
    """Determine if pipeline should continue after packaging.

    Args:
        state: Current state

    Returns:
        "deploy_to_endpoint" if successful, "end" if error
    """
    if state.get("error"):
        return "end"

    if not state.get("bento_packaging_successful", False):
        return "end"

    return "deploy_to_endpoint"


def _should_health_check(state: dict) -> str:
    """Determine if health check should proceed.

    Args:
        state: Current state

    Returns:
        "check_health" if deployment successful, "end" otherwise
    """
    if state.get("error"):
        return "end"

    # Always check health, even if deployment reported as successful
    # to verify endpoint is actually responding
    return "check_health"
