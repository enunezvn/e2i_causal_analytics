"""LangGraph workflow for model_deployer agent.

Enhanced deployment pipeline with planning and rollback support:

  START
    ↓
  Node 1: Deployment Planning (Select strategy, resources)
    ↓
  Node 2: Plan Validation (Validate feasibility)
    ↓
  [Plan valid?]
    ↓ YES
  Node 3: Model Registration (if needed)
    ↓
  Node 4: Promotion Validation
    ↓
  [Promotion allowed?]
    ↓ YES
  Node 5: Stage Promotion
    ↓
  Node 6: Model Packaging (BentoML)
    ↓
  Node 7: Containerization (Docker)
    ↓
  Node 8: Endpoint Deployment (Strategy-based)
    ↓
  Node 9: Health Check
    ↓
  Node 10: Rollback Availability Check
    ↓
  END

Rollback path (when deployment_action == "rollback"):
  START
    ↓
  Execute Rollback
    ↓
  END
"""

from langgraph.graph import END, StateGraph

from .nodes import (
    check_health,
    check_rollback_availability,
    containerize_model,
    deploy_to_endpoint,
    execute_rollback,
    package_model,
    plan_deployment,
    promote_stage,
    register_model,
    validate_deployment_plan,
    validate_promotion,
)
from .state import ModelDeployerState


def create_model_deployer_graph() -> StateGraph:
    """Create model_deployer LangGraph workflow.

    Enhanced pipeline with planning and rollback:
        START
          ↓
        [Rollback action?]
          ↓ YES → execute_rollback → END
          ↓ NO
        plan_deployment (Strategy selection)
          ↓
        validate_deployment_plan (Feasibility check)
          ↓
        [Plan valid?]
          ↓ YES
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
        containerize_model (Docker containerization)
          ↓
        deploy_to_endpoint (Strategy-based deployment)
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
    workflow.add_node("plan_deployment", plan_deployment)
    workflow.add_node("validate_deployment_plan", validate_deployment_plan)
    workflow.add_node("register_model", register_model)
    workflow.add_node("validate_promotion", validate_promotion)
    workflow.add_node("promote_stage", promote_stage)
    workflow.add_node("package_model", package_model)
    workflow.add_node("containerize_model", containerize_model)
    workflow.add_node("deploy_to_endpoint", deploy_to_endpoint)
    workflow.add_node("check_health", check_health)
    workflow.add_node("check_rollback_availability", check_rollback_availability)
    workflow.add_node("execute_rollback", execute_rollback)

    # Define entry point with conditional routing
    workflow.set_entry_point("plan_deployment")

    # Entry routing: Check if this is a rollback action
    workflow.add_conditional_edges(
        "plan_deployment",
        _route_after_planning,
        {
            "validate_plan": "validate_deployment_plan",
            "execute_rollback": "execute_rollback",
            "end": END,
        },
    )

    # Plan validation → Registration
    workflow.add_conditional_edges(
        "validate_deployment_plan",
        _should_continue_after_plan_validation,
        {"register_model": "register_model", "end": END},
    )

    # Registration → Validation
    workflow.add_conditional_edges(
        "register_model",
        _should_continue_after_registration,
        {"validate_promotion": "validate_promotion", "end": END},
    )

    # Validation → Promotion (if allowed)
    workflow.add_conditional_edges(
        "validate_promotion",
        _should_promote,
        {"promote_stage": "promote_stage", "end": END},
    )

    # Promotion → Packaging
    workflow.add_conditional_edges(
        "promote_stage",
        _should_deploy,
        {"package_model": "package_model", "end": END},
    )

    # Packaging → Containerization
    workflow.add_conditional_edges(
        "package_model",
        _should_continue_after_packaging,
        {"containerize_model": "containerize_model", "end": END},
    )

    # Containerization → Deployment
    workflow.add_conditional_edges(
        "containerize_model",
        _should_continue_after_containerization,
        {"deploy_to_endpoint": "deploy_to_endpoint", "end": END},
    )

    # Deployment → Health Check
    workflow.add_conditional_edges(
        "deploy_to_endpoint",
        _should_health_check,
        {"check_health": "check_health", "end": END},
    )

    # Health Check → Rollback Check
    workflow.add_edge("check_health", "check_rollback_availability")

    # Rollback Check → End
    workflow.add_edge("check_rollback_availability", END)

    # Rollback execution → End
    workflow.add_edge("execute_rollback", END)

    return workflow.compile()


def _route_after_planning(state: dict) -> str:
    """Route after planning based on deployment action.

    Args:
        state: Current state

    Returns:
        Next node based on action type
    """
    if state.get("error"):
        return "end"

    # Check if this is a rollback action
    deployment_action = state.get("deployment_action", "deploy")
    if deployment_action == "rollback":
        return "execute_rollback"

    # Continue with normal flow
    return "validate_plan"


def _should_continue_after_plan_validation(state: dict) -> str:
    """Determine if pipeline should continue after plan validation.

    Args:
        state: Current state

    Returns:
        "register_model" if plan valid, "end" if invalid
    """
    if state.get("error"):
        return "end"

    if not state.get("plan_validated", False):
        return "end"

    # Check for validation errors
    validation_errors = state.get("plan_validation_errors", [])
    if validation_errors:
        return "end"

    return "register_model"


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
    # If only registration or promotion was requested, skip packaging/deployment
    deployment_action = state.get("deployment_action", "deploy")
    if deployment_action in ("promote", "register"):
        return "end"

    return "package_model"


def _should_continue_after_packaging(state: dict) -> str:
    """Determine if pipeline should continue after packaging.

    Args:
        state: Current state

    Returns:
        "containerize_model" if successful, "end" if error
    """
    if state.get("error"):
        return "end"

    if not state.get("bento_packaging_successful", False):
        return "end"

    return "containerize_model"


def _should_continue_after_containerization(state: dict) -> str:
    """Determine if pipeline should continue after containerization.

    Args:
        state: Current state

    Returns:
        "deploy_to_endpoint" if successful, "end" if error
    """
    if state.get("error"):
        return "end"

    # Containerization is optional - check if it was attempted
    if state.get("containerization_successful") is False:
        # Containerization explicitly failed
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


# =============================================================================
# Alternative graph creators for specific use cases
# =============================================================================


def create_promotion_only_graph() -> StateGraph:
    """Create a simplified graph for promotion-only operations.

    Pipeline:
        START
          ↓
        validate_promotion
          ↓
        [Promotion allowed?]
          ↓ YES
        promote_stage
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ModelDeployerState)

    workflow.add_node("validate_promotion", validate_promotion)
    workflow.add_node("promote_stage", promote_stage)

    workflow.set_entry_point("validate_promotion")
    workflow.add_conditional_edges(
        "validate_promotion",
        _should_promote,
        {"promote_stage": "promote_stage", "end": END},
    )
    workflow.add_edge("promote_stage", END)

    return workflow.compile()


def create_rollback_graph() -> StateGraph:
    """Create a graph specifically for rollback operations.

    Pipeline:
        START
          ↓
        execute_rollback
          ↓
        check_health
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ModelDeployerState)

    workflow.add_node("execute_rollback", execute_rollback)
    workflow.add_node("check_health", check_health)

    workflow.set_entry_point("execute_rollback")
    workflow.add_conditional_edges(
        "execute_rollback",
        lambda s: "check_health" if s.get("rollback_successful") else "end",
        {"check_health": "check_health", "end": END},
    )
    workflow.add_edge("check_health", END)

    return workflow.compile()


def create_health_check_graph() -> StateGraph:
    """Create a graph for health check only.

    Pipeline:
        START
          ↓
        check_health
          ↓
        check_rollback_availability
          ↓
        END

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(ModelDeployerState)

    workflow.add_node("check_health", check_health)
    workflow.add_node("check_rollback_availability", check_rollback_availability)

    workflow.set_entry_point("check_health")
    workflow.add_edge("check_health", "check_rollback_availability")
    workflow.add_edge("check_rollback_availability", END)

    return workflow.compile()
