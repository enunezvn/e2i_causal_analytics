"""Model Deployer Agent - STANDARD.

Manages model lifecycle from development through production.

Responsibilities:
- Model registration in MLflow
- Stage promotions (dev → staging → shadow → production)
- BentoML deployments
- Health checks
- Rollback management

Outputs:
- DeploymentManifest: Deployment configuration and status
- VersionRecord: MLflow version record
- Rollback availability

Integration:
- Upstream: model_trainer, feature_analyzer
- Downstream: Tier 1-5 agents (via prediction endpoints)
- Database: ml_deployments, ml_model_registry
"""

from typing import Dict, Any
from .graph import create_model_deployer_graph
from .state import ModelDeployerState


class ModelDeployerAgent:
    """Model Deployer: Manage model lifecycle and deployments.

    Handles stage promotions, deployments, and rollbacks.
    """

    tier = 0
    tier_name = "ml_foundation"
    agent_type = "standard"
    sla_seconds = 30

    def __init__(self):
        """Initialize model_deployer agent."""
        self.graph = create_model_deployer_graph()

    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute deployment workflow.

        Args:
            input_data: Input data conforming to ModelDeployerInput contract
                Required fields:
                - model_uri: str (MLflow model URI)
                - experiment_id: str
                - validation_metrics: Dict (ValidationMetrics from training)
                - success_criteria_met: bool
                - deployment_name: str
                Optional fields:
                - shap_analysis_id: str
                - target_environment: "staging" | "shadow" | "production"
                - resources: Dict[str, str] ({"cpu": "2", "memory": "4Gi"})
                - max_batch_size: int
                - max_latency_ms: int
                - deployment_action: "register" | "promote" | "deploy"

        Returns:
            Output data conforming to ModelDeployerOutput contract
        """
        # Validate required inputs
        required_fields = [
            "model_uri",
            "experiment_id",
            "validation_metrics",
            "success_criteria_met",
            "deployment_name",
        ]
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Missing required field: {field}")

        # Prepare initial state
        initial_state: ModelDeployerState = {
            # Input fields
            "model_uri": input_data["model_uri"],
            "experiment_id": input_data["experiment_id"],
            "validation_metrics": input_data["validation_metrics"],
            "success_criteria_met": input_data["success_criteria_met"],
            "deployment_name": input_data["deployment_name"],
            # Optional fields
            "shap_analysis_id": input_data.get("shap_analysis_id"),
            "target_environment": input_data.get("target_environment", "staging"),
            "resources": input_data.get("resources", {"cpu": "2", "memory": "4Gi"}),
            "max_batch_size": input_data.get("max_batch_size", 100),
            "max_latency_ms": input_data.get("max_latency_ms", 100),
            "deployment_action": input_data.get("deployment_action", "deploy"),
            # Shadow mode metrics (for production promotion)
            "shadow_mode_duration_hours": input_data.get("shadow_mode_duration_hours", 0),
            "shadow_mode_requests": input_data.get("shadow_mode_requests", 0),
            "shadow_mode_error_rate": input_data.get("shadow_mode_error_rate", 1.0),
            "shadow_mode_latency_p99_ms": input_data.get("shadow_mode_latency_p99_ms", 999),
        }

        # Execute LangGraph workflow
        final_state = await self.graph.ainvoke(initial_state)

        # Check for errors
        if final_state.get("error"):
            error_msg = final_state["error"]
            error_type = final_state.get("error_type", "unknown")
            raise RuntimeError(f"{error_type}: {error_msg}")

        # Build outputs
        deployment_manifest = self._build_deployment_manifest(final_state)
        version_record = self._build_version_record(final_state)

        # Determine overall status
        promotion_successful = final_state.get("promotion_successful", False)
        deployment_successful = final_state.get("deployment_successful", False)
        deployment_action = final_state.get("deployment_action", "deploy")

        if deployment_action == "promote":
            # Just promotion, no deployment
            overall_status = "completed" if promotion_successful else "failed"
        else:
            # Full deployment
            overall_status = "completed" if (promotion_successful and deployment_successful) else "partial"

        # Build output
        output = {
            # Deployment manifest
            "deployment_manifest": deployment_manifest,
            # Version record
            "version_record": version_record,
            # BentoML tag
            "bentoml_tag": final_state.get("final_bento_tag", ""),
            # Status flags
            "deployment_successful": deployment_successful,
            "health_check_passed": final_state.get("health_check_passed", False),
            "rollback_available": final_state.get("rollback_available", False),
            # Overall status
            "status": overall_status,
        }

        # Store to database (ml_deployments and ml_model_registry)
        await self._store_to_database(output, final_state)

        return output

    def _build_deployment_manifest(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build DeploymentManifest output structure.

        Args:
            state: Final agent state

        Returns:
            DeploymentManifest dict conforming to contract
        """
        return {
            "deployment_id": state.get("deployment_id", ""),
            "experiment_id": state["experiment_id"],
            "model_version": str(state.get("model_version", 1)),
            # Environment
            "environment": state.get("target_environment", "staging"),
            "endpoint_url": state.get("endpoint_url", ""),
            # Resources
            "resources": state.get("resources", {"cpu": "2", "memory": "4Gi"}),
            # Status
            "status": state.get("deployment_status", "pending"),
            "deployed_at": state.get("deployed_at", ""),
            # Health
            "health_check_url": state.get("health_check_url", ""),
            "metrics_url": state.get("metrics_url", ""),
        }

    def _build_version_record(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Build VersionRecord output structure.

        Args:
            state: Final agent state

        Returns:
            VersionRecord dict conforming to contract
        """
        return {
            "registered_model_name": state.get("registered_model_name", ""),
            "version": state.get("model_version", 1),
            "stage": state.get("current_stage", "None"),
            "description": state.get("promotion_reason", "Automated deployment"),
        }

    async def _store_to_database(
        self,
        output: Dict[str, Any],
        state: Dict[str, Any]
    ) -> None:
        """Store deployment to ml_deployments and update ml_model_registry.

        Args:
            output: Agent output to store
            state: Final agent state
        """
        # TODO: Implement database storage
        # This would:
        # 1. Write to ml_deployments table:
        #    - deployment_id
        #    - model_version_id
        #    - experiment_id
        #    - target_stage
        #    - endpoint_name
        #    - endpoint_url
        #    - status
        #    - replicas
        #    - cpu_limit
        #    - memory_limit
        #    - autoscaling (JSONB)
        #    - bento_tag
        #    - deployed_by
        #    - deployed_at
        #    - deployment_duration_seconds
        #
        # 2. Update ml_model_registry table:
        #    - Update stage to current_stage
        #    - Update deployment_id
        #    - Update deployed_at timestamp
        #
        # For now, pass (will be implemented in integration phase)
        pass
