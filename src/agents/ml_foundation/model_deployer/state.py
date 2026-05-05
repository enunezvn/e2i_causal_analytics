"""State definition for model_deployer agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard C of the migration. Inherits from ``BaseAgentSchema``.

This agent manages the model lifecycle from development through production:
1. Model Registration - Register in MLflow
2. Stage Promotion - Promote through stages (dev → staging → shadow → prod)
3. Deployment - Deploy to BentoML endpoints
4. Health Checks - Verify deployment health
5. Rollback - Revert to previous version if needed
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from uuid import UUID, uuid4

from pydantic import Field

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)


class ModelDeployerState(BaseAgentSchema):
    """State for model_deployer agent.

    Follows deployment workflow:
    1. Register model (if not already registered)
    2. Validate promotion criteria
    3. Promote to target stage
    4. Deploy to endpoint (if requested)
    5. Perform health checks
    """

    # === INPUT FIELDS (from model_trainer/feature_analyzer) ===

    model_uri: Optional[str] = None  # MLflow model URI (e.g., "runs:/abc123/model")
    experiment_id: Optional[str] = None

    # Validation metrics (from model_trainer)
    validation_metrics: Optional[Dict[str, Any]] = None  # ValidationMetrics from training
    success_criteria_met: Optional[bool] = None  # Whether model met success criteria

    # SHAP analysis (from feature_analyzer)
    shap_analysis_id: Optional[str] = None  # SHAP analysis ID for explainability

    # === DEPLOYMENT CONFIG ===

    target_environment: Optional[Literal["staging", "shadow", "production"]] = None
    deployment_name: Optional[str] = None

    # Serving configuration
    resources: Optional[Dict[str, str]] = None  # {"cpu": "2", "memory": "4Gi"}
    max_batch_size: Optional[int] = None
    max_latency_ms: Optional[int] = None

    # === DEPLOYMENT PLANNING ===

    # Planning inputs
    model_type: Optional[str] = None  # Model type: classification, regression, causal, ensemble

    # Plan outputs
    deployment_plan: Optional[Dict[str, Any]] = None  # DeploymentPlan as dict
    deployment_strategy: Optional[str] = None  # direct, blue_green, canary, shadow
    service_template: Optional[str] = None  # classification, regression, causal
    health_check_config: Optional[Dict[str, Any]] = None  # Health check configuration
    traffic_config: Optional[Dict[str, Any]] = None  # Traffic routing configuration
    rollback_config: Optional[Dict[str, Any]] = None  # Rollback thresholds

    # Plan validation
    plan_validated: Optional[bool] = None
    plan_validation_errors: Optional[List[str]] = None

    # === DEPLOYMENT ACTION ===

    deployment_action: Optional[Literal["register", "promote", "deploy", "rollback"]] = None

    # === MODEL REGISTRATION ===

    # MLflow registration
    registered_model_name: Optional[str] = None  # Name in MLflow registry
    model_version: Optional[int] = None  # Version number
    current_stage: Optional[str] = None  # Current MLflow stage
    target_stage: Optional[str] = None  # Target MLflow stage

    # Registration result
    registration_successful: Optional[bool] = None
    registration_timestamp: Optional[str] = None  # ISO timestamp of registration
    registration_error: Optional[str] = None

    # === STAGE PROMOTION ===

    # Promotion validation
    promotion_allowed: Optional[bool] = None
    promotion_target_stage: Optional[str] = None  # Target stage for promotion
    promotion_denial_reason: Optional[str] = None
    validation_failures: Optional[List[str]] = None  # Shadow mode validation failures
    promotion_validation_errors: Optional[List[str]] = None

    # Shadow mode validation (for production promotion)
    shadow_mode_duration_hours: Optional[float] = None
    shadow_mode_requests: Optional[int] = None
    shadow_mode_error_rate: Optional[float] = None
    shadow_mode_latency_p99_ms: Optional[float] = None
    shadow_mode_validated: Optional[bool] = None

    # Promotion result
    promotion_successful: Optional[bool] = None
    promotion_timestamp: Optional[str] = None  # ISO timestamp of promotion
    promotion_error: Optional[str] = None
    promotion_reason: Optional[str] = None

    # Previous stage (for version record)
    previous_stage: Optional[str] = None

    # Metrics at promotion — widened preemptively per Shard B lesson
    metrics_at_promotion: Optional[Dict[str, Any]] = None

    # === DEPLOYMENT ===

    # BentoML packaging
    bento_tag: Optional[str] = None  # "e2i_exp123_model:v1.2.3"
    bento_packaging_successful: Optional[bool] = None
    bento_packaging_error: Optional[str] = None

    # Containerization
    container_image: Optional[str] = None  # Docker image tag
    container_config: Optional[Dict[str, Any]] = None  # ContainerConfig as dict
    containerization_successful: Optional[bool] = None
    containerization_error: Optional[str] = None

    # Endpoint deployment
    endpoint_name: Optional[str] = None
    endpoint_url: Optional[str] = None
    deployment_id: Optional[str] = None

    # Deployment configuration
    replicas: Optional[int] = None  # Number of replicas
    cpu_limit: Optional[str] = None  # CPU limit
    memory_limit: Optional[str] = None  # Memory limit
    autoscaling: Optional[Dict[str, Any]] = None  # Autoscaling configuration

    # Deployment status
    deployment_status: Optional[
        Literal["pending", "deploying", "healthy", "unhealthy", "failed"]
    ] = None
    deployment_duration_seconds: Optional[float] = None

    # === HEALTH CHECKS ===

    health_check_url: Optional[str] = None
    health_check_passed: Optional[bool] = None
    health_check_response_time_ms: Optional[float] = None
    health_check_error: Optional[str] = None

    # Metrics endpoint
    metrics_url: Optional[str] = None

    # === ROLLBACK ===

    # Rollback configuration
    rollback_to_deployment_id: Optional[str] = None
    rollback_to_version: Optional[int] = None
    rollback_reason: Optional[str] = None

    # Rollback status
    rollback_successful: Optional[bool] = None
    rollback_error: Optional[str] = None
    rollback_available: Optional[bool] = None  # Whether rollback is possible

    # Previous deployment (for rollback)
    previous_deployment_id: Optional[str] = None
    previous_deployment_url: Optional[str] = None

    # === OUTPUT FIELDS (Final) ===

    # Deployment manifest
    deployment_manifest: Optional[Dict[str, Any]] = None  # DeploymentManifest

    # Version record
    version_record: Optional[Dict[str, Any]] = None  # VersionRecord

    # BentoML tag
    final_bento_tag: Optional[str] = None

    # Status flags
    deployment_successful: Optional[bool] = None
    overall_status: Optional[Literal["completed", "failed", "partial"]] = None

    # === METADATA ===

    deployed_at: Optional[str] = None  # ISO timestamp
    deployed_by: Optional[str] = None  # Agent name

    # === ERROR HANDLING ===

    error: Optional[str] = None
    error_type: Optional[str] = None
    error_details: Optional[Dict[str, Any]] = None

    # === AUDIT CHAIN ===
    audit_workflow_id: UUID = Field(default_factory=uuid4)

    _validate_audit_id = audit_workflow_id_validator()
