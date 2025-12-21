"""State definition for model_deployer agent.

This agent manages the model lifecycle from development through production:
1. Model Registration - Register in MLflow
2. Stage Promotion - Promote through stages (dev → staging → shadow → prod)
3. Deployment - Deploy to BentoML endpoints
4. Health Checks - Verify deployment health
5. Rollback - Revert to previous version if needed
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


class ModelDeployerState(TypedDict, total=False):
    """State for model_deployer agent.

    Follows deployment workflow:
    1. Register model (if not already registered)
    2. Validate promotion criteria
    3. Promote to target stage
    4. Deploy to endpoint (if requested)
    5. Perform health checks
    """

    # === INPUT FIELDS (from model_trainer/feature_analyzer) ===

    model_uri: str  # MLflow model URI (e.g., "runs:/abc123/model")
    experiment_id: str  # Experiment identifier

    # Validation metrics (from model_trainer)
    validation_metrics: Dict[str, Any]  # ValidationMetrics from training
    success_criteria_met: bool  # Whether model met success criteria

    # SHAP analysis (from feature_analyzer)
    shap_analysis_id: Optional[str]  # SHAP analysis ID for explainability

    # === DEPLOYMENT CONFIG ===

    target_environment: Literal["staging", "shadow", "production"]  # Target environment
    deployment_name: str  # Deployment name

    # Serving configuration
    resources: Dict[str, str]  # {"cpu": "2", "memory": "4Gi"}
    max_batch_size: int  # Maximum batch size
    max_latency_ms: int  # Maximum latency in milliseconds

    # === DEPLOYMENT ACTION ===

    deployment_action: Literal["register", "promote", "deploy", "rollback"]

    # === MODEL REGISTRATION ===

    # MLflow registration
    registered_model_name: str  # Name in MLflow registry
    model_version: int  # Version number
    current_stage: str  # Current MLflow stage
    target_stage: str  # Target MLflow stage

    # Registration result
    registration_successful: bool
    registration_timestamp: str  # ISO timestamp of registration
    registration_error: Optional[str]

    # === STAGE PROMOTION ===

    # Promotion validation
    promotion_allowed: bool
    promotion_target_stage: (
        str  # Target stage for promotion (e.g., "Staging", "Shadow", "Production")
    )
    promotion_denial_reason: Optional[str]  # Reason if promotion denied
    validation_failures: List[str]  # Shadow mode validation failures
    promotion_validation_errors: List[str]

    # Shadow mode validation (for production promotion)
    shadow_mode_duration_hours: float
    shadow_mode_requests: int
    shadow_mode_error_rate: float
    shadow_mode_latency_p99_ms: float
    shadow_mode_validated: bool

    # Promotion result
    promotion_successful: bool
    promotion_timestamp: str  # ISO timestamp of promotion
    promotion_error: Optional[str]
    promotion_reason: str

    # Previous stage (for version record)
    previous_stage: str

    # Metrics at promotion
    metrics_at_promotion: Dict[str, float]

    # === DEPLOYMENT ===

    # BentoML packaging
    bento_tag: str  # "e2i_exp123_model:v1.2.3"
    bento_packaging_successful: bool
    bento_packaging_error: Optional[str]

    # Endpoint deployment
    endpoint_name: str
    endpoint_url: str
    deployment_id: str

    # Deployment configuration
    replicas: int  # Number of replicas
    cpu_limit: str  # CPU limit
    memory_limit: str  # Memory limit
    autoscaling: Dict[str, Any]  # Autoscaling configuration

    # Deployment status
    deployment_status: Literal["pending", "deploying", "healthy", "unhealthy", "failed"]
    deployment_duration_seconds: float

    # === HEALTH CHECKS ===

    health_check_url: str
    health_check_passed: bool
    health_check_response_time_ms: float
    health_check_error: Optional[str]

    # Metrics endpoint
    metrics_url: str

    # === ROLLBACK ===

    # Rollback configuration
    rollback_to_deployment_id: Optional[str]
    rollback_to_version: Optional[int]
    rollback_reason: Optional[str]

    # Rollback status
    rollback_successful: bool
    rollback_error: Optional[str]
    rollback_available: bool  # Whether rollback is possible

    # Previous deployment (for rollback)
    previous_deployment_id: Optional[str]
    previous_deployment_url: Optional[str]

    # === OUTPUT FIELDS (Final) ===

    # Deployment manifest
    deployment_manifest: Dict[str, Any]  # DeploymentManifest

    # Version record
    version_record: Dict[str, Any]  # VersionRecord

    # BentoML tag
    final_bento_tag: str

    # Status flags
    deployment_successful: bool
    overall_status: Literal["completed", "failed", "partial"]

    # === METADATA ===

    deployed_at: str  # ISO timestamp
    deployed_by: str  # Agent name

    # === ERROR HANDLING ===

    error: Optional[str]  # Error message if failed
    error_type: Optional[str]  # Error classification
    error_details: Optional[Dict[str, Any]]  # Additional error context
