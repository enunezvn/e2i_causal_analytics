"""Handoff Protocols for Tier 0 Agent Pipeline.

This module defines the formal contracts for data handoffs between
ML Foundation agents. Each protocol specifies:
- Required fields (MUST be present)
- Optional fields (MAY be present)
- Validation rules (MUST be satisfied)
- Error conditions (when handoff is rejected)

Handoff Flow:
    ScopeDefiner ──► DataPreparer ──► ModelSelector ──► ModelTrainer
                                                              │
                                          FeatureAnalyzer ◄───┘
                                                              │
                                          ModelDeployer ◄─────┘
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, TypedDict

# =============================================================================
# Type Definitions for Handoff Contracts
# =============================================================================


class ScopeSpec(TypedDict, total=False):
    """Scope specification from ScopeDefiner.

    Defines the complete ML problem specification.
    """

    # Required fields
    experiment_id: str
    experiment_name: str
    problem_type: str  # binary_classification, regression, causal_inference, etc.
    prediction_target: str
    prediction_horizon_days: int

    # Population
    target_population: str
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]

    # Features
    required_features: List[str]
    excluded_features: List[str]
    feature_categories: List[str]

    # Constraints
    regulatory_constraints: List[str]
    ethical_constraints: List[str]
    technical_constraints: List[str]
    minimum_samples: int

    # Context
    brand: str
    region: str
    use_case: str

    # Metadata
    created_by: str
    created_at: str


class SuccessCriteria(TypedDict, total=False):
    """Success criteria from ScopeDefiner.

    Defines performance thresholds for model evaluation.
    """

    # Required fields
    experiment_id: str

    # Classification metrics
    min_auc: Optional[float]
    min_precision: Optional[float]
    min_recall: Optional[float]
    min_f1: Optional[float]

    # Regression metrics
    max_rmse: Optional[float]
    min_r2: Optional[float]

    # General metrics
    max_inference_latency_ms: int
    min_samples: int
    max_model_size_mb: int

    # Baseline comparison
    min_improvement_over_baseline: float


class QCReport(TypedDict, total=False):
    """QC report from DataPreparer.

    Defines data quality validation results.
    """

    # Required fields
    report_id: str
    experiment_id: str
    status: str  # passed, failed, warning
    overall_score: float  # 0.0 - 1.0
    qc_passed: bool  # CRITICAL: gate status

    # Dimension scores
    completeness_score: float
    validity_score: float
    consistency_score: float
    uniqueness_score: float
    timeliness_score: float

    # Details
    expectation_results: List[Dict[str, Any]]
    failed_expectations: List[str]
    warnings: List[str]
    remediation_steps: List[str]
    blocking_issues: List[str]

    # Data stats
    row_count: int
    column_count: int
    validated_at: str


class BaselineMetrics(TypedDict, total=False):
    """Baseline metrics from DataPreparer.

    Computed from TRAIN split only to prevent data leakage.
    """

    experiment_id: str
    split_type: str  # Must be "train"
    feature_stats: Dict[str, Dict[str, float]]  # mean, std, min, max per feature
    target_rate: Optional[float]  # For classification
    target_distribution: Dict[str, float]
    correlation_matrix: Dict[str, Dict[str, float]]
    computed_at: str
    training_samples: int


class ModelCandidate(TypedDict, total=False):
    """Model candidate from ModelSelector.

    Defines the selected algorithm and configuration.
    """

    # Required fields
    algorithm_name: str  # e.g., "XGBoost", "LightGBM", "CausalForest"
    algorithm_class: str  # Full class path
    algorithm_family: str  # tree, linear, neural, causal

    # Hyperparameters
    default_hyperparameters: Dict[str, Any]
    hyperparameter_search_space: Dict[str, Dict[str, Any]]

    # Expected performance
    expected_performance: Dict[str, float]
    training_time_estimate_hours: float
    estimated_inference_latency_ms: int
    memory_requirement_gb: float

    # Scores
    interpretability_score: float  # 0.0 - 1.0
    scalability_score: float  # 0.0 - 1.0
    selection_score: float  # 0.0 - 1.0


class TrainingResult(TypedDict, total=False):
    """Training result from ModelTrainer.

    Contains trained model and evaluation metrics.
    """

    # Required fields
    training_run_id: str
    model_id: str
    trained_model: Any  # Actual model object

    # Metrics
    train_metrics: Dict[str, float]
    validation_metrics: Dict[str, float]
    test_metrics: Dict[str, float]

    # Classification metrics
    auc_roc: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]

    # Regression metrics
    rmse: Optional[float]
    mae: Optional[float]
    r2: Optional[float]

    # Success criteria
    success_criteria_met: bool
    success_criteria_results: Dict[str, bool]

    # HPO results
    best_hyperparameters: Dict[str, Any]
    hpo_completed: bool
    hpo_trials_run: int

    # MLflow
    mlflow_run_id: Optional[str]
    model_artifact_uri: str

    # Metadata
    training_duration_seconds: float
    algorithm_name: str


class SHAPAnalysis(TypedDict, total=False):
    """SHAP analysis from FeatureAnalyzer.

    Contains feature importance and interpretability results.
    """

    experiment_id: str
    model_version: str
    shap_analysis_id: Optional[str]

    # Importance
    feature_importance: List[Dict[str, Any]]  # [{"feature": str, "importance": float, "rank": int}]
    interactions: List[Dict[str, Any]]  # [{"features": [str, str], "strength": float}]
    top_features: List[str]

    # Interpretation
    interpretation: str
    executive_summary: str
    key_insights: List[str]
    recommendations: List[str]

    # Metadata
    samples_analyzed: int
    computation_time_seconds: float


class DeploymentResult(TypedDict, total=False):
    """Deployment result from ModelDeployer.

    Contains deployment configuration and status.
    """

    # Required fields
    deployment_successful: bool
    health_check_passed: bool
    status: str  # completed, failed

    # Deployment info
    deployment_manifest: Dict[str, Any]
    version_record: Dict[str, Any]
    bentoml_tag: str

    # Endpoint
    endpoint_url: str
    endpoint_name: str

    # Rollback
    rollback_available: bool
    previous_deployment_id: Optional[str]


# =============================================================================
# Handoff Protocol Definitions
# =============================================================================


@dataclass
class HandoffProtocol:
    """Base class for handoff protocols between agents."""

    from_agent: str
    to_agent: str
    required_fields: List[str]
    optional_fields: List[str]
    validation_rules: List[str]

    def validate(self, data: Dict[str, Any]) -> tuple:
        """Validate handoff data against protocol.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        errors = []

        # Check required fields
        for field in self.required_fields:
            if field not in data or data[field] is None:
                errors.append(f"Missing required field: {field}")

        return len(errors) == 0, errors


# Protocol: ScopeDefiner -> DataPreparer
SCOPE_TO_DATA_PROTOCOL = HandoffProtocol(
    from_agent="scope_definer",
    to_agent="data_preparer",
    required_fields=[
        "scope_spec",
        "experiment_id",
    ],
    optional_fields=[
        "success_criteria",
        "experiment_name",
    ],
    validation_rules=[
        "scope_spec.problem_type must be valid",
        "scope_spec.experiment_id must match experiment_id",
        "scope_spec.minimum_samples must be > 0",
    ],
)


# Protocol: DataPreparer -> ModelSelector
DATA_TO_SELECTOR_PROTOCOL = HandoffProtocol(
    from_agent="data_preparer",
    to_agent="model_selector",
    required_fields=[
        "scope_spec",
        "qc_report",
    ],
    optional_fields=[
        "baseline_metrics",
        "algorithm_preferences",
        "excluded_algorithms",
    ],
    validation_rules=[
        "qc_report.qc_passed MUST be True (QC Gate)",
        "qc_report.overall_score >= 0.7 recommended",
        "qc_report.blocking_issues must be empty",
    ],
)


# Protocol: ModelSelector -> ModelTrainer
SELECTOR_TO_TRAINER_PROTOCOL = HandoffProtocol(
    from_agent="model_selector",
    to_agent="model_trainer",
    required_fields=[
        "model_candidate",
        "qc_report",
        "experiment_id",
    ],
    optional_fields=[
        "success_criteria",
        "enable_hpo",
        "hpo_trials",
        "train_data",
        "validation_data",
        "test_data",
    ],
    validation_rules=[
        "model_candidate.algorithm_class must be importable",
        "model_candidate.hyperparameter_search_space must be valid Optuna format",
        "qc_report.qc_passed must still be True",
    ],
)


# Protocol: ModelTrainer -> FeatureAnalyzer
TRAINER_TO_ANALYZER_PROTOCOL = HandoffProtocol(
    from_agent="model_trainer",
    to_agent="feature_analyzer",
    required_fields=[
        "model_uri",
        "experiment_id",
    ],
    optional_fields=[
        "training_run_id",
        "X_sample",
        "y_sample",
        "max_samples",
    ],
    validation_rules=[
        "model_uri must be valid MLflow URI or path",
        "trained_model must support SHAP (tree or linear)",
    ],
)


# Protocol: ModelTrainer -> ModelDeployer
TRAINER_TO_DEPLOYER_PROTOCOL = HandoffProtocol(
    from_agent="model_trainer",
    to_agent="model_deployer",
    required_fields=[
        "model_uri",
        "experiment_id",
        "validation_metrics",
        "success_criteria_met",
    ],
    optional_fields=[
        "deployment_name",
        "target_environment",
        "resources",
        "shadow_mode_duration_hours",
        "shadow_mode_requests",
    ],
    validation_rules=[
        "success_criteria_met should be True for production deployment",
        "For production: shadow_mode_duration_hours >= 24",
        "For production: shadow_mode_requests >= 1000",
    ],
)


# =============================================================================
# Protocol Validation Functions
# =============================================================================


def validate_scope_to_data_handoff(data: Dict[str, Any]) -> tuple:
    """Validate ScopeDefiner -> DataPreparer handoff.

    Args:
        data: Handoff data containing scope_spec

    Returns:
        Tuple of (is_valid, errors)
    """
    is_valid, errors = SCOPE_TO_DATA_PROTOCOL.validate(data)

    # Additional validation
    if "scope_spec" in data:
        scope_spec = data["scope_spec"]
        if scope_spec.get("experiment_id") != data.get("experiment_id"):
            errors.append("scope_spec.experiment_id must match experiment_id")

        valid_problem_types = [
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        ]
        if scope_spec.get("problem_type") not in valid_problem_types:
            errors.append(f"Invalid problem_type: {scope_spec.get('problem_type')}")

    return len(errors) == 0, errors


def validate_data_to_selector_handoff(data: Dict[str, Any]) -> tuple:
    """Validate DataPreparer -> ModelSelector handoff (QC GATE).

    Args:
        data: Handoff data containing qc_report

    Returns:
        Tuple of (is_valid, errors)
    """
    is_valid, errors = DATA_TO_SELECTOR_PROTOCOL.validate(data)

    # QC Gate validation (CRITICAL)
    if "qc_report" in data:
        qc_report = data["qc_report"]

        # QC Gate check
        if not qc_report.get("qc_passed", False):
            errors.append("QC GATE FAILED: qc_report.qc_passed is False")

        # Blocking issues check
        blocking_issues = qc_report.get("blocking_issues", [])
        if blocking_issues:
            errors.append(f"QC blocking issues: {blocking_issues}")

    return len(errors) == 0, errors


def validate_selector_to_trainer_handoff(data: Dict[str, Any]) -> tuple:
    """Validate ModelSelector -> ModelTrainer handoff.

    Args:
        data: Handoff data containing model_candidate

    Returns:
        Tuple of (is_valid, errors)
    """
    is_valid, errors = SELECTOR_TO_TRAINER_PROTOCOL.validate(data)

    # Model candidate validation
    if "model_candidate" in data:
        candidate = data["model_candidate"]

        if not candidate.get("algorithm_name"):
            errors.append("model_candidate.algorithm_name is required")

        if not candidate.get("algorithm_class"):
            errors.append("model_candidate.algorithm_class is required")

        if not candidate.get("hyperparameter_search_space"):
            errors.append("model_candidate.hyperparameter_search_space is required")

    return len(errors) == 0, errors


def validate_trainer_to_deployer_handoff(data: Dict[str, Any]) -> tuple:
    """Validate ModelTrainer -> ModelDeployer handoff.

    Args:
        data: Handoff data containing training outputs

    Returns:
        Tuple of (is_valid, errors)
    """
    is_valid, errors = TRAINER_TO_DEPLOYER_PROTOCOL.validate(data)

    # Production deployment validation
    target_env = data.get("target_environment", "staging")
    if target_env == "production":
        # Success criteria check
        if not data.get("success_criteria_met", False):
            errors.append("Production deployment requires success_criteria_met=True")

        # Shadow mode validation
        shadow_duration = data.get("shadow_mode_duration_hours", 0)
        if shadow_duration < 24:
            errors.append(
                f"Production requires shadow_mode_duration_hours >= 24, got {shadow_duration}"
            )

        shadow_requests = data.get("shadow_mode_requests", 0)
        if shadow_requests < 1000:
            errors.append(
                f"Production requires shadow_mode_requests >= 1000, got {shadow_requests}"
            )

    return len(errors) == 0, errors


# =============================================================================
# Handoff Protocol Interface
# =============================================================================


class HandoffValidator(Protocol):
    """Protocol interface for handoff validators."""

    def validate(self, data: Dict[str, Any]) -> tuple:
        """Validate handoff data.

        Args:
            data: Data to validate

        Returns:
            Tuple of (is_valid: bool, errors: List[str])
        """
        ...


# Export all protocols
ALL_PROTOCOLS = {
    "scope_to_data": SCOPE_TO_DATA_PROTOCOL,
    "data_to_selector": DATA_TO_SELECTOR_PROTOCOL,
    "selector_to_trainer": SELECTOR_TO_TRAINER_PROTOCOL,
    "trainer_to_analyzer": TRAINER_TO_ANALYZER_PROTOCOL,
    "trainer_to_deployer": TRAINER_TO_DEPLOYER_PROTOCOL,
}

ALL_VALIDATORS = {
    "scope_to_data": validate_scope_to_data_handoff,
    "data_to_selector": validate_data_to_selector_handoff,
    "selector_to_trainer": validate_selector_to_trainer_handoff,
    "trainer_to_deployer": validate_trainer_to_deployer_handoff,
}
