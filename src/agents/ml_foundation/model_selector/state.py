"""State definition for model_selector agent.

Migrated from ``TypedDict(total=False)`` to pydantic v2 ``BaseModel``
in Shard C of the migration. Inherits from ``BaseAgentSchema``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
)
from src.agents.ml_foundation.data_preparer.schemas import QCReportSchema
from src.agents.ml_foundation.model_trainer.schemas import OptunaDistribution
from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema


class ModelSelectorState(BaseAgentSchema):
    """State for model_selector agent.

    The model_selector evaluates candidate algorithms and recommends the
    optimal model architecture based on problem scope and constraints.
    """

    # === INPUT FIELDS ===
    # From scope_definer
    # D2.4: typed scope_spec contract; see data_preparer/state.py:50.
    scope_spec: Optional[ScopeSpecSchema] = None  # Complete ScopeSpec
    experiment_id: Optional[str] = None  # Extracted from scope_spec
    kpi_category: Optional[str] = None  # KPI category for domain-specific recommendations

    # From data_preparer
    # D2.2: typed QC contract; see model_trainer/state.py:58.
    qc_report: Optional[QCReportSchema] = None  # Must have passed QC gate
    baseline_metrics: Optional[Dict[str, Any]] = None  # Baseline metrics from training data

    # Feature characteristics (optional, from tier0 feature discovery)
    feature_characteristics: Optional[Dict[str, Any]] = None  # e.g. {"categorical_ratio": 0.6}

    # User preferences (optional)
    algorithm_preferences: Optional[List[str]] = None
    excluded_algorithms: Optional[List[str]] = None
    interpretability_required: Optional[bool] = None  # Whether model must be interpretable

    # Sample data for benchmarking (optional)
    X_sample: Optional[Any] = None  # Feature matrix for cross-validation
    y_sample: Optional[Any] = None  # Target vector for cross-validation

    # Control flags
    skip_benchmarks: Optional[bool] = None  # Skip cross-validation benchmarks
    skip_mlflow: Optional[bool] = None  # Skip MLflow registration

    # === INTERMEDIATE FIELDS ===
    # Problem analysis
    problem_type: Optional[str] = None  # Extracted from scope_spec
    technical_constraints: Optional[List[str]] = None  # Extracted from scope_spec
    row_count: Optional[int] = None  # From qc_report
    column_count: Optional[int] = None  # From qc_report

    # Algorithm filtering
    candidate_algorithms: Optional[List[Dict[str, Any]]] = None  # Filtered candidates
    filtered_by_problem_type: Optional[List[Dict[str, Any]]] = None
    filtered_by_constraints: Optional[List[Dict[str, Any]]] = None
    filtered_by_preferences: Optional[List[Dict[str, Any]]] = None

    # Historical data — widened from Dict[str, float] preemptively per Shard B lesson
    historical_success_rates: Optional[Dict[str, Any]] = None
    similar_experiments: Optional[List[str]] = None  # Similar past experiments

    # Ranking
    ranked_candidates: Optional[List[Dict[str, Any]]] = None  # Ranked by selection score
    selection_scores: Optional[Dict[str, Any]] = None  # Algorithm -> composite score

    # === OUTPUT FIELDS ===
    # Primary selection
    primary_candidate: Optional[Dict[str, Any]] = None  # Selected ModelCandidate
    algorithm_name: Optional[str] = None
    algorithm_class: Optional[str] = None  # Python class path
    algorithm_family: Optional[str] = None  # "causal_ml", "gradient_boosting", etc.

    # Configuration
    default_hyperparameters: Optional[Dict[str, Any]] = None  # Starting hyperparameters
    # D2.1: see model_trainer/state.py:54 — typed Optuna search space.
    hyperparameter_search_space: Optional[Dict[str, OptunaDistribution]] = None

    # Performance expectations — widened
    expected_performance: Optional[Dict[str, Any]] = None  # Expected metrics
    training_time_estimate_hours: Optional[float] = None  # Estimated training time
    estimated_inference_latency_ms: Optional[int] = None  # Expected latency
    memory_requirement_gb: Optional[float] = None  # Memory requirements

    # Characteristics
    interpretability_score: Optional[float] = None  # 0-1 interpretability score
    scalability_score: Optional[float] = None  # 0-1 scalability score
    selection_score: Optional[float] = None  # Overall selection score

    # Alternative candidates
    alternative_candidates: Optional[List[Dict[str, Any]]] = None  # Top 2-3 alternatives

    # Rationale
    selection_rationale: Optional[str] = None
    primary_reason: Optional[str] = None  # Main selection reason
    supporting_factors: Optional[List[str]] = None
    alternatives_considered: Optional[List[Dict[str, Any]]] = None
    constraint_compliance: Optional[Dict[str, bool]] = None  # Constraint check results

    # Baseline comparison — widened
    baseline_to_beat: Optional[Dict[str, Any]] = None  # Baseline model metrics
    baseline_candidates: Optional[List[str]] = None
    baseline_comparison: Optional[Dict[str, Any]] = None  # Full baseline comparison results

    # Benchmarking
    benchmark_results: Optional[Dict[str, Dict[str, Any]]] = None
    benchmark_rankings: Optional[List[Dict[str, Any]]] = None
    benchmarks_skipped: Optional[bool] = None
    benchmark_skip_reason: Optional[str] = None
    benchmark_time_seconds: Optional[float] = None
    combined_score: Optional[float] = None
    benchmark_score: Optional[float] = None
    max_benchmark_candidates: Optional[int] = None  # Max candidates to benchmark
    cv_folds: Optional[int] = None  # Number of CV folds

    # Historical analysis
    historical_data_available: Optional[bool] = None
    historical_experiments_count: Optional[int] = None
    history_recommended_algorithms: Optional[List[str]] = None
    recommendation_source: Optional[str] = None  # "historical" or "prior_knowledge"
    algorithm_trends: Optional[Dict[str, Dict[str, Any]]] = None  # Performance trends

    # MLflow registration
    registered_in_mlflow: Optional[bool] = None
    model_version_id: Optional[str] = None  # MLflow model version ID
    mlflow_run_id: Optional[str] = None
    mlflow_experiment_id: Optional[str] = None
    mlflow_registration_error: Optional[str] = None
    benchmark_logged: Optional[bool] = None  # Whether benchmarks logged to MLflow
    benchmark_log_error: Optional[str] = None

    # Selection summary (for database storage)
    selection_summary: Optional[Dict[str, Any]] = None

    # Stage
    stage: Optional[str] = None  # Model stage: "development"

    # Metadata
    created_at: Optional[str] = None  # ISO timestamp
    created_by: Optional[str] = None  # "model_selector"

    # Error handling
    error: Optional[str] = None
    error_type: Optional[str] = None

    # Audit chain — required: caller MUST provide ``audit_workflow_id``
    # (backlog #1 tightening landed 2026-05-09; PRs #58 / #62 / #65 thread).
    audit_workflow_id: UUID

    _validate_audit_id = audit_workflow_id_validator()
