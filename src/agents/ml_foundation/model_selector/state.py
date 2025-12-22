"""State definition for model_selector agent.

This module defines the TypedDict state used by the model_selector LangGraph.
"""

from typing import Any, Dict, List, Optional, TypedDict


class ModelSelectorState(TypedDict, total=False):
    """State for model_selector agent.

    The model_selector evaluates candidate algorithms and recommends the
    optimal model architecture based on problem scope and constraints.
    """

    # === INPUT FIELDS ===
    # From scope_definer
    scope_spec: Dict[str, Any]  # Complete ScopeSpec
    experiment_id: str  # Extracted from scope_spec
    kpi_category: Optional[str]  # KPI category for domain-specific recommendations

    # From data_preparer
    qc_report: Dict[str, Any]  # Must have passed QC gate
    baseline_metrics: Dict[str, Any]  # Baseline metrics from training data

    # User preferences (optional)
    algorithm_preferences: Optional[List[str]]  # Preferred algorithms
    excluded_algorithms: Optional[List[str]]  # Algorithms to exclude
    interpretability_required: bool  # Whether model must be interpretable

    # Sample data for benchmarking (optional)
    X_sample: Any  # Feature matrix for cross-validation
    y_sample: Any  # Target vector for cross-validation

    # Control flags
    skip_benchmarks: bool  # Skip cross-validation benchmarks
    skip_mlflow: bool  # Skip MLflow registration

    # === INTERMEDIATE FIELDS ===
    # Problem analysis
    problem_type: str  # Extracted from scope_spec
    technical_constraints: List[str]  # Extracted from scope_spec
    row_count: int  # From qc_report
    column_count: int  # From qc_report

    # Algorithm filtering
    candidate_algorithms: List[Dict[str, Any]]  # Filtered candidates
    filtered_by_problem_type: List[Dict[str, Any]]
    filtered_by_constraints: List[Dict[str, Any]]
    filtered_by_preferences: List[Dict[str, Any]]

    # Historical data
    historical_success_rates: Dict[str, float]  # Algorithm -> success rate
    similar_experiments: List[str]  # Similar past experiments

    # Ranking
    ranked_candidates: List[Dict[str, Any]]  # Ranked by selection score
    selection_scores: Dict[str, float]  # Algorithm -> composite score

    # === OUTPUT FIELDS ===
    # Primary selection
    primary_candidate: Dict[str, Any]  # Selected ModelCandidate
    algorithm_name: str  # Selected algorithm name
    algorithm_class: str  # Python class path
    algorithm_family: str  # "causal_ml", "gradient_boosting", etc.

    # Configuration
    default_hyperparameters: Dict[str, Any]  # Starting hyperparameters
    hyperparameter_search_space: Dict[str, Dict[str, Any]]  # Optuna search space

    # Performance expectations
    expected_performance: Dict[str, float]  # Expected metrics
    training_time_estimate_hours: float  # Estimated training time
    estimated_inference_latency_ms: int  # Expected latency
    memory_requirement_gb: float  # Memory requirements

    # Characteristics
    interpretability_score: float  # 0-1 interpretability score
    scalability_score: float  # 0-1 scalability score
    selection_score: float  # Overall selection score

    # Alternative candidates
    alternative_candidates: List[Dict[str, Any]]  # Top 2-3 alternatives

    # Rationale
    selection_rationale: str  # Why this algorithm was selected
    primary_reason: str  # Main selection reason
    supporting_factors: List[str]  # Supporting factors
    alternatives_considered: List[Dict[str, Any]]  # Alternatives with reasons
    constraint_compliance: Dict[str, bool]  # Constraint check results

    # Baseline comparison
    baseline_to_beat: Dict[str, float]  # Baseline model metrics
    baseline_candidates: List[str]  # Baseline models for comparison
    baseline_comparison: Dict[str, Any]  # Full baseline comparison results

    # Benchmarking
    benchmark_results: Dict[str, Dict[str, Any]]  # Algorithm -> benchmark results
    benchmark_rankings: List[Dict[str, Any]]  # Candidates ranked by benchmark
    benchmarks_skipped: bool  # Whether benchmarks were skipped
    benchmark_skip_reason: Optional[str]  # Reason benchmarks skipped
    benchmark_time_seconds: float  # Time spent on benchmarks
    combined_score: Optional[float]  # Combined selection + benchmark score
    benchmark_score: Optional[float]  # Benchmark-only score
    max_benchmark_candidates: int  # Max candidates to benchmark
    cv_folds: int  # Number of CV folds

    # Historical analysis
    historical_data_available: bool  # Whether historical data was found
    historical_experiments_count: int  # Number of historical experiments
    history_recommended_algorithms: List[str]  # Algorithms recommended by history
    recommendation_source: str  # "historical" or "prior_knowledge"
    algorithm_trends: Dict[str, Dict[str, Any]]  # Performance trends

    # MLflow registration
    registered_in_mlflow: bool  # Whether registered in MLflow
    model_version_id: str  # MLflow model version ID
    mlflow_run_id: Optional[str]  # MLflow run ID
    mlflow_experiment_id: Optional[str]  # MLflow experiment ID
    mlflow_registration_error: Optional[str]  # Registration error message
    benchmark_logged: bool  # Whether benchmarks logged to MLflow
    benchmark_log_error: Optional[str]  # Benchmark logging error

    # Selection summary (for database storage)
    selection_summary: Dict[str, Any]  # Complete summary for persistence

    # Stage
    stage: str  # Model stage: "development"

    # Metadata
    created_at: str  # ISO timestamp
    created_by: str  # "model_selector"

    # Error handling
    error: Optional[str]
    error_type: Optional[str]
