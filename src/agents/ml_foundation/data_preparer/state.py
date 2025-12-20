"""State definition for data_preparer agent.

This module defines the TypedDict state used by the data_preparer LangGraph.
"""

from typing import TypedDict, Optional, List, Dict, Any, Literal
from datetime import datetime


class DataPreparerState(TypedDict, total=False):
    """State for data_preparer agent.

    The data_preparer validates data quality, computes baseline metrics,
    and enforces a QC gate that blocks downstream training if quality fails.
    """

    # === INPUT FIELDS ===
    # From scope_definer
    experiment_id: str
    scope_spec: Dict[str, Any]  # ScopeSpec from scope_definer

    # Data source configuration
    data_source: str  # Table/view name
    split_id: Optional[str]  # ML split ID (if using existing split)

    # Validation configuration
    validation_suite: Optional[str]  # Great Expectations suite name
    skip_leakage_check: bool  # Skip data leakage detection (NOT RECOMMENDED)

    # === INTERMEDIATE FIELDS ===
    # Data loading
    train_df: Any  # pandas DataFrame (train split)
    validation_df: Any  # pandas DataFrame (validation split)
    test_df: Any  # pandas DataFrame (test split)
    holdout_df: Any  # pandas DataFrame (holdout split)

    # Quality checks
    expectation_results: List[Dict[str, Any]]  # Great Expectations results
    failed_expectations: List[str]
    warnings: List[str]

    # Dimension scores
    completeness_score: float
    validity_score: float
    consistency_score: float
    uniqueness_score: float
    timeliness_score: float
    overall_score: float

    # Leakage detection
    leakage_detected: bool
    leakage_issues: List[str]

    # Baseline computation
    feature_stats: Dict[str, Dict[str, Any]]  # Per-feature statistics
    target_rate: Optional[float]  # For classification
    target_distribution: Dict[str, Any]
    correlation_matrix: Dict[str, Dict[str, float]]

    # Recommendations
    remediation_steps: List[str]
    blocking_issues: List[str]  # If non-empty, blocks training

    # === OUTPUT FIELDS ===
    # QC Report
    report_id: str
    qc_status: Literal["passed", "failed", "warning", "skipped"]
    row_count: int
    column_count: int
    validated_at: str  # ISO timestamp

    # Data Readiness
    total_samples: int
    train_samples: int
    validation_samples: int
    test_samples: int
    holdout_samples: int
    available_features: List[str]
    missing_required_features: List[str]
    is_ready: bool
    qc_passed: bool
    qc_score: float
    blockers: List[str]

    # Gate decision
    gate_passed: bool  # CRITICAL: blocks model_trainer if False

    # Metadata
    validation_duration_seconds: float
    computed_at: str  # ISO timestamp
    training_samples: int  # For baseline metrics

    # Error handling
    error: Optional[str]
    error_type: Optional[str]
