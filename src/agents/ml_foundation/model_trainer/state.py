"""State definition for model_trainer agent.

This module defines the TypedDict state used by the model_trainer LangGraph.
"""

from typing import Any, Dict, List, Optional, TypedDict
from uuid import UUID


class ModelTrainerState(TypedDict, total=False):
    """State for model_trainer agent.

    The model_trainer executes the complete ML training pipeline with strict
    split enforcement, hyperparameter optimization, and MLflow logging.
    """

    # === INPUT FIELDS ===
    # From model_selector
    model_candidate: Dict[str, Any]  # Complete ModelCandidate
    algorithm_name: str  # Extracted from model_candidate
    algorithm_class: str  # Python class path
    hyperparameter_search_space: Dict[str, Dict[str, Any]]  # Optuna search space
    default_hyperparameters: Dict[str, Any]  # Starting hyperparameters

    # From data_preparer
    qc_report: Dict[str, Any]  # QC validation report
    experiment_id: str  # Experiment identifier

    # From scope_definer
    success_criteria: Dict[str, float]  # Performance thresholds to meet
    problem_type: str  # binary_classification, regression, etc.

    # Training configuration
    enable_hpo: bool  # Whether to run hyperparameter optimization
    hpo_trials: int  # Number of Optuna trials
    hpo_timeout_hours: Optional[float]  # HPO timeout
    early_stopping: bool  # Enable early stopping
    early_stopping_patience: int  # Early stopping patience epochs
    enable_mlflow: bool  # Whether to log to MLflow
    enable_checkpointing: bool  # Whether to save model checkpoints

    # === INTERMEDIATE FIELDS ===
    # QC Gate
    qc_gate_passed: bool  # Whether QC gate check passed
    qc_gate_message: str  # Gate check message

    # Data Splits
    train_data: Dict[str, Any]  # Training split (X, y, row_count)
    validation_data: Dict[str, Any]  # Validation split (X, y, row_count)
    test_data: Dict[str, Any]  # Test split (X, y, row_count)
    holdout_data: Dict[str, Any]  # Holdout split (X, y, row_count) - LOCKED

    # Split Validation
    split_ratios_valid: bool  # Whether splits match expected ratios
    train_samples: int
    validation_samples: int
    test_samples: int
    holdout_samples: int
    total_samples: int
    train_ratio: float  # Actual train ratio (should be ~0.60)
    validation_ratio: float  # Actual validation ratio (should be ~0.20)
    test_ratio: float  # Actual test ratio (should be ~0.15)
    holdout_ratio: float  # Actual holdout ratio (should be ~0.05)
    split_validation_message: str  # Split validation message
    split_ratio_checks: List[str]  # Individual ratio check results
    leakage_warnings: List[str]  # Data leakage warnings

    # Class Imbalance Detection
    imbalance_detected: bool  # Whether imbalance was detected
    imbalance_ratio: float  # Majority/minority ratio (e.g., 10.0 means 10:1)
    minority_ratio: float  # Minority class percentage (e.g., 0.09 for 9%)
    imbalance_severity: str  # none, moderate, severe, extreme
    class_distribution: Dict[int, int]  # {0: 800, 1: 77}
    recommended_strategy: str  # smote, random_oversample, class_weight, etc.
    strategy_rationale: str  # LLM explanation for strategy choice

    # Resampling Results
    X_train_resampled: Any  # Resampled training features
    y_train_resampled: Any  # Resampled training labels
    resampling_applied: bool  # Whether resampling was actually applied
    resampling_strategy: str  # Strategy that was applied
    original_train_shape: tuple  # Shape before resampling
    resampled_train_shape: tuple  # Shape after resampling
    original_distribution: Dict[int, int]  # Class counts before
    resampled_distribution: Dict[int, int]  # Class counts after

    # Feature Names (preserved from data_preparer)
    feature_columns: List[str]  # Original feature names from data_preparer

    # Preprocessing
    preprocessor: Any  # Fitted preprocessing pipeline (fit on train only)
    X_train_preprocessed: Any  # Transformed training data
    X_validation_preprocessed: Any  # Transformed validation data
    X_test_preprocessed: Any  # Transformed test data
    preprocessing_statistics: Dict[str, Any]  # Statistics from train split

    # Hyperparameter Tuning
    hpo_completed: bool  # Whether HPO completed
    hpo_best_trial: Optional[int]  # Best trial number
    best_hyperparameters: Dict[str, Any]  # Best hyperparameters found
    hpo_trials_run: int  # Number of trials actually run
    hpo_duration_seconds: float  # HPO duration

    # Model Training
    trained_model: Any  # Trained model object
    training_duration_seconds: float  # Training duration
    early_stopped: bool  # Whether training stopped early
    final_epoch: Optional[int]  # Final epoch number

    # Model Evaluation
    train_metrics: Dict[str, float]  # Training set metrics
    validation_metrics: Dict[str, float]  # Validation set metrics
    test_metrics: Dict[str, float]  # Test set metrics (FINAL)

    # Classification Metrics (problem-type specific)
    auc_roc: Optional[float]  # AUC-ROC
    precision: Optional[float]  # Precision
    recall: Optional[float]  # Recall
    f1_score: Optional[float]  # F1 score
    pr_auc: Optional[float]  # Precision-Recall AUC
    confusion_matrix: Optional[Dict[str, int]]  # TP, TN, FP, FN

    # Regression Metrics (problem-type specific)
    rmse: Optional[float]  # Root mean squared error
    mae: Optional[float]  # Mean absolute error
    r2: Optional[float]  # R-squared

    # Calibration (classification only)
    brier_score: Optional[float]  # Brier score
    calibration_error: Optional[float]  # Expected calibration error

    # Threshold Analysis
    optimal_threshold: float  # Optimal classification threshold
    precision_at_k: Dict[int, float]  # {100: 0.35, 500: 0.28}

    # Confidence Intervals
    confidence_interval: Dict[str, tuple]  # {'auc': (0.78, 0.85)}
    bootstrap_samples: int  # Number of bootstrap samples

    # Success Criteria Check
    success_criteria_met: bool  # Whether all criteria met
    success_criteria_results: Dict[str, bool]  # Metric -> passed/failed

    # === OUTPUT FIELDS ===
    # Trained Model
    training_run_id: str  # Unique training run ID
    model_id: str  # Model identifier

    # MLflow Integration (populated by log_to_mlflow node)
    mlflow_run_id: Optional[str]  # MLflow run ID
    mlflow_experiment_id: Optional[str]  # MLflow experiment ID
    mlflow_status: str  # MLflow logging status: success, disabled, skipped, failed
    mlflow_model_uri: Optional[str]  # MLflow model URI (runs:/<run_id>/model)
    mlflow_registered: bool  # Whether model was registered in registry
    mlflow_model_version: Optional[str]  # Registered model version
    mlflow_model_name: Optional[str]  # Registered model name
    db_training_run_id: Optional[str]  # Database training run ID (UUID)
    # Legacy fields (kept for compatibility)
    model_artifact_uri: str  # MLflow model artifact URI (deprecated, use mlflow_model_uri)
    preprocessing_artifact_uri: str  # Preprocessing artifact URI
    registered_model_name: str  # MLflow registered model name (deprecated)
    model_version: int  # MLflow model version (deprecated)
    model_stage: str  # MLflow stage (Staging, Production)

    # Artifacts
    model_artifact_path: str  # Local model artifact path
    preprocessing_artifact_path: str  # Local preprocessing artifact path

    # Timing
    training_started_at: str  # ISO timestamp
    training_completed_at: str  # ISO timestamp
    total_training_duration_seconds: float  # Total end-to-end duration

    # Status
    training_status: str  # running, completed, failed
    training_error: Optional[str]  # Error message if failed

    # Metadata
    framework: str  # ML framework (econml, xgboost, sklearn, etc.)
    trained_by: str  # Agent name (model_trainer)
    created_at: str  # ISO timestamp

    # Database
    persisted_to_db: bool  # Whether saved to ml_training_runs table

    # Error handling
    error: Optional[str]
    error_type: Optional[str]

    # Audit chain
    audit_workflow_id: UUID
