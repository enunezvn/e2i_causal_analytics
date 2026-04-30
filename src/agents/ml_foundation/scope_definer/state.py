"""State definition for scope_definer agent.

This module defines the TypedDict state used by the scope_definer LangGraph.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from uuid import UUID


class ScopeDefinerState(TypedDict, total=False):
    """State for scope_definer agent.

    The scope_definer transforms business requirements into formal ML
    problem specifications with success criteria.
    """

    # === INPUT FIELDS ===
    # Business request
    problem_description: str  # Natural language problem description
    business_objective: str  # Business objective this ML model serves
    target_outcome: str  # Target outcome (e.g., "Increase prescriptions")

    # Problem type hint (optional)
    problem_type_hint: Optional[
        Literal[
            "binary_classification",
            "multiclass_classification",
            "regression",
            "causal_inference",
            "time_series",
        ]
    ]

    # Target variable (optional)
    target_variable: Optional[str]  # Target variable name if known

    # Features (optional)
    candidate_features: Optional[List[str]]  # Candidate feature list if known

    # Constraints (optional)
    time_budget_hours: Optional[float]  # Maximum training time budget
    performance_requirements: Optional[Dict[str, float]]  # e.g., {'min_f1': 0.85}

    # Context
    brand: Optional[str]  # Brand context (Remibrutinib, Fabhalta, Kisqali)
    region: Optional[str]  # Region context
    use_case: Optional[str]  # Use case category

    # Temporal anchoring (Block 1B scaffolding; consumed in Block 4+).
    # Inference cutoff time the model predicts from — feeds scope_spec so
    # downstream agents can clip lookback windows and post-prediction filters.
    # Accepts datetime, str, or pd.Timestamp at the API boundary; the scope
    # builder normalises to ISO 8601 string for storage in scope_spec.
    #
    # Distinguish from ``prediction_horizon_days`` (read from state via
    # ``state.get("prediction_horizon_days", 30)`` in scope_builder): the
    # horizon is a *duration* in days, while this field is the *anchor*
    # timestamp. Together they define the prediction window.
    prediction_timestamp: Optional[Any]

    # Cost matrix for business-utility-driven evaluation (Block 5 — finding
    # #10). Keys ``tp``/``fp``/``fn``/``tn`` map confusion-matrix outcomes to
    # their per-prediction monetary value. Optional: when absent, the
    # evaluator skips ``business_utility``. Forwarded verbatim onto
    # ``scope_spec["cost_matrix"]`` so the model_trainer can read it without
    # threading through a separate parameter.
    cost_matrix: Optional[Dict[str, float]]

    # Adaptive success criteria pre-eval inputs (task 05 of
    # adaptive_success_criteria plan). When ``ADAPTIVE_CRITERIA=true`` and
    # all four are present, ``criteria_validator`` stashes them on
    # ``success_criteria['_adaptive_inputs']`` for the evaluator overlay
    # to pick up alongside the live ``baseline_test_auc``. Optional —
    # when any is missing, the validator falls back to fixed thresholds
    # with ``criteria_source="adaptive_fallback_to_fixed"``.
    # NOTE: ``baseline_auc`` is intentionally NOT here — it is computed at
    # eval time inside the evaluator, not at scope-definition time.
    n_samples: Optional[int]            # training-split row count
    prevalence: Optional[float]         # positive-class rate, in [0, 1]
    feature_count: Optional[int]        # post-preprocessing feature count
    regime: Optional[Literal["default", "clean", "adverse"]]

    # === INTERMEDIATE FIELDS ===
    # Problem classification
    inferred_problem_type: str  # Inferred ML problem type
    inferred_target_variable: str  # Inferred target variable name

    # Feature requirements
    required_features: List[str]  # Features required for training
    excluded_features: List[str]  # Features to exclude (PII, leakage risks)
    feature_categories: List[str]  # Feature categories

    # Population criteria
    target_population: str  # Population description
    inclusion_criteria: List[str]  # Data inclusion criteria
    exclusion_criteria: List[str]  # Data exclusion criteria
    minimum_samples: int  # Minimum required samples

    # Constraints
    regulatory_constraints: List[str]  # Regulatory constraints
    ethical_constraints: List[str]  # Ethical constraints
    technical_constraints: List[str]  # Technical constraints

    # Success criteria
    minimum_auc: Optional[float]  # For classification
    minimum_precision: Optional[float]
    minimum_recall: Optional[float]
    minimum_f1: Optional[float]
    minimum_rmse: Optional[float]  # For regression
    minimum_r2: Optional[float]
    baseline_model: str  # Baseline to beat
    minimum_lift_over_baseline: float  # Required improvement

    # Validation
    validation_passed: bool
    validation_warnings: List[str]
    validation_errors: List[str]

    # === OUTPUT FIELDS ===
    # Experiment identification
    experiment_id: str  # Unique experiment identifier
    experiment_name: str  # Human-readable experiment name

    # ScopeSpec (complete specification)
    scope_spec: Dict[str, Any]  # Complete ScopeSpec as dict

    # SuccessCriteria (complete criteria)
    success_criteria: Dict[str, Any]  # Complete SuccessCriteria as dict

    # Metadata
    created_at: str  # ISO timestamp
    created_by: str  # "scope_definer"

    # Error handling
    error: Optional[str]
    error_type: Optional[str]

    # Audit chain
    audit_workflow_id: UUID
