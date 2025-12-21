"""State definition for scope_definer agent.

This module defines the TypedDict state used by the scope_definer LangGraph.
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict


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
