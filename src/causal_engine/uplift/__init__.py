"""
E2I Causal Analytics - Uplift Modeling Module
==============================================

CausalML-based uplift modeling for heterogeneous treatment effect estimation.

Components:
- UpliftRandomForest: Tree-based ensemble for uplift
- UpliftTree: Single tree uplift model
- UpliftGradientBoosting: Gradient boosting meta-learners
- Metrics: AUUC, Qini, cumulative gain

Example:
    >>> from src.causal_engine.uplift import (
    ...     UpliftRandomForest,
    ...     UpliftConfig,
    ...     evaluate_uplift_model,
    ... )
    >>> config = UpliftConfig(n_estimators=100, max_depth=5)
    >>> model = UpliftRandomForest(config)
    >>> result = model.estimate(X, treatment, y)
    >>> metrics = evaluate_uplift_model(result.uplift_scores, treatment, y)
    >>> print(f"AUUC: {metrics.auuc:.4f}, Qini: {metrics.qini_coefficient:.4f}")

Author: E2I Causal Analytics Team
"""

from .base import (
    BaseUpliftModel,
    UpliftConfig,
    UpliftModelType,
    UpliftNormalization,
    UpliftResult,
)
from .gradient_boosting import (
    GradientBoostingMetaLearner,
    GradientBoostingUpliftConfig,
    UpliftGradientBoosting,
)
from .metrics import (
    UpliftMetrics,
    auuc,
    calculate_cumulative_gain,
    calculate_qini_curve,
    calculate_uplift_curve,
    cumulative_gain_auc,
    evaluate_uplift_model,
    qini_auc,
    qini_coefficient,
    treatment_effect_calibration,
    uplift_at_k,
)
from .random_forest import UpliftRandomForest, UpliftTree

__all__ = [
    # Base classes
    "BaseUpliftModel",
    "UpliftConfig",
    "UpliftResult",
    "UpliftModelType",
    "UpliftNormalization",
    # Random Forest
    "UpliftRandomForest",
    "UpliftTree",
    # Gradient Boosting
    "UpliftGradientBoosting",
    "GradientBoostingUpliftConfig",
    "GradientBoostingMetaLearner",
    # Metrics
    "UpliftMetrics",
    "auuc",
    "qini_coefficient",
    "qini_auc",
    "cumulative_gain_auc",
    "uplift_at_k",
    "evaluate_uplift_model",
    "calculate_uplift_curve",
    "calculate_qini_curve",
    "calculate_cumulative_gain",
    "treatment_effect_calibration",
]
