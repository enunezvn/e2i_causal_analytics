"""Custom training objectives for boosting algorithms (Phase 1 W4).

Phase 1 W4 introduces proper-scoring-rule objectives (Brier) for LightGBM
and XGBoost so the gradient-boosting search optimizes calibration directly
rather than via post-hoc isotonic. Reference: shard 17 Week 3 rows
Day 1 + Day 2 of `.claude/plans/adaptive_criteria_v3_followup/`.
"""

from src.mlops.objectives.brier import (
    brier_objective_lightgbm,
    brier_objective_xgboost,
)

__all__ = ["brier_objective_lightgbm", "brier_objective_xgboost"]
