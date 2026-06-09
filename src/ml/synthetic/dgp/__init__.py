"""
E2I Data Generating Processes (DGP)

Each DGP generates synthetic data with known causal effects:
- Simple Linear: TRUE_ATE = 0.40, no confounding
- Confounded: TRUE_ATE = 0.25, requires adjustment
- Heterogeneous: CATE varies by segment
- Time-Series: TRUE_ATE = 0.30, with lag effects
- Selection Bias: TRUE_ATE = 0.35, requires IPW
"""

from .treatment_arm import (
    SEGMENT_HIGH,
    SEGMENT_LOW,
    SEGMENT_MEDIUM,
    assign_segment,
    assign_treatment_arm,
    binary_outcome_with_cate,
    brand_scaled_cate,
    rd_map_from_tau,
)

__all__ = [
    "assign_treatment_arm",
    "brand_scaled_cate",
    "assign_segment",
    "binary_outcome_with_cate",
    "rd_map_from_tau",
    "SEGMENT_HIGH",
    "SEGMENT_MEDIUM",
    "SEGMENT_LOW",
]
