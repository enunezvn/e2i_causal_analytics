"""
E2I Data Generating Processes (DGP)

Each DGP generates synthetic data with known causal effects:
- Simple Linear: TRUE_ATE = 0.40, no confounding
- Confounded: TRUE_ATE = 0.25, requires adjustment
- Heterogeneous: CATE varies by segment
- Time-Series: TRUE_ATE = 0.30, with lag effects
- Selection Bias: TRUE_ATE = 0.35, requires IPW
"""

# DGP implementations will be added in Phase 2
__all__ = []
