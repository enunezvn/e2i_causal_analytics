"""
E2I Synthetic Data Generation Module

Generates synthetic pharmaceutical data with embedded ground truth causal effects
for pipeline validation. Supports 5 Data Generating Processes (DGPs) across 3 brands.

Key Features:
- Known TRUE_ATE values for DoWhy validation
- ML-compliant splits (60/20/15/5)
- Zero data leakage (patient-level isolation)
- Schema-compliant with Supabase tables
"""

from .config import (
    BRANDS,
    DGP_TYPES,
    DGPConfig,
    SplitBoundaries,
    SyntheticDataConfig,
)

__all__ = [
    "SyntheticDataConfig",
    "DGPConfig",
    "SplitBoundaries",
    "BRANDS",
    "DGP_TYPES",
]

__version__ = "1.0.0"
