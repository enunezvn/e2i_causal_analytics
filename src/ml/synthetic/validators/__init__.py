"""
E2I Synthetic Data Validators

Validation framework ensuring:
- Schema compliance with Supabase tables
- Causal effect recovery within tolerance
- ML-compliant splits with no data leakage

Test-only-by-design: this cluster has no production consumers (imported only
by tests).
"""

from .causal_validator import CausalValidationResult, CausalValidator
from .schema_validator import SchemaValidationResult, SchemaValidator
from .split_validator import SplitValidationResult, SplitValidator

__all__ = [
    "SchemaValidator",
    "SchemaValidationResult",
    "CausalValidator",
    "CausalValidationResult",
    "SplitValidator",
    "SplitValidationResult",
]
