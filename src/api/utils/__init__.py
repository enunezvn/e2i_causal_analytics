"""API utility modules.

This package contains utility functions for the API layer.
"""

from src.api.utils.agent_import_guard import (
    guard_or_raise,
    raise_503_for_import_error,
    should_fail_closed_on_import_error,
)
from src.api.utils.data_masking import (
    DEFAULT_PII_FIELDS,
    PIIMaskingMiddleware,
    create_masked_model_response,
    mask_identifier,
    mask_pii,
    mask_response,
    mask_response_dict,
)

__all__ = [
    "DEFAULT_PII_FIELDS",
    "PIIMaskingMiddleware",
    "create_masked_model_response",
    "guard_or_raise",
    "mask_identifier",
    "mask_pii",
    "mask_response",
    "mask_response_dict",
    "raise_503_for_import_error",
    "should_fail_closed_on_import_error",
]
