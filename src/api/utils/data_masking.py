"""Data masking utilities for PII protection in API responses.

This module provides functions to mask sensitive data (patient IDs, HCP IDs)
before returning them in API responses. This ensures compliance with privacy
requirements while still providing useful reference information.

Masking patterns:
- Patient IDs: PAT-2024-001234 -> PAT-****-1234 (preserve prefix and last 4 chars)
- HCP IDs: HCP-NE-5678 -> HCP-**-5678 (preserve prefix and last 4 chars)

Usage:
    from src.api.utils.data_masking import mask_pii, mask_response_dict

    # Mask single value
    masked_id = mask_pii("PAT-2024-001234", "patient_id")

    # Mask entire response dict
    masked_response = mask_response_dict(response, pii_fields=["patient_id", "hcp_id"])
"""

import re
from collections.abc import Set as AbstractSet
from typing import Any, Dict, List, Optional, Union

# Default PII fields to mask in responses
DEFAULT_PII_FIELDS = frozenset(["patient_id", "hcp_id"])

# Minimum characters to preserve at start and end
MIN_PRESERVE_START = 4
MIN_PRESERVE_END = 4
MASK_CHAR = "*"

# A short, human-readable label prefix such as ``PAT-``, ``HCP-``, ``ID-``.
# Bounded to <= 6 letters so that long all-letter blocks (e.g. a UUID head that
# happens to be all hex-letters) are NOT mistaken for a label and exposed.
_LABEL_PREFIX_RE = re.compile(r"^([A-Za-z]{1,6})-")

# UUID (8-4-4-4-12 hex). Such identifiers carry no safe "prefix" — the leading
# block is just as sensitive as the rest, so they are masked head-to-tail.
_UUID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


def _mask_all_but_last(value: str, preserve_end: int) -> str:
    """Mask every character except the last ``preserve_end`` (the safe default).

    For PHI/PII the conservative posture is to expose only a short trailing
    window for human reference and hide everything before it (no leading window).
    """
    if len(value) <= preserve_end:
        # Too short to keep a full trailing window without exposing most of it:
        # keep at most the final character.
        if len(value) <= 1:
            return value
        return MASK_CHAR * (len(value) - 1) + value[-1]
    return MASK_CHAR * (len(value) - preserve_end) + value[-preserve_end:]


def mask_identifier(value: str, preserve_end: int = 4) -> str:
    """Mask an identifier value, preserving a label prefix and last N chars.

    Masking strategy (PII hardening):
    - Labelled IDs (``PAT-…``, ``HCP-…``): keep the short label prefix and the
      last N chars; mask the body in between.
    - UUIDs: no safe prefix — mask all but the last N chars.
    - Numeric / SSN / opaque IDs (no label prefix): mask all but the last N
      chars (NO leading window — the head is the sensitive part).
    - Very short IDs: keep at most the final character.

    Args:
        value: The identifier string to mask
        preserve_end: Number of characters to preserve at the end

    Returns:
        Masked string exposing at most a label prefix + the last N characters.

    Examples:
        >>> mask_identifier("PAT-2024-001234")
        'PAT-*******1234'
        >>> mask_identifier("HCP-NE-5678")
        'HCP-***5678'
        >>> mask_identifier("12345678901234")
        '**********1234'
    """
    if not value or not isinstance(value, str):
        return value

    # Handle very short strings — expose at most the final character.
    if len(value) <= preserve_end + MIN_PRESERVE_START:
        if len(value) <= 1:
            return value
        return MASK_CHAR * (len(value) - 1) + value[-1]

    # UUIDs carry no safe prefix: mask head-to-tail. Checked BEFORE the label
    # prefix so an all-hex-letter UUID head is never exposed as a "label".
    if _UUID_RE.match(value):
        return _mask_all_but_last(value, preserve_end)

    # Short human-readable label prefix (PAT-, HCP-, ID-, ...).
    prefix_match = _LABEL_PREFIX_RE.match(value)
    if prefix_match:
        prefix = prefix_match.group(1) + "-"
        rest = value[len(prefix) :]

        # Preserve last N characters of the rest; mask everything before them.
        if len(rest) <= preserve_end:
            masked_rest = MASK_CHAR * len(rest)
        else:
            masked_portion_len = len(rest) - preserve_end
            masked_rest = MASK_CHAR * masked_portion_len + rest[-preserve_end:]

        return prefix + masked_rest

    # No label prefix (numeric, SSN, opaque): mask all but the trailing window.
    # NO leading window — the head of an MRN/SSN/account number is sensitive.
    return _mask_all_but_last(value, preserve_end)


def mask_pii(
    value: Any,
    field_name: str,
    pii_fields: Optional[AbstractSet[str]] = None,
) -> Any:
    """Mask a PII value based on field name.

    Args:
        value: The value to potentially mask
        field_name: The field name to check against PII fields
        pii_fields: Set of field names considered PII (defaults to DEFAULT_PII_FIELDS)

    Returns:
        Masked value if field is PII, otherwise original value
    """
    if pii_fields is None:
        pii_fields = DEFAULT_PII_FIELDS

    if field_name not in pii_fields:
        return value

    if value is None:
        return None

    if isinstance(value, str):
        return mask_identifier(value)
    elif isinstance(value, list):
        return [mask_identifier(v) if isinstance(v, str) else v for v in value]
    else:
        return value


def mask_response_dict(
    data: Dict[str, Any],
    pii_fields: Optional[AbstractSet[str]] = None,
    recursive: bool = True,
) -> Dict[str, Any]:
    """Mask PII fields in a response dictionary.

    Args:
        data: Dictionary containing response data
        pii_fields: Set of field names to mask (defaults to DEFAULT_PII_FIELDS)
        recursive: Whether to recursively mask nested dicts/lists

    Returns:
        New dictionary with PII fields masked

    Example:
        >>> data = {"patient_id": "PAT-2024-001234", "name": "Test", "hcp_id": "HCP-NE-5678"}
        >>> mask_response_dict(data)
        {'patient_id': 'PAT-****-1234', 'name': 'Test', 'hcp_id': 'HCP-**-5678'}
    """
    if pii_fields is None:
        pii_fields = DEFAULT_PII_FIELDS

    if not isinstance(data, dict):
        return data

    result = {}
    for key, value in data.items():
        if key in pii_fields:
            result[key] = mask_pii(value, key, pii_fields)
        elif recursive:
            if isinstance(value, dict):
                result[key] = mask_response_dict(value, pii_fields, recursive)
            elif isinstance(value, list):
                result[key] = _mask_list(value, pii_fields)
            else:
                result[key] = value
        else:
            result[key] = value

    return result


def _mask_list(
    items: List[Any],
    pii_fields: AbstractSet[str],
) -> List[Any]:
    """Mask PII fields in a list of items.

    Args:
        items: List of items (dicts, strings, or other values)
        pii_fields: Set of field names to mask

    Returns:
        New list with PII fields masked in dict items
    """
    result = []
    for item in items:
        if isinstance(item, dict):
            result.append(mask_response_dict(item, pii_fields))
        else:
            result.append(item)
    return result


def create_masked_model_response(
    model_instance: Any,
    pii_fields: Optional[AbstractSet[str]] = None,
) -> Dict[str, Any]:
    """Create a masked dictionary from a Pydantic model instance.

    Args:
        model_instance: Pydantic model instance with model_dump() method
        pii_fields: Set of field names to mask

    Returns:
        Dictionary with PII fields masked

    Example:
        >>> response = ExplainResponse(patient_id="PAT-2024-001234", ...)
        >>> masked = create_masked_model_response(response)
        >>> masked["patient_id"]
        'PAT-****-1234'
    """
    if hasattr(model_instance, "model_dump"):
        data = model_instance.model_dump()
    elif hasattr(model_instance, "dict"):
        # Pydantic v1 compatibility
        data = model_instance.dict()
    else:
        raise TypeError(f"Expected Pydantic model, got {type(model_instance)}")

    return mask_response_dict(data, pii_fields)


class PIIMaskingMiddleware:
    """Response wrapper for automatic PII masking.

    Use this to wrap response data before returning from API endpoints.

    Example:
        masker = PIIMaskingMiddleware()
        return masker.mask(response_data)
    """

    def __init__(
        self,
        pii_fields: Optional[AbstractSet[str]] = None,
        enabled: bool = True,
    ):
        """Initialize the masking middleware.

        Args:
            pii_fields: Custom set of PII field names to mask
            enabled: Whether masking is enabled (for testing/debug)
        """
        self.pii_fields = pii_fields or DEFAULT_PII_FIELDS
        self.enabled = enabled

    def mask(self, data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
        """Mask PII in the given data.

        Args:
            data: Response data (dict, list, or Pydantic model)

        Returns:
            Masked data
        """
        if not self.enabled:
            return data

        if hasattr(data, "model_dump"):
            return create_masked_model_response(data, self.pii_fields)
        elif isinstance(data, dict):
            return mask_response_dict(data, self.pii_fields)
        elif isinstance(data, list):
            return _mask_list(data, self.pii_fields)
        else:
            return data


# Singleton instance for convenience
_default_masker = PIIMaskingMiddleware()


def mask_response(data: Union[Dict, List, Any]) -> Union[Dict, List, Any]:
    """Convenience function to mask PII using default settings.

    Args:
        data: Response data to mask

    Returns:
        Masked response data
    """
    return _default_masker.mask(data)
