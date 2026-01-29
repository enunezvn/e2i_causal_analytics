"""Contract Validator for Agent Output Validation.

Validates agent outputs against TypedDict state contracts.
"""

from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, get_args, get_origin, get_type_hints

# Python 3.11+ has Required/NotRequired in typing, else use typing_extensions
try:
    from typing import NotRequired, Required
except ImportError:
    from typing_extensions import NotRequired, Required


@dataclass
class ValidationResult:
    """Result of validating an agent output against its contract."""

    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_fields: int = 0
    required_fields_present: int = 0
    required_fields_total: int = 0
    optional_fields_present: int = 0
    optional_fields_total: int = 0
    type_errors: list[dict[str, str]] = field(default_factory=list)
    extra_fields: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Get a summary string of the validation result."""
        if self.valid:
            return f"VALID: {self.required_fields_present}/{self.required_fields_total} required, {len(self.warnings)} warnings"
        return f"INVALID: {len(self.errors)} errors, {len(self.type_errors)} type errors"


class ContractValidator:
    """Validates agent outputs against TypedDict state contracts.

    Usage:
        validator = ContractValidator()
        result = validator.validate_state(output, CausalImpactState)
        if result.valid:
            print("Output matches contract")
        else:
            print(f"Validation errors: {result.errors}")

    NOTE: The 'lenient' mode has been removed. Missing required fields are ALWAYS
    errors. Use QualityGateValidator for per-agent quality thresholds that define
    what fields are truly required for each agent's output.
    """

    def __init__(self, strict: bool = False):
        """Initialize validator.

        Args:
            strict: If True, treat warnings as errors
        """
        self.strict = strict

    def validate_state(
        self,
        state: dict[str, Any],
        state_class: type,
        strict: bool | None = None,
    ) -> ValidationResult:
        """Validate a state dictionary against a TypedDict class.

        Args:
            state: The state dictionary to validate
            state_class: The TypedDict class defining the contract
            strict: Override instance-level strict setting

        Returns:
            ValidationResult with validation details
        """
        use_strict = strict if strict is not None else self.strict

        errors: list[str] = []
        warnings: list[str] = []
        type_errors: list[dict[str, str]] = []

        # Get type hints for the state class
        try:
            hints = get_type_hints(state_class)
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to get type hints for {state_class.__name__}: {e}"],
            )

        # Determine required vs optional fields
        required_keys = self._get_required_keys(state_class)
        optional_keys = set(hints.keys()) - required_keys

        # Check required fields
        required_present = 0
        for field_name in required_keys:
            if field_name not in state:
                # Missing required fields are ALWAYS errors
                errors.append(f"Missing required field: {field_name}")
            else:
                required_present += 1
                # Type check
                type_error = self._check_type(
                    field_name, state[field_name], hints.get(field_name)
                )
                if type_error:
                    type_errors.append(type_error)

        # Check optional fields (presence is not required, but type should match if present)
        optional_present = 0
        for field_name in optional_keys:
            if field_name in state:
                optional_present += 1
                # Type check
                type_error = self._check_type(
                    field_name, state[field_name], hints.get(field_name)
                )
                if type_error:
                    type_errors.append(type_error)

        # Check for extra fields not in the TypedDict
        known_fields = set(hints.keys())
        extra_fields = [k for k in state.keys() if k not in known_fields]
        if extra_fields:
            warnings.append(f"Extra fields not in contract: {extra_fields}")

        # Determine validity
        is_valid = len(errors) == 0 and len(type_errors) == 0
        if use_strict and warnings:
            is_valid = False
            errors.extend(warnings)

        return ValidationResult(
            valid=is_valid,
            errors=errors,
            warnings=warnings,
            checked_fields=len(hints),
            required_fields_present=required_present,
            required_fields_total=len(required_keys),
            optional_fields_present=optional_present,
            optional_fields_total=len(optional_keys),
            type_errors=type_errors,
            extra_fields=extra_fields,
        )

    def _get_required_keys(self, state_class: type) -> set[str]:
        """Get required keys from a TypedDict class.

        TypedDict classes may have __required_keys__ and __optional_keys__
        attributes (Python 3.9+). However, TypedDict doesn't distinguish between
        Optional[X] (value can be None) and NotRequired[X] (key can be missing).

        For practical purposes in agent testing, we treat Optional[X] fields as
        "key not required" since agents typically only populate fields that are
        relevant to the analysis.
        """
        hints = get_type_hints(state_class, include_extras=True)
        required = set()

        for field_name, field_type in hints.items():
            origin = get_origin(field_type)

            # Check if it's explicitly marked as NotRequired
            if origin is NotRequired:
                continue

            # Check if it's Optional (Union[X, None] or X | None)
            # Optional types have the key as not required for practical testing
            if self._is_optional_type(field_type):
                continue

            # Check if it's explicitly marked as Required or has no wrapper
            required.add(field_name)

        return required

    def _is_optional_type(self, field_type: type) -> bool:
        """Check if a type is Optional (i.e., Union[X, None] or X | None).

        Args:
            field_type: The type annotation to check

        Returns:
            True if the type is Optional, False otherwise
        """
        origin = get_origin(field_type)

        # Check for Union types (Optional[X] is Union[X, None])
        if origin is typing.Union:
            args = get_args(field_type)
            # Optional[X] is Union[X, None], so check if None is in the args
            return type(None) in args

        return False

    def _check_type(
        self,
        field_name: str,
        value: Any,
        expected_type: type | None,
    ) -> dict[str, str] | None:
        """Check if a value matches the expected type.

        Returns:
            Type error dict if mismatch, None if OK
        """
        if expected_type is None:
            return None

        # Handle None values
        if value is None:
            # Check if Optional
            origin = get_origin(expected_type)
            if origin is typing.Union:
                args = get_args(expected_type)
                if type(None) in args:
                    return None  # None is allowed for Optional types
            return {
                "field": field_name,
                "expected": str(expected_type),
                "actual": "None",
                "message": f"Field {field_name} is None but not Optional",
            }

        # Unwrap Optional, Required, NotRequired, Annotated
        actual_type = self._unwrap_type(expected_type)

        # Handle basic type checking
        if actual_type is Any:
            return None

        # Get the origin of generic types
        origin = get_origin(actual_type)

        # Handle Union types (including Optional)
        if origin is typing.Union:
            args = get_args(actual_type)
            for arg in args:
                if arg is type(None) and value is None:
                    return None
                if arg is not type(None) and self._is_instance(value, arg):
                    return None
            return {
                "field": field_name,
                "expected": str(expected_type),
                "actual": type(value).__name__,
                "message": f"Value doesn't match any type in Union",
            }

        # Handle list, dict, etc.
        if origin is list:
            if not isinstance(value, list):
                return {
                    "field": field_name,
                    "expected": "list",
                    "actual": type(value).__name__,
                    "message": f"Expected list, got {type(value).__name__}",
                }
            return None

        if origin is dict:
            if not isinstance(value, dict):
                return {
                    "field": field_name,
                    "expected": "dict",
                    "actual": type(value).__name__,
                    "message": f"Expected dict, got {type(value).__name__}",
                }
            return None

        # Handle Literal types
        if origin is typing.Literal:
            allowed_values = get_args(actual_type)
            if value not in allowed_values:
                return {
                    "field": field_name,
                    "expected": f"Literal{allowed_values}",
                    "actual": repr(value),
                    "message": f"Value {value!r} not in allowed values {allowed_values}",
                }
            return None

        # Basic isinstance check
        if not self._is_instance(value, actual_type):
            return {
                "field": field_name,
                "expected": str(expected_type),
                "actual": type(value).__name__,
                "message": f"Type mismatch",
            }

        return None

    def _unwrap_type(self, type_hint: type) -> type:
        """Unwrap type hint wrappers like Optional, Required, NotRequired, Annotated."""
        origin = get_origin(type_hint)

        # Handle Annotated
        if origin is typing.Annotated:
            args = get_args(type_hint)
            if args:
                return self._unwrap_type(args[0])
            return type_hint

        # Handle Required/NotRequired
        if origin in (Required, NotRequired):
            args = get_args(type_hint)
            if args:
                return self._unwrap_type(args[0])
            return type_hint

        return type_hint

    def _is_instance(self, value: Any, expected_type: type) -> bool:
        """Check if value is instance of expected type, handling edge cases."""
        try:
            # Handle string type annotations
            if isinstance(expected_type, str):
                return True  # Can't validate string annotations

            # Get origin for generic types
            origin = get_origin(expected_type)

            # Handle generic types
            if origin is not None:
                # For list[X], dict[K, V], etc., just check the container type
                if origin is list:
                    return isinstance(value, list)
                if origin is dict:
                    return isinstance(value, dict)
                if origin is set:
                    return isinstance(value, set)
                if origin is tuple:
                    return isinstance(value, tuple)
                # For other origins, check against the origin
                return isinstance(value, origin)

            # Standard isinstance check
            return isinstance(value, expected_type)
        except TypeError:
            # If isinstance fails (e.g., for special types), allow it
            return True

    def get_contract_summary(self, state_class: type) -> dict[str, Any]:
        """Get a summary of a TypedDict contract.

        Args:
            state_class: The TypedDict class

        Returns:
            Dict with contract summary including fields, types, required/optional
        """
        hints = get_type_hints(state_class)
        required = self._get_required_keys(state_class)

        return {
            "class_name": state_class.__name__,
            "total_fields": len(hints),
            "required_fields": list(required),
            "optional_fields": [k for k in hints.keys() if k not in required],
            "field_types": {k: str(v) for k, v in hints.items()},
        }
