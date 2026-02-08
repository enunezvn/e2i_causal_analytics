"""Type helper utilities for narrowing external library types.

Provides functions to safely narrow broad union types (e.g., Supabase JSON)
into concrete Python types with runtime validation.
"""

from __future__ import annotations

from typing import Any, Mapping, TypeVar

T = TypeVar("T")


def parse_supabase_row(row: Any) -> dict[str, Any]:
    """Narrow a Supabase JSON row to dict[str, Any].

    Supabase postgrest-py types results as list[JSON] where JSON is a broad union.
    This function narrows each row to a usable dict with runtime validation.

    Args:
        row: A single row from a Supabase query result (result.data[i]).

    Returns:
        The row as dict[str, Any].

    Raises:
        TypeError: If the row is not a Mapping (dict-like).
    """
    if not isinstance(row, Mapping):
        raise TypeError(f"Expected dict row from Supabase, got {type(row).__name__}")
    return dict(row)


def parse_supabase_rows(rows: Any) -> list[dict[str, Any]]:
    """Narrow a list of Supabase JSON rows to list[dict[str, Any]].

    Args:
        rows: The data attribute from a Supabase query result (result.data).

    Returns:
        List of rows as dict[str, Any].

    Raises:
        TypeError: If rows is not iterable or any row is not a Mapping.
    """
    if not isinstance(rows, (list, tuple)):
        raise TypeError(f"Expected list of rows from Supabase, got {type(rows).__name__}")
    return [parse_supabase_row(r) for r in rows]


def safe_get(d: dict[str, Any], key: str, expected_type: type[T], default: T) -> T:
    """Type-safe dict.get() with runtime type check.

    Args:
        d: Dictionary to get the value from.
        key: Key to look up.
        expected_type: Expected type of the value.
        default: Default value if key is missing or wrong type.

    Returns:
        The value cast to expected_type, or default.
    """
    val = d.get(key)
    if val is None:
        return default
    if not isinstance(val, expected_type):
        return default
    return val
