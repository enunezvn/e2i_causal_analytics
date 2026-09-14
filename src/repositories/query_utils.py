"""Shared PostgREST query filters for repository readers and writers.

Moved out of :mod:`src.repositories.expert_review_versions` (#1991 debt 3) so
any repository module that needs to filter on a nullable identity column
reaches for the one definition instead of writing its own ``eq``/``is_``
branch.
"""

from __future__ import annotations

from typing import Any, Optional


def match_nullable_column(query: Any, column: str, value: Optional[str]) -> Any:
    """Filter ``query`` on a NULLABLE identity column, matching NULL EXPLICITLY.

    "Not recorded" (``None``) is a precondition that is CHECKED, never skipped:
    the caller states what it believes the column currently holds -- a known
    value or unknown (``None``) -- and this filter matches exactly that belief,
    row by row, rather than assuming a known value or ignoring the column.

    A known value is an ``eq``. An UNKNOWN one (None) is ``is_``, which postgrest
    2.27 renders as ``<column>=is.null`` (verified: its ``is_`` maps a Python
    None to the string ``"null"`` before building the filter). ``eq`` can NOT
    express this -- PostgREST would compare against the literal text ``"None"``
    and match nothing -- and OMITTING the filter is worse: it matches every row,
    which is the defect being closed.
    """
    if value is None:
        return query.is_(column, "null")
    return query.eq(column, value)
