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


def escape_like_pattern(value: str) -> str:
    """Escape PostgREST/SQL ``LIKE``/``ILIKE`` metacharacters in ``value``.

    ``.ilike`` interprets its argument as a PATTERN, not a literal, so a
    caller-supplied value containing ``%`` or ``_`` broadens the match. Escaping
    them with the default ``\\`` escape character makes the pattern a literal,
    whole-string, case-insensitive match -- the case-insensitivity is what the
    caller wanted from ILIKE and is preserved; the wildcarding is not.

    Backslash is escaped FIRST so the escapes added after it are not themselves
    escaped: doing ``%`` first would turn ``"\\%"`` into ``"\\\\%"``, an escaped
    backslash followed by a LIVE ``%``, which is the broadening this prevents.

    MEASURED read-only against live ``causal_paths`` (109 rows, three brands)
    before this moved here: unescaped ``.ilike("brand","%")`` returned all 109
    across all three brands, and ``"Kis%"``/``"_isqali"`` each returned
    Kisqali's 37 for an ask that named neither. Escaped, all three return 0,
    while ``"Kisqali"`` and ``"kisqali"`` still return their own 37.

    Lives here, beside :func:`match_nullable_column`, for the same #1991-debt-3
    reason that one does: a repository filtering a caller-supplied string
    reaches for the one definition instead of writing its own -- or, as
    ``causal_path`` did in both of its twins, omitting it. Home chosen so no
    layer inverts: ``src/api/repositories/`` may import ``src/repositories/``,
    never the reverse.
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
