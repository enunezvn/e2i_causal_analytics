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


#: The characters a PostgREST ``like``/``ilike`` pattern treats as wildcards.
#:
#: ENUMERATED against the live server, not reasoned about (#2114 r6). PostgREST
#: is Haskell and is not in our venv, so its translation layer cannot be read
#: the way ``postgrest-py`` can -- ``base_request_builder.py:411`` forwards the
#: pattern verbatim, which proves only that the client does nothing. The probe
#: covered a KNOWN-COMPLETE set -- every printable ASCII punctuation mark plus
#: space, i.e. ``string.punctuation`` (32) + ``" "`` = **33 characters** -- each
#: tested for BOTH wildcard roles against the real server, with a positive
#: control (a row whose value contains a literal ``_``) and a negative control.
#: The accounting closes exactly:
#:
#:     3 wildcards  +  1 special-not-wildcard (``\``)  +  29 literal  =  33
#:
#: and ``test_the_wildcard_set_is_the_ENUMERATED_one`` ASSERTS that partition
#: against ``string.punctuation`` rather than restating the count, because a
#: count in a comment is a claim and a set-difference is a check. Exactly these
#: three widen:
#:
#:     ``*``  multi-char -- POSTGREST's own wildcard, translated to ``%``
#:     ``%``  multi-char -- SQL's
#:     ``_``  single-char -- SQL's
#:
#: These 29 are LITERAL and must NOT be escaped, or an honest value containing
#: one would stop matching:
#:     , . : ( ) " ' ? # & = + space | [ ] { } ^ $ ! ~ / @ < > ; - `
#:
#: `-` and `` ` `` were probed LAST: a set-difference of the first pass's own
#: list against printable ASCII punctuation found them never probed, so the
#: closed-class claim had been wider than the evidence under it. `-` is the one
#: that could have mattered -- plausible inside a real hyphenated brand or node
#: value, so had it widened the victim would have been an honest value, not an
#: attacker. Both measured literal. (The totals were then wrong twice more, 30
#: and 31, before being closed against the complete set above; hence asserting
#: the partition instead of counting it.)
#:
#: The r5 version of this helper escaped only the two SQL wildcards, because it
#: reasoned from SQL rather than from the stack. That closed a SHAPE and left the
#: class open for one round -- hence the enumeration.
LIKE_WILDCARDS = ("*", "%", "_")


def escape_like_pattern(value: str) -> str:
    """Escape PostgREST/SQL ``LIKE``/``ILIKE`` metacharacters in ``value``.

    ``.ilike`` interprets its argument as a PATTERN, not a literal, so a
    caller-supplied value containing any of :data:`LIKE_WILDCARDS` broadens the
    match. Escaping them with the default ``\\`` escape character makes the
    pattern a literal, whole-string, case-insensitive match -- the
    case-insensitivity is what the caller wanted from ILIKE and is preserved;
    the wildcarding is not.

    Backslash is escaped FIRST so the escapes added after it are not themselves
    escaped: doing ``%`` first would turn ``"\\%"`` into ``"\\\\%"``, an escaped
    backslash followed by a LIVE ``%``, which is the broadening this prevents.
    Backslash is not a wildcard, but it is not merely cosmetic either -- a
    DANGLING one is an API ERROR rather than a miss (measured: raw ``"\\"``
    raises ``APIError``; escaped, it returns 0 rows), so this function also
    keeps a caller's stray backslash from failing the request outright.

    MEASURED read-only against live ``causal_paths`` (109 rows, three brands):
    unescaped ``"%"`` and ``"*"`` EACH returned all 109 across all three brands,
    and ``"Kis%"``/``"Kis*"``/``"_isqali"`` each returned Kisqali's 37 for an ask
    that named neither. Escaped, every one returns 0, while ``"Kisqali"`` and
    ``"kisqali"`` still return their own 37 -- case-insensitivity is load-bearing
    for callers that pass free-form request-body and LLM-supplied brands, which
    is why this escapes rather than switching the predicate to ``.eq``.

    Verified POSITIVELY, not merely by absence -- zero rows cannot distinguish
    "escaping works" from "pattern broken". On a real value containing a literal
    underscore, ``"acceptance\\_status"`` still matches it (3 rows) while the
    unescaped one-char wildcard ``"acceptanc_status"`` correctly matches nothing.

    Lives here, beside :func:`match_nullable_column`, for the same #1991-debt-3
    reason that one does: a repository filtering a caller-supplied string
    reaches for the one definition instead of writing its own -- or, as
    ``causal_path`` did in both of its twins, omitting it. Home chosen so no
    layer inverts: ``src/api/repositories/`` may import ``src/repositories/``,
    never the reverse.
    """
    out = value.replace("\\", "\\\\")
    for wildcard in LIKE_WILDCARDS:
        out = out.replace(wildcard, "\\" + wildcard)
    return out
