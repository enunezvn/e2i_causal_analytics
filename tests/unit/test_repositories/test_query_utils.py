"""``src.repositories.query_utils`` — the shared PostgREST filter helpers.

``escape_like_pattern`` is the #2114 addition: one definition of LIKE/ILIKE
metacharacter escaping, so a repository filtering on a caller-supplied string
reaches for it instead of writing its own (or forgetting to).
"""

from __future__ import annotations

import string

import pytest

from src.repositories.query_utils import escape_like_pattern


class TestEscapeLikePattern:
    """Measured against live ``causal_paths`` (109 rows, 3 brands) before this
    test was written: unescaped ``.ilike("brand","%")`` returned all 109 rows
    across all three brands, ``"Kis%"`` and ``"_isqali"`` each returned
    Kisqali's 37, and every escaped form returned 0 while the honest
    ``"Kisqali"``/``"kisqali"`` still returned its own 37."""

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Kisqali", "Kisqali"),  # an honest value is untouched
            ("kisqali", "kisqali"),  # casing is ILIKE's business, not ours
            ("%", "\\%"),
            ("_", "\\_"),
            ("Kis%", "Kis\\%"),
            ("_isqali", "\\_isqali"),
            ("%%", "\\%\\%"),
            ("a_b%c", "a\\_b\\%c"),
            # `*` is POSTGREST's wildcard, translated to `%` server-side before
            # SQL is built (#2114 r6). The r5 helper escaped the two SQL ones and
            # left this live, so `.ilike("brand","*")` still matched every brand.
            ("*", "\\*"),
            ("Kis*", "Kis\\*"),
            ("K*sqali", "K\\*sqali"),
            ("**", "\\*\\*"),
            ("*%_", "\\*\\%\\_"),
            ("", ""),
        ],
    )
    def test_metacharacters_become_literals(self, raw: str, expected: str) -> None:
        assert escape_like_pattern(raw) == expected

    def test_backslash_is_escaped_FIRST_so_escapes_are_not_doubled(self) -> None:
        """A literal backslash must not turn the escape we add into a literal.

        Escaping ``%`` before ``\\`` would map ``"\\%"`` to ``"\\\\%"`` -- an
        escaped backslash followed by a LIVE ``%`` wildcard, i.e. the very
        broadening this function exists to stop.
        """
        assert escape_like_pattern("\\") == "\\\\"
        assert escape_like_pattern("\\%") == "\\\\\\%"
        assert escape_like_pattern("50\\%") == "50\\\\\\%"

    def test_every_metacharacter_in_one_value(self) -> None:
        assert escape_like_pattern("\\_%*") == "\\\\\\_\\%\\*"

    def test_the_wildcard_set_is_the_ENUMERATED_one(self) -> None:
        """The three wildcards were enumerated against the LIVE server, not
        reasoned about, and the class is closed against a KNOWN-COMPLETE set:
        every printable ASCII punctuation mark plus space, 33 characters.

        3 wildcards (`*` `%` `_`) + 1 special-not-wildcard (`\\`) + 29 literal
        = 33. The accounting is ASSERTED below rather than stated, because
        stating it is exactly what went wrong: an earlier revision claimed 30,
        then 31, and its prose list disagreed with the loop beneath it. A count
        in a docstring is a claim; a set-difference against ``string.punctuation``
        is a check.

        The literal characters MUST NOT be escaped -- an honest brand or node
        value containing one would stop matching.
        """
        wildcards = set("*%_")
        special = set("\\")
        literal = set(",.:()\"'?#&=+ |[]{}^$!~/@<>;-`")
        complete = set(string.punctuation) | {" "}

        # The class is closed: no candidate unaccounted for, none double-counted.
        assert wildcards | special | literal == complete
        assert not (wildcards & literal) and not (special & literal)
        assert len(complete) == 33 and len(literal) == 29

        for wild in sorted(wildcards):
            assert escape_like_pattern(wild) == "\\" + wild, wild
        for lit in sorted(literal):
            assert escape_like_pattern(lit) == lit, lit
