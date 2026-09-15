"""``src.repositories.query_utils`` — the shared PostgREST filter helpers.

``escape_like_pattern`` is the #2114 addition: one definition of LIKE/ILIKE
metacharacter escaping, so a repository filtering on a caller-supplied string
reaches for it instead of writing its own (or forgetting to).
"""

from __future__ import annotations

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
        assert escape_like_pattern("\\_%") == "\\\\\\_\\%"
