"""#2216: planted-prefix predicates must not treat ``_`` as a wildcard.

``_`` matches any single character under SQL ``LIKE``, so ``hl_<rid>_%`` also
matches ``hlX<rid>Y…``. A foreign row that happens to collide is counted as
planted by the census (the permissive direction) and deleted by the prefix-scoped
teardown. The fix is one helper in the guard that escapes ``\\``, ``%`` and ``_``,
plus an explicit ``ESCAPE '\\'`` on every planted ``LIKE`` -- in the specs and in
the consumer files' own statements.

No database: the LIKE semantics are checked against sqlite, which implements the
same ``_`` / ``%`` / ``ESCAPE`` rules as Postgres for these ASCII patterns and
parses the guard's escape clause verbatim (``'\\'`` is one backslash in both, since
``standard_conforming_strings`` has been Postgres's default since 9.1). The
consumer files are checked by source scan, because they only run in CI against the
live database.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pytest

from tests.integration._prod_write_guard import (
    LIKE_ESCAPE,
    WriteWindowSpec,
    adherence_spec,
    like_literal,
    per_hcp_rollup_spec,
    planted_prefix,
    planted_suffix,
    territory_arrival_spec,
    territory_rollup_spec,
)

_REPO = Path(__file__).resolve().parents[3]
_GUARD = _REPO / "tests" / "integration" / "_prod_write_guard.py"
#: How the clause is spelled in Python SOURCE (the backslash doubled), for the scans.
_LIKE_ESCAPE_SRC = LIKE_ESCAPE.replace("\\", "\\\\")

#: Every file that imports the guard. Each plants rows under a per-run prefix and
#: deletes them by that prefix in its own teardown.
_CONSUMERS = tuple(
    _REPO / "tests" / "integration" / name
    for name in (
        "test_per_hcp_rollup_late_arrival.py",
        "test_territory_metrics_etl_integration.py",
        "test_business_metrics_per_hcp_etl_integration.py",
        "test_patient_adherence_etl_integration.py",
    )
)


def _like(value: str, pattern: str) -> bool:
    """``value LIKE pattern`` with the guard's own escape clause, evaluated by sqlite."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.execute("PRAGMA case_sensitive_like = ON")
        (hit,) = conn.execute(f"SELECT ? LIKE ? {LIKE_ESCAPE}", (value, pattern)).fetchone()
        return bool(hit)
    finally:
        conn.close()


# =============================================================================
# The helper: the pattern it builds matches the planted id and only the planted id
# =============================================================================


def test_planted_prefix_matches_the_real_id_and_not_a_wildcard_collision() -> None:
    rid = "3fa9c1"
    pattern = planted_prefix(f"hl_{rid}_")
    assert _like(f"hl_{rid}_a", pattern)
    assert _like(f"hl_{rid}_", pattern), "the bare prefix is itself a match"
    assert not _like(f"hlX{rid}Ya", pattern), "'_' must not act as a single-char wildcard"
    assert not _like(f"hl_{rid}", pattern), "a shorter foreign id must not match"
    assert not _like(f"xhl_{rid}_a", pattern), "prefix must anchor at the start"


def test_planted_suffix_matches_the_real_id_and_not_a_wildcard_collision() -> None:
    rid = "3fa9c1"
    pattern = planted_suffix(f"_{rid}")
    assert _like(f"T_NE_{rid}", pattern)
    assert not _like(f"T_NEX{rid}", pattern), "'_' must not act as a single-char wildcard"
    assert not _like(f"T_NE_{rid}x", pattern), "suffix must anchor at the end"


@pytest.mark.parametrize(
    ("literal", "collision"),
    [
        ("T_LATE_", "TxLATEy"),
        ("100%", "100x"),
        ("a\\b", "axb"),
        ("50%_off_", "50xxoffx"),
    ],
)
def test_like_literal_escapes_every_metacharacter(literal: str, collision: str) -> None:
    pattern = like_literal(literal) + "%"
    assert _like(literal + "tail", pattern)
    assert not _like(collision + "tail", pattern)


def test_the_unescaped_pattern_is_the_bug_this_file_pins() -> None:
    """Negative control on the oracle: without the helper the collision DOES match,
    so a green above is the helper's doing and not sqlite failing to wildcard."""
    assert _like("hlX3fa9c1Ya", "hl_3fa9c1_%")


# =============================================================================
# The specs: every planted LIKE carries the escape clause
# =============================================================================

_SPECS = (
    per_hcp_rollup_spec(
        test_file="f.py",
        start="2024-01-01",
        end="2024-01-31",
        hcp_like=planted_prefix("hcp_abc_"),
        trigger_like=planted_prefix("tr_abc_"),
    ),
    per_hcp_rollup_spec(
        test_file="f2.py",
        start="2024-01-01",
        end="2024-01-31",
        hcp_like=planted_prefix("hcp_abc_"),
        trigger_like=planted_prefix("tr_abc_"),
        window_column="created_at",
    ),
    territory_rollup_spec(
        test_file="g.py",
        start="2024-06-01",
        end="2024-06-04",
        territory_like=planted_suffix("_abc"),
        teardown_deletes_window=True,
    ),
    territory_rollup_spec(
        test_file="g2.py",
        start="2024-06-01",
        end="2024-06-04",
        territory_like=planted_suffix("_abc"),
        teardown_deletes_window=False,
    ),
    adherence_spec(
        test_file="h.py",
        start="2024-01-01",
        end="2024-01-31",
        journey_like=planted_prefix("pj_abc_"),
    ),
    adherence_spec(
        test_file="h2.py",
        start="2024-01-01",
        end="2024-01-31",
        journey_like=planted_prefix("pj_abc_"),
        window_column="created_at",
    ),
    territory_arrival_spec(
        test_file="arr.py",
        start="2019-01-06 21:45:00+00",
        end="2019-01-14 03:45:00+00",
        hcp_like=planted_prefix("hl_abc_"),
        trigger_like=planted_prefix("trlate_abc_"),
        territory_like=planted_prefix("T_LATE_abc"),
    ),
)

# ``LIKE %(name)s`` optionally preceded by NOT, then the rest of that line.
_BOUND_LIKE = re.compile(r"\b(?:NOT\s+)?LIKE\s+%\((\w+)\)s([^\n]*)")


@pytest.mark.parametrize("spec", _SPECS, ids=lambda s: s.test_file)
def test_every_bound_like_in_every_spec_carries_the_escape_clause(spec: WriteWindowSpec) -> None:
    """The helper escapes with a backslash; the predicate must say so. Postgres's
    default escape IS the backslash, but an implicit default is not something a
    reviewer can see, and it is not what sqlite or another engine would assume."""
    seen = 0
    for q in spec.queries:
        for name, tail in _BOUND_LIKE.findall(q.sql):
            seen += 1
            assert tail.strip().startswith(LIKE_ESCAPE), (spec.test_file, q.leg, name, q.sql)
    assert seen >= 1, f"{spec.test_file}: no planted predicate at all -- census cannot own rows"


def test_no_spec_predicate_carries_a_like_without_a_bound_param() -> None:
    """A ``LIKE '...'`` literal in a spec would be the one place the helper cannot
    reach. There is none today; keep it that way."""
    for spec in _SPECS:
        for q in spec.queries:
            for m in re.finditer(r"\bLIKE\b", q.sql):
                assert re.match(r"\s*%\(\w+\)s", q.sql[m.end() :]), (spec.test_file, q.leg)


# =============================================================================
# The consumer files: their own statements go through the same helper
# =============================================================================

_POSITIONAL_LIKE = re.compile(r"\bLIKE\s+%s(?P<tail>[^\n]*)")
#: An f-string literal that carries both an interpolation and a ``%``: a wildcard
#: pattern built by hand around a planted id, e.g. ``f"hl_{rid}_%"`` or
#: ``f"%_{test_run_id}"``. (No consumer formats a percentage in an f-string.)
_HAND_BUILT_PATTERN = re.compile(
    r"""f"(?=[^"\n]*\{)[^"\n]*%[^"\n]*"|f'(?=[^'\n]*\{)[^'\n]*%[^'\n]*'"""
)


@pytest.mark.parametrize("path", _CONSUMERS, ids=lambda p: p.name)
def test_consumer_statements_escape_every_positional_like(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = list(_POSITIONAL_LIKE.finditer(source))
    assert hits, f"{path.name}: expected at least one planted LIKE (teardown deletes by prefix)"
    for m in hits:
        line = source.count("\n", 0, m.start()) + 1
        assert m.group("tail").strip().startswith(_LIKE_ESCAPE_SRC), (
            f"{path.name}:{line}: 'LIKE %s' without {LIKE_ESCAPE} -- '_' is a wildcard here"
        )


@pytest.mark.parametrize("path", _CONSUMERS, ids=lambda p: p.name)
def test_consumer_patterns_come_from_the_helper_not_from_an_f_string(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    for m in _HAND_BUILT_PATTERN.finditer(source):
        line = source.count("\n", 0, m.start()) + 1
        pytest.fail(
            f"{path.name}:{line}: hand-built LIKE pattern {m.group(0)!r}; use planted_prefix/planted_suffix"
        )
    assert re.search(r"\bplanted_(?:prefix|suffix)\b", source), f"{path.name}: helper not used"


def test_the_escape_clause_is_the_one_both_engines_parse() -> None:
    """Pins the literal so a 'tidy-up' to ``E'\\\\'`` (Postgres-only) or to a
    different escape character cannot land without meeting this file."""
    assert LIKE_ESCAPE == "ESCAPE '\\'"
    guard_src = _GUARD.read_text(encoding="utf-8")
    assert guard_src.count(f'LIKE_ESCAPE: str = "{_LIKE_ESCAPE_SRC}"') == 1
    assert guard_src.count("{LIKE_ESCAPE}") >= 11, "every spec predicate interpolates the constant"
