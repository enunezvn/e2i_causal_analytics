"""Lint-style regression: no migration file may contain a script-level
``BEGIN;`` or ``COMMIT;`` line (issue #186).

``scripts/run_migrations.sh:100`` invokes psql with
``--single-transaction``, which owns the outer transaction (the ``\\i``
of the migration file plus the ``INSERT INTO schema_migrations``
bookkeeping row that follows). An inner ``BEGIN;`` ... ``COMMIT;`` here
would prematurely commit before the bookkeeping insert, leaving the
migration applied but unrecorded if the bookkeeping insert fails —
silent ledger drift on fresh-DB replay or re-application.

This was the codex pass-1 MEDIUM-1 finding on PR #185 (migration 039).
Issue #186 mirrors the fix back onto the two predecessors that shipped
the same anti-pattern:

* ``database/migrations/036_add_payer_category.sql`` (merged via PR #167)
* ``database/migrations/038_drop_brand_from_feedback_loop.sql`` (merged
  via PR #180)

PL/pgSQL function-body ``BEGIN ... END`` blocks are NOT the bug — those
sit inside ``$$ ... $$`` dollar-quoted blocks and do not start at
column 0 with a trailing ``;``. Only the script-level
``^BEGIN;$`` / ``^COMMIT;$`` shape (after stripping leading whitespace
and trailing inline ``--`` comments) is flagged.

This test is filesystem-only — no DB required — so it runs in every CI
lane including the unit-test lane.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "database" / "migrations"

# ``^BEGIN;$`` and ``^COMMIT;$`` after stripping leading whitespace and
# trailing inline ``--`` comments. We deliberately DO NOT match PL/pgSQL
# ``BEGIN`` (no trailing ``;`` — it opens a function body), nor
# ``BEGIN ATOMIC`` / ``BEGIN ISOLATION LEVEL`` (legitimate variants that
# specify isolation or atomicity and which the runner contract doesn't
# care about). The bug is specifically the bare ``BEGIN;`` / ``COMMIT;``
# that psql ``--single-transaction`` interprets as nested-txn boundaries.
_BARE_BEGIN_RE = re.compile(r"^BEGIN\s*;\s*$", re.IGNORECASE)
_BARE_COMMIT_RE = re.compile(r"^COMMIT\s*;\s*$", re.IGNORECASE)


def _strip_inline_comment(line: str) -> str:
    """Strip a trailing ``-- comment`` from a SQL line, preserving the
    SQL text before it. Naive: does not handle ``--`` inside string
    literals, but migration scripts in this project don't use ``--``
    inside string literals at the script level (PL/pgSQL body comments
    live inside ``$$ ... $$`` dollar-quoted blocks, which never start
    at column 0).
    """
    idx = line.find("--")
    if idx >= 0:
        return line[:idx]
    return line


def _scan_for_bare_txn(sql_path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(line_number, line_text)`` for every
    script-level ``BEGIN;`` or ``COMMIT;`` in ``sql_path``.

    "Script-level" means after stripping leading whitespace, the line
    matches ``^BEGIN;$`` or ``^COMMIT;$``. PL/pgSQL function-body
    ``BEGIN ... END`` blocks lack the trailing ``;`` on ``BEGIN`` and
    are therefore excluded.
    """
    findings: list[tuple[int, str]] = []
    text = sql_path.read_text(encoding="utf-8")
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        stripped = _strip_inline_comment(raw_line).strip()
        if not stripped:
            continue
        if _BARE_BEGIN_RE.match(stripped) or _BARE_COMMIT_RE.match(stripped):
            findings.append((lineno, raw_line.rstrip()))
    return findings


def _collect_migration_files() -> list[Path]:
    """Return all ``database/migrations/*.sql`` files sorted by name."""
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


@pytest.mark.parametrize(
    "sql_path",
    _collect_migration_files(),
    ids=lambda p: p.name,
)
def test_migration_has_no_script_level_begin_or_commit(sql_path: Path) -> None:
    """Every ``database/migrations/*.sql`` file must be bare of
    script-level ``BEGIN;`` / ``COMMIT;``.

    The runner (``scripts/run_migrations.sh``) wraps each migration
    invocation with ``psql --single-transaction``; inner txn-control
    statements would prematurely close the runner's transaction before
    the ``INSERT INTO schema_migrations`` bookkeeping row, risking
    silent ledger drift if the bookkeeping insert ever fails.
    """
    findings = _scan_for_bare_txn(sql_path)
    assert not findings, (
        f"{sql_path.name} contains script-level transaction-control "
        f"statements that conflict with psql --single-transaction at "
        f"scripts/run_migrations.sh:100:\n"
        + "\n".join(f"  line {ln}: {tx!r}" for ln, tx in findings)
        + "\nRemove the bare BEGIN; and COMMIT; — the runner owns the "
        "outer txn. See database/migrations/039_drop_triggers_join_from_"
        "feedback_loop.sql for the canonical fixed shape."
    )


def test_migrations_directory_is_non_empty() -> None:
    """Sanity guard: if ``database/migrations/`` ever ends up empty
    (e.g. parametrize matrix collapses to zero files), the lint above
    would pass vacuously. Pin the matrix size explicitly.

    Feedback pattern: empty parametrize matrices pass silently in
    pytest. See ``feedback_pr_merge_workflow.md`` §7.
    """
    files = _collect_migration_files()
    assert files, (
        f"no migration files found under {MIGRATIONS_DIR} — the per-file lint would pass vacuously."
    )


def test_migration_038_specifically_clean() -> None:
    """Pin migration 038 specifically (the file named in issue #186).

    The per-file parametrize above already covers this, but an explicit
    test ensures a future refactor of ``_collect_migration_files`` (e.g.
    accidentally restricting the glob to a subset) cannot silently drop
    coverage of the file the issue actually filed against.
    """
    target = MIGRATIONS_DIR / "038_drop_brand_from_feedback_loop.sql"
    assert target.exists(), f"missing fixture: {target}"
    findings = _scan_for_bare_txn(target)
    assert not findings, "038 still has script-level txn-control: " + "\n".join(
        f"  line {ln}: {tx!r}" for ln, tx in findings
    )


def test_migration_036_specifically_clean() -> None:
    """Pin migration 036 (sibling of 038 — same anti-pattern found
    while fixing issue #186).
    """
    target = MIGRATIONS_DIR / "036_add_payer_category.sql"
    assert target.exists(), f"missing fixture: {target}"
    findings = _scan_for_bare_txn(target)
    assert not findings, "036 still has script-level txn-control: " + "\n".join(
        f"  line {ln}: {tx!r}" for ln, tx in findings
    )


def test_migration_039_canonical_shape_is_already_clean() -> None:
    """Migration 039 was the trigger for the codex pass-1 MEDIUM-1
    finding that produced the new convention. Pin it to make sure the
    canonical-shape file stays canonical.
    """
    target = MIGRATIONS_DIR / "039_drop_triggers_join_from_feedback_loop.sql"
    if not target.exists():
        pytest.skip("039 not present in this checkout")
    findings = _scan_for_bare_txn(target)
    assert not findings, "039 has reintroduced script-level txn-control: " + "\n".join(
        f"  line {ln}: {tx!r}" for ln, tx in findings
    )


def test_scanner_flags_synthetic_bad_input(tmp_path: Path) -> None:
    """Self-test the scanner so a future regression in the scanner
    itself (e.g. someone widens the regex into a no-op) is caught.
    """
    bad_sql = tmp_path / "bad_migration.sql"
    bad_sql.write_text(
        "-- bad migration: inner txn under --single-transaction\n"
        "BEGIN;\n"
        "ALTER TABLE foo ADD COLUMN bar INTEGER;\n"
        "COMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(bad_sql)
    assert len(findings) == 2, f"scanner missed bare txn statements: {findings}"
    assert findings[0][0] == 2  # BEGIN; on line 2
    assert findings[1][0] == 4  # COMMIT; on line 4


def test_scanner_allows_plpgsql_begin_end_blocks(tmp_path: Path) -> None:
    """The scanner must NOT flag PL/pgSQL ``BEGIN`` (no trailing ``;``)
    inside function bodies, nor indented BEGIN/COMMIT inside dollar
    blocks, nor ``BEGIN ATOMIC`` variants. Synthesise the patterns
    inline so a future tightening of the regex (or an over-broad
    rewrite) is caught immediately.
    """
    good_sql = tmp_path / "good_migration.sql"
    good_sql.write_text(
        "CREATE OR REPLACE FUNCTION example() RETURNS void AS $$\n"
        "BEGIN\n"  # PL/pgSQL function body open — no trailing semicolon
        "    RAISE NOTICE 'hello';\n"
        "END;\n"  # PL/pgSQL end of function body — column 0 but `END;` not `COMMIT;`
        "$$ LANGUAGE plpgsql;\n"
        "\n"
        "CREATE OR REPLACE FUNCTION example2() RETURNS void AS $func$\n"
        "    BEGIN\n"  # indented — definitely not script-level
        "        SELECT 1;\n"
        "    END;\n"
        "$func$ LANGUAGE plpgsql;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(good_sql)
    assert findings == [], f"scanner false-positive on PL/pgSQL bodies: {findings}"


def test_strip_inline_comment_preserves_sql() -> None:
    """The comment-stripper must not consume real SQL text before the
    ``--`` marker.
    """
    assert _strip_inline_comment("SELECT 1; -- trailing comment") == "SELECT 1; "
    assert _strip_inline_comment("-- whole line comment") == ""
    assert _strip_inline_comment("BEGIN; -- spurious") == "BEGIN; "
    assert _strip_inline_comment("BEGIN;") == "BEGIN;"
    # Verify the post-strip text still matches the bare regex on the BEGIN; case.
    stripped = _strip_inline_comment("BEGIN; -- spurious").strip()
    assert _BARE_BEGIN_RE.match(stripped) is not None
