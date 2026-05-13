"""Lint-style regression: no migration file may contain a script-level
transaction-control statement (issue #186).

``scripts/run_migrations.sh:100`` invokes psql with
``--single-transaction``, which owns the outer transaction (the ``\\i``
of the migration file plus the ``INSERT INTO schema_migrations``
bookkeeping row that follows). An inner ``BEGIN;`` ... ``COMMIT;`` here
would prematurely commit before the bookkeeping insert, leaving the
migration applied but unrecorded if the bookkeeping insert fails —
silent ledger drift on fresh-DB replay or re-application. An inner
``ROLLBACK;`` is the inverse hazard: it can roll back the migration
DDL while the wrapper still appends the schema_migrations row, creating
ledger drift in the opposite direction.

This was the codex pass-1 MEDIUM-1 finding on PR #185 (migration 039).
Issue #186 mirrors the fix back onto the two predecessors that shipped
the same anti-pattern:

* ``database/migrations/036_add_payer_category.sql`` (merged via PR #167)
* ``database/migrations/038_drop_brand_from_feedback_loop.sql`` (merged
  via PR #180)

The lint flags ANY script-level transaction-control statement:

  - ``BEGIN;`` / ``BEGIN TRANSACTION;`` / ``BEGIN WORK;`` /
    ``START TRANSACTION;``                                  -> outer-txn open
  - ``COMMIT;`` / ``COMMIT TRANSACTION;`` / ``COMMIT WORK;`` -> premature commit
  - ``ROLLBACK;`` / ``ROLLBACK TRANSACTION;`` /
    ``ROLLBACK WORK;`` / ``ABORT;`` / ``ABORT TRANSACTION;`` /
    ``ABORT WORK;``                                         -> premature rollback
  - ``END;`` / ``END TRANSACTION;`` / ``END WORK;`` at the script level
    (PostgreSQL treats bare ``END;`` outside a PL/pgSQL body as a
    synonym for ``COMMIT;``)

PL/pgSQL function-body ``BEGIN ... END`` blocks are NOT flagged: the
scanner tracks ``$tag$ ... $tag$`` dollar-quoted block boundaries and
skips anything inside them. ``BEGIN ATOMIC`` / ``BEGIN ISOLATION LEVEL``
variants (with extra clauses before the semicolon) are also legitimate
and excluded — the scanner only flags the bare statement-head followed
by an optional clause word and the terminating ``;``.

This test is filesystem-only (no DB required) so it runs in the CI
integration lane (``tests/integration``). See ``backend-tests.yml``
``Integration Tests`` job for invocation. Per the
``feedback_pr_merge_workflow.md`` empty-parametrize lesson, the
``test_migrations_directory_is_non_empty`` sanity guard prevents the
parametrize matrix from passing vacuously.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIGRATIONS_DIR = REPO_ROOT / "database" / "migrations"

# Match any script-level transaction-control statement head after
# stripping leading whitespace and trailing inline ``--`` comment.
# Optional clause keyword (``TRANSACTION`` / ``WORK``) is permitted —
# Postgres accepts ``BEGIN TRANSACTION;`` etc. as a synonym for the
# bare form. ``START TRANSACTION;`` is explicitly listed (no bare
# form).
#
# Indentation is NOT the discriminator — the discriminator is whether
# the statement sits inside a ``$tag$ ... $tag$`` dollar-quoted block
# (PL/pgSQL function body), which the scanner tracks separately.
_TXN_STATEMENT_RE = re.compile(
    r"""
    ^                                  # start of stripped line
    (?:
        BEGIN (?:\s+(?:TRANSACTION|WORK))?
        | START\s+TRANSACTION
        | COMMIT (?:\s+(?:TRANSACTION|WORK))?
        | END (?:\s+(?:TRANSACTION|WORK))?
        | ROLLBACK (?:\s+(?:TRANSACTION|WORK))?
        | ABORT (?:\s+(?:TRANSACTION|WORK))?
    )
    \s*;\s*$                           # terminating semicolon (no trailing clauses)
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Dollar-quote delimiter. Tag is optional and may be any
# identifier-like token (Postgres docs: alphanumeric or empty).
# Examples: ``$$``, ``$func$``, ``$BODY$``. We capture the tag so the
# closing delimiter must match the opener — nested dollar blocks with
# different tags are technically legal (though unused in this repo).
_DOLLAR_QUOTE_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")


def _strip_inline_comment(line: str) -> str:
    """Strip a trailing ``-- comment`` from a SQL line, preserving the
    SQL text before it. Naive: does not handle ``--`` inside string
    literals, but migration scripts in this project don't use ``--``
    inside string literals at the script level (PL/pgSQL body comments
    live inside ``$$ ... $$`` dollar-quoted blocks, which the scanner
    tracks separately and never invokes this helper for).
    """
    idx = line.find("--")
    if idx >= 0:
        return line[:idx]
    return line


def _split_line_at_dollar_boundaries(
    raw_line: str,
    in_dollar_block: bool,
    open_tag: str | None,
) -> tuple[list[str], bool, str | None]:
    """Split ``raw_line`` into segments of consecutive script-level
    text (i.e. text NOT inside a dollar-quoted block), returning
    ``(script_segments, new_in_dollar_block, new_open_tag)``.

    The script-level segments are the portions of the line that the
    lint should inspect for transaction-control statements. Text
    inside a dollar-quoted block is omitted from the returned list.
    This correctly handles same-line script SQL after a dollar-block
    close, e.g. ``$$; COMMIT;`` returns ``["; COMMIT;"]`` (after the
    closer) as a script-level segment. Codex pass-2 LOW-2.

    Examples (compact):

      in=False, "$$ BEGIN" -> segments=["$$ BEGIN"]?
      Actually the opener itself counts as outside-then-transition:
      the ``$$`` token is treated as the boundary, so the segments
      collected are the text BEFORE the ``$$`` (outside, returned)
      and the text AFTER is now inside (omitted until close).
    """
    segments: list[str] = []
    cursor = 0
    for match in _DOLLAR_QUOTE_RE.finditer(raw_line):
        boundary_start, boundary_end = match.span()
        tag = match.group(1) or ""
        # Text up to (but not including) the boundary marker:
        prefix = raw_line[cursor:boundary_start]
        if not in_dollar_block:
            # Prefix was script-level; capture it.
            segments.append(prefix)
        # The boundary marker (``$tag$``) itself is part of the
        # dollar-quote machinery — it's neither pure script nor pure
        # body content. We do NOT add it to ``segments``; it carries
        # no transaction-control semantics.
        if in_dollar_block:
            if tag == (open_tag or ""):
                in_dollar_block = False
                open_tag = None
        else:
            in_dollar_block = True
            open_tag = tag
        cursor = boundary_end

    # Tail after the last boundary (or the whole line if no
    # boundaries): script-level iff we ended OUTSIDE a dollar block.
    if not in_dollar_block:
        segments.append(raw_line[cursor:])

    return segments, in_dollar_block, open_tag


def _scan_for_bare_txn(sql_path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(line_number, line_text)`` for every
    script-level transaction-control statement in ``sql_path``.

    Statements inside dollar-quoted PL/pgSQL function bodies are
    skipped. The scanner uses ``_split_line_at_dollar_boundaries`` to
    extract per-line script-level segments, then splits each segment
    on ``;`` to recover individual statements. Each candidate
    statement-head is then matched against the txn regex.

    This handles:

    * The historical bug shape: top-of-line ``BEGIN;`` / ``COMMIT;``.
    * Whole-line-inside-dollar-block: zero script segments → skipped.
    * Mid-line dollar close followed by script SQL on the same line:
      the post-close tail is captured as a script segment, then
      semicolon-split into statements (codex pass-3 LOW-1).
    * Multi-statement-per-line script SQL like
      ``ALTER TABLE foo ADD COLUMN bar INTEGER; COMMIT;`` —
      semicolon-split surfaces the ``COMMIT`` as its own statement
      and flags it.
    """
    findings: list[tuple[int, str]] = []
    text = sql_path.read_text(encoding="utf-8")
    in_dollar_block = False
    open_tag: str | None = None

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        segments, in_dollar_block, open_tag = _split_line_at_dollar_boundaries(
            raw_line,
            in_dollar_block,
            open_tag,
        )
        if not segments:
            continue
        # Strip inline ``--`` comments from each segment BEFORE
        # joining. (After joining, a ``--`` in one segment would
        # consume the next segment's text too, missing real
        # transaction-control statements.)
        cleaned_segments = [_strip_inline_comment(seg) for seg in segments]
        joined = " ".join(cleaned_segments)
        # Split on ``;`` to recover individual statements; each
        # statement is then stripped and matched against the regex
        # WITHOUT requiring a trailing ``;`` (we already split on
        # it). The regex was authored to match the full statement
        # including ``;``; we strip the ``;`` from the regex by
        # using a "head-only" variant. Equivalently: re-append a
        # ``;`` to each non-empty stripped segment before matching.
        for raw_statement in joined.split(";"):
            statement = raw_statement.strip()
            if not statement:
                continue
            # Re-append the ``;`` we split on so the existing regex
            # (which anchors on ``;\s*$``) still matches.
            candidate = statement + ";"
            if _TXN_STATEMENT_RE.match(candidate):
                findings.append((lineno, raw_line.rstrip()))
                # Only one finding per line is useful — additional
                # txn statements on the same line will point at the
                # same line and aren't actionable separately.
                break
    return findings


def _collect_migration_files() -> list[Path]:
    """Return all ``database/migrations/*.sql`` files sorted by name."""
    return sorted(MIGRATIONS_DIR.glob("*.sql"))


@pytest.mark.parametrize(
    "sql_path",
    _collect_migration_files(),
    ids=lambda p: p.name,
)
def test_migration_has_no_script_level_txn_control(sql_path: Path) -> None:
    """Every ``database/migrations/*.sql`` file must be bare of
    script-level transaction-control statements.

    The runner (``scripts/run_migrations.sh``) wraps each migration
    invocation with ``psql --single-transaction``; inner txn-control
    statements would prematurely close (or roll back) the runner's
    transaction before the ``INSERT INTO schema_migrations``
    bookkeeping row, risking silent ledger drift in either direction.
    """
    findings = _scan_for_bare_txn(sql_path)
    assert not findings, (
        f"{sql_path.name} contains script-level transaction-control "
        f"statements that conflict with psql --single-transaction at "
        f"scripts/run_migrations.sh:100:\n"
        + "\n".join(f"  line {ln}: {tx!r}" for ln, tx in findings)
        + "\nRemove the bare BEGIN/COMMIT/ROLLBACK/END/ABORT/START TRANSACTION — "
        "the runner owns the outer txn. See "
        "database/migrations/039_drop_triggers_join_from_feedback_loop.sql "
        "for the canonical fixed shape."
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


@pytest.mark.parametrize(
    "sql_body",
    [
        # Bare two-statement form (the historical bug).
        "BEGIN;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nCOMMIT;\n",
        # ``TRANSACTION`` / ``WORK`` clause variants (Postgres accepts).
        "BEGIN TRANSACTION;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nCOMMIT TRANSACTION;\n",
        "BEGIN WORK;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nCOMMIT WORK;\n",
        # ``START TRANSACTION;`` plus ``END;`` commit alias.
        "START TRANSACTION;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nEND;\n",
        # ``ROLLBACK;`` premature-abort hazard.
        "BEGIN;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nROLLBACK;\n",
        "BEGIN;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nABORT;\n",
        # Mixed-case (Postgres is case-insensitive on keywords).
        "begin;\nALTER TABLE foo ADD COLUMN bar INTEGER;\ncommit;\n",
        # Indented script-level (still a runner-contract violation —
        # the discriminator is dollar-quoting, not indentation).
        "    BEGIN;\nALTER TABLE foo ADD COLUMN bar INTEGER;\n    COMMIT;\n",
    ],
    ids=[
        "bare-begin-commit",
        "begin-transaction-commit-transaction",
        "begin-work-commit-work",
        "start-transaction-end",
        "begin-rollback",
        "begin-abort",
        "lowercase-begin-commit",
        "indented-begin-commit",
    ],
)
def test_scanner_flags_synthetic_bad_input(tmp_path: Path, sql_body: str) -> None:
    """Self-test the scanner across the txn-control variants Postgres
    accepts.

    A regression in the scanner itself (e.g. someone widens the regex
    into a no-op, or narrows it back to just ``^BEGIN;$`` / ``^COMMIT;$``)
    must be caught by these self-tests.
    """
    bad_sql = tmp_path / "bad_migration.sql"
    bad_sql.write_text(sql_body, encoding="utf-8")
    findings = _scan_for_bare_txn(bad_sql)
    # Expect at least 2 findings (the BEGIN-equivalent open + the
    # COMMIT/ROLLBACK/END/ABORT-equivalent close).
    assert len(findings) >= 2, (
        f"scanner missed bare txn statements for body:\n{sql_body}\nfindings: {findings}"
    )


def test_scanner_allows_plpgsql_begin_end_blocks(tmp_path: Path) -> None:
    """The scanner must NOT flag PL/pgSQL ``BEGIN`` (no trailing ``;``)
    nor ``END;`` inside ``$$...$$`` dollar-quoted bodies, nor
    ``BEGIN ATOMIC`` variants. Synthesise the patterns inline so a
    future tightening of the regex (or an over-broad rewrite) is
    caught immediately.
    """
    good_sql = tmp_path / "good_migration.sql"
    good_sql.write_text(
        "CREATE OR REPLACE FUNCTION example() RETURNS void AS $$\n"
        "BEGIN\n"  # PL/pgSQL function body open — no trailing ``;``
        "    RAISE NOTICE 'hello';\n"
        "    BEGIN\n"  # PL/pgSQL nested block — no ``;``, still inside dollar
        "        SELECT 1;\n"
        "    END;\n"  # PL/pgSQL inner end — inside dollar, OK
        "END;\n"  # PL/pgSQL outer end — STILL inside dollar block
        "$$ LANGUAGE plpgsql;\n"
        "\n"
        "CREATE OR REPLACE FUNCTION example2() RETURNS void AS $func$\n"
        "BEGIN\n"
        "    SELECT 1;\n"
        "    COMMIT;\n"  # ``COMMIT;`` inside a function body — legal under PL/pgSQL,
        # not a script-level txn marker, the runner doesn't see it
        "END;\n"
        "$func$ LANGUAGE plpgsql;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(good_sql)
    assert findings == [], f"scanner false-positive on PL/pgSQL bodies: {findings}"


def test_scanner_documented_limit_on_begin_atomic_function_bodies(
    tmp_path: Path,
) -> None:
    """``BEGIN ATOMIC ... END;`` is the SQL-standard function body
    syntax (alternative to ``$$ ... $$``). The scanner does NOT track
    ``BEGIN ATOMIC`` openers as opening a "function-body" context,
    because (a) this repo uses ``$$``-style bodies exclusively, and
    (b) implementing full SQL parsing here is out of scope for a
    filesystem-only lint.

    This test pins the resulting limitation: the closing ``END;`` of
    a ``BEGIN ATOMIC`` body will be flagged as a script-level
    transaction-control statement, even though Postgres parses it as
    the function-body closer. The single-false-positive count is
    pinned so a future scanner change cannot silently swallow the
    limit (raising the count would indicate a new bug; lowering it
    to zero would indicate someone implemented full BEGIN ATOMIC
    tracking, in which case this test should be updated rather than
    silently weakened).

    If this repo ever adopts ``BEGIN ATOMIC`` function bodies, widen
    the scanner to track them and update / remove this pin.
    """
    sql = tmp_path / "atomic_migration.sql"
    sql.write_text(
        "CREATE FUNCTION add_one(x integer) RETURNS integer\n"
        "    LANGUAGE SQL\n"
        "    IMMUTABLE\n"
        "BEGIN ATOMIC\n"
        "    SELECT x + 1;\n"
        "END;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, (
        f"expected single END; false-positive (documented limit on "
        f"BEGIN ATOMIC function bodies — see test docstring), got: {findings}"
    )
    assert "END;" in findings[0][1].upper()


def test_scanner_detects_txn_control_after_same_line_dollar_close(
    tmp_path: Path,
) -> None:
    """Codex pass-2 LOW-2 + pass-3 LOW-1: if a dollar-quoted block
    closes mid-line and the same line then has script-level
    transaction-control SQL, that SQL must NOT be silently exempted.

    Postgres allows ``$$ LANGUAGE plpgsql; COMMIT;`` on a single line
    (the function body closes at ``$$``, the ``;`` ends the
    ``CREATE FUNCTION`` statement, ``COMMIT;`` is then a separate
    script-level statement). Closed by adding semicolon-level
    statement splitting in ``_scan_for_bare_txn`` (pass-3).
    """
    sql = tmp_path / "same_line_close_then_commit.sql"
    sql.write_text(
        "CREATE OR REPLACE FUNCTION fn() RETURNS void AS $$\n"
        "BEGIN\n"
        "    SELECT 1;\n"
        "END;\n"
        "$$ LANGUAGE plpgsql; COMMIT;\n",  # Same-line close + script COMMIT;
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected 1 finding on line 5, got: {findings}"
    assert findings[0][0] == 5
    assert "COMMIT;" in findings[0][1].upper()


def test_scanner_detects_multi_statement_per_line_txn_control(tmp_path: Path) -> None:
    """Codex pass-3 LOW-1: multi-statement-per-line script SQL must
    surface txn-control as its own statement. The semicolon-level
    splitter handles this.

    Example pattern (deliberately ugly, but valid SQL):

        ALTER TABLE foo ADD COLUMN bar INTEGER; COMMIT;

    The legitimate ``ALTER`` doesn't trip the regex; ``COMMIT;``
    does.
    """
    sql = tmp_path / "multi_statement_line.sql"
    sql.write_text(
        "ALTER TABLE foo ADD COLUMN bar INTEGER; COMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected 1 finding on line 1, got: {findings}"
    assert findings[0][0] == 1
    assert "COMMIT;" in findings[0][1].upper()


def test_scanner_detects_isolated_script_commit_after_dollar_close(
    tmp_path: Path,
) -> None:
    """Companion to the same-line-close test: the standard pattern of
    a dollar-closer line followed by a SEPARATE script-level
    ``COMMIT;`` MUST be flagged. This is the actual runner-contract
    violation pattern we care about.
    """
    sql = tmp_path / "next_line_commit.sql"
    sql.write_text(
        "CREATE OR REPLACE FUNCTION fn() RETURNS void AS $$\n"
        "BEGIN\n"
        "    SELECT 1;\n"
        "END;\n"
        "$$ LANGUAGE plpgsql;\n"
        "COMMIT;\n",  # Script-level COMMIT on its own line — IS flagged
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected COMMIT; flag, got: {findings}"
    assert findings[0][0] == 6
    assert "COMMIT;" in findings[0][1].upper()


def test_scanner_handles_custom_dollar_tags(tmp_path: Path) -> None:
    """Dollar-quoting with a custom tag (``$outer$ ... $outer$``).
    Inside such a block, embedded ``$$`` literals are nested dollar
    blocks (Postgres parses them as a separate inner dollar-quoted
    string) that open + close on the same line without affecting the
    outer block. The matching close of the outer is the literal
    ``$outer$`` token; anything in between is function-body content.
    """
    sql = tmp_path / "custom_tag.sql"
    sql.write_text(
        "CREATE FUNCTION outer_fn() RETURNS void AS $outer$\n"
        "BEGIN\n"
        "    -- comment inside the function body\n"
        "    EXECUTE $inner$SELECT 1;$inner$;\n"
        "    COMMIT;\n"  # COMMIT inside $outer$ — must not be flagged
        "END;\n"
        "$outer$ LANGUAGE plpgsql;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert findings == [], f"scanner false-positive on custom dollar tag: {findings}"


def test_strip_inline_comment_preserves_sql() -> None:
    """The comment-stripper must not consume real SQL text before the
    ``--`` marker.
    """
    assert _strip_inline_comment("SELECT 1; -- trailing comment") == "SELECT 1; "
    assert _strip_inline_comment("-- whole line comment") == ""
    assert _strip_inline_comment("BEGIN; -- spurious") == "BEGIN; "
    assert _strip_inline_comment("BEGIN;") == "BEGIN;"
    # Verify the post-strip text still matches the txn-statement regex
    # on the BEGIN; case.
    stripped = _strip_inline_comment("BEGIN; -- spurious").strip()
    assert _TXN_STATEMENT_RE.match(stripped) is not None
