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
  - ``BEGIN ISOLATION LEVEL ...;`` / ``BEGIN READ ONLY;`` /
    ``START TRANSACTION READ WRITE;`` and other transaction-mode
    variants                                               -> outer-txn open
  - ``COMMIT;`` / ``COMMIT TRANSACTION;`` / ``COMMIT WORK;`` /
    ``COMMIT AND CHAIN;``                                   -> premature commit
  - ``ROLLBACK;`` / ``ROLLBACK TRANSACTION;`` /
    ``ROLLBACK WORK;`` / ``ROLLBACK AND NO CHAIN;`` /
    ``ABORT;`` / ``ABORT TRANSACTION;`` /
    ``ABORT WORK;``                                         -> premature rollback
  - ``END;`` / ``END TRANSACTION;`` / ``END WORK;`` at the script level
    (PostgreSQL treats bare ``END;`` outside a PL/pgSQL body as a
    synonym for ``COMMIT;``)

PL/pgSQL function-body ``BEGIN ... END`` blocks are NOT flagged: the
scanner tracks ``$tag$ ... $tag$`` dollar-quoted block boundaries and
skips anything inside them. Multi-statement-per-line script SQL like
``ALTER ...; COMMIT;`` is correctly split by a semicolon-level
tokenizer that respects single-quoted string literals (so
``VALUES ('a;b')`` does not produce a fake ``b'`` statement).
Mid-statement fragments without a terminating ``;`` (e.g. lines of a
multi-line ``CASE ... END AS col`` expression) are not classified as
complete statements — the lint only fires on statements that actually
end with ``;`` in the source.

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

# Match any script-level transaction-control statement head, including
# Postgres-accepted clause/mode variants (codex pass-5 LOW-2). The
# runner contract under ``psql --single-transaction`` is violated by
# ANY of:
#
#   - Bare forms: ``BEGIN;``, ``COMMIT;``, ``ROLLBACK;``, ``END;``,
#     ``ABORT;``, ``START TRANSACTION;``.
#   - Synonym clauses: ``BEGIN TRANSACTION;``, ``COMMIT WORK;`` etc.
#   - Transaction-mode clauses: ``BEGIN ISOLATION LEVEL SERIALIZABLE;``,
#     ``BEGIN READ ONLY;``, ``BEGIN NOT DEFERRABLE;``, etc.
#   - Chaining clauses: ``COMMIT AND CHAIN;``, ``COMMIT AND NO CHAIN;``,
#     same for ``ROLLBACK``.
#   - ``START TRANSACTION READ WRITE;``, ``START TRANSACTION ISOLATION
#     LEVEL SERIALIZABLE;``, etc.
#
# The regex accepts arbitrary trailing tokens after the statement
# head and before the terminating ``;``. The ``head`` itself is the
# constrained part: one of ``BEGIN``, ``START TRANSACTION``, ``COMMIT``,
# ``END``, ``ROLLBACK``, ``ABORT``. Followed by ANY content (greedy
# match) plus the terminating ``;``.
#
# Indentation is NOT the discriminator — the discriminator is whether
# the statement sits inside a ``$tag$ ... $tag$`` dollar-quoted block
# (PL/pgSQL function body), which the scanner tracks separately.
_TXN_STATEMENT_RE = re.compile(
    r"""
    ^                                  # start of stripped statement
    (?:
        BEGIN
        | START\s+TRANSACTION
        | COMMIT
        | END
        | ROLLBACK
        | ABORT
    )
    (?:\s+[^;]*)?                      # optional trailing clauses (TRANSACTION,
                                       # WORK, ISOLATION LEVEL ..., AND CHAIN,
                                       # READ ONLY, etc.)
    \s*;\s*$                           # terminating semicolon
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Dollar-quote delimiter. Tag is optional and may be any
# identifier-like token (Postgres docs: alphanumeric or empty).
# Examples: ``$$``, ``$func$``, ``$BODY$``. We capture the tag so the
# closing delimiter must match the opener — nested dollar blocks with
# different tags are technically legal (though unused in this repo).
_DOLLAR_QUOTE_RE = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*)?\$")

# ``BEGIN ATOMIC`` opener (codex pass-7 MEDIUM-1). This is the
# SQL-standard alternative to PL/pgSQL ``$$ ... $$`` function bodies.
# We need to recognize it because the closing ``END;`` of an atomic
# body is a function-body terminator, NOT a script-level commit.
# We don't track ``BEGIN ATOMIC`` via dollar-quote machinery because
# it isn't dollar-quoted — it's a SQL keyword sequence inside the
# ``CREATE FUNCTION`` statement.
_BEGIN_ATOMIC_RE = re.compile(r"\bBEGIN\s+ATOMIC\b", re.IGNORECASE)

# Bare ``END;`` (no clauses) — the typical function-body terminator
# for ``BEGIN ATOMIC ... END;``. Used to detect exit from atomic
# mode. We DELIBERATELY do not match ``END LOOP;`` or ``END CASE;``
# here because those are PL/pgSQL constructs, always inside ``$$``
# blocks, and never seen at script level.
_BARE_END_RE = re.compile(r"^END\s*;\s*$", re.IGNORECASE)


def _tokenize_script_statements(text: str) -> list[tuple[int, str]]:
    """Unified SQL tokenizer (codex pass-8 MEDIUM-1). Walk the full
    file once and emit script-level statements as
    ``(terminator_lineno, statement_text)`` tuples.

    The tokenizer maintains five mutually-exclusive lexical states:

    1. ``normal``: outside any quoting or comment construct.
    2. ``in_line_comment``: between ``--`` and end-of-line.
    3. ``in_block_comment``: between ``/*`` and ``*/``.
    4. ``in_single_quote``: inside a ``'...'`` string literal
       (with ``''`` doubled-quote escape).
    5. ``in_dollar_quote``: inside a ``$tag$ ... $tag$`` dollar-quoted
       block.

    State transitions are recognised ONLY from the ``normal`` state
    (e.g. a ``$$`` inside a ``'...'`` string is just literal text, not
    a dollar-block opener). This is critical to avoid the pass-8
    false-negatives where embedded ``$$``, ``--``, and ``/* */``
    inside string literals were being interpreted as state
    transitions.

    Statements end at every ``;`` encountered in ``normal`` state. The
    returned list contains only terminated statements; any trailing
    non-terminated text is dropped (the lint cares about complete
    statements with terminators).

    Each statement is reduced to its body text with comments
    stripped (comments are skipped during tokenization). Single-
    and dollar-quoted strings are PRESERVED in the body text so the
    matchers downstream can still match keywords correctly — keywords
    are never inside string literals at the statement-head position.
    """
    statements: list[tuple[int, str]] = []
    body: list[str] = []
    terminator_lineno = 1
    current_lineno = 1
    state = "normal"
    open_tag = ""
    i = 0
    n = len(text)

    while i < n:
        ch = text[i]
        if ch == "\n":
            current_lineno += 1

        if state == "normal":
            # Try state transitions in priority order.
            if ch == "'":
                body.append(ch)
                state = "in_single_quote"
                i += 1
                continue
            if ch == "-" and i + 1 < n and text[i + 1] == "-":
                state = "in_line_comment"
                i += 2
                continue
            if ch == "/" and i + 1 < n and text[i + 1] == "*":
                state = "in_block_comment"
                i += 2
                continue
            if ch == "$":
                # Try to match `$tag$` at this position.
                dollar_match = _DOLLAR_QUOTE_RE.match(text, i)
                if dollar_match is not None:
                    open_tag = dollar_match.group(1) or ""
                    body.append(dollar_match.group(0))
                    i = dollar_match.end()
                    state = "in_dollar_quote"
                    continue
            if ch == ";":
                # End of statement.
                statement = "".join(body).strip()
                if statement:
                    statements.append((current_lineno, statement))
                body = []
                terminator_lineno = current_lineno
                i += 1
                continue
            body.append(ch)
            i += 1
            continue

        if state == "in_line_comment":
            if ch == "\n":
                state = "normal"
                # Newline itself is whitespace at script level.
                body.append(" ")
            i += 1
            continue

        if state == "in_block_comment":
            if ch == "*" and i + 1 < n and text[i + 1] == "/":
                state = "normal"
                body.append(" ")
                i += 2
                continue
            i += 1
            continue

        if state == "in_single_quote":
            body.append(ch)
            if ch == "'":
                # SQL `''` escape: doubled quote = stay inside.
                if i + 1 < n and text[i + 1] == "'":
                    body.append(text[i + 1])
                    i += 2
                    continue
                state = "normal"
            i += 1
            continue

        if state == "in_dollar_quote":
            # Try to match the closing tag at this position.
            if ch == "$":
                dollar_match = _DOLLAR_QUOTE_RE.match(text, i)
                if dollar_match is not None:
                    tag = dollar_match.group(1) or ""
                    if tag == open_tag:
                        body.append(dollar_match.group(0))
                        i = dollar_match.end()
                        state = "normal"
                        open_tag = ""
                        continue
            body.append(ch)
            i += 1
            continue

        # Unreachable — defensive.
        i += 1

    # Drop any non-terminated trailing text — the lint only fires on
    # statements that actually have a `;`.
    _ = terminator_lineno  # silence unused-var; line citation lives in tuples
    return statements


def _scan_for_bare_txn(sql_path: Path) -> list[tuple[int, str]]:
    """Return a list of ``(line_number, line_text)`` for every
    script-level transaction-control statement in ``sql_path``.

    Uses the unified ``_tokenize_script_statements`` to extract
    properly-tokenized script-level statements (respecting all of
    single quotes, dollar quotes, line comments, block comments). For
    each statement, decides whether it matches the txn regex; tracks
    ``BEGIN ATOMIC`` function-body context across statements.
    """
    findings: list[tuple[int, str]] = []
    text = sql_path.read_text(encoding="utf-8")
    source_lines = text.splitlines()

    # ``BEGIN ATOMIC`` function-body tracking (codex pass-7 MEDIUM).
    in_atomic_body = False

    for terminator_lineno, statement in _tokenize_script_statements(text):
        candidate = statement + ";"

        # Atomic-body opener?
        if not in_atomic_body and _BEGIN_ATOMIC_RE.search(statement):
            in_atomic_body = True
            continue

        if in_atomic_body:
            if _BARE_END_RE.match(candidate):
                in_atomic_body = False
            continue

        if _TXN_STATEMENT_RE.match(candidate):
            cited_line = (
                source_lines[terminator_lineno - 1]
                if 0 < terminator_lineno <= len(source_lines)
                else ""
            )
            findings.append((terminator_lineno, cited_line.rstrip()))

    return findings
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
        # Transaction modes (codex pass-5 LOW-2).
        "BEGIN ISOLATION LEVEL SERIALIZABLE;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nCOMMIT;\n",
        "START TRANSACTION READ WRITE;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nCOMMIT;\n",
        # Chaining clauses (codex pass-5 LOW-2).
        "BEGIN;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nCOMMIT AND CHAIN;\n",
        "BEGIN;\nALTER TABLE foo ADD COLUMN bar INTEGER;\nROLLBACK AND NO CHAIN;\n",
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
        "begin-isolation-level-serializable",
        "start-transaction-read-write",
        "commit-and-chain",
        "rollback-and-no-chain",
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


def test_scanner_allows_begin_atomic_function_bodies(tmp_path: Path) -> None:
    """``BEGIN ATOMIC ... END;`` is the SQL-standard function body
    syntax (alternative to ``$$ ... $$``). The scanner tracks
    ``BEGIN ATOMIC`` openers as a function-body context: subsequent
    statements (and the closing ``END;``) are exempt from
    transaction-control flagging.

    Codex pass-7 MEDIUM-1: closed the pass-2 limit. The scanner now
    enters "atomic mode" when a candidate statement contains
    ``BEGIN ATOMIC`` and exits when it sees a bare ``END;``.
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
    assert findings == [], f"scanner false-positive on BEGIN ATOMIC function body: {findings}"


def test_scanner_resumes_flagging_after_begin_atomic_body(tmp_path: Path) -> None:
    """After a ``BEGIN ATOMIC ... END;`` body exits, the scanner must
    resume flagging script-level txn-control. A migration with a
    legitimate atomic function body followed by a stray script-level
    ``COMMIT;`` should flag the COMMIT.
    """
    sql = tmp_path / "atomic_then_commit.sql"
    sql.write_text(
        "CREATE FUNCTION add_one(x integer) RETURNS integer\n"
        "    LANGUAGE SQL\n"
        "    IMMUTABLE\n"
        "BEGIN ATOMIC\n"
        "    SELECT x + 1;\n"
        "END;\n"
        "COMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected COMMIT; finding, got: {findings}"
    assert findings[0][0] == 7
    assert "COMMIT;" in findings[0][1].upper()


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


def test_scanner_ignores_semicolons_inside_single_quoted_strings(
    tmp_path: Path,
) -> None:
    """Codex pass-4 LOW-1: semicolons inside SQL string literals must
    NOT be treated as statement separators. Otherwise a benign
    ``INSERT INTO audit_log VALUES ('before; COMMIT; after');``
    would surface a fake ``COMMIT;`` finding.

    The fix is the ``_split_statements_outside_strings`` tokenizer
    which tracks single-quote open/close state.
    """
    sql = tmp_path / "string_literal_semicolons.sql"
    sql.write_text(
        "INSERT INTO audit_log(message) VALUES ('before; COMMIT; after');\n"
        "SELECT 'before; BEGIN; after';\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert findings == [], (
        f"scanner false-positive on semicolons inside string literals: {findings}"
    )


def test_scanner_handles_doubled_quote_escape_in_strings(tmp_path: Path) -> None:
    """SQL standard ``''`` (two single quotes) is the escape for a
    literal apostrophe inside a string. A doubled quote must NOT be
    interpreted as a string close-then-reopen, otherwise the
    tokenizer's state would drift and a subsequent ``;`` would be
    treated as a statement separator.

    Pattern: ``VALUES ('it''s a test; COMMIT; here')`` — the ``''``
    is an escaped apostrophe, the whole literal is one string, and
    the ``;`` characters are inside it.
    """
    sql = tmp_path / "escaped_quotes.sql"
    sql.write_text(
        "INSERT INTO audit_log(message) VALUES ('it''s a test; COMMIT; here');\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert findings == [], f"scanner false-positive on ``''`` quote escape: {findings}"


def test_scanner_handles_multi_line_string_literals(
    tmp_path: Path,
) -> None:
    """Codex pass-5 LOW-1 + pass-6 MEDIUM: the scanner threads
    single-quote state across line boundaries via the cross-line
    script buffer, so a multi-line SQL string literal containing a
    ``;`` on its own line is correctly NOT flagged.

    The pass-6 fix introduced a cross-line accumulator: characters
    are buffered (with their source line numbers) until the script
    tokenizer finds a real statement-terminator ``;`` (i.e. outside
    a single-quoted string). This closed the pass-5 LOW-1 limit as
    a side effect.
    """
    sql = tmp_path / "multi_line_string.sql"
    sql.write_text(
        "INSERT INTO audit_log(message) VALUES ('first line\nCOMMIT;\nlast line');\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert findings == [], f"scanner false-positive on multi-line string literal: {findings}"


def test_scanner_detects_txn_control_split_across_lines(tmp_path: Path) -> None:
    """Codex pass-6 MEDIUM: a transaction-control statement whose
    body and terminator are on different physical lines (e.g.
    ``BEGIN\\n;``) IS syntactically equivalent to the same-line
    form and must be flagged.

    The pass-6 cross-line accumulator handles this: characters are
    buffered with line numbers, and when the tokenizer finds the
    statement-terminating ``;`` the citation walks back to the
    correct line.
    """
    sql = tmp_path / "split_across_lines.sql"
    sql.write_text(
        "BEGIN\n;\nALTER TABLE foo ADD COLUMN bar int;\nCOMMIT\n;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    # Two findings: ``BEGIN ;`` closes on line 2, ``COMMIT ;`` on
    # line 5.
    assert len(findings) == 2, f"expected 2 findings, got: {findings}"
    assert findings[0][0] == 2  # ``;`` after BEGIN
    assert findings[1][0] == 5  # ``;`` after COMMIT


def test_scanner_detects_isolation_level_split_across_lines(
    tmp_path: Path,
) -> None:
    """``BEGIN\\n  ISOLATION LEVEL SERIALIZABLE;`` — head on one
    line, mode + terminator on the next. Same pattern as above.
    """
    sql = tmp_path / "split_isolation.sql"
    sql.write_text(
        "BEGIN\n  ISOLATION LEVEL SERIALIZABLE;\nSELECT 1;\nCOMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 2, f"expected 2 findings, got: {findings}"
    assert findings[0][0] == 2  # `;` after ISOLATION LEVEL SERIALIZABLE
    assert findings[1][0] == 4  # `;` after COMMIT


def test_tokenize_script_statements_basic_cases() -> None:
    """Direct unit-test of ``_tokenize_script_statements`` to pin the
    unified tokenizer's behavior.

    Return shape: ``[(terminator_lineno, statement_text)]`` —
    statements are emitted only when terminated by ``;``; trailing
    non-terminated text is dropped.
    """
    # No string, two statements (one terminator).
    assert _tokenize_script_statements("BEGIN;COMMIT") == [(1, "BEGIN")]
    # Trailing semicolon → 1 terminated statement, no tail.
    assert _tokenize_script_statements("BEGIN;") == [(1, "BEGIN")]
    # Both terminated.
    assert _tokenize_script_statements("BEGIN;COMMIT;") == [
        (1, "BEGIN"),
        (1, "COMMIT"),
    ]
    # Semicolon inside single-quoted string (no split, no
    # terminator → no statement emitted).
    assert _tokenize_script_statements("VALUES ('a;b')") == []
    # Doubled quote escape (no split, no terminator).
    assert _tokenize_script_statements("VALUES ('it''s;a;test')") == []
    # Multiple strings + a real terminator.
    assert _tokenize_script_statements("INSERT INTO t VALUES ('a;b'), ('c;d');COMMIT;") == [
        (1, "INSERT INTO t VALUES ('a;b'), ('c;d')"),
        (1, "COMMIT"),
    ]
    # Empty input.
    assert _tokenize_script_statements("") == []
    # Mid-statement fragment without terminator: dropped.
    assert _tokenize_script_statements("    END AS patient_volume_tier,") == []
    # Line-number tracking: terminator on line 2.
    assert _tokenize_script_statements("BEGIN\n;") == [(2, "BEGIN")]
    # Block comment stripped, no statement.
    assert _tokenize_script_statements("/* hello */ ALTER TABLE foo ADD bar int;") == [
        (1, "ALTER TABLE foo ADD bar int")
    ]
    # Line comment stripped.
    assert _tokenize_script_statements("ALTER TABLE foo -- ignored\nADD bar int;") == [
        (2, "ALTER TABLE foo  ADD bar int")
    ]


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


def test_scanner_handles_block_comments_around_txn(tmp_path: Path) -> None:
    """Codex pass-8 MEDIUM-1: ``/* ... */`` block comments must be
    stripped during tokenization, otherwise ``/* harmless */ COMMIT;``
    would have its statement-head be ``/* harmless */ COMMIT`` (not
    ``COMMIT``) and miss the txn regex.
    """
    sql = tmp_path / "block_comment_around_commit.sql"
    sql.write_text(
        "/* comment block */ COMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected COMMIT; flag, got: {findings}"
    assert findings[0][0] == 1


def test_scanner_handles_line_comment_inside_string_literal(
    tmp_path: Path,
) -> None:
    """Codex pass-8 MEDIUM-1: ``--`` inside a single-quoted string
    must NOT terminate a "line comment" — the whole string is a
    literal. A subsequent ``COMMIT;`` IS a real script statement and
    must flag.
    """
    sql = tmp_path / "dash_dash_in_string.sql"
    sql.write_text(
        "SELECT '-- not a comment'; COMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected COMMIT; flag, got: {findings}"
    assert findings[0][0] == 1


def test_scanner_handles_dollar_dollar_inside_string_literal(
    tmp_path: Path,
) -> None:
    """Codex pass-8 MEDIUM-1: ``$$`` inside a single-quoted string
    must NOT open a "dollar-quoted block" — the whole string is a
    literal. A subsequent ``COMMIT;`` IS a real script statement and
    must flag.
    """
    sql = tmp_path / "dollar_dollar_in_string.sql"
    sql.write_text(
        "SELECT '$$'; COMMIT;\n",
        encoding="utf-8",
    )
    findings = _scan_for_bare_txn(sql)
    assert len(findings) == 1, f"expected COMMIT; flag, got: {findings}"
    assert findings[0][0] == 1
