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


def _split_statements_outside_strings(text: str) -> list[tuple[str, bool]]:
    """Split ``text`` on ``;`` boundaries that are NOT inside a
    single-quoted SQL string literal. Handles SQL's standard
    ``''`` escape (a doubled single-quote represents a literal
    apostrophe inside a string, and does NOT close the string).

    Returns a list of ``(statement_text, terminated)`` tuples, where
    ``terminated`` is ``True`` if the statement was followed by a
    ``;`` in the source (a complete statement) and ``False`` if it's
    a trailing partial. Callers must only treat ``terminated=True``
    entries as full statements — otherwise the lint would fire on
    multi-line statements that haven't reached their semicolon yet
    (e.g. ``CASE ... END AS col,`` on a line by itself, codex pass-5
    follow-up).

    This is a minimal SQL tokenizer scoped to the lint's needs:

    * Tracks ``'`` open/close. Doubled ``''`` toggles in then back
      out, which is equivalent to "stay open" (per SQL spec).
    * Does NOT handle PostgreSQL's ``E'\\''`` C-style escape syntax;
      this repo doesn't use it in migrations. If a future migration
      starts using ``E'...'``, the tokenizer will need a second
      state for that quoting style.
    * Does NOT handle ``"..."`` identifier quoting (no transaction
      keywords are valid identifiers in standard contexts; double-
      quoted identifiers can contain ``;`` but never as
      statement-separator).
    * Does NOT handle ``/* ... */`` block comments. Migrations in
      this repo use ``--`` line comments exclusively.
    * Dollar-quoted strings are already filtered upstream by
      ``_split_line_at_dollar_boundaries`` — by the time text
      reaches this function, all content is script-level (outside
      function bodies). Within a script-level context, dollar
      quoting is rare (it's primarily a function-body construct)
      but if it appeared, this tokenizer would NOT skip semicolons
      inside it. Codex pass-4 documented limitation.
    * Single-quote state does NOT persist across the caller-supplied
      text boundary. The scanner currently calls this once per
      source line, so a multi-line string literal with a ``;`` on
      its own line would false-positive. Codex pass-5 documented
      limitation; not a current issue because migrations in this
      repo do not split string literals across lines.

    Codex pass-4 LOW-1; pass-5 follow-up tuple shape.
    """
    statements: list[tuple[str, bool]] = []
    current: list[str] = []
    in_single_quote = False
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "'":
            current.append(ch)
            if in_single_quote:
                # SQL ``''`` escape: if next char is also ``'``,
                # consume it and stay inside the string.
                if i + 1 < n and text[i + 1] == "'":
                    current.append(text[i + 1])
                    i += 2
                    continue
                in_single_quote = False
            else:
                in_single_quote = True
            i += 1
            continue
        if ch == ";" and not in_single_quote:
            statements.append(("".join(current), True))
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    # Tail (any trailing text after the last ``;`` or whole text if no
    # ``;`` found). ``terminated=False`` — this is a partial statement
    # waiting for its semicolon on a subsequent line.
    if current:
        statements.append(("".join(current), False))
    return statements


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
    # Codex pass-7 MEDIUM: ``BEGIN ATOMIC`` opens a SQL-standard
    # function body. Inside it, ``;`` separates body statements and
    # the closing ``END;`` ends the outer ``CREATE FUNCTION``. We do
    # not flag txn-control inside ``BEGIN ATOMIC``, nor on the
    # function-body-closing ``END;``.
    in_atomic_body = False

    # Accumulator for script-level text that spans multiple physical
    # lines (codex pass-6 MEDIUM: ``BEGIN\n;\n`` is syntactically
    # equivalent to ``BEGIN;`` in Postgres). Pair each character with
    # its source line number so findings can cite the line where the
    # offending statement closes.
    script_buffer: list[tuple[int, str]] = []

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        segments, in_dollar_block, open_tag = _split_line_at_dollar_boundaries(
            raw_line,
            in_dollar_block,
            open_tag,
        )
        if not segments:
            continue
        # Strip inline ``--`` comments from each segment BEFORE
        # accumulating. (After accumulating across lines, a ``--``
        # in one line would consume the next line's text too,
        # missing real transaction-control statements.)
        cleaned_segments = [_strip_inline_comment(seg) for seg in segments]
        # Pair each segment character with this line's line-number
        # so findings cite the right line when the statement closes
        # mid-buffer.
        for seg in cleaned_segments:
            for ch in seg:
                script_buffer.append((lineno, ch))
            # Replace inter-segment boundaries with a synthetic
            # whitespace so adjacent segments don't run together.
            script_buffer.append((lineno, " "))

        # Drain complete statements from the buffer (text up to and
        # including each script-level ``;``). The drainer respects
        # single-quote string state. A non-terminated tail remains
        # in the buffer for the next line to potentially complete.
        buffer_text = "".join(ch for _, ch in script_buffer)
        statements = _split_statements_outside_strings(buffer_text)
        if not statements:
            continue
        # Reconstruct line-number for each statement: each statement
        # ends at the position of its terminating ``;`` (or the buffer
        # tail). Walk the buffer's (lineno, ch) pairs to find each
        # ``;`` position.
        new_buffer: list[tuple[int, str]] = []
        cursor = 0
        for raw_statement, terminated in statements:
            statement_len = len(raw_statement) + (1 if terminated else 0)
            statement_end = cursor + statement_len
            if not terminated:
                # Leftover tail — preserve for next iteration.
                new_buffer = script_buffer[cursor:]
                break
            # Cite the line where the terminating ``;`` appears
            # (statement_end - 1 is the ``;``).
            terminator_lineno = script_buffer[statement_end - 1][0]
            cursor = statement_end
            statement = raw_statement.strip()
            if not statement:
                continue
            candidate = statement + ";"

            # Track ``BEGIN ATOMIC`` function-body opener/closer
            # (codex pass-7 MEDIUM). A statement that CONTAINS
            # ``BEGIN ATOMIC`` (case-insensitive) enters atomic mode;
            # while inside, we do NOT flag any txn-control. The
            # bare ``END;`` that exits the function body takes us
            # out of atomic mode (without flagging).
            if not in_atomic_body and _BEGIN_ATOMIC_RE.search(statement):
                in_atomic_body = True
                continue

            if in_atomic_body:
                # Inside ``BEGIN ATOMIC ... END;`` — every internal
                # ``;`` separates body statements, and the closing
                # ``END;`` (with no other clauses) exits the body.
                if _BARE_END_RE.match(candidate):
                    in_atomic_body = False
                continue

            if _TXN_STATEMENT_RE.match(candidate):
                # Recover the raw line text for the citation: re-read
                # from the source text by line number.
                source_lines = text.splitlines()
                cited_line = (
                    source_lines[terminator_lineno - 1]
                    if 0 < terminator_lineno <= len(source_lines)
                    else ""
                )
                findings.append((terminator_lineno, cited_line.rstrip()))
        else:
            # All statements terminated; clear buffer.
            new_buffer = []
        script_buffer = new_buffer

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


def test_split_statements_outside_strings_basic_cases() -> None:
    """Direct unit-test of ``_split_statements_outside_strings`` to
    pin the tokenizer's behavior.

    Return shape: ``[(statement_text, terminated_bool)]`` where
    ``terminated`` is ``True`` iff the source had a ``;`` after the
    statement.
    """
    # No string, two statements — first is terminated by `;`, second
    # is a trailing fragment without `;`.
    assert _split_statements_outside_strings("BEGIN;COMMIT") == [
        ("BEGIN", True),
        ("COMMIT", False),
    ]
    # Trailing semicolon → empty tail dropped (no current text):
    assert _split_statements_outside_strings("BEGIN;") == [("BEGIN", True)]
    # Semicolon inside single-quoted string (no split):
    assert _split_statements_outside_strings("VALUES ('a;b')") == [
        ("VALUES ('a;b')", False),
    ]
    # Doubled quote escape (no split):
    assert _split_statements_outside_strings("VALUES ('it''s;a;test')") == [
        ("VALUES ('it''s;a;test')", False),
    ]
    # Multiple strings + a real terminator:
    assert _split_statements_outside_strings("INSERT INTO t VALUES ('a;b'), ('c;d');COMMIT") == [
        ("INSERT INTO t VALUES ('a;b'), ('c;d')", True),
        ("COMMIT", False),
    ]
    # Empty input:
    assert _split_statements_outside_strings("") == []
    # Mid-statement fragment with no terminator (codex pass-5
    # follow-up: must not be classified as terminated):
    assert _split_statements_outside_strings("    END AS patient_volume_tier,") == [
        ("    END AS patient_volume_tier,", False),
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
