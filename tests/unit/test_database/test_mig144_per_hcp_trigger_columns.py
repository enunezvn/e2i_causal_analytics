"""Migration 144: the per_hcp_rollup trigger counts get honest names — by EXPAND, not rename.

Hermetic — reads the migration FILES, never a database. That boundary is the point:
a file can show that every statement is written so a second application is a no-op,
but only a live re-application can show that it *is* one. The rehearsal
(BEGIN / apply / apply again / ROLLBACK) is what proves idempotency; these tests
prove the file is SHAPED so that it can be idempotent, and refuse the shapes that
cannot be.

WHY THIS IS AN EXPAND AND NOT A RENAME (codex iter1 HIGH-1, owner-approved
2026-09-18). 144 used to RENAME business_metrics.{trx_count,nrx_count,
total_rx_count} to the triggers_* names. ``.github/workflows/deploy.yml`` applies
migrations at :956 while the OLD containers are still serving, replaces Feast at
:1044 and the app services only at :1085 — and a post-flip health failure at :1104
rolls back ONLY the app services. A rename therefore leaves pre-lane code, which
reads and WRITES the legacy names (52 references on origin/main, including
``src/etl/business_metrics_per_hcp_etl.py``'s ``ON CONFLICT DO UPDATE SET
trx_count = EXCLUDED.trx_count``), running against a schema that no longer has
them — permanently, with no automated way back.

So 144 now ADDs the canonical columns beside the legacy ones, backfills them, and
installs a BIDIRECTIONAL row trigger so either name may be read or written by
either code version for as long as both exist. The legacy columns are retired
later by ``database/migrations/146_*`` (applied by hand 2026-09-20, issue #2167) — see
``test_mig146_contract_legacy_per_hcp_columns.py``, which pins that the runner
cannot apply the contract half in the same deploy as this one.

Live census 2026-09-18 (read-only) behind the numbers above: ``business_metrics``
holds 22,043 rows, 12,143 of them ``metric_type='per_hcp_rollup'``, and each of
the three legacy columns is non-NULL on exactly those 12,143 — so the columns are
populated on per-HCP rollup rows and nowhere else. All three are nullable INTEGER
with no default, no index, no constraint and no RLS policy, and the table carries
no other user trigger. The four ``v_{train,test,validation,holdout}_business_metrics``
views are ``SELECT *`` snapshots that are ALREADY seven columns behind the table
and have ZERO consumers outside ``database/`` (measured across src/, tests/,
scripts/, feature_repo/, frontend/src) — which is why the expand leaves them
untouched and the contract migration recreates them.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "144_per_hcp_trigger_count_columns.sql"
ROLLBACK = MIGRATION.parent / "rollback_144_per_hcp_trigger_count_columns.sql"
#: legacy name -> canonical name. The expand adds the value beside the key.
PAIRS = {
    "trx_count": "triggers_delivered_count",
    "nrx_count": "triggers_accepted_count",
    "total_rx_count": "triggers_total_count",
}
SYNC_FUNCTION = "business_metrics_sync_legacy_trigger_counts"


def _sql() -> str:
    return MIGRATION.read_text()


def _executable_sql(text: str | None = None) -> str:
    """The file with ``--`` line comments stripped.

    Every REFUSAL below ("must not rename", "must not drop", "must not go
    dynamic") is asked of this, not of the raw text: a prose comment that merely
    NAMES a forbidden shape — and 144's header names several, because explaining
    why they are forbidden is the point — is not that shape. Matching the raw file
    would be a text proxy for a structural property, satisfiable while the real
    condition is untouched and, worse, failable while the file is correct.
    ``scripts/run_migrations.sh`` strips ``--`` the same way before its own
    keyword detection, and for the same reason. Positive assertions keep using
    the raw text, where a comment can only be an extra match, never a false one.
    """
    return _strip_comments(_sql() if text is None else text)


#: A dollar-quote open/close tag: ``$$`` or ``$tag$``.
_DOLLAR_TAG = re.compile(r"\$(?:[A-Za-z_][A-Za-z_0-9]*)?\$")


def _lex(text: str) -> list[tuple[str, str]]:
    """Split ``text`` into ``(kind, chunk)`` where kind is code, string or comment.

    ONE pass, because **anything that decides what is code and what is text has to
    make that decision in one place**. This module learned that three times, each
    time one construct later:

    * iter5: ``;`` inside a literal split a valid statement in two.
    * iter6 HIGH-1: ``--`` was stripped by a regex that ran BEFORE anything knew
      where the literals were, so the allowlist parsed a different program from the
      one psql would execute.
    * iter7 HIGH-2: ``/* */`` was not a construct at all. A ``*/`` parked after a
      ``--`` on the same line deletes the statement behind it from the guard's view
      while psql runs it -- measured, ``test_the_expand_renames_nothing`` passed on a
      144 that renamed ``trx_count``. The same hole fails a CORRECT file: a block
      comment naming a forbidden shape reads as that shape, and one apostrophe inside
      a block comment flips the string parity for the rest of the file.

    Each of those was fixed by adding the construct that had just been used against
    the previous fix. The generalisation is this function: the four things psql's own
    lexer distinguishes -- ``'...'`` literals (with ``''`` escapes, and backslash
    escapes after an ``E`` prefix), ``$tag$...$tag$`` bodies, ``--`` to end of line,
    and NESTED ``/* */`` -- decided once, in the order the server decides them, with
    every caller reading the result rather than re-deciding it.

    Unterminated literals, dollar bodies and block comments are REFUSED (AssertionError)
    rather than guessed at: each would silently swallow every statement after it, and
    a guard that swallows statements is worse than no guard.
    """
    out: list[tuple[str, str]] = []
    code: list[str] = []
    i, n = 0, len(text)

    def flush() -> None:
        if code:
            out.append(("code", "".join(code)))
            code.clear()

    while i < n:
        ch = text[i]
        pair = text[i : i + 2]
        if pair == "--":
            flush()
            nl = text.find("\n", i)
            end = n if nl == -1 else nl  # the newline itself stays code
            out.append(("comment", text[i:end]))
            i = end
        elif pair == "/*":
            flush()
            depth, j = 0, i
            while j < n:
                if text[j : j + 2] == "/*":
                    depth += 1
                    j += 2
                elif text[j : j + 2] == "*/":
                    depth -= 1
                    j += 2
                    if depth == 0:
                        break
                else:
                    j += 1
            assert depth == 0, "the file has an unterminated /* block comment"
            out.append(("comment", text[i:j]))
            i = j
        elif ch == "'":
            flush()
            # E'...' honours backslash escapes; a plain literal does not
            # (standard_conforming_strings), so E'\'' is ONE literal and 'a\' is not.
            prev, before = text[i - 1 : i], text[i - 2 : i - 1]
            escapes = prev in ("E", "e") and not (before.isalnum() or before == "_")
            j, closed = i + 1, False
            while j < n:
                c = text[j]
                if escapes and c == "\\":
                    j += 2
                    continue
                if c == "'":
                    if text[j + 1 : j + 2] == "'":  # '' is an escaped quote
                        j += 2
                        continue
                    j += 1
                    closed = True
                    break
                j += 1
            assert closed, "the file has an unterminated string literal"
            out.append(("string", text[i:j]))
            i = j
        elif ch == "$" and (m := _DOLLAR_TAG.match(text, i)):
            flush()
            tag = m.group(0)
            end = text.find(tag, m.end())
            assert end != -1, f"the file has an unterminated {tag} dollar-quoted body"
            j = end + len(tag)
            out.append(("string", text[i:j]))
            i = j
        else:
            code.append(ch)
            i += 1
    flush()
    return out


def _strip_comments(text: str) -> str:
    """The file with BOTH comment syntaxes replaced by the whitespace they are.

    See :func:`_executable_sql` for why every refusal is asked of this rather than
    of the raw file, and :func:`_lex` for why the decision is made once.

    A comment becomes blanks, not nothing: to psql a comment IS whitespace, so
    ``SELECT/**/1`` is two tokens, and deleting the comment outright would weld them
    into a ``SELECT1`` the server never sees. Newlines inside a comment are kept so
    line-anchored patterns over the result still line up with the file.
    """
    return "".join(
        re.sub(r"[^\n]", " ", chunk) if kind == "comment" else chunk for kind, chunk in _lex(text)
    )


#: The three column-name alternations the allowlists are built from, so that a
#: pattern cannot drift away from :data:`PAIRS` without the file being edited.
_LEGACY = "|".join(PAIRS)
_CANONICAL = "|".join(PAIRS.values())
_ANY_COUNT_COLUMN = "|".join([*PAIRS, *PAIRS.values()])
#: A SQL string literal, INCLUDING the ``''`` escape the lexer implements.
#: ``'[^']*'`` was a narrower language than the one being parsed (ultracode iter7
#: MED): a restored comment reading ``'the HCP''s counts'`` is a single valid
#: literal, and the pattern would have reported that correct statement unapproved.
_LITERAL = r"'(?:[^']|'')*'"

#: Every statement ``rollback_144`` is ALLOWED to contain, as an anchored pattern.
#: Whitespace is collapsed before matching, so a statement may wrap across lines.
_ROLLBACK_ALLOWED = (
    r"DROP TRIGGER IF EXISTS business_metrics_sync_legacy_trigger_counts_trg "
    r"ON public\.business_metrics",
    r"DROP FUNCTION IF EXISTS public\.business_metrics_sync_legacy_trigger_counts\(\)",
    rf"ALTER TABLE public\.business_metrics DROP COLUMN IF EXISTS (?:{_CANONICAL})",
    rf"COMMENT ON COLUMN public\.business_metrics\.(?:{_LEGACY}) IS {_LITERAL}",
    r"DELETE FROM public\.schema_migrations "
    r"WHERE filename = '144_per_hcp_trigger_count_columns\.sql'",
    r"NOTIFY pgrst, 'reload schema'",
)

#: The sync trigger statement, EXACTLY — every word of it is load-bearing, so the
#: pattern is anchored rather than sampled. ``BEFORE`` (an AFTER trigger cannot
#: change NEW), ``INSERT OR UPDATE`` with no ``OF <column>`` list (``UPDATE OF
#: engagement_score`` leaves the sync dead for every other write — ultracode iter7
#: HIGH-3, which moved ``pg_trigger.tgattr`` while the catalog fingerprint stayed
#: byte-identical), ``FOR EACH ROW`` (a statement trigger has no NEW), the function
#: it is bound to, and — by the anchoring alone — the ABSENCE of a ``WHEN`` clause,
#: which is how a trigger is disabled while still existing under its own name.
_SYNC_TRIGGER_STATEMENT = (
    rf"CREATE OR REPLACE TRIGGER {SYNC_FUNCTION}_trg "
    r"BEFORE INSERT OR UPDATE ON public\.business_metrics "
    r"FOR EACH ROW "
    rf"EXECUTE FUNCTION public\.{SYNC_FUNCTION}\(\)"
)

#: Every statement migration 144 is ALLOWED to contain.
#:
#: ultracode iter7 MED/HIGH: 144's "purely additive" guard used to be
#: ``not re.search(r"\b(DROP|DELETE|TRUNCATE)\b", ...)`` — the exact verb blacklist
#: :func:`_unapproved_statements` documents as unable to be a capability check, and
#: 144 is the half ``deploy.yml`` applies UNATTENDED while the rollback is applied by
#: hand. The allowlist protected only the file a human was already reading. Every
#: payload that walked past the rollback blacklist (``DROP TABLE``,
#: ``ALTER TABLE ... DROP COLUMN``, a concatenated ``'DE' || 'LETE'`` inside a DO
#: block) walked past this one for the same reason, and is refused here by the same
#: inversion: a statement must be a form 144 is known to need.
_EXPAND_ALLOWED = (
    rf"ALTER TABLE public\.business_metrics ADD COLUMN IF NOT EXISTS "
    rf"(?:{_CANONICAL}) INTEGER",
    rf"UPDATE public\.business_metrics SET (?:{_CANONICAL}) = (?:{_LEGACY}) "
    rf"WHERE (?:{_LEGACY}) IS NOT NULL "
    rf"AND (?:{_CANONICAL}) IS DISTINCT FROM (?:{_LEGACY})",
    # The plpgsql body is a dollar-quoted STRING to the lexer, so its `;`s do not
    # split the statement. `[^$]*` keeps a nested `$x$` body — the dynamic-EXECUTE
    # shape — out of it; what the body may SAY is asserted line by line against
    # :data:`_SYNC_BODY_LINE_ALLOWED`, in the test that owns the trigger.
    rf"CREATE OR REPLACE FUNCTION public\.{SYNC_FUNCTION}\(\) "
    rf"RETURNS TRIGGER LANGUAGE plpgsql AS \$sync\$[^$]*\$sync\$",
    _SYNC_TRIGGER_STATEMENT,
    rf"COMMENT ON COLUMN public\.business_metrics\.(?:{_ANY_COUNT_COLUMN}) IS {_LITERAL}",
    r"NOTIFY pgrst, 'reload schema'",
)

#: Every LINE the sync function's body is allowed to contain. The statement
#: allowlist can only say "a CREATE FUNCTION with a $sync$ body"; this says what may
#: be inside it, by the same inversion — an ``EXECUTE 'DELETE ...'``, a ``PERFORM``,
#: a second table's name or a silent ``RETURN OLD`` matches nothing here.
_SYNC_BODY_LINE_ALLOWED = (
    r"BEGIN",
    r"END",
    r"ELSE",
    r"END IF;",
    r"RETURN NEW;",
    r"IF TG_OP = 'INSERT' THEN",
    rf"(?:ELS)?IF NEW\.(?:{_ANY_COUNT_COLUMN}) IS NULL THEN",
    rf"(?:ELS)?IF NEW\.(?:{_ANY_COUNT_COLUMN}) IS DISTINCT FROM "
    rf"OLD\.(?:{_ANY_COUNT_COLUMN}) THEN",
    rf"NEW\.(?:{_ANY_COUNT_COLUMN}) := NEW\.(?:{_ANY_COUNT_COLUMN});",
)


def _statements(sql: str) -> list[str]:
    """The file's executable statements, comments stripped and whitespace collapsed.

    The split walks :func:`_lex`'s tokens, so a ``;`` only ends a statement when it is
    CODE. A naive ``body.split(";")`` is wrong here and this file proves it:
    ``COMMENT ON COLUMN ... IS '... (misnamed by migration 033; never prescriptions)'``
    carries a semicolon inside the literal, so the naive form cut a valid statement in
    two and reported both halves as unapproved. A splitter that mis-parses the file it
    guards is not a guard -- it fails on correct input and, on hostile input, whoever
    controls a string literal controls where the statement boundaries appear to be.

    Dollar-quoted bodies used to be REFUSED here rather than parsed, on the grounds
    that ``$`` needs real nesting rules and the rollback needs no such statement.
    144 does: its plpgsql body is a ``$sync$`` literal whose ``;``s must not split it,
    and 144 is the half the deploy applies unattended. :func:`_lex` now carries the
    tag-matching rule, so a dollar body is a string like any other -- and a body this
    file has no business containing (a ``DO $x$ ... $x$`` block, say) is still refused,
    by the ALLOWLIST, which is where "what may this file contain" belongs.
    """
    out: list[str] = []
    cur: list[str] = []
    for kind, chunk in _lex(sql):
        if kind == "comment":
            cur.append(" ")
        elif kind == "string":
            cur.append(chunk)
        else:
            pieces = chunk.split(";")
            for k, piece in enumerate(pieces):
                if k:
                    out.append("".join(cur))
                    cur = []
                cur.append(piece)
    out.append("".join(cur))
    return [re.sub(r"\s+", " ", s).strip() for s in out if s.strip()]


def _unapproved_statements(sql: str, allowed: tuple[str, ...] = _ROLLBACK_ALLOWED) -> list[str]:
    """Statements in ``sql`` that match none of ``allowed``.

    codex iter5 HIGH-1. Two rounds of BLACKLIST failed here, and the second failure
    is the instructive one. iter4 replaced a line-anchored ``^\\s*DELETE FROM`` with a
    detector that counted the DELETE/TRUNCATE verb anywhere -- which closed the three
    payloads codex had shown me and nothing else. Measured against the iter4 form:

        DROP TABLE public.business_metrics;                      -- not counted
        ALTER TABLE public.business_metrics DROP COLUMN metric_id;  -- not counted
        DO $x$ BEGIN EXECUTE 'DE' || 'LETE FROM public.business_metrics'; END $x$;
                                                                 -- not counted
        COMMENT ON TABLE public.business_metrics IS 'Never DELETE FROM ...';
                                                                 -- FALSE positive

    I had fixed the three examples instead of the question. **A blacklist of
    destructive spellings can never be a capability check**: the language has
    unboundedly many ways to spell destruction (other verbs, string concatenation,
    dynamic execution), so any such list is a proxy that is satisfiable while the
    real condition is false. An ALLOWLIST inverts the quantifier -- every statement
    must be one of the forms this rollback is known to need -- so an unapproved form
    fails by construction, whatever it is called.

    If this file legitimately gains a statement, add its pattern above. That edit is
    the review the blacklist never forced.
    """
    return [s for s in _statements(sql) if not any(re.fullmatch(p, s) for p in allowed)]


def _sync_function_body() -> str:
    """The RAW text inside 144's ``$sync$ ... $sync$``, line structure intact.

    Taken from :func:`_lex`'s token stream rather than by a regex over the file, so
    the body this returns is exactly the one psql receives — the same decision about
    where the literal starts and ends, made once.
    """
    bodies = [
        chunk[len("$sync$") : -len("$sync$")]
        for kind, chunk in _lex(_sql())
        if kind == "string" and chunk.startswith("$sync$")
    ]
    assert len(bodies) == 1, f"expected exactly one $sync$ body in 144, found {len(bodies)}"
    return bodies[0]


def test_the_canonical_columns_are_added_statically():
    """The three ADDs must be STATIC, not dynamic SQL.

    Same reason the renames they replace were un-loop-ed: an
    ``EXECUTE format('... ADD COLUMN %I ...', ...)`` hides the column names behind
    placeholders, and the hermetic Feast guard
    (tests/unit/test_feature_repo/test_data_sources_columns_exist.py) models the
    canonical schema by TEXT-PARSING these files. Under a dynamic form it would
    not know business_metrics had gained triggers_delivered_count at all and would
    report the Feast source columns as "absent". That text IS the contract the
    static reader consumes, so assert the statements verbatim.
    """
    sql = _sql()
    for canonical in PAIRS.values():
        assert (
            f"ALTER TABLE public.business_metrics ADD COLUMN IF NOT EXISTS {canonical} INTEGER;"
            in sql
        ), (
            f"the {canonical} ADD is not statically declared — a dynamic ADD is "
            "invisible to the Feast guard's text parser"
        )
    assert not re.search(r"EXECUTE format\([^)]*ADD COLUMN", _executable_sql()), (
        "a column ADD went dynamic; keep committed schema additions statically declarable"
    )


def test_the_expand_renames_nothing():
    """THE HIGH-1 REGRESSION GUARD.

    A rename is what made 144 deploy-incompatible: it retires the legacy name at
    :956 while the containers that read and write it are still up, and the
    :1104 rollback restores those containers onto the migrated schema. The expand
    half must therefore contain no rename at all — of a table column or a view
    column. Retiring the legacy names is the deferred contract migration's job.
    """
    sql = _executable_sql()
    assert "RENAME COLUMN" not in sql.upper(), (
        "144 renames again — that is exactly the shape codex HIGH-1 rejected; the "
        "legacy names must survive this deploy so pre-lane containers keep working"
    )
    assert "ALTER VIEW" not in sql.upper(), (
        "the expand must not touch the split views: they are SELECT * snapshots "
        "with zero consumers, and recreating them is the contract migration's job"
    )


def test_every_legacy_value_is_backfilled_into_its_canonical_column():
    """Adding a column leaves it NULL on all 12,143 existing rollup rows. Without
    the backfill the new code reads NULL where a real count exists — a
    silently-wrong value, which is worse than the mislabel being fixed."""
    sql = _sql()
    for legacy, canonical in PAIRS.items():
        assert re.search(
            rf"UPDATE public\.business_metrics\s*\n?\s*SET\s+{canonical}\s*=\s*{legacy}\b",
            sql,
        ), f"no backfill of {legacy} -> {canonical}"
        # Re-running must touch no row: the WHERE clause has to exclude rows that
        # already agree, or the second application rewrites the whole table.
        assert f"{canonical} IS DISTINCT FROM {legacy}" in sql, (
            f"the {canonical} backfill has no is-distinct guard, so it is not a no-op on re-run"
        )


def test_a_bidirectional_sync_trigger_keeps_both_names_true():
    """Old containers write the legacy names, new containers write the canonical
    ones, and BOTH may be live against this schema (the deploy window, and
    permanently after a :1104 app-only rollback). The trigger is what stops either
    writer from leaving the other name stale.

    ultracode iter7 HIGH-4: the previous version of this test could not fail for any
    of the three ways the sync actually breaks. Measured, each on a mutated 144, each
    with the mutation itself verified to have landed:

    * the function's entire UPDATE branch deleted (6 ``OLD.`` references -> 0) — the
      path the per-HCP ETL's own ``ON CONFLICT DO UPDATE`` takes. GREEN: every
      ``NEW.x := NEW.y`` substring it looked for still existed, in the INSERT branch.
    * the trigger given ``WHEN (false)`` — never fires, catalog-identical on every
      field the prover then bound. GREEN: the regex sampled words, so extra ones were
      free.
    * the trigger rebound to a different function. GREEN: nothing tied the trigger to
      the function created ten lines above it.

    A substring is a PROXY for a behaviour. The three assertions below ask for the
    behaviour's structure instead: the trigger statement matched WHOLE (so a WHEN
    clause, an ``UPDATE OF`` column list, AFTER-instead-of-BEFORE, FOR EACH STATEMENT
    or another function's name all fail by construction), the body's lines matched
    against a closed set of forms, and the six assignments demanded of EACH branch
    separately. The capability itself — that a write to either name reaches the other
    — is settled where only a write can settle it, by ``SYNC_PROBE_SQL`` in the
    prover and the live certification.
    """
    sql = _sql()
    stmts = _statements(sql)

    # 1. The trigger, WHOLE. CREATE OR REPLACE TRIGGER is PostgreSQL 14+; the droplet
    #    runs 15.8 (confirmed 2026-09-18). It keeps the expand free of any DROP.
    triggers = [s for s in stmts if s.startswith("CREATE OR REPLACE TRIGGER")]
    assert len(triggers) == 1, f"expected exactly one trigger statement, found {len(triggers)}"
    assert re.fullmatch(_SYNC_TRIGGER_STATEMENT, triggers[0]), (
        "the sync trigger is not the exact statement this expand needs — check for a "
        f"WHEN clause, an UPDATE OF column list, or another function:\n{triggers[0]}"
    )

    # 2. The function it names, and nothing else, may supply the body.
    functions = [s for s in stmts if s.startswith("CREATE OR REPLACE FUNCTION")]
    assert len(functions) == 1, f"expected exactly one function statement, found {len(functions)}"
    assert functions[0].startswith(f"CREATE OR REPLACE FUNCTION public.{SYNC_FUNCTION}()"), (
        "the trigger is bound to a function this migration does not create"
    )

    # 3. Every line of the body is one of the forms a sync function needs.
    body = _sync_function_body()
    unapproved = [
        line
        for line in (raw.strip() for raw in body.splitlines())
        if line and not any(re.fullmatch(pattern, line) for pattern in _SYNC_BODY_LINE_ALLOWED)
    ]
    assert unapproved == [], (
        f"the sync body contains lines this trigger has no use for: {unapproved}"
    )

    # 4. BOTH branches, each carrying all six assignments. The INSERT branch fills in
    #    whichever side the writer left empty; the UPDATE branch follows the side that
    #    actually changed, which is what distinguishes an old-code write from a new one
    #    — so it must read OLD, and the INSERT branch must not (there is no OLD).
    branches = re.split(r"^\s*ELSE\s*$", body, flags=re.MULTILINE)
    assert len(branches) == 2, (
        "the sync body no longer has exactly one ELSE separating its INSERT branch "
        f"from its UPDATE branch (found {len(branches) - 1})"
    )
    on_insert, on_update = branches
    assert "IF TG_OP = 'INSERT' THEN" in on_insert, "the INSERT branch is not guarded by TG_OP"
    assert "OLD." not in on_insert, "the INSERT branch reads OLD, which does not exist on INSERT"
    assert on_update.count("OLD.") == 2 * len(PAIRS), (
        "the UPDATE branch must compare all six columns against OLD to tell which "
        f"side the writer moved; it makes {on_update.count('OLD.')} such comparisons"
    )
    for branch, which in ((on_insert, "INSERT"), (on_update, "UPDATE")):
        for legacy, canonical in PAIRS.items():
            assert f"NEW.{canonical} := NEW.{legacy};" in branch, (
                f"the {which} branch does not propagate {legacy} -> {canonical}"
            )
            assert f"NEW.{legacy} := NEW.{canonical};" in branch, (
                f"the {which} branch does not propagate {canonical} -> {legacy}"
            )


def test_the_expand_is_purely_additive():
    """Nothing may be dropped, deleted or truncated by the expand half — that is the
    entire safety property being bought, and 144 is the half ``deploy.yml`` applies
    UNATTENDED at :956 while the old containers are still serving.

    ultracode iter7: this used to be ``not re.search(r"\b(DROP|DELETE|TRUNCATE)\b")``
    — the verb blacklist that :func:`_unapproved_statements` documents, from two
    rounds of measured failure, as unable to be a capability check. Every payload
    that walked past the rollback's blacklist walks past a blacklist here for the
    same reason, and the allowlist refuses each of them by construction; the
    dedicated exercise is in :func:`test_the_statement_allowlist_cannot_be_walked_past`.
    That the guarded file was the hand-applied one while the auto-applied one kept
    the weaker check is the part worth remembering: the strength of a guard should
    follow how the change reaches production, not which file was reviewed first.

    ``CREATE OR REPLACE`` is used for the function and the trigger precisely so that
    not even a DROP TRIGGER appears, and :data:`_EXPAND_ALLOWED` has no DROP form at
    all — so a DROP cannot enter this file without the allowlist being edited, which
    is the review a blacklist never forced.
    """
    unapproved = _unapproved_statements(_sql(), _EXPAND_ALLOWED)
    assert unapproved == [], f"144 contains statements the expand does not need: {unapproved}"

    # ...and the allowlist must be EXERCISED per subject, not per form: it constrains
    # what MAY appear and never what MUST, so without this 144 could shrink to a bare
    # NOTIFY and still pass. The ADDs and the backfills are asserted by subject in
    # their own tests; here it is the two statements that have no other owner.
    stmts = _statements(_sql())
    for pattern, want, what in (
        (r"CREATE OR REPLACE FUNCTION\b", 1, "the sync function"),
        (r"CREATE OR REPLACE TRIGGER\b", 1, "the sync trigger"),
        (r"NOTIFY\b", 1, "the PostgREST schema reload"),
    ):
        got = sum(1 for s in stmts if re.match(pattern, s))
        assert got == want, f"expected {want} statement installing {what}, found {got}"

    # The migration must not own its own transaction: scripts/run_migrations.sh wraps
    # the file with psql --single-transaction, and a self-managed COMMIT would split
    # the schema change from the ledger row that records it.
    assert not re.search(r"^\s*COMMIT\s*;", _executable_sql(), re.IGNORECASE | re.MULTILINE)
    assert "NOTIFY pgrst, 'reload schema';" in _sql()


def test_both_the_canonical_and_the_legacy_columns_are_documented():
    """The canonical columns say what they count; the legacy ones say they are
    deprecated aliases kept alive by the trigger until the contract migration, so
    the next reader of ``\\d business_metrics`` is not left guessing which is real."""
    sql = _sql()
    for legacy, canonical in PAIRS.items():
        assert f"COMMENT ON COLUMN public.business_metrics.{canonical} IS" in sql
        assert f"COMMENT ON COLUMN public.business_metrics.{legacy} IS" in sql
    assert sql.upper().count("DEPRECATED") >= len(PAIRS)


def test_the_rollback_removes_exactly_what_the_expand_added():
    """The rollback is no longer part of deploy recovery — that was the rename's
    problem and the expand does not have it. It exists to return the schema to its
    pre-144 shape, so it drops the three ADDED columns, the trigger and the
    function, and touches neither the legacy columns nor their data.
    """
    rollback = ROLLBACK.read_text()
    for legacy, canonical in PAIRS.items():
        assert (
            f"ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS {canonical};" in rollback
        ), f"{canonical} is not removed by the rollback"
        assert not re.search(rf"DROP COLUMN[^;]*\b{legacy}\b", rollback), (
            f"the rollback drops the LEGACY column {legacy} — it must never touch the "
            "columns that carry the pre-lane data"
        )
    assert "DROP TRIGGER IF EXISTS" in rollback
    assert f"DROP FUNCTION IF EXISTS public.{SYNC_FUNCTION}()" in rollback

    # EVERY statement must be one of the forms this rollback is known to need. The one
    # DELETE among them is against the migration ledger: the rollback retires its own
    # schema_migrations row in the same transaction as the schema change (codex iter3
    # HIGH-2). Doing it as a second psql invocation left a window where the columns
    # were gone while the runner still believed 144 was applied, so the next deploy
    # would skip re-applying it.
    #
    # This is an ALLOWLIST because two rounds of blacklisting destructive spellings
    # both failed (codex iter3 HIGH-2 narrowed a ban that iter4 then walked through;
    # iter4's replacement counted the DELETE/TRUNCATE verb and iter5 walked through
    # THAT with DROP TABLE, ALTER TABLE DROP COLUMN and a concatenated 'DE'||'LETE').
    # See :func:`_unapproved_statements` for why no blacklist here can be anything but
    # a proxy.
    assert _unapproved_statements(rollback) == [], _unapproved_statements(rollback)

    # ...and the allowlist must be EXERCISED, per subject, not per form. An allowlist
    # constrains what MAY appear and never what MUST, so without this the rollback
    # could shrink to a single NOTIFY and still pass.
    #
    # codex iter6 HIGH-4: the first version of this counted distinct FORMS, and the
    # COMMENT form is generic over the three legacy columns. Deleting the nrx_count
    # and total_rx_count comments left all six forms present and the test green, while
    # the rollback would leave two columns still labelled "DEPRECATED" after 144 was
    # removed -- the mislabelling this whole lane exists to end. Count the SUBJECTS.
    stmts = _statements(rollback)
    dropped = {
        mm.group(1)
        for mm in (re.match(r"ALTER TABLE \S+ DROP COLUMN IF EXISTS (\w+)", s) for s in stmts)
        if mm
    }
    assert dropped == set(PAIRS.values()), (
        f"the rollback drops {sorted(dropped)}; it must drop every column 144 added"
    )
    commented = {
        mm.group(1)
        for mm in (
            re.match(r"COMMENT ON COLUMN public\.business_metrics\.(\w+) IS", s) for s in stmts
        )
        if mm
    }
    assert commented == set(PAIRS), (
        f"the rollback restores the comment on {sorted(commented)}; it must restore all "
        "three legacy columns' comments, or removing 144 leaves them reading DEPRECATED"
    )

    # ...and the restored text must actually STOP saying it (ultracode iter7 MED). The
    # assertion above checks WHICH columns are re-commented and the message it fails
    # with names the condition it never tested: a rollback that re-COMMENTed all three
    # with 144's own DEPRECATED wording passed, and left a rolled-back schema
    # advertising a deprecation, a sync trigger and a migration 146 that no longer
    # exist. The subject is not the property; assert the property.
    restored = {
        mm.group(1): mm.group(2)
        for mm in (
            re.match(rf"COMMENT ON COLUMN public\.business_metrics\.(\w+) IS ({_LITERAL})", s)
            for s in stmts
        )
        if mm
    }
    for column, text in restored.items():
        assert "DEPRECATED" not in text.upper(), (
            f"the rollback leaves {column} labelled DEPRECATED — after 144 is removed there "
            "is nothing to be deprecated in favour of, and no trigger keeping a canonical "
            "column in sync"
        )
        assert SYNC_FUNCTION not in text, (
            f"the rollback's comment on {column} still points at {SYNC_FUNCTION}, which it "
            "has just dropped"
        )
    for pattern, want, what in (
        (r"DROP TRIGGER\b", 1, "the sync trigger"),
        (r"DROP FUNCTION\b", 1, "the sync function"),
        (r"DELETE FROM public\.schema_migrations\b", 1, "its own ledger row"),
        (r"NOTIFY\b", 1, "the PostgREST schema reload"),
    ):
        got = sum(1 for s in stmts if re.match(pattern, s))
        assert got == want, f"expected {want} statement retiring {what}, found {got}"

    # ...and the flag that makes "in the same transaction" true (codex iter4 MED-1).
    # The ledger DELETE following every schema statement (only NOTIFY comes after
    # it) is only atomicity if psql is told
    # to wrap the file in one transaction; without the flag it commits statement by
    # statement and the DROPs can land while the ledger row survives -- the very
    # split the DELETE was moved here to close. It must be in the APPLY COMMAND:
    # this header ALSO explains the flag in prose two paragraphs above, so a
    # substring check over the header passes while the command has lost it (that
    # exact proxy was codex iter3 LOW-2 on the contract migration). Reconstruct the
    # command from its comment lines -- it wraps across two with a backslash -- and
    # look inside it.
    command_text = " ".join(re.sub(r"^--\s?", "", ln) for ln in rollback.splitlines())
    assert re.search(
        r"docker exec.*?psql.*?--single-transaction.*?<\s*database/migrations/rollback_144",
        command_text,
    ), (
        "the documented apply COMMAND does not use --single-transaction, so nothing "
        "couples the ledger DELETE to the schema change it records"
    )


def test_the_statement_allowlist_cannot_be_walked_past():
    """THE GUARD ON THE GUARD.

    Every payload below is valid PostgreSQL that destroys or exfiltrates data, and
    every one of them passed some earlier version of this check (codex iter3 HIGH-2,
    iter4 HIGH-2, iter5 HIGH-1, iter6 HIGH-1 -- four rounds, because each fix closed
    the examples it was shown instead of the class). They live here rather than in a
    reviewer's scratch directory so the next person to touch :func:`_statements` or
    :data:`_ROLLBACK_ALLOWED` finds out immediately.

    The last two matter most: they are not "extra statements" but attacks on the
    PARSER, which is what a text-based guard really rests on.
    """
    base = ROLLBACK.read_text()
    # The loop below treats a lexer refusal as a rejection, which is right for a
    # payload and catastrophic for the base: if rollback_144 itself ever tripped a
    # lexer guard, EVERY payload would take that branch and the whole test would pass
    # while asserting nothing. Establish first, outside the try, that the real file
    # parses and is approved — so a vacuous run is a failure rather than a green.
    assert _unapproved_statements(base) == [], (
        "rollback_144 does not itself pass the allowlist, so every payload below would "
        "be 'rejected' for the wrong reason and this test would go vacuous"
    )

    payloads = {
        "DROP TABLE": "DROP TABLE public.business_metrics;",
        "ALTER TABLE ... DROP COLUMN": (
            "ALTER TABLE public.business_metrics DROP COLUMN metric_id;"
        ),
        "DELETE inside a CTE": (
            "WITH e AS (DELETE FROM public.business_metrics RETURNING metric_id) SELECT 1;"
        ),
        "DELETE split across lines": "DELETE\nFROM public.business_metrics;",
        "TRUNCATE": "TRUNCATE public.business_metrics;",
        "COPY out to a file": "COPY public.business_metrics TO '/tmp/x.csv';",
        "UPDATE zeroing the data": "UPDATE public.business_metrics SET trx_count = 0;",
        "statement hidden after a real literal": (
            "COMMENT ON COLUMN public.business_metrics.trx_count IS 'x';"
            " DROP TABLE public.business_metrics;"
        ),
        # PARSER attacks: a '--' the server reads as literal content, and a DELETE
        # spelled by concatenation inside a dollar-quoted body.
        "'--' inside a string literal": (
            "COMMENT ON COLUMN public.business_metrics.trx_count IS 'safe -- harmless';\n"
            "DROP TABLE public.business_metrics;\nSELECT ';\n-- ';"
        ),
        "dollar-quoted dynamic DELETE": (
            "DO $x$ BEGIN EXECUTE 'DE' || 'LETE FROM public.business_metrics'; END $x$;"
        ),
        "unterminated literal swallowing the rest": (
            "COMMENT ON COLUMN public.business_metrics.trx_count IS 'oops;"
        ),
        # ...and the same three attacks one comment syntax over (ultracode iter7
        # HIGH-2). The allowlist survived these even before the lexer knew what `/*`
        # was — the `/*` residue always landed outside the six anchored patterns, which
        # is the allowlist's whole point — but 144's ABSENCE-based refusals did not,
        # and nothing here would have noticed the day this file gained one.
        "statement hidden behind a block comment's close": (
            "/* rehearsal note\n-- */ DROP TABLE public.business_metrics;"
        ),
        "statement wrapped so only the guard sees a comment": (
            "/* -- */ TRUNCATE public.business_metrics;"
        ),
        "unterminated block comment swallowing the rest": (
            "/* never closed\nDROP TABLE public.business_metrics;"
        ),
    }
    for name, payload in payloads.items():
        try:
            unapproved = _unapproved_statements(f"{base}\n{payload}\n")
        except AssertionError:
            continue  # refused by the lexer's own guards, which is a rejection too
        assert unapproved, f"the allowlist accepts a rollback containing: {name}"

    # ...and it must not cry wolf: a COMMENT whose TEXT names a forbidden statement is
    # not that statement. The iter4 detector failed this, which is how a guard gets
    # weakened -- a check that fails on correct input is one someone will loosen.
    harmless = (
        f"{base}\nCOMMENT ON COLUMN public.business_metrics.nrx_count IS "
        "'Never DELETE FROM public.business_metrics';\n"
    )
    assert _unapproved_statements(harmless) == []
    assert _statements("COMMENT ON COLUMN public.t.c IS 'a -- b';") == [
        "COMMENT ON COLUMN public.t.c IS 'a -- b'"
    ], "a legitimate '--' inside a literal is being eaten as a comment"


def test_the_rollback_is_never_applied_as_a_forward_migration():
    """The recovery file undoes 144. If the forward runner picked it up, applying
    migrations would add the columns and immediately drop them — a migration that
    silently does nothing. The runner's exclusion is what makes shipping the file
    safe, so assert the runner still excludes this NAME rather than trusting the
    convention.
    """
    runner = (REPO / "scripts" / "run_migrations.sh").read_text()
    patterns = re.search(r"^\s*(\*_validation_queries\.sql\|[^)]*)\)\s*continue", runner, re.M)
    assert patterns, "run_migrations.sh no longer has the skip-case this file relies on"
    globs = patterns.group(1).split("|")
    assert "rollback_*.sql" in globs, globs
    assert ROLLBACK.name.startswith("rollback_") and ROLLBACK.name.endswith(".sql")


def test_the_comment_lexer_cannot_be_walked_past():
    """THE GUARD ON THE OTHER GUARD — the one the allowlist does not cover.

    :func:`_unapproved_statements` protects the rollback, and it is walk-proof by
    construction: an unapproved statement fails whatever it is called. The EXPAND's
    refusals (``test_the_expand_renames_nothing``, ``test_the_expand_is_purely_additive``)
    rest on something weaker -- ``X not in _executable_sql()`` -- and an absence
    assertion is only as strong as the lexer that decides what "executable" means.
    144 is also the half ``deploy.yml`` applies UNATTENDED, so this is the guard that
    has to hold.

    PostgreSQL has TWO comment syntaxes and the lexer knew one (ultracode iter7 HIGH-2).
    Every payload below is a shape where the lexer and the server disagree about what
    is code -- in BOTH directions, because a guard that fires on a correct file gets
    loosened by the next maintainer just as surely as one that misses a hostile file.
    """
    # 1. FAIL-OPEN. `*/` parked after a `--` on the same line: the server ends the
    #    block comment there and runs the statement behind it; a `--`-only lexer
    #    deletes to end of line and never sees it. Measured: this made
    #    test_the_expand_renames_nothing PASS on a 144 that renames trx_count.
    hidden = (
        "/* rehearsal note: re-enable after the window\n"
        "-- */ ALTER TABLE public.business_metrics"
        " RENAME COLUMN trx_count TO triggers_delivered_count;\n"
    )
    assert "RENAME COLUMN" in _strip_comments(hidden).upper(), (
        "a statement hidden behind a block comment's closing */ is invisible to the "
        "lexer while psql executes it — the expand's renames-nothing guard is blind"
    )

    # 2. FAIL-CLOSED. A legitimate block comment that NAMES a forbidden shape is not
    #    that shape — the whole reason the refusals are asked of the executable text.
    prose = "/* Do NOT add an ALTER TABLE ... RENAME COLUMN here — see the header. */\n"
    assert "RENAME COLUMN" not in _strip_comments(prose).upper(), (
        "a block comment's prose is being read as executable SQL, so a correct file "
        "fails the refusal guards"
    )

    # 3. PARITY. An apostrophe inside a block comment is prose, not a string opener.
    #    Reading it as one flips the lexer's string state for the REST OF THE FILE:
    #    every later `--` stops being stripped and every later statement boundary moves.
    apostrophe = (
        "/* we don't rebuild the split views here. */\n"
        "-- this line must still be stripped\n"
        "ALTER TABLE public.business_metrics ADD COLUMN IF NOT EXISTS x INTEGER;\n"
    )
    stripped = _strip_comments(apostrophe)
    assert "must still be stripped" not in stripped, (
        "an apostrophe inside a block comment flipped the string parity, so later "
        "`--` comments stopped being comments"
    )
    assert _statements(apostrophe) == [
        "ALTER TABLE public.business_metrics ADD COLUMN IF NOT EXISTS x INTEGER"
    ], _statements(apostrophe)

    # 4. The server's block comments NEST; a `/*` inside one does not close it early.
    nested = "/* outer /* inner */ still a comment */ SELECT 1;\n"
    assert _statements(nested) == ["SELECT 1"], _statements(nested)

    # 5. ...and a comment opener inside a STRING is content, exactly as `--` is.
    in_literal = "COMMENT ON COLUMN public.t.c IS 'a /* b */ c';"
    assert _statements(in_literal) == ["COMMENT ON COLUMN public.t.c IS 'a /* b */ c'"], (
        "a /* inside a string literal is being treated as a comment opener"
    )

    # 6. A comment is WHITESPACE, not nothing: removing it must not weld two tokens
    #    into one the server never sees.
    assert _statements("SELECT/**/1;") == ["SELECT 1"], _statements("SELECT/**/1;")

    # 7. An unterminated block comment is refused rather than guessed at — it would
    #    otherwise swallow every statement after it, which is the `'oops;` failure
    #    one syntax over.
    try:
        _statements("/* never closed\nDROP TABLE public.business_metrics;\n")
    except AssertionError:
        pass
    else:
        raise AssertionError("an unterminated /* block comment was not refused")
