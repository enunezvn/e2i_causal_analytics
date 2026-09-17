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
later, by hand, by ``database/deferred/146_*`` — see
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


def _strip_comments(text: str) -> str:
    """Drop ``--`` line comments. See :func:`_executable_sql` for why every refusal
    is asked of the executable text rather than the raw file."""
    return re.sub(r"--.*$", "", text, flags=re.M)


#: Every statement ``rollback_144`` is ALLOWED to contain, as an anchored pattern.
#: Whitespace is collapsed before matching, so a statement may wrap across lines.
_ROLLBACK_ALLOWED = (
    r"DROP TRIGGER IF EXISTS business_metrics_sync_legacy_trigger_counts_trg "
    r"ON public\.business_metrics",
    r"DROP FUNCTION IF EXISTS public\.business_metrics_sync_legacy_trigger_counts\(\)",
    r"ALTER TABLE public\.business_metrics DROP COLUMN IF EXISTS "
    r"(?:triggers_delivered_count|triggers_accepted_count|triggers_total_count)",
    r"COMMENT ON COLUMN public\.business_metrics\."
    r"(?:trx_count|nrx_count|total_rx_count) IS '[^']*'",
    r"DELETE FROM public\.schema_migrations "
    r"WHERE filename = '144_per_hcp_trigger_count_columns\.sql'",
    r"NOTIFY pgrst, 'reload schema'",
)


def _statements(sql: str) -> list[str]:
    """The file's executable statements, comments stripped and whitespace collapsed.

    The split is quote-aware. A naive ``body.split(";")`` is wrong here and this file
    proves it: ``COMMENT ON COLUMN ... IS '... (misnamed by migration 033; never
    prescriptions)'`` carries a semicolon INSIDE the literal, so the naive form cut a
    valid statement in two and reported both halves as unapproved. A splitter that
    mis-parses the file it guards is not a guard -- it fails on correct input and, on
    hostile input, an attacker who controls a string literal controls where the
    statement boundaries appear to be.

    Dollar-quoting is refused rather than parsed: ``$`` bodies would need real
    nesting rules, and a ``DO $x$ ... $x$`` block or an inline function body is an
    unapproved statement form here anyway, so the assertion and :data:`_ROLLBACK_ALLOWED`
    agree about what this file may contain. An unterminated literal is refused for the
    same reason -- it would silently swallow every statement after it.
    """
    body = _strip_comments(sql)
    assert "$" not in body, (
        "the rollback contains dollar-quoting; a DO block or a function body is not "
        "an approved statement form here, and it would also break the statement split"
    )
    out: list[str] = []
    cur: list[str] = []
    in_str = False
    i = 0
    while i < len(body):
        ch = body[i]
        if in_str:
            if ch == "'" and body[i + 1 : i + 2] == "'":  # '' is an escaped quote
                cur.append("''")
                i += 2
                continue
            if ch == "'":
                in_str = False
            cur.append(ch)
        elif ch == "'":
            in_str = True
            cur.append(ch)
        elif ch == ";":
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
        i += 1
    assert not in_str, "the rollback has an unterminated string literal"
    out.append("".join(cur))
    return [re.sub(r"\s+", " ", s).strip() for s in out if s.strip()]


def _unapproved_statements(sql: str) -> list[str]:
    """Statements in the rollback that match none of :data:`_ROLLBACK_ALLOWED`.

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
    return [s for s in _statements(sql) if not any(re.fullmatch(p, s) for p in _ROLLBACK_ALLOWED)]


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
    writer from leaving the other name stale."""
    sql = _sql()
    assert f"CREATE OR REPLACE FUNCTION public.{SYNC_FUNCTION}()" in sql
    # CREATE OR REPLACE TRIGGER is PostgreSQL 14+; the droplet runs 15.8
    # (confirmed 2026-09-18). It keeps the expand free of any DROP.
    assert re.search(
        r"CREATE OR REPLACE TRIGGER\s+\w+\s*\n?\s*BEFORE INSERT OR UPDATE ON public\.business_metrics",
        sql,
    ), "no BEFORE INSERT OR UPDATE row trigger on business_metrics"
    assert "FOR EACH ROW" in sql
    for legacy, canonical in PAIRS.items():
        assert f"NEW.{canonical} := NEW.{legacy}" in sql, f"no {legacy} -> {canonical} propagation"
        assert f"NEW.{legacy} := NEW.{canonical}" in sql, f"no {canonical} -> {legacy} propagation"


def test_the_expand_is_purely_additive():
    """Nothing may be dropped, deleted or truncated by the expand half — that is
    the entire safety property being bought. ``CREATE OR REPLACE`` is used for the
    function and the trigger precisely so that not even a DROP TRIGGER appears."""
    sql = _executable_sql()
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE)\b", sql, re.IGNORECASE)
    assert not re.search(r"^\s*COMMIT\s*;", sql, re.IGNORECASE | re.MULTILINE)
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
    # ...and the allowlist must actually be exercised: every approved form present.
    # Without this the test would still pass if the rollback were emptied to a single
    # NOTIFY -- an allowlist constrains what MAY appear, never what MUST.
    kinds = {
        next(i for i, p in enumerate(_ROLLBACK_ALLOWED) if re.fullmatch(p, s))
        for s in _statements(rollback)
    }
    assert kinds == set(range(len(_ROLLBACK_ALLOWED))), (
        f"the rollback no longer uses every approved statement form: {sorted(kinds)}"
    )

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
