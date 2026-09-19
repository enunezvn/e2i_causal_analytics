"""Migration 146: the CONTRACT half of the per_hcp_rollup column expand/contract.

144 adds ``business_metrics.triggers_{delivered,accepted,total}_count`` beside the
legacy ``{trx,nrx,total_rx}_count`` and keeps both true with a row trigger. 146
retires the legacy three.

(It is 146, not 145: ``database/migrations/145_drop_trx_share_patient_axis_variants.sql``
is already on origin/main and applied live. Different directories, so the runner keys
the two apart today — but the intended end state moves this file into
``database/migrations/``, where a duplicate number makes apply order ambiguous.
``test_the_contract_number_collides_with_no_other_migration`` pins both halves of
that.)

THE WHOLE POINT OF THIS MODULE IS THAT 146 MUST NOT RUN IN THE SAME DEPLOY AS 144.
``scripts/run_migrations.sh`` applies EVERY pending forward ``*.sql`` in each of its
``MIGRATION_DIRS`` in one pass, so a ``database/migrations/146_*.sql`` committed
beside 144 would be applied seconds after it — the legacy columns would be gone
before a single container was replaced, and the deploy would be exactly as unsafe
as the rename that codex HIGH-1 rejected. Expand/contract is only expand/contract
if the two halves land in two different deploys.

The separation is STRUCTURAL rather than a naming convention: 146 lives in
``database/deferred/``, a directory the runner's ``MIGRATION_DIRS`` does not list,
so no filename typo and no new skip-pattern can arm it. These tests pin that
directory out of the runner's scope, and pin the runner's scope as the ONE source
of it (the runner's own header: "MIGRATION_DIRS is the only source of that scope
-- do not restate the dir count here, it has gone stale twice").
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
RUNNER = REPO / "scripts" / "run_migrations.sh"
DEFERRED_DIR = REPO / "database" / "deferred"
CONTRACT = DEFERRED_DIR / "146_drop_legacy_per_hcp_count_columns.sql"
EXPAND = REPO / "database" / "migrations" / "144_per_hcp_trigger_count_columns.sql"
LEGACY = ("trx_count", "nrx_count", "total_rx_count")
CANONICAL = ("triggers_delivered_count", "triggers_accepted_count", "triggers_total_count")
#: split view -> the ``data_split`` value it must keep selecting. Read off the LIVE
#: definitions on 2026-09-18 rather than inferred from the view names:
#: ``pg_get_viewdef`` shows each of the four filtering on
#: ``data_split = '<its own split>'::data_split_type``.
VIEWS = {
    "v_train_business_metrics": "train",
    "v_test_business_metrics": "test",
    "v_validation_business_metrics": "validation",
    "v_holdout_business_metrics": "holdout",
}


def _runner_dirs() -> list[str]:
    """The ``$PROJECT_ROOT``-relative directories run_migrations.sh applies.

    Parsed from the script rather than restated here: a restated list is a PROXY
    for the runner's scope and is satisfiable while the real scope has moved. The
    runner's own header says the array is the only source of it.
    """
    body = RUNNER.read_text()
    block = re.search(r"^MIGRATION_DIRS=\((.*?)^\)", body, re.M | re.S)
    assert block, "run_migrations.sh no longer declares MIGRATION_DIRS as an array literal"
    dirs = re.findall(r'"\$PROJECT_ROOT/([^":]+)::', block.group(1))
    assert dirs, f"parsed no directories out of MIGRATION_DIRS: {block.group(1)!r}"
    return dirs


def test_the_runner_scope_parser_is_not_vacuous():
    """A parser that silently found nothing would pass every assertion below."""
    dirs = _runner_dirs()
    assert "database/migrations" in dirs, dirs
    assert len(dirs) >= 8, dirs


def test_the_contract_migration_is_outside_every_directory_the_runner_applies():
    """THE GUARD. If ``database/deferred`` ever joins MIGRATION_DIRS — or 146 is
    moved into a directory already on it — the contract half starts shipping in
    the same deploy as the expand half, and the deploy-safety property bought by
    codex HIGH-1 is silently gone."""
    assert CONTRACT.exists(), f"missing contract migration: {CONTRACT}"
    applied = {(REPO / d).resolve() for d in _runner_dirs()}
    assert CONTRACT.parent.resolve() not in applied, (
        f"{CONTRACT.name} sits in a directory the migration runner applies — it "
        "would run in the same deploy as 144 and drop the legacy columns while the "
        "pre-lane containers still read them"
    )
    assert EXPAND.parent.resolve() in applied, (
        "the EXPAND half must still be auto-applied; only the contract is deferred"
    )


def test_the_contract_number_collides_with_no_other_migration():
    """The contract was first written as 145 — and 145 was ALREADY TAKEN by
    ``database/migrations/145_drop_trx_share_patient_axis_variants.sql``, which is
    on origin/main and applied in the live database. Nothing caught it: the two
    files sit in different directories, so the runner keys them differently and no
    test compared numbers. It surfaced only from reading the live
    ``schema_migrations`` ledger.

    It matters because the intended end state (codex iter2 MED-1) is to MOVE this
    file into ``database/migrations/`` in a later PR, where a duplicate number
    makes the apply order ambiguous and the history unreadable. Guard the number
    across every directory under ``database/``, not just this one.
    """
    number = CONTRACT.name.split("_", 1)[0]
    assert number.isdigit(), CONTRACT.name
    clashes = [
        q.relative_to(REPO).as_posix()
        for q in sorted((REPO / "database").rglob(f"{number}_*.sql"))
        if q.resolve() != CONTRACT.resolve()
    ]
    assert not clashes, (
        f"migration number {number} is already used by {clashes} — renumber the "
        "contract before it is moved into database/migrations/"
    )
    # ...and the guard must be able to SEE a clash: the file this contract WOULD
    # have collided with is still there, under its own number.
    #
    # codex iter4 LOW-1: this used to be `assert sorted(database.rglob("145_*.sql"))`
    # -- a PROXY. It asks "does any file numbered 145 exist anywhere under
    # database/?", when the condition it claims to prove is "the specific migration
    # that made 145 unusable is still on main". Deleting
    # 145_drop_trx_share_patient_axis_variants.sql and dropping any unrelated
    # 145_dummy.sql anywhere under database/ satisfied the old form while the trap it
    # documents had evaporated. Pin the path.
    collision = REPO / "database" / "migrations" / "145_drop_trx_share_patient_axis_variants.sql"
    assert collision.exists(), (
        f"{collision.relative_to(REPO)} is gone -- that file (already on origin/main "
        "and applied live) is WHY this contract is numbered 146 rather than 145. If it "
        "really was removed, re-derive the next free number instead of deleting this "
        "assertion: the guard below is only meaningful while a real collision exists."
    )


def test_the_contract_retires_the_legacy_columns_and_the_sync_machinery():
    sql = CONTRACT.read_text()
    for legacy in LEGACY:
        assert f"ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS {legacy};" in sql, (
            f"{legacy} is not retired"
        )
    for canonical in CANONICAL:
        assert not re.search(rf"DROP COLUMN[^;]*\b{canonical}\b", sql), (
            f"the contract drops the CANONICAL column {canonical}"
        )
    assert "DROP TRIGGER IF EXISTS" in sql, "the sync trigger outlives the columns it syncs"
    assert "DROP FUNCTION IF EXISTS public.business_metrics_sync_legacy_trigger_counts()" in sql


def _recreates_its_own_split(sql: str, view: str, split: str) -> bool:
    """Whether ``sql`` recreates ``view`` as ``SELECT *`` filtered to ``split``.

    The literal may carry the ``::data_split_type`` cast ``pg_get_viewdef`` prints for
    the enum column (ultracode iter8 HIGH-b) -- both forms select the same rows.
    """
    return (
        re.search(
            rf"CREATE OR REPLACE VIEW public\.{view} AS\s*\n?\s*"
            rf"SELECT \* FROM public\.business_metrics WHERE data_split = '{split}'"
            rf"(?:::(?:public\.)?data_split_type)?;",
            sql,
        )
        is not None
    )


def test_the_contract_recreates_the_split_views():
    """``DROP COLUMN`` fails while a view depends on the column, and all four
    ``SELECT *`` split views expand to the legacy names. They are dropped and
    recreated around the ALTERs — never dropped with CASCADE, which would take the
    views away and leave nothing behind."""
    sql = CONTRACT.read_text()
    assert not re.search(r"\bCASCADE\b", sql, re.IGNORECASE), (
        "CASCADE would silently drop the dependent views instead of recreating them"
    )
    for view, split in VIEWS.items():
        assert f"DROP VIEW IF EXISTS public.{view};" in sql, view
        # ...and recreated selecting ITS OWN split. The predicate is the only thing
        # that distinguishes these four views from each other and from the table, and
        # the previous version of this assertion stopped at
        # `SELECT * FROM public.business_metrics` (ultracode iter7 MED): a contract
        # that rebuilt all four as `WHERE data_split = 'train'` passed this test, and
        # passed prove_state's contract_state too, which counts four views. Three
        # consumers would then have been silently served the training split — the
        # holdout view, whose entire purpose is to be the split nothing trains on,
        # reading the split everything trains on.
        assert _recreates_its_own_split(sql, view, split), (
            f"{view} is not recreated as the {split!r} split"
        )
        assert sql.index(f"DROP VIEW IF EXISTS public.{view};") < sql.index(
            f"CREATE OR REPLACE VIEW public.{view}"
        ), f"{view} is recreated before it is dropped"

    # The four predicates must also be DISTINCT — the assertion above is per view, and
    # a mapping that repeated a split would satisfy it four times over only if VIEWS
    # itself were wrong. Pin that here rather than trusting the constant.
    assert len(set(VIEWS.values())) == len(VIEWS), f"VIEWS repeats a split: {VIEWS}"


def test_the_split_view_assertion_accepts_the_cast_pg_get_viewdef_emits():
    """``data_split`` is the ``data_split_type`` ENUM, so ``pg_get_viewdef`` prints
    each live predicate as ``data_split = 'train'::data_split_type`` (read off the
    live ``v_train_business_metrics`` on 2026-09-19). A maintainer who copies that
    form into 146 has written a CORRECT contract; a guard that rejects it invites
    being loosened (ultracode iter8 HIGH-b). The cast must be accepted -- and only
    for the view's OWN split, so the per-view teeth survive the relaxation."""
    sql = CONTRACT.read_text()
    for view, split in VIEWS.items():
        cast_form = sql.replace(
            f"WHERE data_split = '{split}';", f"WHERE data_split = '{split}'::data_split_type;"
        )
        assert cast_form != sql, f"{view}: the rewrite did not land"
        assert _recreates_its_own_split(cast_form, view, split), (
            f"{view}: the ::data_split_type form pg_get_viewdef emits was rejected"
        )
        wrong = next(s for s in VIEWS.values() if s != split)
        crossed = sql.replace(
            f"WHERE data_split = '{split}';", f"WHERE data_split = '{wrong}'::data_split_type;"
        )
        assert not _recreates_its_own_split(crossed, view, split), (
            f"{view}: a cast predicate selecting {wrong!r} was accepted"
        )


def test_the_contract_is_ordered_drop_views_then_columns_then_recreate():
    sql = CONTRACT.read_text()
    last_view_drop = max(sql.index(f"DROP VIEW IF EXISTS public.{v};") for v in VIEWS)
    first_col_drop = min(
        sql.index(f"ALTER TABLE public.business_metrics DROP COLUMN IF EXISTS {c};") for c in LEGACY
    )
    first_view_create = min(sql.index(f"CREATE OR REPLACE VIEW public.{v}") for v in VIEWS)
    assert last_view_drop < first_col_drop < first_view_create, (
        "the contract must drop every dependent view, then the columns, then rebuild the views"
    )


def test_the_contract_records_itself_in_the_migration_ledger_atomically():
    """Nothing else writes a ``schema_migrations`` row for this file — the runner
    never sees it (that is the point of ``database/deferred/``). An earlier draft
    left the INSERT as a second, separate command in the header, which can be
    forgotten or fail on its own and leave the ledger disagreeing with the schema.
    It now follows every schema statement in the file (only ``NOTIFY`` comes after
    it), so under the header's own ``--single-transaction`` apply command it commits
    with the change it records, or not at all.

    codex iter3 LOW-2: an earlier version of this test asserted the ordering but
    NOT the flag that makes the ordering matter — deleting ``--single-transaction``
    from the documented command left every assertion green, while the file would
    then apply statement-by-statement and could commit the DROPs without the ledger
    row. The flag is the atomicity; it is asserted here.
    """
    sql = CONTRACT.read_text()
    m = re.search(
        r"INSERT INTO public\.schema_migrations\(filename\)\s*\n?\s*"
        r"VALUES \('deferred/146_drop_legacy_per_hcp_count_columns\.sql'\)",
        sql,
    )
    assert m, "the contract does not record itself in public.schema_migrations"
    assert "ON CONFLICT DO NOTHING" in sql, "re-applying it would raise on the ledger row"

    # THE flag: without it psql commits each statement on its own and the ledger row
    # is no longer coupled to the change. It must be in the APPLY COMMAND, not merely
    # somewhere in the header — the header also explains the flag in prose, and a
    # substring check over the whole header is satisfied by that prose while the
    # command itself has lost the flag. Reconstruct the command from its comment
    # lines (it wraps across two with a backslash) and look inside it.
    head = sql[: sql.index("DROP VIEW")]
    command_text = " ".join(re.sub(r"^--\s?", "", ln) for ln in head.splitlines())
    assert re.search(
        r"docker exec.*?psql.*?--single-transaction.*?<\s*database/deferred/146", command_text
    ), (
        "the documented apply COMMAND does not use --single-transaction, so nothing "
        "couples the ledger row to the schema change it records"
    )

    # The INSERT must come after EVERY schema statement, not merely after one of
    # them; NOTIFY is the only thing allowed to follow.
    schema_stmts = [
        mm.start()
        for mm in re.finditer(
            r"^\s*(ALTER TABLE|DROP VIEW|DROP TRIGGER|DROP FUNCTION|CREATE OR REPLACE VIEW)",
            sql,
            re.M,
        )
    ]
    assert schema_stmts, "no schema statements found — the parser, not the file, is wrong"
    assert m.start() > max(schema_stmts), (
        "the ledger row is written before some schema change; a failure in between "
        "would leave the ledger claiming work that did not happen"
    )
    tail = sql[m.end() :]
    leftovers = [
        line.strip()
        for line in tail.splitlines()
        if line.strip() and not line.strip().startswith("--")
    ]
    assert all(ln.startswith(("ON CONFLICT", "VALUES", "NOTIFY", ")", ";")) for ln in leftovers), (
        f"unexpected statements after the ledger row: {leftovers}"
    )


def test_the_contract_carries_its_own_manual_apply_instructions():
    """Nothing applies this file automatically, so the file itself has to say who
    applies it and what must be true first — otherwise it rots in the tree and the
    legacy columns live forever."""
    sql = CONTRACT.read_text()
    head = sql[: sql.index("DROP VIEW")] if "DROP VIEW" in sql else sql
    assert "run_migrations.sh" in head, "the header does not say why the runner skips it"
    assert "144" in head, "the header does not name the expand half it completes"
    assert re.search(r"psql|docker exec", head), "the header gives no apply command"
    assert not re.search(r"^\s*(COMMIT|ROLLBACK)\s*;", sql, re.I | re.M)


def test_the_contract_refuses_a_schema_where_the_expand_has_not_run():
    """THE PRECONDITION. 146 is the destructive half, applied BY HAND, possibly weeks
    later and possibly by someone who was not here for 144.

    ultracode iter8 HIGH. Every statement in this file is ``IF EXISTS`` / ``OR
    REPLACE`` — deliberately, so a re-run is a no-op — and the consequence nobody had
    drawn is that the file runs happily to completion on a schema where 144 was NEVER
    applied, or was rolled back. It drops trx_count / nrx_count / total_rx_count with
    no canonical column holding the values, reports **exit 0**, and the per-HCP trigger
    counts are simply gone.

    Reproduced 2026-09-18 against the live database inside BEGIN/ROLLBACK — which was a
    FAITHFUL environment for it, because production genuinely has not had 144 applied:
    3 legacy columns and 12,143 rows carrying counts before, **0 legacy columns, 0
    canonical columns, 33 columns and psql exit 0** after. Nothing raised; only two
    NOTICEs about skipping a trigger and a function that were never there.

    The only thing standing between that and a hand-applied contract was header prose,
    and prose is not a precondition — it is a hope about who is reading. The check has
    to be IN THE FILE, and it has to fail CLOSED.

    What the block asserts is the CAPABILITY, in the order the two conditions can be
    asked. First: the three canonical columns exist at all. That must raise before
    anything references them, because the row-level query below names those columns and
    would otherwise fail with a bare ``column does not exist`` — a refusal, technically,
    but one an operator cannot act on, which this lane has already classified as its own
    defect. plpgsql plans a statement on first execution, not at block entry, so an
    early RAISE keeps the second query from ever being planned; that ordering is load
    bearing and is verified live, not assumed. Second: no row holds a legacy count the
    canonical column does not carry — the actual condition under which dropping the
    legacy three loses information.
    """
    sql = CONTRACT.read_text()

    block = re.search(r"DO \$precondition\$(.*?)\$precondition\$;", sql, re.DOTALL)
    assert block, (
        "146 has no DO $precondition$ block — it will drop the legacy columns on a "
        "schema where 144 never ran, silently and exit 0"
    )
    body = block.group(1)

    # It must come before ANY destructive statement, or it is a post-mortem.
    destructive = [
        m.start()
        for m in re.finditer(
            r"^\s*(DROP|ALTER TABLE \S+ DROP|TRUNCATE|DELETE)\b", sql, re.MULTILINE
        )
    ]
    assert destructive, "no destructive statement found — this test is looking at the wrong file"
    assert block.start() < min(destructive), (
        "the precondition block runs after the first destructive statement, so it can "
        "only report the loss it was supposed to prevent"
    )

    # Condition 1: the canonical columns exist, asked of the catalog.
    for canonical in CANONICAL:
        assert canonical in body, f"the precondition does not check for {canonical}"
    assert re.search(r"information_schema\.columns|pg_attribute", body), (
        "the precondition does not ask the catalog whether the canonical columns exist"
    )

    # Condition 2: every legacy value is actually carried by its canonical column.
    # strict=True: LEGACY and CANONICAL are parallel tuples, and a silent zip
    # truncation would drop a pair from this check without failing anything.
    for legacy, canonical in zip(LEGACY, CANONICAL, strict=True):
        assert re.search(rf"{canonical}\s+IS DISTINCT FROM\s+{legacy}", body), (
            f"the precondition does not compare {canonical} against {legacy} row by row"
        )

    # ...and both must REFUSE, not warn. A NOTICE would let the drops proceed.
    assert body.count("RAISE EXCEPTION") >= 2, (
        "the precondition must RAISE EXCEPTION for both conditions; a RAISE NOTICE or a "
        "logged warning still lets the columns be dropped"
    )
    assert "RAISE WARNING" not in body and "RAISE NOTICE" not in body, (
        "the precondition downgrades a refusal to a warning"
    )


def test_the_documented_apply_command_keeps_the_precondition_armed():
    """``ON_ERROR_STOP=1`` is what turns the precondition's RAISE into a refusal.

    Without it psql prints the error and RUNS THE NEXT STATEMENT, which is the first
    DROP — so the block would raise, be ignored, and the columns would go anyway. The
    flag is therefore part of the guard, exactly as ``--single-transaction`` is part of
    the ledger's atomicity (codex iter4 MED-1), and it is asserted in the COMMAND rather
    than anywhere in the header prose: this header explains both flags in sentences too,
    so a substring check over the whole file passes while the command has lost them.
    That exact proxy was codex iter3 LOW-2 on this same file.
    """
    head = CONTRACT.read_text()
    command_text = " ".join(re.sub(r"^--\s?", "", ln) for ln in head.splitlines())
    assert re.search(
        r"docker exec.*?psql.*?ON_ERROR_STOP=1.*?<\s*database/deferred/146", command_text
    ), (
        "the documented apply COMMAND does not set ON_ERROR_STOP=1, so the precondition "
        "block would raise and psql would carry on into the DROPs"
    )
