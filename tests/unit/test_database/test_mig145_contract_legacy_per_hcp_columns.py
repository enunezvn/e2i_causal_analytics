"""Migration 145: the CONTRACT half of the per_hcp_rollup column expand/contract.

144 adds ``business_metrics.triggers_{delivered,accepted,total}_count`` beside the
legacy ``{trx,nrx,total_rx}_count`` and keeps both true with a row trigger. 145
retires the legacy three.

THE WHOLE POINT OF THIS MODULE IS THAT 145 MUST NOT RUN IN THE SAME DEPLOY AS 144.
``scripts/run_migrations.sh`` applies EVERY pending forward ``*.sql`` in each of its
``MIGRATION_DIRS`` in one pass, so a ``database/migrations/145_*.sql`` committed
beside 144 would be applied seconds after it — the legacy columns would be gone
before a single container was replaced, and the deploy would be exactly as unsafe
as the rename that codex HIGH-1 rejected. Expand/contract is only expand/contract
if the two halves land in two different deploys.

The separation is STRUCTURAL rather than a naming convention: 145 lives in
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
CONTRACT = DEFERRED_DIR / "145_drop_legacy_per_hcp_count_columns.sql"
EXPAND = REPO / "database" / "migrations" / "144_per_hcp_trigger_count_columns.sql"
LEGACY = ("trx_count", "nrx_count", "total_rx_count")
CANONICAL = ("triggers_delivered_count", "triggers_accepted_count", "triggers_total_count")
VIEWS = (
    "v_train_business_metrics",
    "v_test_business_metrics",
    "v_validation_business_metrics",
    "v_holdout_business_metrics",
)


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
    """THE GUARD. If ``database/deferred`` ever joins MIGRATION_DIRS — or 145 is
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


def test_the_contract_recreates_the_split_views():
    """``DROP COLUMN`` fails while a view depends on the column, and all four
    ``SELECT *`` split views expand to the legacy names. They are dropped and
    recreated around the ALTERs — never dropped with CASCADE, which would take the
    views away and leave nothing behind."""
    sql = CONTRACT.read_text()
    assert not re.search(r"\bCASCADE\b", sql, re.IGNORECASE), (
        "CASCADE would silently drop the dependent views instead of recreating them"
    )
    for view in VIEWS:
        assert f"DROP VIEW IF EXISTS public.{view};" in sql, view
        assert re.search(
            rf"CREATE OR REPLACE VIEW public\.{view} AS\s*\n?\s*SELECT \* FROM public\.business_metrics",
            sql,
        ), f"{view} is dropped but not recreated"
        assert sql.index(f"DROP VIEW IF EXISTS public.{view};") < sql.index(
            f"CREATE OR REPLACE VIEW public.{view}"
        ), f"{view} is recreated before it is dropped"


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
