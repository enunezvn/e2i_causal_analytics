"""Migration 144: the per_hcp_rollup trigger counts get honest names (canonical TRx lane).

Hermetic — reads the migration FILES, never a database. That boundary is the point:
a file can show that every rename is written inside an existence guard, but only a
live re-application can show that the second run is a no-op. The rehearsal
(BEGIN / apply / apply again / ROLLBACK) is what proves idempotency; these tests
prove the file is SHAPED so that it can be idempotent, and refuse the shapes that
cannot be.

Live census 2026-09-17 (read-only), which is what the renames are aimed at:
``business_metrics`` plus ``v_{train,test,validation,holdout}_business_metrics``
carry all three legacy columns — 15 relation-columns in total — and nothing else
does. No index, constraint, default, RLS policy or kpi_query_registry statement
references them. One function body matched a word-boundary grep,
``assign_truth_script_conversion``, and reading it showed the match is its OWN CTE
alias (``COUNT(*) as nrx_count_window`` over ``treatment_events``, surfaced as
``nrx_count``) — not a reference to ``business_metrics``. A grep is the proxy; the
body is the capability.
"""

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "144_per_hcp_trigger_count_columns.sql"
ROLLBACK = MIGRATION.parent / "rollback_144_per_hcp_trigger_count_columns.sql"
RENAMES = {
    "trx_count": "triggers_delivered_count",
    "nrx_count": "triggers_accepted_count",
    "total_rx_count": "triggers_total_count",
}
VIEWS = (
    "v_train_business_metrics",
    "v_test_business_metrics",
    "v_validation_business_metrics",
    "v_holdout_business_metrics",
)


def _sql() -> str:
    return MIGRATION.read_text()


def test_every_column_pair_is_renamed():
    """The TABLE renames must be STATIC, the VIEW renames may be dynamic.

    The three table renames were deliberately un-loop-ed: a dynamic
    ``EXECUTE format('... RENAME COLUMN %I TO %I', ...)`` hides the column names
    behind placeholders, and the hermetic Feast guard
    (tests/unit/test_feature_repo/test_data_sources_columns_exist.py) models the
    canonical schema by TEXT-PARSING these files. Under the dynamic form it still
    believed business_metrics carried trx_count/nrx_count/total_rx_count and
    reported the renamed Feast source columns as "absent". So assert the literal
    static statements, not a placeholder: that text IS the contract the static
    reader consumes. The view pair literals stay because the view loop is still
    dynamic and no static reader models view columns.
    """
    sql = _sql()
    for old, new in RENAMES.items():
        assert f"['{old}', '{new}']" in sql, f"view-loop pair literal missing for {old}"
        assert f"ALTER TABLE public.business_metrics RENAME COLUMN {old} TO {new};" in sql, (
            f"the {old} table rename is not statically declared — a dynamic rename is "
            "invisible to the Feast guard's text parser"
        )
    assert "ALTER TABLE public.business_metrics RENAME COLUMN %I TO %I" not in sql, (
        "a table rename went back to the dynamic form; keep committed table renames "
        "statically declarable"
    )


def test_every_split_view_output_column_is_renamed():
    """A table column rename keeps a view VALID (views bind by attnum) but leaves
    the view's OUTPUT column under the old name, so each view is renamed too."""
    sql = _sql()
    for view in VIEWS:
        assert f"'{view}'" in sql, view
    assert "ALTER VIEW public.%I RENAME COLUMN %I TO %I" in sql


def test_every_rename_is_inside_an_existence_guard():
    """Guard-to-rename PARITY, not a marker count.

    ``count("IF EXISTS") >= 3`` is satisfiable by three occurrences in a comment,
    so it cannot fail for the reason it exists. Counting the executable RENAMEs and
    requiring at least one guard apiece is still structural, but an unguarded
    rename can no longer hide behind guards belonging to other statements — and an
    unguarded rename is exactly what makes a re-run raise instead of no-op.
    """
    sql = _sql()
    dynamic = len(re.findall(r"EXECUTE format\('ALTER (?:TABLE|VIEW)[^']*RENAME COLUMN", sql))
    static = len(re.findall(r"^\s*ALTER TABLE [^;]*RENAME COLUMN ", sql, re.M))
    guards = len(re.findall(r"IF EXISTS \(\s*\n\s*SELECT 1 FROM information_schema\.columns", sql))
    renames = dynamic + static
    # PostgreSQL has no RENAME COLUMN IF EXISTS, so idempotency is carried by an
    # explicit guard per rename: three static table renames + the one view-loop body.
    assert static == len(RENAMES), f"expected {len(RENAMES)} static table renames, got {static}"
    assert dynamic == 1, f"expected exactly the view loop to rename dynamically, got {dynamic}"
    assert guards >= renames, f"{renames} renames but only {guards} existence guards"


def test_nothing_destructive_and_postgrest_reloads():
    sql = _sql()
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE)\b", sql, re.IGNORECASE)
    assert not re.search(r"^\s*COMMIT\s*;", sql, re.IGNORECASE | re.MULTILINE)
    assert "NOTIFY pgrst, 'reload schema';" in sql


def test_columns_are_documented():
    sql = _sql()
    for new in RENAMES.values():
        assert f"COMMENT ON COLUMN public.business_metrics.{new} IS" in sql


def test_the_rollback_reverses_every_rename():
    rollback = ROLLBACK.read_text()
    for old, new in RENAMES.items():
        assert f"['{new}', '{old}']" in rollback
    assert "ALTER VIEW public.%I RENAME COLUMN %I TO %I" in rollback
    assert not re.search(r"\b(DROP|DELETE|TRUNCATE)\b", rollback, re.IGNORECASE)


def test_the_rollback_is_never_applied_as_a_forward_migration():
    """The recovery file undoes 144. If the forward runner picked it up, applying
    migrations would rename the columns and immediately rename them back — a
    migration that silently does nothing. The runner's exclusion is what makes
    shipping the file safe, so assert the runner still excludes this NAME rather
    than trusting the convention.
    """
    runner = (REPO / "scripts" / "run_migrations.sh").read_text()
    patterns = re.search(r"^\s*(\*_validation_queries\.sql\|[^)]*)\)\s*continue", runner, re.M)
    assert patterns, "run_migrations.sh no longer has the skip-case this file relies on"
    globs = patterns.group(1).split("|")
    assert "rollback_*.sql" in globs, globs
    assert ROLLBACK.name.startswith("rollback_") and ROLLBACK.name.endswith(".sql")
