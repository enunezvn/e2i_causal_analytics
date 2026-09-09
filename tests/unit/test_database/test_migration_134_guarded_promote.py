"""Migration 134 ships the guarded causal_paths promote (lane 1, spec §4.3)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "134_guarded_causal_path_promote.sql"

FUNCTIONS = (
    "public.dag_structure_rejected(text, text)",
    "public.promote_causal_path_guarded(text, text, text[], text, text)",
)


@pytest.mark.unit
def test_migration_defines_both_functions():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "CREATE OR REPLACE FUNCTION public.dag_structure_rejected(" in sql
    assert "CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded(" in sql


@pytest.mark.unit
def test_rejection_is_evaluated_inside_the_update_statement():
    """The whole point: no window between reading the verdict and writing the status."""
    sql = MIGRATION.read_text(encoding="utf-8")
    update = re.search(r"UPDATE public\.causal_paths.*?;", sql, re.S)
    assert update is not None
    assert "NOT public.dag_structure_rejected(" in update.group(0)
    assert "validation_status = ANY (p_allowed_current)" in update.group(0)


@pytest.mark.unit
def test_promote_takes_a_table_share_lock_before_the_update():
    """A rejection racing the promote must either be seen by the UPDATE or wait
    for it; the STABLE predicate alone leaves a statement-sized window and a
    row lock cannot cover a review row inserted meanwhile (pre-execution review
    iter-2 + iter-3, codex HIGH)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    fn = sql[sql.index("CREATE OR REPLACE FUNCTION public.promote_causal_path_guarded(") :]
    lock = "LOCK TABLE public.expert_reviews IN SHARE MODE;"
    assert lock in fn
    assert fn.index(lock) < fn.index("UPDATE public.causal_paths")


@pytest.mark.unit
def test_empty_string_brand_means_no_brand():
    """``get_reviews_for_dag`` filters with ``if brand:`` -- '' is unfiltered. The
    SQL must read '' the same way or a same-hash rejection under another brand
    is missed (pre-execution review 2026-09-08, codex HIGH)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert sql.count("NULLIF(p_brand, '') IS NULL OR") == 2
    assert "(p_brand IS NULL OR" not in sql


@pytest.mark.unit
def test_service_role_only():
    sql = MIGRATION.read_text(encoding="utf-8")
    for fn in FUNCTIONS:
        assert f"REVOKE ALL ON FUNCTION {fn} FROM PUBLIC, anon, authenticated;" in sql
        assert f"GRANT EXECUTE ON FUNCTION {fn} TO service_role;" in sql
    assert "has_function_privilege" in sql  # the migration asserts its own grants


@pytest.mark.unit
def test_created_at_is_made_not_null_so_the_chronology_is_total():
    """A NULL created_at sorts FIRST under ``ORDER BY created_at DESC`` (the Python
    probe reads it as newest) while ``NULL > ts`` is UNKNOWN in the SQL predicate
    (read as "not newer"): the two readers could disagree on such a row. No live
    row has one and both writers rely on DEFAULT now(), so the migration closes
    the class (lane 1 codex iter-1, MED)."""
    sql = MIGRATION.read_text(encoding="utf-8")
    alter = "ALTER TABLE public.expert_reviews ALTER COLUMN created_at SET NOT NULL;"
    assert alter in sql
    assert sql.index(alter) < sql.index("CREATE OR REPLACE FUNCTION public.dag_structure_rejected(")
