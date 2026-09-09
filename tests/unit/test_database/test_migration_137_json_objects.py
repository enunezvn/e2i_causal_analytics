"""Migration 137 decodes string-shaped expert_reviews rows into JSON objects (#1992)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "137_expert_reviews_json_objects.sql"


@pytest.mark.unit
def test_backfills_all_four_columns_and_asserts_none_remain():
    # Whitespace-normalised: the migration formats SET / WHERE on separate lines.
    sql = " ".join(MIGRATION.read_text(encoding="utf-8").split())
    for col in (
        "checklist_json",
        "comments_json",
        "agent_assessment_json",
        "dag_structure_json",
    ):
        assert f"SET {col} = ({col} #>> '{{}}')::jsonb WHERE jsonb_typeof({col}) = 'string'" in sql
    assert "RAISE EXCEPTION 'migration 137: string-shaped expert_reviews rows remain'" in sql


@pytest.mark.unit
def test_no_constraint_that_would_break_the_old_image_during_the_deploy_swap():
    """A CHECK (jsonb_typeof = 'object') would make the OLD writer fail between the
    migration run and the container flip; the writer fix + this backfill are the
    guarantee, and live verification certifies zero string rows after the deploy."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "ALTER TABLE" not in sql
    assert "CHECK (" not in sql
