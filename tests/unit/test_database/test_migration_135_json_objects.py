"""Migration 135 decodes string-shaped evidence rows into JSON objects (lane 1, owner decision 6)."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "135_causal_validations_json_objects.sql"


@pytest.mark.unit
def test_backfills_both_columns_and_asserts_none_remain():
    # Whitespace-normalised: the migration formats SET / WHERE on separate lines.
    sql = " ".join(MIGRATION.read_text(encoding="utf-8").split())
    for col in ("details_json", "test_config"):
        assert f"SET {col} = ({col} #>> '{{}}')::jsonb WHERE jsonb_typeof({col}) = 'string'" in sql
    assert "RAISE EXCEPTION 'migration 135: string-shaped evidence rows remain'" in sql


@pytest.mark.unit
def test_no_constraint_that_would_break_the_old_image_during_the_deploy_swap():
    """A CHECK (jsonb_typeof = 'object') would make the OLD writer fail between the
    migration run and the container flip; the writer fix + this backfill are the
    guarantee, and Task 14 certifies zero string rows after the deploy."""
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "ALTER TABLE" not in sql
    assert "CHECK (" not in sql
