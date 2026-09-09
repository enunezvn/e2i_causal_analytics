"""Migration 136 adds ``expert_reviews.resolved_at`` (lane 1, codex whole-diff HIGH F1).

The resolution time for BOTH statuses, written by ``submit_review`` together
with the resolver's ``reviewer_name`` / ``reviewer_email``. Rows resolved before
the migration keep NULL: ``updated_at`` is trigger-maintained and is NOT a
decision time (the one live rejected row had its cached assessment written
after its rejection), so there is nothing honest to backfill from.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
MIGRATION = REPO / "database" / "migrations" / "136_expert_reviews_resolved_at.sql"

ADD_COLUMN = "ALTER TABLE public.expert_reviews ADD COLUMN IF NOT EXISTS resolved_at TIMESTAMPTZ;"


@pytest.mark.unit
def test_adds_the_column_idempotently():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert ADD_COLUMN in sql


@pytest.mark.unit
def test_column_comment_states_the_contract():
    """The comment is the column's contract: both statuses, written by
    ``submit_review`` with the resolver identity, NULL before the migration."""
    sql = MIGRATION.read_text(encoding="utf-8")
    comment = re.search(
        r"COMMENT ON COLUMN public\.expert_reviews\.resolved_at IS\s*(.*?);", sql, re.S
    )
    assert comment is not None
    text = comment.group(1)
    assert "submit_review" in text
    assert "reviewer_name" in text and "reviewer_email" in text
    assert "approved" in text and "rejected" in text
    assert "NULL" in text


@pytest.mark.unit
def test_no_backfill_unknown_stays_unknown():
    """``updated_at`` is not a decision time; a backfill would fabricate one."""
    sql = MIGRATION.read_text(encoding="utf-8")
    statements = [s for s in sql.split("\n") if not s.lstrip().startswith("--")]
    body = "\n".join(statements)
    assert not re.search(r"\bUPDATE\b", body, re.I)
    assert not re.search(r"\bSET\s+resolved_at\b", body, re.I)
    # No DEFAULT either: a default would stamp future rows at INSERT time.
    assert not re.search(r"\bDEFAULT\b", body, re.I)


@pytest.mark.unit
def test_runner_wraps_it_no_own_transaction():
    """scripts/run_migrations.sh wraps a file in --single-transaction unless it
    manages its own; a COMMIT here would end the wrapper's transaction."""
    sql = MIGRATION.read_text(encoding="utf-8")
    body = "\n".join(s for s in sql.split("\n") if not s.lstrip().startswith("--"))
    assert not re.search(r"^\s*(BEGIN|COMMIT)\s*;", body, re.I | re.M)
