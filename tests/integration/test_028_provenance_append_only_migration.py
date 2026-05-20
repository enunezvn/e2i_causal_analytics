"""Migration 028 provenance append-only enforcement tests (#391 security box 1).

Two layers of test:

1. Static-content checks — always run. Verify the SQL has the
   IF-EXISTS guards / CREATE OR REPLACE / DROP+CREATE pattern that
   make the migration idempotent and reversible.

2. Integration tests — run only when ``TEST_POSTGRES_URL`` env + local
   ``psql`` are available. Applies the migration to a real Postgres
   instance and verifies:
     * UPDATE on ``audit_chain_entries`` raises (trigger fires)
     * DELETE on ``audit_chain_entries`` raises
     * INSERT on ``audit_chain_entries`` still works
     * UPDATE on an active ``executive_insights`` row (invalidated_at IS NULL)
       still works (the trigger is conditional, not blanket)
     * UPDATE on an invalidated ``executive_insights`` row raises
     * DELETE on an invalidated ``executive_insights`` row raises
     * Re-applying the migration is a no-op (idempotency)

Pattern mirrors ``tests/integration/test_021_insight_lifecycle_migration.py``
which is the established convention for migration tests in this repo.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent / "database" / "memory" / "028_provenance_append_only.sql"
)

# Sibling migrations the integration test needs to bootstrap the
# referenced tables (audit_chain_entries from 011_audit_chain_tables.sql,
# executive_insights from 021_insight_lifecycle.sql).
AUDIT_CHAIN_MIGRATION = (
    Path(__file__).parent.parent.parent / "database" / "audit" / "011_audit_chain_tables.sql"
)
INSIGHT_LIFECYCLE_MIGRATION = (
    Path(__file__).parent.parent.parent / "database" / "memory" / "021_insight_lifecycle.sql"
)


# ---------------------------------------------------------------------------
# Static-content checks (no DB needed)
# ---------------------------------------------------------------------------


def test_migration_file_exists() -> None:
    """The migration file must exist at the expected path."""
    assert MIGRATION_PATH.exists(), f"Migration not found at {MIGRATION_PATH}"


def test_migration_uses_drop_trigger_if_exists_for_idempotency() -> None:
    """Both triggers must be dropped-if-exists before re-creation so the
    migration re-applies cleanly. Postgres does NOT support
    CREATE TRIGGER IF NOT EXISTS, so DROP+CREATE is the canonical
    idempotent pattern.
    """
    content = MIGRATION_PATH.read_text().lower()
    assert "drop trigger if exists trg_audit_chain_entries_append_only" in content, (
        "Missing DROP TRIGGER IF EXISTS for trg_audit_chain_entries_append_only"
    )
    assert "drop trigger if exists trg_executive_insights_invalidation_set_once" in content, (
        "Missing DROP TRIGGER IF EXISTS for trg_executive_insights_invalidation_set_once"
    )


def test_migration_uses_create_or_replace_function() -> None:
    """Trigger functions use CREATE OR REPLACE FUNCTION for idempotency.

    Without CREATE OR REPLACE the second apply would fail with
    'function already exists' since we don't DROP FUNCTION first.
    """
    content = MIGRATION_PATH.read_text().lower()
    assert "create or replace function prevent_update_delete_audit_chain" in content, (
        "Missing CREATE OR REPLACE FUNCTION prevent_update_delete_audit_chain"
    )
    assert "create or replace function prevent_change_invalidated_executive_insight" in content, (
        "Missing CREATE OR REPLACE FUNCTION prevent_change_invalidated_executive_insight"
    )


def test_migration_uses_raise_exception_with_informative_message() -> None:
    """Trigger functions must RAISE EXCEPTION with operator-friendly
    messages (TG_OP, the entry id, why it's frozen). A bare RAISE EXCEPTION
    without context makes operational debugging painful.
    """
    content = MIGRATION_PATH.read_text().lower()
    # Should mention the table NAME in the error message so operators
    # see "audit_chain_entries is append-only" not just a generic violation.
    assert "audit_chain_entries is append-only" in content, (
        "Error message missing for audit_chain trigger"
    )
    assert "executive_insights row is invalidated and frozen" in content, (
        "Error message missing for invalidated executive_insights trigger"
    )


def test_migration_wraps_in_transaction() -> None:
    """The migration runs in a transaction so partial failure rolls back.

    BEGIN/COMMIT brackets are the established pattern in
    021_insight_lifecycle.sql et al.
    """
    content = MIGRATION_PATH.read_text().lower()
    assert content.lstrip().startswith("begin;") or "\nbegin;" in content, (
        "Migration should start with BEGIN; per repo convention"
    )
    assert "commit;" in content, "Migration should end with COMMIT;"


def test_executive_insights_trigger_is_conditional_on_invalidated_at() -> None:
    """The executive_insights trigger must be conditional — only frozen
    rows reject; active rows (invalidated_at IS NULL) are still mutable
    by the normal lifecycle write path.

    A blanket UPDATE-block would break recall/recall_reason updates and
    is NOT what we want.
    """
    content = MIGRATION_PATH.read_text().lower()
    assert "old.invalidated_at is not null" in content, (
        "executive_insights trigger MUST guard on OLD.invalidated_at IS NOT NULL "
        "so active rows remain mutable."
    )


def test_migration_targets_only_documented_tables() -> None:
    """The migration must NOT silently apply triggers to tables
    outside the documented scope (audit_chain_entries +
    executive_insights). Drift here is dangerous.
    """
    content = MIGRATION_PATH.read_text().lower()
    # Count the trigger declarations on each table.
    assert content.count("on audit_chain_entries") >= 2, (
        "Expect >=2 references to audit_chain_entries (DROP TRIGGER + "
        "CREATE TRIGGER); migration likely refactored — review test."
    )
    assert content.count("on executive_insights") >= 2, (
        "Expect >=2 references to executive_insights (DROP TRIGGER + "
        "CREATE TRIGGER); migration likely refactored — review test."
    )


def test_migration_creates_crystal_narrative_audits_table() -> None:
    """Codex iter-0 M2 closure: migration 028 also creates the
    ``crystal_narrative_audits`` table so the offline PHI scanner
    has a real DB surface to read ``input_prompt`` from. The audit
    harness script SQL JOIN against this table is currently illusory
    without it.
    """
    content = MIGRATION_PATH.read_text().lower()
    assert "create table if not exists crystal_narrative_audits" in content, (
        "Migration 028 must create the crystal_narrative_audits table "
        "(see #391 box 4 + codex iter-0 M2)."
    )
    # The input_prompt column is the load-bearing PHI-audit surface.
    assert "input_prompt" in content, (
        "crystal_narrative_audits MUST have an input_prompt column — "
        "this is the LLM-input audit surface for #391 box 4."
    )
    # FK to executive_insights with cascade keeps audit rows in sync
    # with the parent. References must NOT be removed.
    assert "references executive_insights(insight_id)" in content, (
        "crystal_narrative_audits must FK insight_id to executive_insights"
    )
    assert "on delete cascade" in content, (
        "crystal_narrative_audits FK must cascade-delete with parent (orphan rows are unwanted)."
    )


# ---------------------------------------------------------------------------
# Integration tests (real Postgres, gated)
# ---------------------------------------------------------------------------


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


_TEST_DB_URL = os.environ.get("TEST_POSTGRES_URL")

pytestmark_integration = pytest.mark.skipif(
    not _TEST_DB_URL or not _have_psql(),
    reason="needs TEST_POSTGRES_URL env + local psql for full migration apply",
)


def _exec_sql(url: str, sql: str, *, label: str = "") -> subprocess.CompletedProcess:
    """Pipe SQL to psql. Returns the CompletedProcess (caller asserts)."""
    return subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1"],
        input=sql,
        capture_output=True,
        text=True,
    )


def _apply_migration(url: str, path: Path) -> None:
    result = subprocess.run(
        ["psql", url, "-v", "ON_ERROR_STOP=1", "-f", str(path)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"Migration apply ({path.name}) failed:\nstderr={result.stderr}\nstdout={result.stdout}"
    )


@pytestmark_integration
def test_migration_028_applies_cleanly_after_bootstrap() -> None:
    """Apply migrations 011 (audit_chain) + 021 (executive_insights) +
    028 in order and assert no errors. This is the minimum end-to-end
    apply that proves the trigger functions can be created on tables
    that already exist.
    """
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)


@pytestmark_integration
def test_migration_028_is_idempotent() -> None:
    """Apply the migration twice — second apply must succeed.

    This relies on DROP TRIGGER IF EXISTS and CREATE OR REPLACE
    FUNCTION; if either is missing the second apply errors.
    """
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)


@pytestmark_integration
def test_audit_chain_entries_update_is_rejected() -> None:
    """Update on audit_chain_entries → trigger fires, raises exception."""
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)

    # Seed one row, then try to update.
    seed = """
        INSERT INTO audit_chain_entries (
            workflow_id, sequence_number, agent_name, agent_tier,
            action_type, entry_hash
        ) VALUES (
            gen_random_uuid(), 1, 'test_agent', 0, 'test_action', 'h0'
        );
    """
    seed_result = _exec_sql(_TEST_DB_URL, seed, label="seed audit_chain")
    assert seed_result.returncode == 0, f"Seed failed: {seed_result.stderr}"

    update_sql = "UPDATE audit_chain_entries SET agent_name = 'tampered';"
    update_result = _exec_sql(_TEST_DB_URL, update_sql)
    assert update_result.returncode != 0, (
        "UPDATE on audit_chain_entries should have been blocked by trigger; "
        f"stderr={update_result.stderr} stdout={update_result.stdout}"
    )
    assert "append-only" in update_result.stderr.lower(), (
        f"Expected 'append-only' in error message; got: {update_result.stderr}"
    )


@pytestmark_integration
def test_audit_chain_entries_delete_is_rejected() -> None:
    """Delete on audit_chain_entries → trigger fires."""
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)

    seed = """
        INSERT INTO audit_chain_entries (
            workflow_id, sequence_number, agent_name, agent_tier,
            action_type, entry_hash
        ) VALUES (
            gen_random_uuid(), 1, 'test_agent', 0, 'test_action', 'h0'
        );
    """
    seed_result = _exec_sql(_TEST_DB_URL, seed)
    assert seed_result.returncode == 0

    delete_sql = "DELETE FROM audit_chain_entries;"
    delete_result = _exec_sql(_TEST_DB_URL, delete_sql)
    assert delete_result.returncode != 0, (
        f"DELETE on audit_chain_entries should have been blocked; stderr={delete_result.stderr}"
    )


@pytestmark_integration
def test_audit_chain_entries_insert_still_works() -> None:
    """INSERT on audit_chain_entries must still succeed (the trigger is
    BEFORE UPDATE OR DELETE only — not INSERT)."""
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)

    insert_sql = """
        INSERT INTO audit_chain_entries (
            workflow_id, sequence_number, agent_name, agent_tier,
            action_type, entry_hash
        ) VALUES (
            gen_random_uuid(), 1, 'test_agent_insert', 0,
            'test_action_insert', 'h-insert'
        );
    """
    result = _exec_sql(_TEST_DB_URL, insert_sql)
    assert result.returncode == 0, (
        f"INSERT on audit_chain_entries failed unexpectedly: stderr={result.stderr}"
    )


@pytestmark_integration
def test_executive_insights_update_active_row_still_works() -> None:
    """An active (invalidated_at IS NULL) executive_insights row remains
    mutable. The trigger only freezes invalidated rows.
    """
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)

    seed = """
        INSERT INTO executive_insights (
            title, narrative, brand
        ) VALUES (
            'test active', 'narrative', 'test-brand-active'
        );
    """
    seed_result = _exec_sql(_TEST_DB_URL, seed)
    assert seed_result.returncode == 0

    update_sql = """
        UPDATE executive_insights
        SET narrative = 'updated narrative'
        WHERE brand = 'test-brand-active';
    """
    update_result = _exec_sql(_TEST_DB_URL, update_sql)
    assert update_result.returncode == 0, (
        f"UPDATE on active executive_insights row should succeed; stderr={update_result.stderr}"
    )


@pytestmark_integration
def test_executive_insights_update_invalidated_row_is_rejected() -> None:
    """An invalidated row (invalidated_at IS NOT NULL) is frozen.

    This is the security contract: once invalidated, no UPDATE may
    clear the invalidated_at back to NULL or otherwise mutate the row.
    """
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)

    # Seed a row WITH invalidated_at set (skip the trigger by inserting
    # the invalidated state up-front — the trigger is BEFORE UPDATE OR
    # DELETE only, not BEFORE INSERT).
    seed = """
        INSERT INTO executive_insights (
            title, narrative, brand, invalidated_at, invalidation_reason
        ) VALUES (
            'frozen', 'frozen narrative', 'test-brand-frozen',
            now(), 'manual freeze for test'
        );
    """
    seed_result = _exec_sql(_TEST_DB_URL, seed)
    assert seed_result.returncode == 0, f"Seed failed: {seed_result.stderr}"

    update_sql = """
        UPDATE executive_insights
        SET invalidated_at = NULL
        WHERE brand = 'test-brand-frozen';
    """
    update_result = _exec_sql(_TEST_DB_URL, update_sql)
    assert update_result.returncode != 0, (
        "UPDATE on invalidated executive_insights row should be blocked; "
        f"stderr={update_result.stderr}"
    )
    assert "frozen" in update_result.stderr.lower(), (
        f"Expected 'frozen' in error message; got: {update_result.stderr}"
    )


@pytestmark_integration
def test_executive_insights_delete_invalidated_row_is_rejected() -> None:
    """DELETE on an invalidated row is rejected (admin can't drop frozen rows)."""
    assert _TEST_DB_URL is not None
    _apply_migration(_TEST_DB_URL, AUDIT_CHAIN_MIGRATION)
    _apply_migration(_TEST_DB_URL, INSIGHT_LIFECYCLE_MIGRATION)
    _apply_migration(_TEST_DB_URL, MIGRATION_PATH)

    seed = """
        INSERT INTO executive_insights (
            title, narrative, brand, invalidated_at, invalidation_reason
        ) VALUES (
            'frozen del', 'frozen narrative', 'test-brand-frozen-del',
            now(), 'manual freeze for delete test'
        );
    """
    seed_result = _exec_sql(_TEST_DB_URL, seed)
    assert seed_result.returncode == 0

    delete_sql = """
        DELETE FROM executive_insights WHERE brand = 'test-brand-frozen-del';
    """
    delete_result = _exec_sql(_TEST_DB_URL, delete_sql)
    assert delete_result.returncode != 0, (
        "DELETE on invalidated executive_insights row should be blocked; "
        f"stderr={delete_result.stderr}"
    )
