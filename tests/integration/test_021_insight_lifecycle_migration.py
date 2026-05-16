"""Migration 021 smoke tests.

These tests check that the migration is self-contained:

- It does not depend on the target database having pgcrypto pre-enabled
  (some self-hosted Postgres / older Supabase projects don't enable it
  by default). Migration declares ``CREATE EXTENSION IF NOT EXISTS
  pgcrypto`` so ``gen_random_uuid()`` resolves.

- Re-applying the migration is a no-op (idempotency). All `CREATE TABLE`
  and `CREATE INDEX` statements use `IF NOT EXISTS` and the ENUM types
  use the `DO $$ ... EXCEPTION WHEN duplicate_object` pattern.

The static-content checks run anywhere. The full-apply idempotency test
needs `TEST_POSTGRES_URL` env + local `psql`; it skips otherwise (and
the skip is documented via the standard ``pytest.mark.skipif`` mechanism,
NOT via a self-declared "deferred" — per
feedback_verification_step_evidence_gate).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent / "database" / "memory" / "021_insight_lifecycle.sql"
)


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


def test_migration_declares_pgcrypto_explicitly() -> None:
    """The migration must ``CREATE EXTENSION IF NOT EXISTS pgcrypto;``.

    ``gen_random_uuid()`` is provided by pgcrypto. Supabase enables this
    by default but older / self-hosted projects may not, so the migration
    must declare the extension explicitly.
    """
    assert MIGRATION_PATH.exists(), f"Migration file not found at {MIGRATION_PATH}"
    content = MIGRATION_PATH.read_text().lower()
    assert "create extension if not exists pgcrypto" in content, (
        "Migration uses gen_random_uuid() which is in pgcrypto. The extension "
        "must be declared explicitly with CREATE EXTENSION IF NOT EXISTS "
        "pgcrypto so the migration works on projects where pgcrypto isn't "
        "pre-enabled."
    )


def test_pgcrypto_declaration_precedes_first_gen_random_uuid() -> None:
    """The extension must be declared BEFORE the first USE of gen_random_uuid().

    Otherwise the function reference fails when the migration is applied to
    a project where pgcrypto isn't already enabled.

    Searches for ``DEFAULT gen_random_uuid()`` (the DDL pattern) rather than
    the bare string, so a documentation comment that names the function
    (e.g., "pgcrypto provides gen_random_uuid()") doesn't false-trip.
    """
    content = MIGRATION_PATH.read_text().lower()
    pgcrypto_idx = content.find("create extension if not exists pgcrypto")
    # Match the DDL pattern, not the bare function reference, so prose
    # comments don't false-trip.
    first_uuid_use_idx = content.find("default gen_random_uuid()")
    assert pgcrypto_idx >= 0, "pgcrypto declaration missing"
    assert first_uuid_use_idx >= 0, "no DEFAULT gen_random_uuid() in migration"
    assert pgcrypto_idx < first_uuid_use_idx, (
        f"pgcrypto extension declared at char {pgcrypto_idx} but first "
        f"DEFAULT gen_random_uuid() is at char {first_uuid_use_idx} "
        f"(must come BEFORE)"
    )


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL") or not _have_psql(),
    reason="needs TEST_POSTGRES_URL env + local psql for full migration apply",
)
def test_migration_is_idempotent() -> None:
    """Applying the migration twice in a row must not fail.

    Falsifiable: if any `CREATE TABLE` or `CREATE INDEX` is missing
    `IF NOT EXISTS`, or if the ENUM block doesn't have the
    duplicate_object exception handler, the second apply errors out.
    """
    url = os.environ["TEST_POSTGRES_URL"]
    for i in range(2):
        result = subprocess.run(
            ["psql", url, "-f", str(MIGRATION_PATH)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Migration apply (iter {i}) failed:\nstderr={result.stderr}\nstdout={result.stdout}"
        )
