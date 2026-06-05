"""Migration 058 — REVOKE anon/authenticated over-grant + DROP execute_custom_sql.

Closes M9 / #703: the Supabase default `GRANT ALL TO anon, authenticated` left
~101 no-RLS public tables + 105 anon-granted views readable/writable by the anon
role through the internet-bound Kong gateway (PHI tables included), and an
anon-callable `execute_custom_sql(text)` arbitrary-SQL-as-postgres function
bypassed any REVOKE entirely.

Static-content checks run anywhere (CI-runnable). The functional test needs
``TEST_POSTGRES_URL`` + local ``psql``; it skips otherwise (standard
``pytest.mark.skipif`` — NOT a self-declared "deferred"). The DECISIVE evidence
is the faithful droplet run documented in the PR: under the OLD state an anon-key
``GET /rest/v1/<phi_table>`` returned 206 with row counts; after this migration
(+ the service-role code change) anon is denied while the backend (service-role)
still reads.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

MIGRATION_PATH = (
    Path(__file__).parent.parent.parent
    / "database"
    / "migrations"
    / "058_revoke_anon_authenticated_overgrant.sql"
)


def _have_psql() -> bool:
    return subprocess.run(["which", "psql"], capture_output=True).returncode == 0


def _content() -> str:
    assert MIGRATION_PATH.exists(), f"Migration file not found at {MIGRATION_PATH}"
    return MIGRATION_PATH.read_text()


# ---------------------------------------------------------------------------
# Static-content checks (run anywhere)
# ---------------------------------------------------------------------------


def test_drops_execute_custom_sql() -> None:
    """The anon-callable arbitrary-SQL RPC must be DROPped (idempotently)."""
    content = _content().lower()
    assert "drop function if exists public.execute_custom_sql(text)" in content, (
        "Migration must DROP FUNCTION IF EXISTS public.execute_custom_sql(text) "
        "(the anon-executable arbitrary-SQL-as-postgres bypass)."
    )


def test_revokes_all_tables_from_anon_and_authenticated() -> None:
    """Every public table+view must be revoked from BOTH anon and authenticated."""
    content = _content().lower()
    assert "revoke all on all tables in schema public from anon, authenticated" in content, (
        "Migration must REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon, authenticated."
    )


def test_handles_materialized_views() -> None:
    """Materialized views aren't covered by ON ALL TABLES; must be revoked too."""
    content = _content().lower()
    assert "relkind = 'm'" in content and "revoke all on public." in content, (
        "Migration must also REVOKE on materialized views (relkind='m'), which "
        "ON ALL TABLES does not cover."
    )


def test_fixes_default_privileges_for_both_owner_roles() -> None:
    """Default privileges must be neutralised for BOTH postgres and supabase_admin.

    Both roles carry the public-schema default that re-grants ALL to
    anon/authenticated on every new table (verified via pg_default_acl); missing
    either lets future tables silently re-open the hole.
    """
    content = _content().lower()
    for role in ("postgres", "supabase_admin"):
        assert f"alter default privileges for role {role} in schema public" in content, (
            f"Missing ALTER DEFAULT PRIVILEGES FOR ROLE {role} IN SCHEMA public"
        )
    assert "revoke all on tables from anon, authenticated" in content, (
        "Default-privilege change must REVOKE ALL ON TABLES FROM anon, authenticated."
    )


def test_no_script_level_txn_control_and_idempotent() -> None:
    """No bare BEGIN;/COMMIT;/ROLLBACK; — run_migrations.sh wraps each migration in
    ``psql --single-transaction`` (the runner owns the outer txn; enforced by
    test_migrations_no_inner_txn.py). PL/pgSQL DO-block ``BEGIN``/``END $$;`` are
    fine (they don't end a script-level statement with a bare keyword+semicolon).
    Idempotent via DROP IF EXISTS + (inherently re-runnable) REVOKE.
    """
    content = _content()
    # Only consider lines that END a statement (`;`); the DO blocks' `BEGIN`
    # opener has no trailing `;`, and `END $$;` is not a bare `END;`.
    stmt_lines = {ln.strip().lower() for ln in content.splitlines() if ln.strip().endswith(";")}
    offenders = stmt_lines & {"begin;", "commit;", "rollback;", "end;", "abort;"}
    assert not offenders, (
        f"Migration must NOT contain script-level transaction control {offenders} — "
        "run_migrations.sh owns the outer txn via psql --single-transaction."
    )
    assert "DROP FUNCTION IF EXISTS" in content, (
        "Use DROP FUNCTION IF EXISTS for idempotent re-apply."
    )


# ---------------------------------------------------------------------------
# Functional (needs TEST_POSTGRES_URL + psql)
# ---------------------------------------------------------------------------

_SETUP_SQL = """
BEGIN;
-- Ensure the supabase roles exist (no-op on a real Supabase DB).
DO $$ BEGIN CREATE ROLE anon NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE authenticated NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;
DO $$ BEGIN CREATE ROLE supabase_admin NOLOGIN; EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- A public table granted to anon (mimics the over-grant) + the dangerous RPC.
CREATE TABLE IF NOT EXISTS public.t058_phi_probe (id int primary key, secret text);
GRANT ALL ON public.t058_phi_probe TO anon, authenticated;
CREATE OR REPLACE FUNCTION public.execute_custom_sql(sql_query text)
RETURNS jsonb LANGUAGE plpgsql SECURITY DEFINER AS $fn$
BEGIN RETURN '[]'::jsonb; END; $fn$;
GRANT EXECUTE ON FUNCTION public.execute_custom_sql(text) TO anon;
COMMIT;
"""

_ASSERT_SQL = """
SELECT 'ANON_TABLE_SELECT:' ||
       has_table_privilege('anon', 'public.t058_phi_probe', 'SELECT')::text;
SELECT 'AUTH_TABLE_SELECT:' ||
       has_table_privilege('authenticated', 'public.t058_phi_probe', 'SELECT')::text;
SELECT 'EXEC_FN_EXISTS:' ||
       (to_regprocedure('public.execute_custom_sql(text)') IS NOT NULL)::text;
"""

_CLEANUP_SQL = "DROP TABLE IF EXISTS public.t058_phi_probe;"


@pytest.mark.skipif(
    not os.environ.get("TEST_POSTGRES_URL") or not _have_psql(),
    reason="needs TEST_POSTGRES_URL env + local psql",
)
def test_functional_anon_loses_table_access_and_rpc_dropped() -> None:
    """Apply 058 and prove anon/authenticated lose table SELECT and the RPC is gone."""
    url = os.environ["TEST_POSTGRES_URL"]

    def _run(sql: str, *, by_file: str | None = None):
        cmd = ["psql", url, "-v", "ON_ERROR_STOP=1", "-t", "-A"]
        cmd += ["-f", by_file] if by_file else ["-c", sql]
        return subprocess.run(cmd, capture_output=True, text=True)

    try:
        setup = _run(_SETUP_SQL)
        assert setup.returncode == 0, f"setup failed:\n{setup.stderr}"

        apply = _run("", by_file=str(MIGRATION_PATH))
        assert apply.returncode == 0, f"058 apply failed:\n{apply.stderr}\n{apply.stdout}"

        out = _run(_ASSERT_SQL)
        assert out.returncode == 0, f"assert query failed:\n{out.stderr}"
        body = out.stdout
        assert "ANON_TABLE_SELECT:false" in body, f"anon must lose SELECT; got:\n{body}"
        assert "AUTH_TABLE_SELECT:false" in body, f"authenticated must lose SELECT; got:\n{body}"
        assert "EXEC_FN_EXISTS:false" in body, f"execute_custom_sql must be dropped; got:\n{body}"
    finally:
        _run(_CLEANUP_SQL)
