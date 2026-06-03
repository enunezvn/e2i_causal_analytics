#!/bin/bash
# =============================================================================
# Migration Idempotency / Reconciliation Harness
# =============================================================================
# Faithful, isolated regression test for the partial-migration reconciliation
# (see docs/reports/memory-system-review-20260603.md "Reconciliation Plan").
#
# Several schema dirs (audit/chat/ml) were applied to the droplet by hand and
# left in an inconsistent partial state (an enum exists but its table does not,
# 4 of 5 monitoring tables missing, a policy referencing a non-existent table,
# etc.). Re-running them via scripts/run_migrations.sh would ABORT the deploy
# because the original files are NOT idempotent (bare CREATE TYPE / CREATE TABLE
# / CREATE INDEX / CREATE TRIGGER / CREATE POLICY).
#
# This harness PROVES each migration is safe to (re-)apply by running it against
# a THROWAWAY database in the SAME Postgres instance as the droplet:
#   - FAITHFUL: same server version (PG 15.x) + same extensions (vector, pgcrypto)
#   - ISOLATED: a dedicated `migration_idem_test` DB, dropped after; the prod
#     `postgres` database is NEVER written to (only DROP/CREATE of the test DB).
#
# For each migration it:
#   1. seeds the real droplet PRECONDITION (FK dependency tables + any
#      already-present partial state)
#   2. applies the migration ONCE   -> must succeed AND create expected objects
#   3. applies the migration AGAIN  -> must succeed (idempotency)
#
# Exit code 0 = all migrations create their objects and re-apply cleanly.
# Exit code 1 = at least one migration is non-idempotent or broken.
#
# Usage:
#   scripts/test_migration_idempotency.sh            # run all
#   SUPABASE_DB_CONTAINER=supabase-db scripts/test_migration_idempotency.sh
# =============================================================================

set -uo pipefail

DB_CONTAINER="${SUPABASE_DB_CONTAINER:-supabase-db}"
TEST_DB="migration_idem_test"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP="$(mktemp -d)"
FAIL=0

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

# supabase_admin is the instance superuser (the `postgres` role cannot CREATE
# EXTENSION vector). On the droplet the extensions are pre-installed, so the
# real run_migrations.sh (as `postgres`) only USES vector(...); here we install
# it into the throwaway DB as supabase_admin to reproduce that precondition.
PSQL_USER="${MIGRATION_TEST_PSQL_USER:-supabase_admin}"
psql_admin() { docker exec -i "$DB_CONTAINER" psql -U "$PSQL_USER" -d postgres -v ON_ERROR_STOP=1 -q "$@"; }
psql_test()  { docker exec -i "$DB_CONTAINER" psql -U "$PSQL_USER" -d "$TEST_DB" -v ON_ERROR_STOP=1 -q "$@"; }

reset_db() {
  psql_admin -c "DROP DATABASE IF EXISTS $TEST_DB WITH (FORCE);" >/dev/null
  psql_admin -c "CREATE DATABASE $TEST_DB;" >/dev/null
  psql_test  -c "CREATE EXTENSION IF NOT EXISTS vector; CREATE EXTENSION IF NOT EXISTS pgcrypto;" >/dev/null
  # Supabase provides an `auth` schema with auth.uid() (current user from JWT).
  # It exists on the droplet; stub it here so RLS policies referencing it are
  # creatable. Returns NULL (no JWT in a DDL idempotency test — we assert the
  # policy is CREATABLE/re-runnable, not that it filters rows).
  psql_test  -c "CREATE SCHEMA IF NOT EXISTS auth; CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS \$fn\$ SELECT NULL::uuid \$fn\$;" >/dev/null
}

# test_one NAME MIGRATION_FILE PRECONDITION_SQL ASSERT_SQL
test_one() {
  local name="$1" file="$2" precond="$3" assert="$4"
  printf "=== %s (%s) ===\n" "$name" "$file"

  if [ ! -f "$ROOT/$file" ]; then
    printf "  ${RED}MISSING FILE${NC}\n"; FAIL=1; return
  fi

  reset_db

  if [ -n "$precond" ]; then
    if ! printf '%s' "$precond" | psql_test >"$TMP/pre.log" 2>&1; then
      printf "  ${RED}PRECONDITION FAILED${NC}\n"; sed 's/^/    /' "$TMP/pre.log"; FAIL=1; return
    fi
  fi

  # Run 1 — first application
  if ! psql_test <"$ROOT/$file" >"$TMP/run1.log" 2>&1; then
    printf "  ${RED}RUN 1 FAILED${NC} (migration errors on first apply)\n"
    sed 's/^/    /' "$TMP/run1.log" | tail -15; FAIL=1; return
  fi

  # Assert expected objects exist after first apply
  if ! printf '%s' "$assert" | psql_test >"$TMP/assert.log" 2>&1; then
    printf "  ${RED}ASSERT FAILED${NC} (expected objects not created)\n"
    sed 's/^/    /' "$TMP/assert.log" | tail -15; FAIL=1; return
  fi

  # Run 2 — re-application MUST be a clean no-op (idempotency)
  if ! psql_test <"$ROOT/$file" >"$TMP/run2.log" 2>&1; then
    printf "  ${RED}RUN 2 FAILED${NC} (migration is NOT idempotent)\n"
    sed 's/^/    /' "$TMP/run2.log" | tail -15; FAIL=1; return
  fi

  printf "  ${GREEN}PASS${NC} (creates expected objects + idempotent re-apply)\n"
}

# Assertion helper: raises if a named object is absent.
assert_exists() {
  # kind name [extra]
  cat <<SQL
DO \$\$ BEGIN IF NOT EXISTS ($1) THEN RAISE EXCEPTION 'MISSING: $2'; END IF; END \$\$;
SQL
}

# ---------------------------------------------------------------------------
# chat/036_user_roles.sql  — FULLY APPLIED on droplet (untracked).
#   Precondition: base chatbot_user_profiles with is_admin (RBAC target table).
# ---------------------------------------------------------------------------
test_one "chat/036 user_roles" "database/chat/036_user_roles.sql" \
"CREATE TABLE chatbot_user_profiles (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), is_admin boolean DEFAULT false);" \
"DO \$\$ BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='user_role') THEN RAISE EXCEPTION 'MISSING enum user_role'; END IF;
   IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name='chatbot_user_profiles' AND column_name='role') THEN RAISE EXCEPTION 'MISSING column chatbot_user_profiles.role'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_indexes WHERE indexname='idx_chatbot_user_profiles_role') THEN RAISE EXCEPTION 'MISSING index idx_chatbot_user_profiles_role'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname='role_level') THEN RAISE EXCEPTION 'MISSING function role_level'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname='has_role') THEN RAISE EXCEPTION 'MISSING function has_role'; END IF;
 END \$\$;"

# ---------------------------------------------------------------------------
# audit/012_security_audit_log.sql — NOT APPLIED; RLS policy references a
#   user_roles table that nothing creates. Reconciled to use chatbot_user_profiles.
#   Precondition: chatbot_user_profiles WITH role col + user_role enum.
# ---------------------------------------------------------------------------
test_one "audit/012 security_audit_log" "database/audit/012_security_audit_log.sql" \
"CREATE TYPE user_role AS ENUM ('viewer','analyst','operator','admin');
 CREATE TABLE chatbot_user_profiles (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), is_admin boolean DEFAULT false, role user_role DEFAULT 'viewer' NOT NULL);" \
"DO \$\$ BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='security_audit_log') THEN RAISE EXCEPTION 'MISSING table security_audit_log'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='security_audit_log' AND policyname='security_audit_admin_read') THEN RAISE EXCEPTION 'MISSING policy admin_read'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='security_audit_log' AND policyname='security_audit_service_insert') THEN RAISE EXCEPTION 'MISSING policy service_insert'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE tablename='security_audit_log' AND policyname='security_audit_user_own_read') THEN RAISE EXCEPTION 'MISSING policy user_own_read'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname='check_ip_should_block') THEN RAISE EXCEPTION 'MISSING function check_ip_should_block'; END IF;
 END \$\$;"

# ---------------------------------------------------------------------------
# ml/013_tool_composer_tables.sql — FULLY APPLIED on droplet (untracked).
#   Self-contained (creates its own types/tables); needs only vector ext.
# ---------------------------------------------------------------------------
test_one "ml/013 tool_composer" "database/ml/013_tool_composer_tables.sql" \
"" \
"DO \$\$ BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='routing_pattern') THEN RAISE EXCEPTION 'MISSING enum routing_pattern'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='composition_status') THEN RAISE EXCEPTION 'MISSING enum composition_status'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='tool_registry') THEN RAISE EXCEPTION 'MISSING table tool_registry'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='composer_episodes') THEN RAISE EXCEPTION 'MISSING table composer_episodes'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='composition_steps') THEN RAISE EXCEPTION 'MISSING table composition_steps'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='trg_tool_registry_updated') THEN RAISE EXCEPTION 'MISSING trigger trg_tool_registry_updated'; END IF;
   IF (SELECT count(*) FROM tool_registry) < 13 THEN RAISE EXCEPTION 'MISSING seed rows in tool_registry (expected 13)'; END IF;
 END \$\$;"

# ---------------------------------------------------------------------------
# ml/017_model_monitoring_tables.sql — HALF-APPLIED (drift_history + 3 enums
#   present; 4 monitoring tables + alert_status_enum missing). FK deps required.
# ---------------------------------------------------------------------------
test_one "ml/017 model_monitoring" "database/ml/017_model_monitoring_tables.sql" \
"CREATE TABLE ml_model_registry (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), model_name varchar, stage text);
 CREATE TABLE ml_experiments (id uuid PRIMARY KEY DEFAULT gen_random_uuid());
 CREATE TABLE ml_deployments (id uuid PRIMARY KEY DEFAULT gen_random_uuid());
 CREATE TABLE ml_training_runs (id uuid PRIMARY KEY DEFAULT gen_random_uuid());" \
"DO \$\$ BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='alert_status_enum') THEN RAISE EXCEPTION 'MISSING enum alert_status_enum'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ml_performance_metrics') THEN RAISE EXCEPTION 'MISSING table ml_performance_metrics'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ml_monitoring_alerts') THEN RAISE EXCEPTION 'MISSING table ml_monitoring_alerts'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ml_monitoring_runs') THEN RAISE EXCEPTION 'MISSING table ml_monitoring_runs'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ml_retraining_history') THEN RAISE EXCEPTION 'MISSING table ml_retraining_history'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='trigger_create_drift_alert') THEN RAISE EXCEPTION 'MISSING trigger trigger_create_drift_alert'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_views WHERE viewname='ml_model_health_dashboard') THEN RAISE EXCEPTION 'MISSING view ml_model_health_dashboard'; END IF;
 END \$\$;"

# ---------------------------------------------------------------------------
# ml/021_ab_results_tables.sql — NOT APPLIED. FK deps ml_experiments + twin_simulations.
# ---------------------------------------------------------------------------
test_one "ml/021 ab_results" "database/ml/021_ab_results_tables.sql" \
"CREATE TABLE ml_experiments (id uuid PRIMARY KEY DEFAULT gen_random_uuid(), experiment_name text, prediction_target text);
 CREATE TABLE twin_simulations (simulation_id uuid PRIMARY KEY DEFAULT gen_random_uuid());" \
"DO \$\$ BEGIN
   IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='ab_analysis_type') THEN RAISE EXCEPTION 'MISSING enum ab_analysis_type'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname='srm_severity') THEN RAISE EXCEPTION 'MISSING enum srm_severity'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ab_experiment_results') THEN RAISE EXCEPTION 'MISSING table ab_experiment_results'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ab_srm_checks') THEN RAISE EXCEPTION 'MISSING table ab_srm_checks'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_tables WHERE tablename='ab_fidelity_comparisons') THEN RAISE EXCEPTION 'MISSING table ab_fidelity_comparisons'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_trigger WHERE tgname='tr_fidelity_grade') THEN RAISE EXCEPTION 'MISSING trigger tr_fidelity_grade'; END IF;
   IF NOT EXISTS (SELECT 1 FROM pg_proc WHERE proname='ab_calculate_fidelity_grade') THEN RAISE EXCEPTION 'MISSING function ab_calculate_fidelity_grade'; END IF;
 END \$\$;"

# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------
psql_admin -c "DROP DATABASE IF EXISTS $TEST_DB WITH (FORCE);" >/dev/null 2>&1
rm -rf "$TMP"

echo
if [ "$FAIL" -eq 0 ]; then
  printf "${GREEN}ALL MIGRATIONS IDEMPOTENT & COMPLETE${NC}\n"
  exit 0
else
  printf "${RED}IDEMPOTENCY FAILURES PRESENT${NC}\n"
  exit 1
fi
