#!/bin/bash
# =============================================================================
# Migration Apply-Time Transaction Handling Test
# =============================================================================
# The dry-run coverage test (test_migration_scour_coverage.sh) lists files but
# never APPLIES them, so apply-time transaction aborts are invisible to it. This
# test runs the REAL scripts/run_migrations.sh in APPLY mode against a temp
# PROJECT_ROOT of fixture migrations on a prod-faithful ephemeral Postgres, and
# asserts the runner correctly handles the statements that cannot live inside a
# single wrapping transaction:
#
#   CONCURRENTLY  — CREATE/DROP INDEX CONCURRENTLY aborts under --single-transaction
#                   ('cannot run inside a transaction block'). The runner must
#                   detect it and run the file UN-WRAPPED. (MED-1)
#   SELF-COMMIT   — a file with its own BEGIN;..COMMIT; truncates the wrapper; the
#                   appended tracking INSERT must NOT ride inside / be skipped by
#                   the file's committed txn. A self-COMMIT file that FAILS after
#                   its COMMIT must be left UNTRACKED (so the next deploy retries),
#                   never falsely recorded as applied. (MED-2)
#   ADD VALUE     — existing enum un-wrap path stays green (regression guard).
#
# RED before the runner fix: the CONCURRENTLY fixture aborts the run (exit 1) and
# nothing is tracked. GREEN after: all good fixtures apply + are tracked; the
# post-COMMIT-failure fixture aborts AND is left untracked.
# =============================================================================
set -uo pipefail

SRC_RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_migrations.sh"
# Deterministic single-phase-init Postgres: the runner logic under test
# (--single-transaction wrapping, CONCURRENTLY, self-COMMIT truncation, tracking)
# is vanilla PG transaction semantics, identical to the supabase/postgres prod
# image, which only adds extensions/roles. The supabase image RESTARTS mid-init
# (its pg_isready healthcheck passes before that restart), wiping state applied
# in the window — a test-harness race, not a runner bug. The faithful-to-prod
# check is the separate real-droplet dry-run, not this unit test.
IMAGE="postgres:15-alpine"
CTR="e2i_mig_txn_test_$$"
ROOT="$(mktemp -d)"
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
fail=0
note() { printf "  %b%s%b %s\n" "$1" "$2" "$NC" "$3"; }

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; rm -rf "$ROOT"; }
trap cleanup EXIT

# ---- fixture migration tree -------------------------------------------------
mkdir -p "$ROOT/scripts" "$ROOT/database/migrations"
cp "$SRC_RUNNER" "$ROOT/scripts/run_migrations.sh"
cat > "$ROOT/database/migrations/001_normal.sql"        <<'SQL'
CREATE TABLE IF NOT EXISTS txn_t1 (id int);
SQL
cat > "$ROOT/database/migrations/002_concurrently.sql"  <<'SQL'
CREATE INDEX CONCURRENTLY IF NOT EXISTS txn_ix1 ON txn_t1 (id);
SQL
cat > "$ROOT/database/migrations/003_self_commit_ok.sql" <<'SQL'
BEGIN;
CREATE TABLE IF NOT EXISTS txn_t2 (id int);
COMMIT;
SQL
cat > "$ROOT/database/migrations/004_addvalue.sql" <<'SQL'
DO $$ BEGIN CREATE TYPE txn_enum AS ENUM ('a'); EXCEPTION WHEN duplicate_object THEN null; END $$;
ALTER TYPE txn_enum ADD VALUE IF NOT EXISTS 'b';
SQL

run_q() { docker exec -i "$CTR" psql -U postgres -d postgres -tAc "$1" 2>/dev/null; }

# ---- ephemeral prod-faithful Postgres --------------------------------------
echo "=== starting ephemeral Postgres ($IMAGE) ==="
docker run -d --name "$CTR" -e POSTGRES_PASSWORD=test "$IMAGE" >/dev/null 2>&1 || {
  echo -e "${RED}could not start ephemeral postgres${NC}"; exit 2; }
wait_ready() {                       # image-agnostic: a written sentinel must SURVIVE
  local stable=0                     # (guards against a mid-init restart wiping state)
  for _ in $(seq 1 90); do
    docker exec -i "$CTR" psql -U postgres -d postgres -q -c \
      'CREATE TABLE IF NOT EXISTS _ready_probe(x int);' >/dev/null 2>&1
    if [ "$(run_q "SELECT to_regclass('public._ready_probe') IS NOT NULL")" = "t" ]; then
      stable=$((stable + 1)); [ "$stable" -ge 3 ] && break
    else stable=0; fi
    sleep 1
  done
  docker exec -i "$CTR" psql -U postgres -d postgres -q -c 'DROP TABLE IF EXISTS _ready_probe;' >/dev/null 2>&1
  [ "$stable" -ge 3 ]
}
wait_ready || { echo -e "${RED}ephemeral postgres never stabilised${NC}"; exit 2; }
echo "    ready."

# ---- APPLY the four good fixtures -------------------------------------------
echo "=== apply 4 fixtures (normal / CONCURRENTLY / self-COMMIT / ADD VALUE) ==="
out="$(SUPABASE_DB_CONTAINER="$CTR" bash "$ROOT/scripts/run_migrations.sh" 2>&1)"; rc=$?
clean="$(printf '%s' "$out" | sed -r 's/\x1B\[[0-9;]*[mK]//g')"

if [ "$rc" -eq 0 ]; then note "$GREEN" "PASS" "runner applied all fixtures without aborting (exit 0)"
else note "$RED" "FAIL" "runner aborted (exit $rc) — CONCURRENTLY likely wrapped in a txn"; printf '%s\n' "$clean" | grep -iE 'error|cannot|fail' | head -3; fail=1; fi

for f in 001_normal.sql 002_concurrently.sql 003_self_commit_ok.sql 004_addvalue.sql; do
  if [ "$(run_q "SELECT count(*) FROM public.schema_migrations WHERE filename='$f'")" = "1" ]; then
    note "$GREEN" "PASS" "tracked: $f"
  else note "$RED" "FAIL" "NOT tracked: $f"; fail=1; fi
done
# objects actually created
[ "$(run_q "SELECT to_regclass('public.txn_t1') IS NOT NULL")" = "t" ] && note "$GREEN" "PASS" "table txn_t1 created" || { note "$RED" "FAIL" "txn_t1 missing"; fail=1; }
[ "$(run_q "SELECT to_regclass('public.txn_ix1') IS NOT NULL")" = "t" ] && note "$GREEN" "PASS" "index txn_ix1 created (CONCURRENTLY applied)" || { note "$RED" "FAIL" "txn_ix1 missing"; fail=1; }
[ "$(run_q "SELECT to_regclass('public.txn_t2') IS NOT NULL")" = "t" ] && note "$GREEN" "PASS" "table txn_t2 created (self-COMMIT applied)" || { note "$RED" "FAIL" "txn_t2 missing"; fail=1; }

# ---- self-COMMIT file that FAILS after its COMMIT must NOT be tracked -------
echo "=== self-COMMIT file failing AFTER its COMMIT must be left UNTRACKED ==="
cat > "$ROOT/database/migrations/005_post_commit_fail.sql" <<'SQL'
BEGIN;
CREATE TABLE IF NOT EXISTS txn_t3 (id int);
COMMIT;
SELECT 1/0;
SQL
out2="$(SUPABASE_DB_CONTAINER="$CTR" bash "$ROOT/scripts/run_migrations.sh" 2>&1)"; rc2=$?
cnt005="$(run_q "SELECT count(*) FROM public.schema_migrations WHERE filename='005_post_commit_fail.sql'")"
if [ "$rc2" -ne 0 ]; then note "$GREEN" "PASS" "runner failed the post-COMMIT-error file (exit $rc2)"
else note "$RED" "FAIL" "runner did NOT fail on a post-COMMIT error (exit 0)"; fail=1; fi
if [ "$cnt005" = "0" ]; then
  note "$GREEN" "PASS" "005 left UNTRACKED — next deploy will retry (no committed-but-tracked desync)"
else note "$RED" "FAIL" "005 was FALSELY tracked despite failing — wedge risk"; fail=1; fi

echo
if [ "$fail" -eq 0 ]; then echo -e "${GREEN}APPLY-TIME TXN HANDLING: ALL PASS${NC}"; exit 0
else echo -e "${RED}APPLY-TIME TXN TEST FAILURES PRESENT${NC}"; exit 1; fi
