#!/bin/bash
# =============================================================================
# Migration Scour Coverage + Safety Test  (embeddings step: re-enable all-dirs)
# =============================================================================
# Behavioral guard for the deploy migration runner once the all-dirs scour is
# re-enabled (the other schema dirs were reconciled in PR #676 + #682, so the
# runner may now cover them — see docs/reports/memory-system-review-20260603.md).
#
# It runs the REAL scripts/run_migrations.sh in --dry-run against an EMPTY,
# prod-faithful ephemeral Postgres (same supabase/postgres image as the droplet).
# With an empty schema_migrations table, --dry-run lists EVERY file the runner
# would apply on a fresh deploy — i.e. the exact surface a real deploy touches.
# We assert two contracts on that surface:
#
#   COVERAGE  — the six reconciled dirs (core/ml/causal/chat/rag/audit) ARE now
#               scanned (each contributes at least one <dir>/ key).
#   SAFETY    — NO destructive/utility file is ever in the apply set:
#                 * rollback_*.sql / *_rollback.sql   (DROP live tables — #682
#                   near-miss: ml/rollback_023.sql DROPs estimator_evaluations)
#                 * *_validation_queries.sql          (read-only query bundles)
#
# RED before the runner change (the six dirs are not scanned → COVERAGE fails).
# GREEN after (dirs added + rollback exclusion → COVERAGE + SAFETY hold).
# =============================================================================
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Deterministic single-phase-init Postgres (the supabase/postgres prod image
# restarts mid-init, racing the readiness gate). The runner's dir-scan +
# exclusion logic is image-independent; the faithful-to-prod check is the
# separate real-droplet dry-run, not this unit test.
IMAGE="postgres:15-alpine"
CTR="e2i_mig_scour_test_$$"
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
fail=0
note() { printf "  %b%s%b %s\n" "$1" "$2" "$NC" "$3"; }

cleanup() { docker rm -f "$CTR" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "=== starting ephemeral Postgres ($IMAGE) ==="
docker run -d --name "$CTR" -e POSTGRES_PASSWORD=test "$IMAGE" >/dev/null 2>&1 || {
  echo -e "${RED}could not start ephemeral postgres${NC}"; exit 2; }

# Image-agnostic readiness: a written sentinel must SURVIVE (guards a mid-init restart).
stable=0
for _ in $(seq 1 90); do
  docker exec -i "$CTR" psql -U postgres -d postgres -q -c 'CREATE TABLE IF NOT EXISTS _ready_probe(x int);' >/dev/null 2>&1
  if [ "$(docker exec -i "$CTR" psql -U postgres -d postgres -tAc "SELECT to_regclass('public._ready_probe') IS NOT NULL" 2>/dev/null)" = "t" ]; then
    stable=$((stable + 1)); [ "$stable" -ge 3 ] && break
  else stable=0; fi
  sleep 1
done
docker exec -i "$CTR" psql -U postgres -d postgres -q -c 'DROP TABLE IF EXISTS _ready_probe;' >/dev/null 2>&1
[ "$stable" -ge 3 ] || { echo -e "${RED}ephemeral postgres never stabilised${NC}"; exit 2; }
echo "    ready."

echo "=== run_migrations.sh --dry-run against EMPTY db (docker-exec mode) ==="
out="$(SUPABASE_DB_CONTAINER="$CTR" bash "$ROOT/scripts/run_migrations.sh" --dry-run 2>&1)"
rc=$?
# strip ANSI colour codes for matching
clean="$(printf '%s' "$out" | sed -r 's/\x1B\[[0-9;]*[mK]//g')"

[ "$rc" -eq 0 ] || { echo -e "${RED}runner exited $rc${NC}"; printf '%s\n' "$clean" | tail -20; fail=1; }

echo "--- COVERAGE: each reconciled dir contributes a pending key ---"
for d in core ml causal chat rag audit; do
  if printf '%s\n' "$clean" | grep -qE "\[PENDING\][[:space:]]+$d/"; then
    note "$GREEN" "PASS" "dir scanned: $d/"
  else
    note "$RED" "FAIL" "dir NOT scanned: $d/  (expected pending $d/* keys)"; fail=1
  fi
done

echo "--- SAFETY: no destructive/utility file is ever in the apply set ---"
for pat in 'rollback_' '_rollback\.sql' '_validation_queries\.sql'; do
  if printf '%s\n' "$clean" | grep -qE "\[PENDING\].*$pat"; then
    hit="$(printf '%s\n' "$clean" | grep -E "\[PENDING\].*$pat" | head -1)"
    note "$RED" "FAIL" "destructive/utility file in apply set ($pat): $hit"; fail=1
  else
    note "$GREEN" "PASS" "excluded from apply set: $pat"
  fi
done

echo
if [ "$fail" -eq 0 ]; then echo -e "${GREEN}SCOUR COVERAGE + SAFETY: ALL PASS${NC}"; exit 0
else echo -e "${RED}SCOUR TEST FAILURES PRESENT${NC}"; exit 1; fi
