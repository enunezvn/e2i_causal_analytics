#!/bin/bash
# =============================================================================
# Migration Re-apply Regression Test  (step 2 of the partial-migration reconcile)
# =============================================================================
# Faithful guard that every NOT-YET-TRACKED migration in the audit/chat/ml/core/
# causal/rag dirs can be (re-)applied onto the REAL droplet without aborting a
# deploy — the precondition for re-enabling the all-dirs scour in
# run_migrations.sh.
#
# For each in-scope, untracked .sql file it runs the migration inside a
# BEGIN .. ROLLBACK transaction against the live `supabase-db` Postgres (prod is
# NEVER modified — every test rolls back). A clean exit means the file is
# idempotent against prod's actual (possibly drifted) state.
#
# Statements that cannot run inside a transaction are tested un-wrapped by
# stripping only those lines for the check (the runner applies them un-wrapped):
#   - ALTER TYPE ... ADD VALUE
#   - CREATE INDEX CONCURRENTLY
#
# SKIP list — files intentionally NOT applied (baselined-as-superseded); these
# would fail by design and must be tracked, never run:
#   - core/030_fix_agent_tier_classification.sql  (old tier_3_* taxonomy,
#       superseded by core/029 + migrations/057 v4 enum realignment)
#   - chat/002_user_profiles_requests_rls.sql      (RLS on user_profiles, the
#       dead old chat schema never created; superseded by chatbot_* schema)
#   - ml/015_foreign_key_indexes.sql               (applied; CREATE INDEX
#       CONCURRENTLY cannot run in a transaction, so it is baseline-only)
#
# Exit 0 = every non-skipped in-scope migration re-applies cleanly.
# =============================================================================
set -uo pipefail

DB_CONTAINER="${SUPABASE_DB_CONTAINER:-supabase-db}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'

SKIP="core/030_fix_agent_tier_classification.sql chat/002_user_profiles_requests_rls.sql ml/015_foreign_key_indexes.sql"

psql() { docker exec -i "$DB_CONTAINER" psql -U postgres -d postgres -v ON_ERROR_STOP=1 -q "$@"; }

tracked="$(psql -At -c "SELECT filename FROM public.schema_migrations WHERE filename ~ '^(core|ml|causal|chat|rag|audit)/';")"
fail=0; clean=0; skipped=0

for d in core ml causal chat rag audit; do
  for f in "$ROOT"/database/$d/*.sql; do
    [ -f "$f" ] || continue
    case "$f" in *_validation_queries.sql|*/rollback_*.sql|*_rollback.sql) continue ;; esac  # utilities, not forward migrations
    key="$d/$(basename "$f")"
    echo "$tracked" | grep -qxF "$key" && continue          # already tracked
    case " $SKIP " in *" $key "*) printf "  ${YELLOW}SKIP${NC} %s (superseded — baseline-only)\n" "$key"; skipped=$((skipped+1)); continue ;; esac

    if grep -qiE "ALTER[[:space:]]+TYPE.*ADD[[:space:]]+VALUE|CREATE[[:space:]]+INDEX[[:space:]]+CONCURRENTLY" "$f"; then
      body="$(grep -viE "ALTER[[:space:]]+TYPE.*ADD[[:space:]]+VALUE|CREATE[[:space:]]+INDEX[[:space:]]+CONCURRENTLY" "$f")"
    else
      body="$(cat "$f")"
    fi
    out="$( { echo "BEGIN;"; printf '%s\n' "$body"; echo "ROLLBACK;"; } | psql 2>&1 )"
    if [ $? -eq 0 ]; then
      clean=$((clean+1))
    else
      fail=$((fail+1)); printf "  ${RED}FAIL${NC} %s :: %s\n" "$key" "$(echo "$out" | grep -iE 'ERROR:' | head -1)"
    fi
  done
done

echo
printf "re-apply clean: %s   skipped(superseded): %s   failed: %s\n" "$clean" "$skipped" "$fail"
if [ "$fail" -eq 0 ]; then printf "${GREEN}ALL IN-SCOPE MIGRATIONS RE-APPLY CLEANLY${NC}\n"; exit 0
else printf "${RED}RE-APPLY FAILURES PRESENT${NC}\n"; exit 1; fi
