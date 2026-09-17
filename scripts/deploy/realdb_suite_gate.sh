#!/usr/bin/env bash
# Deploy gate (#2065): the learning-loop real-DB suite, on the droplet, BEFORE migrations.
#
# deploy.yml calls this right after the tree is reset to NEW_SHA and before
# `bash scripts/run_migrations.sh`; a non-zero exit fails the deploy with nothing flipped or
# migrated. Why here: CI's GitHub-hosted runners cannot reach the droplet's database, so
# tests/unit/test_database/learning_loop/ skips there; the deploy is the one automated run on
# the box. The suite derives what is pending from prod's ledger, so before a deploy that adds
# migrations it rehearses them on a throwaway copy of prod (upgrade-path tests), and it always
# runs the behaviour tests against the post-deploy schema. It writes only to throwaway
# containers (e2i-learnloop-pg-*, 1 GiB cap); prod is read inside READ ONLY transactions.
#
# Fails closed, never open. A red suite, a timeout, a suite that passed nothing, an unreadable
# memory reading or too little memory to run it all refuse the deploy. Skipping the gate on a
# busy box would ship exactly the deploy it exists to check; the recovery is to free memory
# (or wait for the other load) and re-run the deploy: `gh workflow run deploy.yml`.
#
# Budget, measured 2026-09-17 on this box: 3m27s wall, 1.06 GiB peak pytest RSS, throwaway
# Postgres ~130 MiB. MIN_AVAILABLE_MIB leaves ~0.8 GiB above that for the serving stack.
# SUITE_TIMEOUT plus the slowest measured rollout (10.5 min) stays inside the rollout's 30m
# SSH command_timeout (tests/unit/test_docker/test_deploy_realdb_suite_gate_2065.py).
set -uo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$PROJECT_DIR/.venv/bin/python"
SUITE="tests/unit/test_database/learning_loop/"
SUITE_TIMEOUT="12m"
MIN_AVAILABLE_MIB=2048
# Test seam only (the meta-tests point it at a fake meminfo); the deploy never sets it.
MEMINFO="${E2I_REALDB_GATE_MEMINFO:-/proc/meminfo}"

refuse() {
  echo "==> ERROR: real-DB suite gate (#2065): $*"
  exit 1
}

avail_kib=$(awk '/^MemAvailable:/ {print $2}' "$MEMINFO" 2>/dev/null)
[ -n "$avail_kib" ] || refuse "cannot read MemAvailable from ${MEMINFO}; refusing to guess"
avail_mib=$((avail_kib / 1024))
if [ "$avail_mib" -lt "$MIN_AVAILABLE_MIB" ]; then
  refuse "MemAvailable ${avail_mib} MiB is below the ${MIN_AVAILABLE_MIB} MiB the suite needs" \
    "(measured peak ~1.2 GiB). Free memory on the droplet, then re-run the deploy."
fi
[ -x "$PYTHON" ] || refuse "no interpreter at ${PYTHON}; the suite runs on the host venv"

JUNIT="$(mktemp -t e2i-realdb-gate-XXXXXX.xml)"
trap 'rm -f "$JUNIT"' EXIT

echo "==> real-DB suite gate: MemAvailable ${avail_mib} MiB; running ${SUITE} (timeout ${SUITE_TIMEOUT})"
# The rehearsal knob and live-LLM spend are removed from the environment, whatever the caller has.
(
  cd "$PROJECT_DIR" \
    && env -u E2I_DB_SIMULATE_PENDING -u E2I_LIVE_LLM E2I_DB_INTEGRATION=1 \
      timeout "$SUITE_TIMEOUT" "$PYTHON" -m pytest -n 0 -p no:cacheprovider -q -rs \
      --junitxml="$JUNIT" "$SUITE"
)
rc=$?

if [ "$rc" -ne 0 ]; then
  # A killed run cannot run its finalizers; remove the throwaway containers it left.
  (cd "$PROJECT_DIR" && "$PYTHON" -m tests.unit.test_database.learning_loop._pg --reap) || true
  [ "$rc" -eq 124 ] && refuse "the suite timed out after ${SUITE_TIMEOUT}"
  refuse "the suite failed (pytest exit ${rc}); its output is above"
fi

counts=$("$PYTHON" - "$JUNIT" <<'PY'
import sys
import xml.etree.ElementTree as ET

root = ET.parse(sys.argv[1]).getroot()
suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
total = {k: sum(int(s.get(k, 0)) for s in suites) for k in ("tests", "failures", "errors", "skipped")}
passed = total["tests"] - total["failures"] - total["errors"] - total["skipped"]
print(passed, total["skipped"], total["failures"] + total["errors"])
PY
) || refuse "could not read the suite's JUnit report"
read -r passed skipped broken <<< "$counts"
[ "$broken" -eq 0 ] || refuse "the suite reported ${broken} failure(s) or error(s)"
[ "$passed" -gt 0 ] || refuse "the suite ran nothing (0 passed, ${skipped} skipped); a skip is not coverage"

echo "==> real-DB suite gate: ${passed} passed, ${skipped} skipped"
