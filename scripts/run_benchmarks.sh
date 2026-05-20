#!/usr/bin/env bash
# Run the performance-slice benchmark harnesses (issue #391, boxes 1-3).
#
# Each benchmark file is a `pytest -m benchmark` invocation against a
# specific test file. We separate them into individual pytest runs so a
# single benchmark's failure doesn't blow up the rest, and so the
# JUnit-XML artifacts upload cleanly per benchmark.
#
# Per [[causal-role-propagation-phases-2-7-close-20260518]]: benchmarks
# MUST run with `-p no:xdist` because xdist can starve async retriever
# calls; we also clear addopts so the pyproject default doesn't sneak
# xdist back in.
#
# Failure-isolation contract (codex iter-0 M1 closure): with the `all`
# argument, EVERY benchmark runs even if a prior one fails. We capture
# each invocation's exit code and exit at the end with the max — so the
# overall script still fails if any benchmark fails (CI signal preserved),
# but every benchmark gets a chance to write its junit-xml artifact. We
# explicitly disable `-e` for the run-all path because `set -e` would
# abort after the first non-zero exit and the comment block above would
# be false in practice.
#
# Usage:
#   scripts/run_benchmarks.sh           # run all 3 benchmark files
#   scripts/run_benchmarks.sh cascade   # run just the cascade benchmark
#   scripts/run_benchmarks.sh hybrid    # run just the hybrid-retriever benchmark
#   scripts/run_benchmarks.sh bm25      # run just the bm25-rebuild benchmark

set -uo pipefail

# Resolve repo root regardless of cwd (the script lives at scripts/).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p test-results

# Common pytest invocation shape. Per the comment block above:
#   -p no:xdist + -o "addopts=" = no parallelism (benchmark stability)
#   -m benchmark                = select only @pytest.mark.benchmark tests
#   --junitxml + junit_family   = stable artifact shape for CI consumers
PYTEST_BASE=(
  pytest
  -m benchmark
  -v
  --tb=short
  -p no:xdist
  -o "addopts="
  -o junit_family=xunit2
)

run_cascade() {
  echo "==> Cascade BFS latency benchmark (issue #391 box 1)"
  "${PYTEST_BASE[@]}" \
    tests/benchmarks/test_cascade_latency.py \
    --junitxml=test-results/benchmark-cascade.xml
}

run_hybrid() {
  echo "==> HybridRetriever fused-search latency benchmark (issue #391 box 2)"
  "${PYTEST_BASE[@]}" \
    tests/benchmarks/test_hybrid_retriever_latency.py \
    --junitxml=test-results/benchmark-hybrid.xml
}

run_bm25() {
  echo "==> BM25 rebuild-time benchmark (issue #391 box 3)"
  "${PYTEST_BASE[@]}" \
    tests/benchmarks/test_bm25_rebuild_time.py \
    --junitxml=test-results/benchmark-bm25.xml
}

# Run a single benchmark target. Echoes the outcome and updates the global
# `worst_exit` so the script can surface the worst result at the end.
worst_exit=0
run_target() {
  local name="$1"
  shift
  echo
  echo "--- $name --------------------------------------------------------------"
  "$@"
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "!!! $name FAILED (exit $rc)"
    if [ "$rc" -gt "$worst_exit" ]; then
      worst_exit=$rc
    fi
  fi
  return 0
}

case "${1:-all}" in
  cascade) run_cascade; exit $? ;;
  hybrid)  run_hybrid; exit $? ;;
  bm25)    run_bm25; exit $? ;;
  all)
    # `set +e` is implicit (we already removed -e above). Run each target
    # via the `run_target` helper so an early failure does NOT abort the
    # remaining benchmarks. The script's overall exit code is the worst
    # individual exit so CI still flags failures.
    run_target cascade run_cascade
    run_target hybrid  run_hybrid
    run_target bm25    run_bm25
    if [ "$worst_exit" -ne 0 ]; then
      echo
      echo "==> One or more benchmarks failed (worst exit=$worst_exit)."
    fi
    exit "$worst_exit"
    ;;
  *)
    echo "Usage: $0 [cascade|hybrid|bm25|all]" >&2
    exit 2
    ;;
esac
