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
# Usage:
#   scripts/run_benchmarks.sh           # run all 3 benchmark files
#   scripts/run_benchmarks.sh cascade   # run just the cascade benchmark
#   scripts/run_benchmarks.sh hybrid    # run just the hybrid-retriever benchmark
#   scripts/run_benchmarks.sh bm25      # run just the bm25-rebuild benchmark

set -euo pipefail

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

case "${1:-all}" in
  cascade) run_cascade ;;
  hybrid)  run_hybrid ;;
  bm25)    run_bm25 ;;
  all)
    run_cascade
    run_hybrid
    run_bm25
    ;;
  *)
    echo "Usage: $0 [cascade|hybrid|bm25|all]" >&2
    exit 2
    ;;
esac
