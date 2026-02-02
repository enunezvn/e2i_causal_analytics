#!/bin/bash
# =============================================================================
# E2I Causal Analytics — Frontend Batched Test Runner with RAM Monitoring
# =============================================================================
# Runs frontend unit tests (Vitest) and optionally E2E tests (Playwright)
# in directory-level batches, monitors available RAM between batches, and
# reports per-batch results.
#
# Usage:
#   ./scripts/run_frontend_tests_batched.sh              # Run all unit batches
#   ./scripts/run_frontend_tests_batched.sh --batch 3    # Run single batch by number
#   ./scripts/run_frontend_tests_batched.sh --from 5     # Resume from batch 5
#   ./scripts/run_frontend_tests_batched.sh --e2e        # Include E2E batches
#   ./scripts/run_frontend_tests_batched.sh --list       # List all batches
#   ./scripts/run_frontend_tests_batched.sh --dry-run    # Show what would run
#   ./scripts/run_frontend_tests_batched.sh --coverage   # Enable coverage
#
# RAM thresholds (adjustable via env vars):
#   RAM_THRESHOLD_HIGH=3000  — proceed normally above this (MB)
#   RAM_THRESHOLD_LOW=2000   — proceed with warning (MB)
#   RAM_CRITICAL=1000        — pause and wait for recovery (MB)
# =============================================================================

set -o pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------
PROJECT="${E2I_PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
FRONTEND_DIR="$PROJECT/frontend"
RESULTS_DIR="${E2I_RESULTS_DIR:-$PROJECT/docs/results}"
RESULTS_FILE="$RESULTS_DIR/frontend_test_batch_results_$(date +%Y%m%d_%H%M%S).txt"
RAM_THRESHOLD_HIGH="${RAM_THRESHOLD_HIGH:-3000}"
RAM_THRESHOLD_LOW="${RAM_THRESHOLD_LOW:-2000}"
RAM_CRITICAL="${RAM_CRITICAL:-1000}"
VITEST_TIMEOUT="${VITEST_TIMEOUT:-10000}"  # 10s per test (ms)

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
RUN_SINGLE_BATCH=""
RUN_FROM_BATCH=""
DRY_RUN=false
LIST_ONLY=false
INCLUDE_E2E=false
WITH_COVERAGE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)  RUN_SINGLE_BATCH="$2"; shift 2 ;;
        --from)   RUN_FROM_BATCH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --list)   LIST_ONLY=true; shift ;;
        --e2e)    INCLUDE_E2E=true; shift ;;
        --coverage) WITH_COVERAGE=true; shift ;;
        --help|-h)
            head -18 "$0" | tail -16
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Batch definitions — ordered lightest to heaviest
# ---------------------------------------------------------------------------
# Format: "type|glob_or_dir|label"
#   type: "vitest" or "playwright"
#   glob_or_dir: file glob relative to frontend/src (vitest) or spec file (playwright)
#   label: human-readable name
BATCHES=(
    # Group 1: Config & lib (lightweight, no components)
    "vitest|src/config/env.test.ts|config/env"
    "vitest|src/lib/api-schemas.test.ts src/lib/api-client.test.ts|lib/api"

    # Group 2: API layer
    "vitest|src/api/kpi.test.ts src/api/predictions.test.ts src/api/monitoring.test.ts|api"

    # Group 3: Hooks — utility (no React rendering)
    "vitest|src/hooks/use-e2i-filters.test.ts src/hooks/use-e2i-highlights.test.ts src/hooks/use-e2i-validation.test.ts src/hooks/use-user-preferences.test.ts|hooks/e2i-utilities"
    "vitest|src/hooks/use-cytoscape.test.ts src/hooks/use-d3.test.ts src/hooks/use-websocket.test.ts src/hooks/use-websocket-cache-sync.test.tsx src/hooks/use-query-error.test.ts|hooks/integration"

    # Group 4: Hooks — API (MSW-backed)
    "vitest|src/hooks/api/use-cognitive.test.ts src/hooks/api/use-explain.test.ts src/hooks/api/use-graph.test.ts src/hooks/api/use-kpi.test.ts|hooks/api-1"
    "vitest|src/hooks/api/use-memory.test.ts src/hooks/api/use-monitoring.test.ts src/hooks/api/use-rag.test.ts src/hooks/api/use-predictions.test.ts src/hooks/api/use-digital-twin.test.ts|hooks/api-2"

    # Group 5: UI components (small, isolated)
    "vitest|src/components/ui/PriorityBadge.test.tsx src/components/ui/query-error-state.test.tsx src/components/chat/ValidationBadge.test.tsx|components/ui"

    # Group 6: Digital twin components
    "vitest|src/components/digital-twin/RecommendationCards.test.tsx src/components/digital-twin/ScenarioResults.test.tsx src/components/digital-twin/SimulationPanel.test.tsx|components/digital-twin"

    # Group 7: Visualization components — part 1
    "vitest|src/components/visualizations/agents/agents.test.tsx src/components/visualizations/dashboard/dashboard.test.tsx src/components/visualizations/charts/charts.test.tsx src/components/visualizations/charts/MultiAxisLineChart.test.tsx|visualizations/agents-dashboard-charts"

    # Group 8: Visualization components — part 2
    "vitest|src/components/visualizations/graph/graph.test.tsx src/components/visualizations/graph/GraphFilters.test.tsx src/components/visualizations/graph/CytoscapeGraph.test.tsx src/components/visualizations/experiments/ExperimentCard.test.tsx src/components/visualizations/drift/DriftVisualization.test.tsx|visualizations/graph-experiments-drift"

    # Group 9: Visualization components — part 3 (heavier: causal, SHAP)
    "vitest|src/components/visualizations/causal/causal.test.tsx src/components/visualizations/shap/shap.test.tsx|visualizations/causal-shap"

    # Group 10: Providers
    "vitest|src/providers/E2ICopilotProvider.test.tsx|providers"

    # Group 11: Pages — group 1
    "vitest|src/pages/Home.test.tsx src/pages/DataQuality.test.tsx src/pages/Monitoring.test.tsx src/pages/TimeSeries.test.tsx src/pages/FeatureImportance.test.tsx src/pages/KPIDictionary.test.tsx src/pages/SystemHealth.test.tsx|pages/core"

    # Group 12: Pages — group 2 (heavier: full agent/ML pages)
    "vitest|src/pages/InterventionImpact.test.tsx src/pages/ModelPerformance.test.tsx src/pages/PredictiveAnalytics.test.tsx src/pages/KnowledgeGraph.test.tsx src/pages/CausalDiscovery.test.tsx src/pages/MemoryArchitecture.test.tsx src/pages/DigitalTwin.test.tsx src/pages/AgentOrchestration.test.tsx|pages/analytics"
)

# E2E batches — only included with --e2e flag
E2E_BATCHES=(
    # Group 13: E2E — core navigation
    "playwright|e2e/specs/home.spec.ts e2e/specs/kpi-dictionary.spec.ts e2e/specs/data-quality.spec.ts e2e/specs/monitoring.spec.ts|e2e/core-navigation"

    # Group 14: E2E — analytics pages
    "playwright|e2e/specs/causal-discovery.spec.ts e2e/specs/intervention-impact.spec.ts e2e/specs/feature-importance.spec.ts e2e/specs/time-series.spec.ts|e2e/analytics"

    # Group 15: E2E — ML & agent pages
    "playwright|e2e/specs/model-performance.spec.ts e2e/specs/predictive-analytics.spec.ts e2e/specs/agent-orchestration.spec.ts e2e/specs/ai-insights.spec.ts|e2e/ml-agents"

    # Group 16: E2E — system & advanced
    "playwright|e2e/specs/system-health.spec.ts e2e/specs/memory-architecture.spec.ts e2e/specs/knowledge-graph.spec.ts e2e/specs/digital-twin.spec.ts|e2e/system-advanced"
)

if $INCLUDE_E2E; then
    BATCHES+=("${E2E_BATCHES[@]}")
fi

TOTAL_BATCHES=${#BATCHES[@]}

# ---------------------------------------------------------------------------
# --list: print batch table and exit
# ---------------------------------------------------------------------------
if $LIST_ONLY; then
    printf "%-6s %-8s %-42s %s\n" "Batch" "Type" "Label" "Files"
    printf "%-6s %-8s %-42s %s\n" "-----" "------" "-----" "-----"
    for i in "${!BATCHES[@]}"; do
        IFS='|' read -r type files label <<< "${BATCHES[$i]}"
        file_count=$(echo "$files" | wc -w)
        printf "%-6s %-8s %-42s %s file(s)\n" "$((i + 1))" "$type" "$label" "$file_count"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# Verify frontend directory exists
# ---------------------------------------------------------------------------
if [ ! -d "$FRONTEND_DIR" ]; then
    echo "ERROR: Frontend directory not found: $FRONTEND_DIR"
    exit 1
fi

# ---------------------------------------------------------------------------
# RAM monitoring helpers
# ---------------------------------------------------------------------------
get_available_ram_mb() {
    free -m | awk '/Mem:/{print $7}'
}

wait_for_ram() {
    local ram=$(get_available_ram_mb)
    local waited=0
    while [ "$ram" -lt "$RAM_CRITICAL" ] && [ "$waited" -lt 120 ]; do
        echo "  [!] RAM critical (${ram}MB). Pausing 30s for recovery..."
        sleep 30
        waited=$((waited + 30))
        ram=$(get_available_ram_mb)
    done
    if [ "$ram" -lt "$RAM_CRITICAL" ]; then
        echo "  [!] RAM still critical after 120s (${ram}MB). Proceeding anyway."
    fi
}

ram_status_label() {
    local ram=$(get_available_ram_mb)
    if [ "$ram" -gt "$RAM_THRESHOLD_HIGH" ]; then
        echo "ok"
    elif [ "$ram" -gt "$RAM_THRESHOLD_LOW" ]; then
        echo "low"
    else
        echo "critical"
    fi
}

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
total_passed=0
total_failed=0
total_skipped=0
failed_batches=()

# ---------------------------------------------------------------------------
# run_vitest_batch — execute a vitest batch
# ---------------------------------------------------------------------------
run_vitest_batch() {
    local batch_num="$1"
    local files="$2"
    local label="$3"

    # Build file list with full paths
    local file_args=()
    for f in $files; do
        local full_path="$FRONTEND_DIR/$f"
        if [ ! -f "$full_path" ]; then
            echo "  [WARN] File not found: $full_path"
            continue
        fi
        file_args+=("$f")
    done

    if [ ${#file_args[@]} -eq 0 ]; then
        echo "  [SKIP] $label — no test files found"
        echo "$batch_num|vitest|$label|SKIP|0|0|0|0s|N/A" >> "$RESULTS_FILE"
        echo ""
        return
    fi

    wait_for_ram
    local ram=$(get_available_ram_mb)
    local ram_label=$(ram_status_label)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Batch $batch_num/$TOTAL_BATCHES: $label [vitest]"
    echo "  RAM: ${ram}MB ($ram_label) | Files: ${#file_args[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local coverage_flag=""
    if $WITH_COVERAGE; then
        coverage_flag="--coverage"
    fi

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: npx vitest run ${file_args[*]} $coverage_flag"
        echo ""
        return
    fi

    local start_time=$(date +%s)

    local output
    output=$(cd "$FRONTEND_DIR" && npx vitest run \
        --reporter=verbose \
        --testTimeout="$VITEST_TIMEOUT" \
        $coverage_flag \
        "${file_args[@]}" 2>&1) || true

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Strip ANSI escape codes before parsing
    local clean_output
    clean_output=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')

    # Parse vitest results from the "Tests" summary line (not "Test Files")
    local summary_line
    summary_line=$(echo "$clean_output" | grep -E '^\s+Tests\s+' | tail -1)

    local passed=$(echo "$summary_line" | grep -oP '\d+(?=\s+passed)' || echo 0)
    local failed=$(echo "$summary_line" | grep -oP '\d+(?=\s+failed)' || echo 0)
    local skipped=$(echo "$summary_line" | grep -oP '\d+(?=\s+skipped)' || echo 0)

    [ -z "$passed" ] && passed=0
    [ -z "$failed" ] && failed=0
    [ -z "$skipped" ] && skipped=0

    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))
    total_skipped=$((total_skipped + skipped))

    local status="PASS"
    if [ "$failed" -gt 0 ]; then
        status="FAIL"
        failed_batches+=("Batch $batch_num: $label ($failed failed)")
    fi

    # Check for vitest crash (no summary line found)
    if [ "$passed" -eq 0 ] && [ "$failed" -eq 0 ]; then
        local has_error=$(echo "$clean_output" | grep -c -E "^(Error|FAIL|TypeError|ReferenceError|SyntaxError)" || true)
        if [ "$has_error" -gt 0 ]; then
            status="ERROR"
            failed_batches+=("Batch $batch_num: $label (vitest error — no summary)")
        fi
    fi

    local ram_after=$(get_available_ram_mb)
    echo "  Result: $passed passed, $failed failed, $skipped skipped (${elapsed}s)"
    echo "  RAM after: ${ram_after}MB"

    if [ "$status" = "FAIL" ] || [ "$status" = "ERROR" ]; then
        echo "  FAILURES:"
        echo "$clean_output" | grep -E "FAIL|AssertionError|Error:" | head -20 | sed 's/^/    /'
    fi

    echo "$batch_num|vitest|$label|$status|$passed|$failed|$skipped|${elapsed}s|${ram}MB>${ram_after}MB" >> "$RESULTS_FILE"
    echo ""
}

# ---------------------------------------------------------------------------
# run_playwright_batch — execute a playwright batch
# ---------------------------------------------------------------------------
run_playwright_batch() {
    local batch_num="$1"
    local files="$2"
    local label="$3"

    # Build file list with full paths
    local file_args=()
    for f in $files; do
        local full_path="$FRONTEND_DIR/$f"
        if [ ! -f "$full_path" ]; then
            echo "  [WARN] File not found: $full_path"
            continue
        fi
        file_args+=("$f")
    done

    if [ ${#file_args[@]} -eq 0 ]; then
        echo "  [SKIP] $label — no test files found"
        echo "$batch_num|playwright|$label|SKIP|0|0|0|0s|N/A" >> "$RESULTS_FILE"
        echo ""
        return
    fi

    wait_for_ram
    local ram=$(get_available_ram_mb)
    local ram_label=$(ram_status_label)

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Batch $batch_num/$TOTAL_BATCHES: $label [playwright]"
    echo "  RAM: ${ram}MB ($ram_label) | Files: ${#file_args[@]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: npx playwright test ${file_args[*]}"
        echo ""
        return
    fi

    local start_time=$(date +%s)

    local output
    output=$(cd "$FRONTEND_DIR" && npx playwright test \
        --reporter=list \
        "${file_args[@]}" 2>&1) || true

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Strip ANSI escape codes before parsing
    local clean_output
    clean_output=$(echo "$output" | sed 's/\x1b\[[0-9;]*m//g')

    # Parse playwright results from summary line
    # Playwright outputs: "  2 passed (5.2s)" or "  1 failed  2 passed (8s)"
    local passed=$(echo "$clean_output" | grep -oP '\d+(?=\s+passed)' | tail -1 || echo 0)
    local failed=$(echo "$clean_output" | grep -oP '\d+(?=\s+failed)' | tail -1 || echo 0)
    local skipped=$(echo "$clean_output" | grep -oP '\d+(?=\s+skipped)' | tail -1 || echo 0)

    [ -z "$passed" ] && passed=0
    [ -z "$failed" ] && failed=0
    [ -z "$skipped" ] && skipped=0

    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))
    total_skipped=$((total_skipped + skipped))

    local status="PASS"
    if [ "$failed" -gt 0 ]; then
        status="FAIL"
        failed_batches+=("Batch $batch_num: $label ($failed failed)")
    fi

    local ram_after=$(get_available_ram_mb)
    echo "  Result: $passed passed, $failed failed, $skipped skipped (${elapsed}s)"
    echo "  RAM after: ${ram_after}MB"

    if [ "$status" = "FAIL" ]; then
        echo "  FAILURES:"
        echo "$clean_output" | grep -E "^\s+\d+\).*FAIL|Error:" | head -20 | sed 's/^/    /'
    fi

    echo "$batch_num|playwright|$label|$status|$passed|$failed|$skipped|${elapsed}s|${ram}MB>${ram_after}MB" >> "$RESULTS_FILE"
    echo ""
}

# ---------------------------------------------------------------------------
# run_batch — dispatch to vitest or playwright runner
# ---------------------------------------------------------------------------
run_batch() {
    local batch_num="$1"
    local type="$2"
    local files="$3"
    local label="$4"

    case "$type" in
        vitest)     run_vitest_batch "$batch_num" "$files" "$label" ;;
        playwright) run_playwright_batch "$batch_num" "$files" "$label" ;;
        *)          echo "  [ERROR] Unknown batch type: $type"; return 1 ;;
    esac
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
> "$RESULTS_FILE"

echo "=============================================="
echo "  E2I Frontend Test Suite — Batched Runner"
echo "  Started: $(date)"
echo "  Project: $PROJECT"
echo "  Frontend: $FRONTEND_DIR"
echo "  Available RAM: $(get_available_ram_mb)MB"
echo "  RAM thresholds: high=$RAM_THRESHOLD_HIGH low=$RAM_THRESHOLD_LOW critical=$RAM_CRITICAL"
echo "  Results: $RESULTS_FILE"
if [ -n "$RUN_SINGLE_BATCH" ]; then
    echo "  Mode: single batch (#$RUN_SINGLE_BATCH)"
elif [ -n "$RUN_FROM_BATCH" ]; then
    echo "  Mode: resume from batch #$RUN_FROM_BATCH"
elif $DRY_RUN; then
    echo "  Mode: dry run"
else
    echo "  Mode: full run ($TOTAL_BATCHES batches)"
fi
if $INCLUDE_E2E; then
    echo "  E2E: included"
else
    echo "  E2E: excluded (use --e2e to include)"
fi
if $WITH_COVERAGE; then
    echo "  Coverage: enabled"
fi
echo "=============================================="
echo ""

for i in "${!BATCHES[@]}"; do
    batch_num=$((i + 1))
    IFS='|' read -r type files label <<< "${BATCHES[$i]}"

    # --batch: run only the specified batch
    if [ -n "$RUN_SINGLE_BATCH" ] && [ "$batch_num" -ne "$RUN_SINGLE_BATCH" ]; then
        continue
    fi

    # --from: skip batches before the start point
    if [ -n "$RUN_FROM_BATCH" ] && [ "$batch_num" -lt "$RUN_FROM_BATCH" ]; then
        continue
    fi

    run_batch "$batch_num" "$type" "$files" "$label"
done

echo "=============================================="
echo "  FINAL SUMMARY"
echo "  Finished: $(date)"
echo "  RAM: $(get_available_ram_mb)MB available"
echo "=============================================="
echo "  Passed:  $total_passed"
echo "  Failed:  $total_failed"
echo "  Skipped: $total_skipped"
echo "  Results: $RESULTS_FILE"
echo "=============================================="
echo ""

if [ ${#failed_batches[@]} -gt 0 ]; then
    echo "FAILED BATCHES:"
    for fb in "${failed_batches[@]}"; do
        echo "  $fb"
    done
    exit 1
else
    echo "ALL BATCHES PASSED"
    exit 0
fi
