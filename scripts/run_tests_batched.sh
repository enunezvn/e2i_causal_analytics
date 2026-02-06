#!/bin/bash
# =============================================================================
# E2I Causal Analytics — Batched Test Runner with RAM Monitoring
# =============================================================================
# Runs unit tests in directory-level batches, monitors available RAM between
# batches, and dynamically adjusts pytest parallelism to prevent OOM/hangs.
#
# Usage:
#   ./scripts/run_tests_batched.sh              # Run all batches
#   ./scripts/run_tests_batched.sh --batch 22   # Run single batch by number
#   ./scripts/run_tests_batched.sh --from 20    # Resume from batch 20
#   ./scripts/run_tests_batched.sh --list       # List all batches
#   ./scripts/run_tests_batched.sh --dry-run    # Show what would run
#
# RAM thresholds (adjustable via env vars):
#   RAM_THRESHOLD_HIGH=4000  — use 2 workers above this (MB)
#   RAM_THRESHOLD_LOW=2500   — use 1 worker above this (MB)
#   RAM_CRITICAL=1500        — pause and wait for recovery (MB)
# =============================================================================

set -o pipefail

# ---------------------------------------------------------------------------
# Configuration (override via env vars)
# ---------------------------------------------------------------------------
PROJECT="${E2I_PROJECT:-$(cd "$(dirname "$0")/.." && pwd)}"
VENV="${E2I_VENV:-$PROJECT/.venv/bin}"
RESULTS_DIR="${E2I_RESULTS_DIR:-$PROJECT/docs/results}"
RESULTS_FILE="$RESULTS_DIR/test_batch_results_$(date +%Y%m%d_%H%M%S).txt"
RAM_THRESHOLD_HIGH="${RAM_THRESHOLD_HIGH:-4000}"
RAM_THRESHOLD_LOW="${RAM_THRESHOLD_LOW:-2500}"
RAM_CRITICAL="${RAM_CRITICAL:-1500}"
PYTEST_TIMEOUT="${PYTEST_TIMEOUT:-30}"

# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------
RUN_SINGLE_BATCH=""
RUN_FROM_BATCH=""
DRY_RUN=false
LIST_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch)  RUN_SINGLE_BATCH="$2"; shift 2 ;;
        --from)   RUN_FROM_BATCH="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --list)   LIST_ONLY=true; shift ;;
        --help|-h)
            head -16 "$0" | tail -14
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Batch definitions — ordered lightest to heaviest
# ---------------------------------------------------------------------------
# Format: "directory|label"
BATCHES=(
    # Group 1: Utils & lightweight
    "tests/unit/test_utils|unit/test_utils"
    "tests/unit/test_tool_registry|unit/test_tool_registry"
    "tests/unit/test_scripts|unit/test_scripts"
    "tests/unit/test_tasks|unit/test_tasks"
    "tests/unit/test_testing|unit/test_testing"
    # Group 2: Database & repositories
    "tests/unit/test_database|unit/test_database"
    "tests/unit/test_repositories|unit/test_repositories"
    "tests/unit/test_services|unit/test_services"
    # Group 3: API layer
    "tests/unit/test_api|unit/test_api"
    "tests/unit/api|unit/api"
    # Group 4: NLP & ontology
    "tests/unit/test_nlp|unit/test_nlp"
    "tests/unit/test_ontology|unit/test_ontology"
    "tests/unit/test_kpi|unit/test_kpi"
    # Group 5: Memory & RAG
    "tests/unit/test_memory|unit/test_memory"
    "tests/unit/test_rag|unit/test_rag"
    # Group 6: Feature store & ML core
    "tests/unit/test_feature_store|unit/test_feature_store"
    "tests/unit/test_ml|unit/test_ml"
    "tests/unit/test_mlops|unit/test_mlops"
    # Group 7: Causal engine
    "tests/unit/test_causal_engine|unit/test_causal_engine"
    # Group 8: Digital twin
    "tests/unit/test_digital_twin|unit/test_digital_twin"
    # Group 9: Agents (split by tier — heaviest)
    "tests/unit/test_agents/test_base|agents/test_base"
    "tests/unit/test_agents/test_ml_foundation|agents/ml_foundation"
    "tests/unit/test_agents/test_orchestrator|agents/orchestrator"
    "tests/unit/test_agents/test_tool_composer|agents/tool_composer"
    "tests/unit/test_agents/test_causal_impact|agents/causal_impact"
    "tests/unit/test_agents/test_gap_analyzer|agents/gap_analyzer"
    "tests/unit/test_agents/test_heterogeneous_optimizer|agents/heterogeneous_optimizer"
    "tests/unit/test_agents/test_drift_monitor|agents/drift_monitor"
    "tests/unit/test_agents/test_experiment_designer|agents/experiment_designer"
    "tests/unit/test_agents/test_experiment_monitor|agents/experiment_monitor"
    "tests/unit/test_agents/test_health_score|agents/health_score"
    "tests/unit/test_agents/test_prediction_synthesizer|agents/prediction_synthesizer"
    "tests/unit/test_agents/test_resource_optimizer|agents/resource_optimizer"
    "tests/unit/test_agents/test_explainer|agents/explainer"
    "tests/unit/test_agents/test_feedback_learner|agents/feedback_learner"
    "tests/unit/test_agents/test_cohort_constructor|agents/cohort_constructor"
    "tests/unit/test_agents/test_tier2_signal_routing|agents/tier2_signal_routing"
    # Group 10: Remaining
    "tests/unit/test_optimization|unit/test_optimization"
    "tests/unit/test_synthetic|unit/test_synthetic"
    "tests/unit/test_workers|unit/test_workers"
    "tests/unit/test_skills|unit/test_skills"
    "tests/unit/observability|unit/observability"
    "tests/unit/security|unit/security"
)

TOTAL_BATCHES=${#BATCHES[@]}

# ---------------------------------------------------------------------------
# --list: print batch table and exit
# ---------------------------------------------------------------------------
if $LIST_ONLY; then
    printf "%-6s %-40s %s\n" "Batch" "Label" "Directory"
    printf "%-6s %-40s %s\n" "-----" "-----" "---------"
    for i in "${!BATCHES[@]}"; do
        IFS='|' read -r dir label <<< "${BATCHES[$i]}"
        printf "%-6s %-40s %s\n" "$((i + 1))" "$label" "$dir"
    done
    exit 0
fi

# ---------------------------------------------------------------------------
# RAM monitoring helpers
# ---------------------------------------------------------------------------
get_available_ram_mb() {
    free -m | awk '/Mem:/{print $7}'
}

get_workers() {
    local ram=$(get_available_ram_mb)
    if [ "$ram" -gt "$RAM_THRESHOLD_HIGH" ]; then
        echo 2
    elif [ "$ram" -gt "$RAM_THRESHOLD_LOW" ]; then
        echo 1
    else
        echo 0  # sequential
    fi
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
        echo "  [!] RAM still critical after 120s (${ram}MB). Proceeding in sequential mode."
    fi
}

# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
total_passed=0
total_failed=0
total_errors=0
total_skipped=0
failed_batches=()

# ---------------------------------------------------------------------------
# run_batch — execute a single batch with RAM-adaptive parallelism
# ---------------------------------------------------------------------------
run_batch() {
    local batch_num="$1"
    local test_dir="$PROJECT/$2"
    local label="$3"

    # Skip if directory doesn't exist
    if [ ! -d "$test_dir" ]; then
        echo "  [SKIP] $label — directory not found: $test_dir"
        echo "$batch_num|$label|SKIP|0|0|0|0|0s|N/A" >> "$RESULTS_FILE"
        echo ""
        return
    fi

    # Check RAM before each batch
    wait_for_ram
    local ram=$(get_available_ram_mb)
    local workers=$(get_workers)

    local worker_flag=""
    local worker_label="sequential"
    if [ "$workers" -ge 2 ]; then
        worker_flag="-n 2 --dist=loadscope"
        worker_label="2 workers"
    elif [ "$workers" -eq 1 ]; then
        worker_flag="-n 1"
        worker_label="1 worker"
    fi

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Batch $batch_num/$TOTAL_BATCHES: $label"
    echo "  RAM: ${ram}MB available | Mode: $worker_label"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if $DRY_RUN; then
        echo "  [DRY RUN] Would run: pytest $test_dir $worker_flag --timeout=$PYTEST_TIMEOUT"
        echo ""
        return
    fi

    local start_time=$(date +%s)

    local output
    output=$($VENV/python -m pytest "$test_dir" \
        $worker_flag \
        --timeout="$PYTEST_TIMEOUT" \
        --tb=short \
        -q \
        --no-header \
        2>&1) || true

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))

    # Parse results from summary line
    local result_line=$(echo "$output" | tail -5 | grep -E "passed|failed|error|no tests")

    local passed=$(echo "$result_line" | grep -oP '\d+(?= passed)' || echo 0)
    local failed=$(echo "$result_line" | grep -oP '\d+(?= failed)' || echo 0)
    local errors=$(echo "$result_line" | grep -oP '\d+(?= error)' || echo 0)
    local skipped=$(echo "$result_line" | grep -oP '\d+(?= skipped)' || echo 0)

    [ -z "$passed" ] && passed=0
    [ -z "$failed" ] && failed=0
    [ -z "$errors" ] && errors=0
    [ -z "$skipped" ] && skipped=0

    total_passed=$((total_passed + passed))
    total_failed=$((total_failed + failed))
    total_errors=$((total_errors + errors))
    total_skipped=$((total_skipped + skipped))

    local status="PASS"
    if [ "$failed" -gt 0 ] || [ "$errors" -gt 0 ]; then
        status="FAIL"
        failed_batches+=("Batch $batch_num: $label ($failed failed, $errors errors)")
    fi

    local ram_after=$(get_available_ram_mb)
    echo "  Result: $passed passed, $failed failed, $errors errors, $skipped skipped (${elapsed}s)"
    echo "  RAM after: ${ram_after}MB"

    if [ "$status" = "FAIL" ]; then
        echo "  FAILURES:"
        echo "$output" | grep -E "^FAILED|^ERROR" | head -20 | sed 's/^/    /'
    fi

    echo "$batch_num|$label|$status|$passed|$failed|$errors|$skipped|${elapsed}s|${ram}MB>${ram_after}MB" >> "$RESULTS_FILE"
    echo ""
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
mkdir -p "$RESULTS_DIR"
> "$RESULTS_FILE"

echo "=============================================="
echo "  E2I Test Suite — Batched Runner"
echo "  Started: $(date)"
echo "  Project: $PROJECT"
echo "  Venv:    $VENV"
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
echo "=============================================="
echo ""

for i in "${!BATCHES[@]}"; do
    batch_num=$((i + 1))
    IFS='|' read -r dir label <<< "${BATCHES[$i]}"

    # --batch: run only the specified batch
    if [ -n "$RUN_SINGLE_BATCH" ] && [ "$batch_num" -ne "$RUN_SINGLE_BATCH" ]; then
        continue
    fi

    # --from: skip batches before the start point
    if [ -n "$RUN_FROM_BATCH" ] && [ "$batch_num" -lt "$RUN_FROM_BATCH" ]; then
        continue
    fi

    run_batch "$batch_num" "$dir" "$label"
done

echo "=============================================="
echo "  FINAL SUMMARY"
echo "  Finished: $(date)"
echo "  RAM: $(get_available_ram_mb)MB available"
echo "=============================================="
echo "  Passed:  $total_passed"
echo "  Failed:  $total_failed"
echo "  Errors:  $total_errors"
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
