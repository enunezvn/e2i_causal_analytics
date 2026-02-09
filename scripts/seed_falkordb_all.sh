#!/bin/bash
# =============================================================================
# E2I Causal Analytics - FalkorDB Graph Seeding Wrapper
# =============================================================================
# Seeds both FalkorDB graphs (e2i_causal + e2i_semantic) if they're empty.
#
# Usage:
#   ./scripts/seed_falkordb_all.sh              # Seed empty graphs (host port 6381)
#   ./scripts/seed_falkordb_all.sh --force      # Clear and re-seed both graphs
#   ./scripts/seed_falkordb_all.sh --docker     # Use internal port 6379 (inside Docker)
#
# Environment:
#   FALKORDB_PASSWORD   Required. Read from .env if not set.
#   FALKORDB_HOST       Default: localhost
#   FALKORDB_PORT       Default: 6381 (host) or 6379 (--docker)
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Parse flags
FORCE=false
DOCKER_MODE=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --force) FORCE=true; shift ;;
        --docker) DOCKER_MODE=true; shift ;;
        --help|-h)
            echo "Usage: $0 [--force] [--docker]"
            echo "  --force   Clear and re-seed both graphs"
            echo "  --docker  Use internal port 6379 (for use inside Docker network)"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Source .env for FALKORDB_PASSWORD if not already set
if [ -z "${FALKORDB_PASSWORD:-}" ] && [ -f "$PROJECT_DIR/.env" ]; then
    # shellcheck disable=SC1091
    set -a
    source "$PROJECT_DIR/.env"
    set +a
fi

if [ -z "${FALKORDB_PASSWORD:-}" ]; then
    echo "ERROR: FALKORDB_PASSWORD is not set. Export it or add to .env"
    exit 1
fi

# Connection defaults
FALKORDB_HOST="${FALKORDB_HOST:-localhost}"
if $DOCKER_MODE; then
    FALKORDB_PORT="${FALKORDB_PORT:-6379}"
else
    FALKORDB_PORT="${FALKORDB_PORT:-6381}"
fi

echo "=== FalkorDB Graph Seeding ==="
echo "Host: $FALKORDB_HOST:$FALKORDB_PORT"
echo "Force: $FORCE"
echo ""

# Activate venv if available (host-side execution)
if [ -d "$PROJECT_DIR/.venv" ]; then
    # shellcheck disable=SC1091
    source "$PROJECT_DIR/.venv/bin/activate"
fi

# ---------------------------------------------------------------------------
# Helper: count nodes in a graph
# ---------------------------------------------------------------------------
count_nodes() {
    local graph_name="$1"
    python3 -c "
from falkordb import FalkorDB
import sys
try:
    db = FalkorDB(host='$FALKORDB_HOST', port=$FALKORDB_PORT, password='$FALKORDB_PASSWORD')
    g = db.select_graph('$graph_name')
    result = g.query('MATCH (n) RETURN count(n)')
    print(result.result_set[0][0])
except Exception as e:
    # Graph doesn't exist yet or connection error
    print(0)
" 2>/dev/null || echo 0
}

# ---------------------------------------------------------------------------
# Seed e2i_causal
# ---------------------------------------------------------------------------
CAUSAL_COUNT=$(count_nodes "e2i_causal")
echo "e2i_causal: $CAUSAL_COUNT nodes"

if $FORCE || [ "$CAUSAL_COUNT" -eq 0 ]; then
    if $FORCE && [ "$CAUSAL_COUNT" -gt 0 ]; then
        echo "  Forcing re-seed (clearing $CAUSAL_COUNT nodes)..."
        CLEAR_FLAG="--clear-first"
    else
        echo "  Graph is empty -- seeding..."
        CLEAR_FLAG="--clear-first"
    fi

    python3 "$PROJECT_DIR/scripts/seed_falkordb.py" \
        --host "$FALKORDB_HOST" \
        --port "$FALKORDB_PORT" \
        $CLEAR_FLAG || {
        echo "ERROR: e2i_causal seeding failed"
        exit 1
    }
else
    echo "  Already has $CAUSAL_COUNT nodes -- skipping"
fi

# ---------------------------------------------------------------------------
# Seed e2i_semantic
# ---------------------------------------------------------------------------
SEMANTIC_COUNT=$(count_nodes "e2i_semantic")
echo ""
echo "e2i_semantic: $SEMANTIC_COUNT nodes"

if $FORCE || [ "$SEMANTIC_COUNT" -eq 0 ]; then
    if $FORCE && [ "$SEMANTIC_COUNT" -gt 0 ]; then
        echo "  Forcing re-seed (clearing $SEMANTIC_COUNT nodes)..."
        CLEAR_FLAG="--clear-first"
    else
        echo "  Graph is empty -- seeding..."
        CLEAR_FLAG="--clear-first"
    fi

    FALKORDB_HOST="$FALKORDB_HOST" \
    FALKORDB_PORT="$FALKORDB_PORT" \
    FALKORDB_PASSWORD="$FALKORDB_PASSWORD" \
    python3 "$PROJECT_DIR/scripts/seed_semantic_graph.py" \
        $CLEAR_FLAG || {
        echo "ERROR: e2i_semantic seeding failed"
        exit 1
    }
else
    echo "  Already has $SEMANTIC_COUNT nodes -- skipping"
fi

# ---------------------------------------------------------------------------
# Verify both graphs
# ---------------------------------------------------------------------------
echo ""
echo "=== Verification ==="
CAUSAL_FINAL=$(count_nodes "e2i_causal")
SEMANTIC_FINAL=$(count_nodes "e2i_semantic")

echo "e2i_causal:   $CAUSAL_FINAL nodes"
echo "e2i_semantic: $SEMANTIC_FINAL nodes"

FAILED=false
if [ "$CAUSAL_FINAL" -eq 0 ]; then
    echo "ERROR: e2i_causal is still empty!"
    FAILED=true
fi
if [ "$SEMANTIC_FINAL" -eq 0 ]; then
    echo "ERROR: e2i_semantic is still empty!"
    FAILED=true
fi

if $FAILED; then
    echo ""
    echo "Seeding FAILED. Check logs above."
    exit 1
fi

echo ""
echo "=== Seeding complete ==="
