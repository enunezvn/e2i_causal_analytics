#!/usr/bin/env bash
# =============================================================================
# E2I BentoML Model Deployment Helper
# =============================================================================
#
# Deploys a trained model to the persistent BentoML service.
#
# Usage:
#   ./scripts/bentoml/deploy_model.sh --model-tag tier0_abc123:v5
#   ./scripts/bentoml/deploy_model.sh --model-name tier0_tier0_e2
#   ./scripts/bentoml/deploy_model.sh                   # auto-discover latest
#
# What it does:
#   1. Validates the model exists in the BentoML store
#   2. Updates the env file with the model selection
#   3. Restarts e2i-bentoml.service
#   4. Waits for health check to pass
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ENV_FILE="${PROJECT_DIR}/deploy/e2i-bentoml.env"
SERVICE_NAME="e2i-bentoml"
HEALTH_URL="http://localhost:3000/healthz"
MAX_WAIT=30

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info()    { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# Parse arguments
MODEL_TAG=""
MODEL_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --model-tag)
            MODEL_TAG="$2"
            shift 2
            ;;
        --model-name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--model-tag TAG] [--model-name NAME]"
            echo ""
            echo "Options:"
            echo "  --model-tag TAG    Exact BentoML model tag (e.g. tier0_abc123:v5)"
            echo "  --model-name NAME  Model name (uses :latest version)"
            echo ""
            echo "If no options given, auto-discovers the latest model."
            exit 0
            ;;
        *)
            error "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Verify bentoml is available
BENTOML_BIN="${PROJECT_DIR}/.venv/bin/bentoml"
if [[ ! -x "${BENTOML_BIN}" ]]; then
    # Try system/venv path
    BENTOML_BIN="$(command -v bentoml 2>/dev/null || true)"
    if [[ -z "${BENTOML_BIN}" ]]; then
        error "bentoml not found. Ensure the venv is set up."
        exit 1
    fi
fi

# Validate model exists in store
if [[ -n "${MODEL_TAG}" ]]; then
    info "Verifying model tag: ${MODEL_TAG}"
    if ! "${BENTOML_BIN}" models get "${MODEL_TAG}" &>/dev/null; then
        error "Model not found in BentoML store: ${MODEL_TAG}"
        echo "  Available models:"
        "${BENTOML_BIN}" models list 2>/dev/null | head -20
        exit 1
    fi
    info "Model verified: ${MODEL_TAG}"
elif [[ -n "${MODEL_NAME}" ]]; then
    info "Verifying model name: ${MODEL_NAME}"
    if ! "${BENTOML_BIN}" models get "${MODEL_NAME}:latest" &>/dev/null; then
        error "Model not found in BentoML store: ${MODEL_NAME}:latest"
        echo "  Available models:"
        "${BENTOML_BIN}" models list 2>/dev/null | head -20
        exit 1
    fi
    info "Model verified: ${MODEL_NAME}:latest"
else
    info "No model specified — service will auto-discover latest model"
fi

# Update env file with model selection
info "Updating ${ENV_FILE}"

# Read existing env file and update/add model vars
TMPFILE=$(mktemp)
# Copy non-model lines
grep -v '^E2I_BENTOML_MODEL_TAG=' "${ENV_FILE}" 2>/dev/null | \
    grep -v '^E2I_BENTOML_MODEL_NAME=' > "${TMPFILE}" || true

if [[ -n "${MODEL_TAG}" ]]; then
    echo "E2I_BENTOML_MODEL_TAG=${MODEL_TAG}" >> "${TMPFILE}"
    info "Set E2I_BENTOML_MODEL_TAG=${MODEL_TAG}"
elif [[ -n "${MODEL_NAME}" ]]; then
    echo "E2I_BENTOML_MODEL_NAME=${MODEL_NAME}" >> "${TMPFILE}"
    info "Set E2I_BENTOML_MODEL_NAME=${MODEL_NAME}"
fi

cp "${TMPFILE}" "${ENV_FILE}"
rm -f "${TMPFILE}"

# Check if service is installed
if ! systemctl list-unit-files "${SERVICE_NAME}.service" &>/dev/null; then
    warn "Service ${SERVICE_NAME} not installed. Install with:"
    echo "  sudo cp ${PROJECT_DIR}/deploy/e2i-bentoml.service /etc/systemd/system/"
    echo "  sudo systemctl daemon-reload"
    echo "  sudo systemctl enable ${SERVICE_NAME}"
    exit 1
fi

# Restart service
info "Restarting ${SERVICE_NAME} service..."
sudo systemctl restart "${SERVICE_NAME}"

# Wait for health
info "Waiting for health check (max ${MAX_WAIT}s)..."
for i in $(seq 1 "${MAX_WAIT}"); do
    if curl -sf "${HEALTH_URL}" &>/dev/null; then
        info "Service healthy after ${i}s"

        # Show model info
        echo ""
        info "Service status:"
        curl -sf http://localhost:3000/model_info 2>/dev/null | python3 -m json.tool 2>/dev/null || true
        echo ""
        info "Deployment complete."
        exit 0
    fi
    sleep 1
done

error "Service failed to become healthy within ${MAX_WAIT}s"
echo "  Check logs: sudo journalctl -u ${SERVICE_NAME} -n 50 --no-pager"
sudo systemctl status "${SERVICE_NAME}" --no-pager || true
exit 1
