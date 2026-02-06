#!/usr/bin/env bash
# ==============================================================================
# E2I Causal Analytics - Docker Deployment Script
# ==============================================================================
# DEPRECATED: Use scripts/deploy.sh instead.
#
# This script is kept for backward compatibility with existing CI references.
# It now delegates to the new deploy script.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "[deploy-docker] DEPRECATED: Use scripts/deploy.sh instead"
exec "$SCRIPT_DIR/deploy.sh" "$@"
