#!/usr/bin/env bash
# Export OpenAPI spec and generate Redoc HTML documentation.
#
# Usage:
#   ./scripts/generate_api_docs.sh          # spec + HTML
#   ./scripts/generate_api_docs.sh --spec   # spec only (no npx needed)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUT_DIR="$PROJECT_DIR/docs/api"

mkdir -p "$OUT_DIR"

# Export OpenAPI spec
python -m scripts.export_openapi --output "$OUT_DIR/openapi.json"
echo "OpenAPI spec written to docs/api/openapi.json"

if [[ "${1:-}" == "--spec" ]]; then
    exit 0
fi

# Generate Redoc static HTML
npx --yes @redocly/cli build-docs "$OUT_DIR/openapi.json" \
    --output "$OUT_DIR/index.html" \
    --title "E2I Causal Analytics API"

echo "Done. Open docs/api/index.html in a browser."
