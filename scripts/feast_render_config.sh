#!/usr/bin/env bash
# =============================================================================
# scripts/feast_render_config.sh
# =============================================================================
# Renders feature_repo/feature_store.yaml.tmpl into feature_repo/feature_store.yaml
# for LOCAL DEV use only — running `feast` from the project venv against a live
# offline/online store (e.g., when iterating on data_sources.py from the host).
#
# In production / CI / the container, the rendering is done by
# docker/feast/entrypoint.sh — never use this script there.
#
# The rendered file is gitignored (.gitignore) and chmod 600.
#
# Usage:
#   ./scripts/feast_render_config.sh
#
# Required env (sourced from .env if present):
#   SUPABASE_POSTGRES_PASSWORD
#   REDIS_PASSWORD
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMPL="${PROJECT_ROOT}/feature_repo/feature_store.yaml.tmpl"
OUT="${PROJECT_ROOT}/feature_repo/feature_store.yaml"
ENV_FILE="${PROJECT_ROOT}/.env"

if [ -f "$ENV_FILE" ]; then
    set -a
    # shellcheck disable=SC1090
    . "$ENV_FILE"
    set +a
fi

: "${SUPABASE_POSTGRES_PASSWORD:?SUPABASE_POSTGRES_PASSWORD must be set (in env or .env)}"
: "${REDIS_PASSWORD:?REDIS_PASSWORD must be set (in env or .env)}"

if [ ! -f "$TMPL" ]; then
    echo "[render] ERROR: template not found at $TMPL" >&2
    exit 1
fi

python3 - "$TMPL" "$OUT" <<'PY'
import os, sys
src, dst = sys.argv[1], sys.argv[2]
content = open(src).read()
for var in ('SUPABASE_POSTGRES_PASSWORD', 'REDIS_PASSWORD'):
    placeholder = '${' + var + '}'
    if placeholder in content:
        val = os.environ.get(var)
        if not val:
            sys.exit(f'[render] ERROR: {var} required by template but not set')
        content = content.replace(placeholder, val)
open(dst, 'w').write(content)
os.chmod(dst, 0o600)
print(f'[render] wrote {dst} (chmod 600)')
PY
