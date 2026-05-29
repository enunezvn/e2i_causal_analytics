# shellcheck shell=sh
# =============================================================================
# Shared Feast repo population (sourced, not executed)
# =============================================================================
# Defines populate_feast(): validate required env + the /feast-src bind mount,
# copy the source .py files from /feast-src into /feast (a writable container
# layer), and render feature_store.yaml from the template with secrets
# substituted in. Sourced by BOTH docker/feast/entrypoint.sh (serve) and
# docker/feast/materializer-entrypoint.sh (#556 scheduled materialize) so the
# two stay in lockstep — a change to how the repo is populated can never drift
# between the serving sidecar and the materializer.
#
# Copy (not symlink) because Feast 0.43.0's parse_repo resolves symlinks to
# their target and then runs path.relative_to(cwd), which fails for targets
# outside /feast. /feast/data is a separate volume (registry + materialization
# output) and is left untouched. The caller is expected to have `set -e`.
# =============================================================================

populate_feast() {
    : "${SUPABASE_POSTGRES_PASSWORD:?SUPABASE_POSTGRES_PASSWORD must be set}"
    : "${REDIS_PASSWORD:?REDIS_PASSWORD must be set}"

    if [ ! -d /feast-src ]; then
        echo "[populate] ERROR: /feast-src not bind-mounted (compose must mount ../feature_repo:/feast-src:ro)" >&2
        exit 1
    fi
    if [ ! -f /feast-src/feature_store.yaml.tmpl ]; then
        echo "[populate] ERROR: /feast-src/feature_store.yaml.tmpl missing — bind mount points at the wrong directory" >&2
        exit 1
    fi

    mkdir -p /feast/features
    cp -f /feast-src/entities.py     /feast/entities.py
    cp -f /feast-src/data_sources.py /feast/data_sources.py
    for f in /feast-src/features/*.py; do
        cp -f "$f" "/feast/features/$(basename "$f")"
    done

    # Render feature_store.yaml. Use Python (robust to special chars in password).
    python3 - <<'PY'
import os, sys
src = '/feast-src/feature_store.yaml.tmpl'
dst = '/feast/feature_store.yaml'
content = open(src).read()
for var in ('SUPABASE_POSTGRES_PASSWORD', 'REDIS_PASSWORD'):
    placeholder = '${' + var + '}'
    if placeholder in content:
        val = os.environ.get(var)
        if not val:
            sys.exit(f'[populate] ERROR: {var} required by template but not set')
        content = content.replace(placeholder, val)
open(dst, 'w').write(content)
os.chmod(dst, 0o600)
print('[populate] rendered /feast/feature_store.yaml from template')
PY
}
