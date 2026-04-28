#!/bin/sh
# =============================================================================
# Feast container entrypoint
# =============================================================================
# 1. Validates required env vars (SUPABASE_POSTGRES_PASSWORD, REDIS_PASSWORD).
# 2. Validates that the feature-repo source has been bind-mounted at /feast-src.
# 3. Populates /feast (writable container layer) by COPYING the source
#    Python files in from /feast-src and rendering feature_store.yaml from
#    feature_store.yaml.tmpl with the secret env vars substituted in. The
#    rendered yaml lives only in the running container — it is never written
#    back to the host bind mount and is therefore never committed.
#    Symlinks are NOT used: Feast 0.43.0's parse_repo resolves symlinks to
#    their target and then runs path.relative_to(cwd), which fails because
#    the symlink target /feast-src/*.py is outside /feast (the working dir).
# 4. Runs `feast apply --skip-source-validation` (idempotent — see
#    tests/integration/test_feast_apply_idempotent.py). `--skip-source-validation`
#    avoids needing offline-store (Postgres) reachability on cold start; live
#    serving still validates sources at materialization time.
# 5. Exec's `feast serve` on 0.0.0.0:6566.
# =============================================================================
set -e

: "${SUPABASE_POSTGRES_PASSWORD:?SUPABASE_POSTGRES_PASSWORD must be set}"
: "${REDIS_PASSWORD:?REDIS_PASSWORD must be set}"

if [ ! -d /feast-src ]; then
    echo "[entrypoint] ERROR: /feast-src not bind-mounted (compose must mount ../feature_repo:/feast-src:ro)" >&2
    exit 1
fi
if [ ! -f /feast-src/feature_store.yaml.tmpl ]; then
    echo "[entrypoint] ERROR: /feast-src/feature_store.yaml.tmpl missing — bind mount points at the wrong directory" >&2
    exit 1
fi

# Populate /feast by copying source .py files in from /feast-src. /feast/data
# is a separate volume (registry + materialization output) and is left alone.
# Copy (not symlink) because Feast 0.43.0 parse_repo rejects symlinks whose
# targets resolve outside the working dir.
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
            sys.exit(f'[entrypoint] ERROR: {var} required by template but not set')
        content = content.replace(placeholder, val)
open(dst, 'w').write(content)
os.chmod(dst, 0o600)
print('[entrypoint] rendered /feast/feature_store.yaml from template')
PY

echo "[entrypoint] feast version: $(feast version)"
# Hard-fail on Feast SDK drift — the registry-proto layout, parse_repo
# semantics, and Pydantic warnings are all version-coupled.
# Brace block (NOT subshell): keeps everything in the current shell — clearer
# intent, no subshell overhead. With 'set -e' active a subshell 'exit 1'
# would still abort the script via the failed pipeline, but the brace form
# is more direct and avoids the subshell footgun.
# Block 6B-infra-4.
feast version | grep -q "0.43.0" || { echo "[entrypoint] FATAL: Feast SDK version drift (expected 0.43.0)"; exit 1; }
echo "[entrypoint] feast apply (skip source validation)..."
feast --chdir /feast apply --skip-source-validation

echo "[entrypoint] starting feast serve on 0.0.0.0:6566"
exec feast --chdir /feast serve --host 0.0.0.0 --port 6566
