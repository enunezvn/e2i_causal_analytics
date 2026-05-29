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

# Populate /feast (env + bind-mount validation, copy source, render yaml).
# Shared with the materializer (#556) via the sourced helper so the two cannot
# drift in how the repo is populated.
. /_populate_feast.sh
populate_feast

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
# #556: serialize registry writes against the feast-materializer. Both share the
# feast_registry volume (/feast/data); flock on a lockfile there ensures `apply`
# (here, on every serve-container start/restart) and the materializer's
# `materialize-incremental` never write the file-based registry concurrently
# (Feast's local registry is not safe for concurrent writers).
flock /feast/data/.registry.lock feast --chdir /feast apply --skip-source-validation

echo "[entrypoint] starting feast serve on 0.0.0.0:6566"
exec feast --chdir /feast serve --host 0.0.0.0 --port 6566
