#!/bin/sh
# =============================================================================
# Feast materializer (#556)
# =============================================================================
# Runs in the SAME image as the feast serving sidecar (Dockerfile.feast — the
# only image that can `import feast`; the app/worker image cannot, per #307's
# tenacity conflict). Its job is to keep the Redis ONLINE store populated for
# the batch-sourced feature views, which otherwise decay (the scheduled Celery
# materialize beats are structural no-ops on the app image).
#
# Design:
#   - MATERIALIZE-ONLY. It does NOT `feast apply`; the feast serving service owns
#     apply on the shared feast_registry volume (/feast/data). This materializer
#     reads that same registry (shared named volume) and writes feature values to
#     the Redis online store — keeping a single registry and avoiding two writers
#     racing on the sqlite registry file. `depends_on: feast (healthy)` guarantees
#     apply has run before the first materialize.
#   - Cadence comes from config/feast_materialization.yaml (schedule.interval_hours,
#     default 6h), mounted read-only at /feast/feast_materialization.yaml. A single
#     global loop; per-view sub-global cadence (e.g. trigger_response hourly) is a
#     tracked follow-up.
#   - Fail-SOFT per cycle (a transient materialize error is logged and retried next
#     cycle rather than crash-looping the container); sustained staleness is
#     surfaced by the online-store freshness probe, not by this loop dying.
# =============================================================================
set -e

# Populate /feast (shared helper — identical to the serving sidecar).
. /_populate_feast.sh
populate_feast

# Hard-fail on Feast SDK drift — must match the serving container + production.
feast version | grep -q "0.43.0" || { echo "[materializer] FATAL: Feast SDK version drift (expected 0.43.0)"; exit 1; }

# Resolve the global materialize cadence from the mounted config (fallback 6h).
INTERVAL_HOURS="$(python3 - <<'PY'
try:
    import yaml
    cfg = yaml.safe_load(open('/feast/feast_materialization.yaml')) or {}
    hours = (cfg.get('schedule') or {}).get('interval_hours', 6)
    print(int(hours) if hours else 6)
except Exception:
    print(6)
PY
)"
[ -n "$INTERVAL_HOURS" ] || INTERVAL_HOURS=6
INTERVAL_SECONDS=$(( INTERVAL_HOURS * 3600 ))
echo "[materializer] online-store materialize loop: every ${INTERVAL_HOURS}h (${INTERVAL_SECONDS}s)"

# Heartbeat for the container healthcheck: a hung loop (process alive but not
# cycling) is otherwise invisible. Written at startup and after every cycle; the
# healthcheck flags the container unhealthy if the heartbeat goes older than one
# interval + buffer. A crash-loop is already visible via the container restarting.
HEARTBEAT=/tmp/materializer_heartbeat
date -u +%s > "$HEARTBEAT"

while true; do
    NOW="$(date -u +%Y-%m-%dT%H:%M:%S)"
    echo "[materializer] feast materialize-incremental ${NOW}"
    if feast --chdir /feast materialize-incremental "${NOW}"; then
        echo "[materializer] cycle complete"
    else
        echo "[materializer] WARNING: materialize-incremental failed this cycle; will retry after sleep" >&2
    fi
    date -u +%s > "$HEARTBEAT"
    sleep "${INTERVAL_SECONDS}"
done
