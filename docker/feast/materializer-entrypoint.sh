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
# cycling) is otherwise invisible. The healthcheck flags the container unhealthy
# if the heartbeat goes older than one interval + buffer. Per #556 M1, the
# heartbeat is NOT written until the FIRST materialize succeeds — so a container
# that has never populated Redis does not report healthy. A crash-loop is already
# visible via the container restarting.
HEARTBEAT=/tmp/materializer_heartbeat
LOCK=/feast/data/.registry.lock
INITIAL_RETRY_SECONDS=60

# One materialize cycle under the shared registry lock (#556 H2: serialize
# against the feast service's `apply` on the shared feast_registry volume; Feast's
# file registry is not safe for concurrent writers).
materialize_once() {
    NOW="$(date -u +%Y-%m-%dT%H:%M:%S)"
    echo "[materializer] feast materialize-incremental ${NOW}"
    flock "$LOCK" feast --chdir /feast materialize-incremental "${NOW}"
}

# #556 M1: retry the FIRST cycle with short backoff (not a full interval) and do
# not mark healthy until it succeeds — otherwise a slow-to-start offline store
# leaves Redis cold for a full interval while the healthcheck reports green.
until materialize_once; do
    echo "[materializer] WARNING: initial materialize failed; retrying in ${INITIAL_RETRY_SECONDS}s (Redis still cold; healthcheck stays red until first success)" >&2
    sleep "$INITIAL_RETRY_SECONDS"
done
echo "[materializer] initial materialize succeeded"
date -u +%s > "$HEARTBEAT"

# Steady state: Redis is populated; a transient per-cycle failure is logged and
# retried next cycle. The heartbeat advances each cycle (liveness), since the
# store was already populated by the initial success.
while true; do
    sleep "${INTERVAL_SECONDS}"
    if materialize_once; then
        echo "[materializer] cycle complete"
    else
        echo "[materializer] WARNING: materialize-incremental failed this cycle; will retry next cycle" >&2
    fi
    date -u +%s > "$HEARTBEAT"
done
