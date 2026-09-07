# ADR-016: Long causal-discovery jobs are cancelled cooperatively and orphan-detected by heartbeat, not by signals or a startup sweep

**Date**: 2026-09-05 | **Status**: Accepted | **Implemented by**: PRs #1898 (cooperative cancel), #1899 (liveness heartbeat + poll-time read-repair)

## Context

`POST /api/causal/discover-effects` runs for minutes: a question's estimators execute **synchronously** in a worker thread and cannot be interrupted mid-question. The job runs as a FastAPI `BackgroundTask`, so the request that would cancel it — arriving at a different worker, in prod always one of two — has no handle on it at all.

Two failures followed from that shape:

- **No cancel.** An analyst who started the wrong discovery run had to wait it out. A `cancel_requested` field written onto the job row would need a read-modify-write that the task's next wholesale `set` overwrites.
- **Orphans.** Every deploy restarts the API; gunicorn also recycles workers on `--max-requests`. A run in flight vanished, and its row stayed `running` until the 8 h job TTL with the page polling forever.

## Decision

1. **Cancellation is a sidecar marker, not a signal.** The cancel route raises `"<prefix>:<id>:cancel"` via `DurableJobStore.set_marker`; the task polls `has_marker` **at question boundaries only**. A sidecar key has no read-modify-write race with the task publishing its row wholesale, and it crosses workers because it lives in Redis. Cancel is idempotent and a no-op on a terminal job.
2. **The boundary is honest about what it can stop.** The question in flight finishes (up to a few minutes); the run then keeps every finished row and marks the unrun ones `cancelled` — status only, no estimate, no summary, nothing fabricated.
3. **The row flag is the fallback, never the primary.** `cancel_requested` on the row is honoured too, because the route's marker `SET` can fail transiently on its own worker (the marker then lives only in that process's memory) while its row write still reaches Redis and the route has already answered 200. The API must never acknowledge a cancel it then ignores.
4. **Liveness is a timestamped heartbeat, read at poll time.** The task re-stamps an `alive` marker every 15 s (`touch_marker`); a poll on any worker reads its **age**. `_repair_if_orphaned` closes a non-terminal row whose stamp is older than the TTL (or absent) to `failed`, with the reason, keeping finished effects and closing unfinished ones, and **persists** it so every later poll on any worker agrees and the frontend stops polling. The heartbeat's TTL is 120 s = the gunicorn worker timeout (`--timeout 120`): an event loop stalled that long is killed anyway, so a gap that long means the worker is gone, never a live run.
5. **Read-repair, not a startup sweep.** Prod runs two workers, and a freshly restarted worker cannot tell whether the *other* worker's jobs are still alive — it would kill live runs. A heartbeat can tell, and a poll is the moment someone actually needs the answer. Repair is therefore idempotent and lazy.
6. **A repaired row stands.** If a poll already closed a run as `failed` (e.g. this worker's beats could not reach Redis for the whole budget), the task stops at its next boundary **without publishing**: the page has stopped polling, and resurrecting the run would burn minutes per question for nobody.
7. **Two invariants inside the task.** The first heartbeat is stamped **inline** before anything else — a `create_task` beat only runs once the coroutine suspends, and nothing before it is guaranteed to suspend. And a failed `touch` never ends the beat loop: a silently stopped heartbeat would declare a live run dead.

## Consequences

- (+) A cancel is acknowledged immediately and honoured at the next boundary, cross-worker, with no interrupt machinery around synchronous estimators.
- (+) An interrupted run reaches a terminal `failed` state on the next poll instead of polling forever; certified live at 121 s after a prod `docker restart e2i_api`, and the pre-existing orphan was read-repaired idempotently.
- (−) Cancel latency equals the remaining time of the in-flight question. The UI must say so rather than implying an immediate stop.
- (−) Liveness is inferred from a marker age, so it is a Redis-availability judgement too: a worker whose beats cannot reach Redis for 120 s has its live run declared dead. Rule 6 makes that outcome consistent rather than contradictory, not impossible.
- (−) In `DurableJobStore`'s degraded (in-memory) mode a marker is visible only to the worker that wrote it — cancel and liveness both silently become per-worker. The store's `_last_durable` is what surfaces that on `/health`.
- (−) The frontend maps an in-flight run repaired this way to **Failed**, not Cancelled; badge expectations must be derived from that mapping.

## References

- `src/api/dependencies/durable_job_store.py` — `set_marker` / `has_marker` (flags) and `touch_marker` / `marker_age_seconds` (timestamped heartbeats)
- `src/api/routes/causal.py` — `_DISCOVERY_CANCEL_MARKER`, `_DISCOVERY_ALIVE_MARKER`, `_repair_if_orphaned`, `_boundary_stop_reason`, `cancel_discover_causal_effects`
