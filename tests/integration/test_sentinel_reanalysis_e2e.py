"""
End-to-end integration test for the sentinel → reanalysis Celery+Redis chain (#378).

This module asserts the FULL production chain merged in PR #383:

    notify_and_queue_reanalysis (src/tasks/sentinel_actions.py:150)
        → celery_app.send_task("src.tasks.insight_lifecycle_tasks.reanalyze_finding", ...)
            → real Celery worker subprocess against real Redis broker
                → reanalyze_finding (src/tasks/insight_lifecycle_tasks.py:171)
                    → _publish_reanalysis_signal (src/tasks/insight_lifecycle_tasks.py:93)
                        → redis.publish(channel="reanalysis:e2i:{brand}", payload)

NOTHING in this module is mocked: the Celery broker, the Celery worker, the
Redis pub/sub subscriber, and the redis-py publish are all real. Unit-test
coverage (tests/unit/test_tasks/test_insight_lifecycle_tasks.py +
test_sentinel_actions.py) already proves the per-component contract; this
test proves they actually wire together against a real broker.

Skip semantics
--------------
* A Redis instance must answer at ``E2I_TEST_REDIS_URL`` (or the default
  ``redis://localhost:6379`` if that env var is unset). Otherwise the
  test skips at module import.
* ``REDIS_URL`` is intentionally NOT consulted — the project conftest
  loads ``.env`` with ``override=True`` (tests/conftest.py:51), which
  silently rebinds ``REDIS_URL`` to the dev value (port 6382). The
  ``E2I_TEST_REDIS_URL`` key is a fixture-explicit escape hatch that
  ``.env`` does not set.
* The test is marked ``integration`` and is excluded from the default
  unit suite.

Run locally::

    docker run -d -p 6379:6379 redis:7-alpine
    E2I_TEST_REDIS_URL=redis://localhost:6379 \
        pytest tests/integration/test_sentinel_reanalysis_e2e.py -v
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

import pytest

# ---------------------------------------------------------------------------
# MODULE-LEVEL SKIP GUARD
# ---------------------------------------------------------------------------
#
# Resolution order (identical to the ``redis_url`` fixture — do not drift):
#   1. ``E2I_TEST_REDIS_URL`` — fixture-explicit escape hatch. The .env
#      file does not set this key, so the conftest's
#      ``load_dotenv(override=True)`` cannot silently rebind it.
#   2. ``_DEFAULT_REDIS_URL`` (``redis://localhost:6379``).
#
# We check at module import so collection skips cleanly in environments
# without a broker.

_DEFAULT_REDIS_URL = "redis://localhost:6379"


def _redis_reachable(url: str) -> bool:
    """Best-effort TCP probe of the Redis URL host/port."""
    try:
        # Strip scheme + optional db suffix: redis://host:port[/db]
        rest = url.split("://", 1)[1]
        hostport = rest.split("/", 1)[0]
        if "@" in hostport:
            hostport = hostport.split("@", 1)[1]
        if ":" in hostport:
            host, port_s = hostport.rsplit(":", 1)
            port = int(port_s)
        else:
            host, port = hostport, 6379
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except Exception:
        return False


# Skip-guard broker resolution MUST match the ``redis_url`` fixture so the
# guard does not let collection proceed against a broker the test will
# never actually reach. (Codex iter-3 M1.)
#
# Resolution order — IDENTICAL to ``redis_url`` fixture (do not drift):
#   1. ``E2I_TEST_REDIS_URL`` — fixture-explicit escape hatch.
#   2. ``_DEFAULT_REDIS_URL`` (``redis://localhost:6379``).
#
# Note: ``REDIS_URL`` is intentionally NOT consulted here. The fixture
# does not consult it either (see fixture docstring). Keying the guard
# off ``REDIS_URL`` would let collection proceed even when the actual
# test target (localhost:6379) is unreachable.
_skip_target_url = os.environ.get("E2I_TEST_REDIS_URL") or _DEFAULT_REDIS_URL
if not _redis_reachable(_skip_target_url):
    pytest.skip(
        f"requires reachable Redis at {_skip_target_url} "
        "(set E2I_TEST_REDIS_URL or start redis on localhost:6379)",
        allow_module_level=True,
    )


pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_redis,
    # Worker subprocess + sleep-based polling — keep tests on one xdist worker
    # to avoid parallel Redis-channel cross-talk.
    pytest.mark.xdist_group(name="sentinel_reanalysis_e2e"),
    # The default per-test timeout in pyproject.toml is 30s. This test spins
    # up a Celery worker subprocess (~10-15s boot), then dispatches a task and
    # polls the pub/sub channel for up to 15s. 30s is too tight; lift to 90s
    # to give worker boot + dispatch + execute + publish full headroom.
    pytest.mark.timeout(90),
]


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def redis_url() -> str:
    """
    The Redis URL the worker + this test process both use.

    Precedence is FIXTURE-EXPLICIT — we do NOT trust ``os.environ`` for
    this value at fixture-setup time because ``tests/conftest.py:51``
    runs ``load_dotenv(override=True)`` at module import, which silently
    rebinds any ``REDIS_URL`` the test runner inherited to the dev value
    (``redis://:changeme@localhost:6382``). Reading the env here would
    therefore make the test connect to whatever the developer's ``.env``
    happens to say, NOT what this test pins.

    Resolution order:
      1. ``E2I_TEST_REDIS_URL`` env var — explicit test-broker pin that
         the conftest's ``load_dotenv`` does NOT override (the .env does
         not set this key).
      2. Module-level ``_DEFAULT_REDIS_URL`` (``redis://localhost:6379``).
    """
    return os.environ.get("E2I_TEST_REDIS_URL") or _DEFAULT_REDIS_URL


@pytest.fixture(scope="module")
def broker_env(redis_url: str) -> Dict[str, str]:
    """
    Environment for the Celery worker subprocess pinned to the test broker.

    The worker boots the SAME ``src.workers.celery_app:celery_app`` the
    sentinel action handler uses; we override only the broker URLs so this
    test never depends on a separately-configured port 6382 dev broker.

    We start from ``os.environ.copy()`` so the worker inherits Python path,
    PYTHONUNBUFFERED, etc., but then unconditionally REBIND the three
    broker-related keys to ``redis_url``. Without this rebind the worker
    inherits the ``.env``-loaded broker URL (port 6382 in this repo),
    which silently desynchronises the worker's consume target from the
    test process's send_task target.
    """
    env = os.environ.copy()
    env["CELERY_BROKER_URL"] = redis_url
    env["CELERY_RESULT_BACKEND"] = redis_url
    env["REDIS_URL"] = redis_url
    # Defensive: if the worker subprocess also imports tests/conftest.py
    # (it should not, but pytest plugins occasionally route through it),
    # the override=True load_dotenv would re-rebind these. Drop the
    # PYTEST_* identifiers so the worker boots cleanly as a Celery
    # process, not a pytest subprocess.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_VERSION", None)
    return env


@pytest.fixture
def celery_worker_subprocess(
    broker_env: Dict[str, str],
) -> Iterator[subprocess.Popen[str]]:
    """
    Spin up a real Celery worker subprocess against the test broker.

    The worker:
      * imports ``src.workers.celery_app:celery_app`` (which autodiscovers
        ``src.tasks``, registering ``reanalyze_finding`` per
        ``src.tasks.__init__``).
      * consumes the default queue (where ``send_task`` lands without
        explicit routing).
      * runs solo pool (no prefork) so subprocess shutdown is clean.

    The fixture polls the worker's stdout for the readiness marker
    (``"celery@... ready."``) before yielding, so the test body never races
    against worker boot.
    """
    cmd = [
        sys.executable,
        "-m",
        "celery",
        "-A",
        "src.workers.celery_app:celery_app",
        "worker",
        "--loglevel=INFO",
        "--pool=solo",  # single-threaded; deterministic shutdown
        "--concurrency=1",
        "--queues=default",
        "--without-heartbeat",
        "--without-gossip",
        "--without-mingle",
    ]
    proc = subprocess.Popen(
        cmd,
        env=broker_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
    )

    # Wait up to 30s for "ready." in the worker's log output.
    ready = False
    deadline = time.monotonic() + 30.0
    stdout_lines: List[str] = []
    assert proc.stdout is not None
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            if proc.poll() is not None:
                # Worker exited before becoming ready.
                break
            time.sleep(0.05)
            continue
        stdout_lines.append(line)
        if "ready." in line.lower() or "celery@" in line.lower() and "ready" in line.lower():
            ready = True
            break

    if not ready:
        # Tear the worker down FIRST so the diagnostic read below sees
        # EOF instead of blocking on a still-running process. Calling
        # ``proc.stdout.read()`` on a live process is a guaranteed
        # deadlock — that's what bit iter-1 when the timeout fired
        # mid-fixture-setup (pyproject.toml ``timeout=30`` default;
        # iter-2 lifts the per-test timeout to 90s and orders the
        # teardown-before-read correctly here so the diagnostic surfaces
        # in the failure message).
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)
        # Now safe to drain remaining buffered output for the failure trace.
        try:
            extra = proc.stdout.read() or ""
        except Exception:
            extra = ""
        stdout_lines.append(extra)
        pytest.fail(
            "Celery worker subprocess did not reach 'ready.' state within 30s. "
            f"Last stdout lines:\n{''.join(stdout_lines[-30:])}"
        )

    try:
        yield proc
    finally:
        # Loud cleanup — never swallow exceptions in teardown.
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5.0)


@pytest.fixture
async def pubsub_listener(redis_url: str) -> AsyncIterator[Any]:
    """
    Async Redis pub/sub subscriber for the reanalysis channel.

    Uses a NEW redis-py async client (not the cached factory singleton) so
    the subscription does not interfere with other tests sharing the
    factory. The caller is responsible for psubscribe()'ing to a channel
    pattern before triggering the action; this fixture only sets up the
    client/pubsub pair and tears them down.
    """
    import redis.asyncio as redis_async

    client = redis_async.from_url(redis_url, decode_responses=True)
    # Sanity-ping so a Redis outage fails the test loudly, not midway through.
    await client.ping()
    pubsub = client.pubsub()
    try:
        yield pubsub
    finally:
        try:
            await pubsub.aclose()
        except Exception:
            pass
        try:
            await client.aclose()
        except Exception:
            pass


@pytest.fixture
def pin_in_process_broker(redis_url: str) -> Iterator[None]:
    """
    Pin the in-process ``celery_app`` config + ``get_redis_client`` cache
    to the SAME broker URL the worker subprocess is configured against,
    with explicit teardown restoration.

    Without this fixture's restoration step, the direct mutations of
    ``celery_app.conf.broker_url`` / ``celery_app.conf.result_backend``
    and ``svc_factories._redis_client`` would leak past this test —
    later tests on the same xdist worker would inherit a Celery app and
    cached Redis singleton both pointed at this test's broker URL,
    creating a hard-to-trace ordering dependency. (Codex iter-3 M2.)

    Connection-pool invalidation (iter-5 CI fix)
    --------------------------------------------
    Setting ``celery_app.conf.broker_url`` and
    ``celery_app.conf.result_backend`` updates the Celery configuration,
    but Celery caches the AMQP producer pool and the result-backend
    object lazily at first use. Once cached, those objects ignore later
    config rebinds and keep retrying against the old URL — which on CI
    is the unreachable ``redis://localhost:6382`` default. The visible
    symptom is ``RuntimeError: Retry limit exceeded while trying to
    reconnect to the Celery redis result store backend``.

    Fix: call ``celery_app.close()`` after rebinding conf — this drops
    the AMQP producer pool and forces a fresh lazy init against the
    new URL. We also invalidate the cached result-backend object via
    ``__dict__.pop("backend", None)`` because ``backend`` is a cached
    descriptor property on the Celery app.

    Environment vars are kept in scope by the caller's ``monkeypatch``
    fixture, which auto-restores on test teardown — only the live config
    object + the lazy singletons need explicit save/restore here.
    """
    from src.memory.services import factories as svc_factories
    from src.workers.celery_app import celery_app

    # Snapshot previous values so we can restore on teardown.
    prev_broker_url = celery_app.conf.broker_url
    prev_result_backend = celery_app.conf.result_backend
    prev_redis_client = svc_factories._redis_client  # type: ignore[attr-defined]

    # Apply test pin.
    celery_app.conf.broker_url = redis_url
    celery_app.conf.result_backend = redis_url
    svc_factories._redis_client = None  # type: ignore[attr-defined]

    # Drop cached AMQP producer pool + cached backend object so the next
    # ``send_task`` call rebuilds them against the new URL.
    try:
        celery_app.close()
    except Exception:
        # ``close()`` may raise if there was no open pool; harmless.
        pass
    # ``Celery.backend`` is a cached @property — pop it from __dict__ so
    # the next access re-creates it from the new ``result_backend``.
    celery_app.__dict__.pop("backend", None)

    try:
        yield
    finally:
        # Restore previous Celery config. Failure here is fatal — we do
        # NOT swallow because a silent restore-fail would taint the
        # remainder of the xdist worker's tests.
        celery_app.conf.broker_url = prev_broker_url
        celery_app.conf.result_backend = prev_result_backend
        # Drop the test-URL pool/backend so the next test sees a clean
        # lazy-init against the restored config.
        try:
            celery_app.close()
        except Exception:
            pass
        celery_app.__dict__.pop("backend", None)

        # Close any redis client the test created against the test URL
        # before restoring the previous singleton, so we do not leak
        # connections to the test broker. Use the explicit-loop pattern
        # (PR #217 commit a321b64f) rather than bare ``asyncio.run`` —
        # the project's ``tests/integration/test_no_bare_asyncio_run_in_
        # integration_tests.py`` enforces this against the RAGAS
        # nest_asyncio pollution chain (issue #220 / #218 / #215).
        leaked = svc_factories._redis_client  # type: ignore[attr-defined]
        if leaked is not None and leaked is not prev_redis_client:
            try:
                loop = asyncio.new_event_loop()
                try:
                    loop.run_until_complete(leaked.aclose())
                finally:
                    loop.close()
            except Exception:
                # Best-effort: a connection-close failure must not mask
                # the test result. We still restore the previous client
                # below.
                pass
        svc_factories._redis_client = prev_redis_client  # type: ignore[attr-defined]


@pytest.fixture
def stale_finding() -> Dict[str, Any]:
    """
    Realistic stale-finding fixture matching the shape
    ``notify_and_queue_reanalysis`` expects in ``trigger_data["stale_findings"]``.

    The shape mirrors what the sentinel evaluator builds in
    ``src.memory.sentinels.registry`` for the staleness_threshold gate:
    a list of dicts with ``finding_id`` (pk), optional ``brand``, optional
    ``staleness_score``, plus arbitrary metadata fields.

    We use a uuid-suffixed finding_id so concurrent test runs do not
    cross-pollute each others' published messages.
    """
    finding_id = f"e2e-finding-{uuid.uuid4().hex[:12]}"
    return {
        "finding_id": finding_id,
        "brand": "e2i",
        "staleness_score": 0.95,
        "last_modified": "2026-05-19T12:00:00Z",
        "source_table": "causal_paths",
    }


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------


async def _poll_for_message(
    pubsub: Any,
    *,
    timeout: float,
    match: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Poll the pub/sub subscriber for a real message up to ``timeout`` seconds.

    Returns the parsed JSON payload on the FIRST message matching every
    key/value in ``match`` (or the first parseable payload when ``match``
    is None), or None on timeout. Filters out subscribe-confirmation
    messages (type=="subscribe"/"psubscribe") and skips messages with
    malformed payloads or that fail the match filter.

    The match filter (Codex iter-3 L1) lets the test ignore concurrent
    publishers on the same brand channel — e.g., a parallel sentinel
    dispatcher or a dev script — by pinning on this test's uuid-suffixed
    ``finding_id``. Without it, the helper would return the FIRST
    parseable payload and a concurrent publisher would cause a spurious
    assertion failure instead of being ignored while the test waits for
    its OWN reanalysis signal.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        message = await pubsub.get_message(
            ignore_subscribe_messages=True,
            timeout=0.5,
        )
        if message is None:
            await asyncio.sleep(0.05)
            continue
        if message.get("type") not in ("message", "pmessage"):
            continue
        data = message.get("data")
        if isinstance(data, bytes):
            data = data.decode("utf-8")
        try:
            payload = dict(json.loads(data))
        except (TypeError, ValueError):
            # Malformed payload — keep polling; caller's assert will surface.
            continue
        if match is not None and not all(payload.get(k) == v for k, v in match.items()):
            # Not OUR message — keep polling. A concurrent publisher on the
            # same brand channel does NOT fail the test.
            continue
        return payload
    return None


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------


async def test_notify_and_queue_reanalysis_publishes_reanalysis_signal_e2e(
    celery_worker_subprocess: subprocess.Popen[str],
    pubsub_listener: Any,
    stale_finding: Dict[str, Any],
    redis_url: str,
    pin_in_process_broker: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Exercises the full production chain:

      notify_and_queue_reanalysis
        → celery_app.send_task(reanalyze_finding, ...)
          → [real Celery worker subprocess via real Redis broker]
            → reanalyze_finding
              → _publish_reanalysis_signal
                → redis.publish("reanalysis:e2i:e2i", payload)

    The test asserts that a payload arrives on the brand-scoped pub/sub
    channel within a bounded timeout AND that the payload carries the
    expected ``finding_id`` / ``brand`` / ``triggered_by`` keys.

    NO production module is patched. The only thing we override is the
    REDIS_URL / CELERY_BROKER_URL environment to point both the in-process
    ``notify_and_queue_reanalysis`` AND the worker subprocess at the SAME
    real broker, which is precisely how production wires them.

    Mutable-state isolation
    -----------------------
    The ``pin_in_process_broker`` fixture (above) saves the live
    ``celery_app.conf`` + ``svc_factories._redis_client`` BEFORE this
    test runs and restores them on teardown — so this test cannot leak
    a test-broker-pinned Celery config to later tests on the same xdist
    worker. The ``monkeypatch.setenv`` calls below cover the env-var
    side of the same contract.
    """
    # ----- 0. Re-set the env vars so the lazy redis-client factory picks
    # up the test URL on next call. The Celery config + redis singleton
    # have already been re-pinned by the ``pin_in_process_broker`` fixture
    # (which saves+restores them around this test). Env vars are restored
    # by ``monkeypatch`` on teardown.
    #
    # ``tests/conftest.py:51`` runs ``load_dotenv(override=True)`` which
    # rebinds REDIS_URL/CELERY_BROKER_URL to the .env values at module
    # import. By the time this test runs, ``os.environ["REDIS_URL"]`` is
    # the .env value (e.g. ``redis://:changeme@localhost:6382``) and NOT
    # what the fixture's ``redis_url`` says. Forcibly re-set those env
    # keys here so any code path that re-reads them lands on the test URL.
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("CELERY_BROKER_URL", redis_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", redis_url)

    # ----- 1. Subscribe to the brand-scoped channel BEFORE triggering.
    brand = str(stale_finding["brand"])
    channel = f"reanalysis:e2i:{brand}"
    await pubsub_listener.subscribe(channel)

    # Drain the subscribe-confirmation message so the polling helper starts
    # from a clean slate.
    confirmation = await pubsub_listener.get_message(timeout=2.0)
    assert confirmation is not None
    assert confirmation.get("type") == "subscribe"

    # ----- 2. Trigger the action handler. This issues the actual
    # ``celery_app.send_task`` against the real broker; the worker
    # subprocess picks it up, runs ``reanalyze_finding``, and publishes
    # the signal on the channel we are subscribed to.
    from src.tasks.sentinel_actions import notify_and_queue_reanalysis

    sentinel_id = f"e2e-sentinel-{uuid.uuid4().hex[:8]}"
    result = await notify_and_queue_reanalysis(
        sentinel_id=sentinel_id,
        brands=[brand],
        trigger_data={"stale_findings": [stale_finding]},
    )

    # The action handler's own contract: 1 notified, 1 queued (broker is up).
    assert result["notified_for_reanalysis"] == 1, result
    assert result["queued_for_reanalysis"] == 1, result
    assert result["stale_findings_count"] == 1, result

    # ----- 3. Poll the subscriber for the SPECIFIC signal carrying this
    # test's finding_id. The match filter ignores any concurrent publisher
    # on the same brand channel (Codex iter-3 L1) — only OUR uuid-suffixed
    # finding_id satisfies the match. Bounded 15s timeout covers worker
    # dispatch + execute + publish; typical wall time is <2s.
    expected_finding_id = stale_finding["finding_id"]
    payload = await _poll_for_message(
        pubsub_listener,
        timeout=15.0,
        match={
            "type": "reanalysis_requested",
            "finding_id": expected_finding_id,
        },
    )

    assert payload is not None, (
        f"No reanalysis_requested signal with finding_id={expected_finding_id!r} "
        f"received on channel {channel!r} within 15s. The Celery worker "
        "either did not execute reanalyze_finding or did not publish the signal."
    )

    # ----- 4. Assert remaining payload shape. ``type`` and ``finding_id``
    # are already pinned by the match filter; verify the remaining fields
    # the downstream orchestrator subscribes for.
    assert payload["brand"] == brand, payload
    assert payload["triggered_by"] == "sentinel:staleness", payload
    assert "requested_at" in payload, payload
