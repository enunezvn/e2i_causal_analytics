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
* ``REDIS_URL`` must be set OR a Redis instance must answer at
  ``redis://localhost:6379``. Otherwise the test skips at module import.
* The test is marked ``integration`` and is excluded from the default unit
  suite.

Run locally::

    docker run -d -p 6379:6379 redis:7-alpine
    REDIS_URL=redis://localhost:6379 \
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
# The test requires a reachable Redis on REDIS_URL (or the default
# localhost:6379). We check at module import so collection skips cleanly
# in environments without a broker.

_DEFAULT_REDIS_URL = "redis://localhost:6379"


def _redis_url() -> str:
    return os.environ.get("REDIS_URL") or _DEFAULT_REDIS_URL


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


if not os.environ.get("REDIS_URL") and not _redis_reachable(_DEFAULT_REDIS_URL):
    pytest.skip(
        "requires REDIS_URL or reachable redis://localhost:6379",
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
) -> Optional[Dict[str, Any]]:
    """
    Poll the pub/sub subscriber for a real message up to ``timeout`` seconds.

    Returns the parsed JSON payload on success, or None on timeout. Filters
    out subscribe-confirmation messages (type=="subscribe"/"psubscribe").
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
            return dict(json.loads(data))
        except (TypeError, ValueError):
            # Malformed payload — keep polling; caller's assert will surface.
            continue
    return None


# ---------------------------------------------------------------------------
# TESTS
# ---------------------------------------------------------------------------


async def test_notify_and_queue_reanalysis_publishes_reanalysis_signal_e2e(
    celery_worker_subprocess: subprocess.Popen[str],
    pubsub_listener: Any,
    stale_finding: Dict[str, Any],
    redis_url: str,
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
    """
    # ----- 0. Pin the in-process Celery app + redis-client factory to the
    # SAME broker URL the worker subprocess is configured against.
    #
    # ``tests/conftest.py:51`` runs ``load_dotenv(override=True)`` which
    # rebinds REDIS_URL/CELERY_BROKER_URL to the .env values at module
    # import. By the time this test runs, ``os.environ["REDIS_URL"]`` is
    # the .env value (e.g. ``redis://:changeme@localhost:6382``) and NOT
    # what the fixture's ``redis_url`` says. We forcibly re-set those env
    # keys here so the lazy redis-client factory picks up the test URL,
    # AND we rebind the celery_app.conf object directly so the in-process
    # ``send_task`` lands on the same broker the worker subprocess is
    # consuming. Without both rebinds, the test would silently call
    # ``send_task`` against the wrong broker and the assertion would
    # time out.
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("CELERY_BROKER_URL", redis_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", redis_url)
    from src.workers.celery_app import celery_app

    celery_app.conf.broker_url = redis_url
    celery_app.conf.result_backend = redis_url

    # Reset the cached redis singleton so the action's lazy ``get_redis_client``
    # constructs a new client against the test URL on next call.
    from src.memory.services import factories as svc_factories

    svc_factories._redis_client = None  # type: ignore[attr-defined]

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

    # ----- 3. Poll the subscriber for the published signal. Bounded
    # timeout: worker boot is in the fixture, so we only wait for task
    # dispatch + execute + publish — typically <2s, generous cap at 15s.
    payload = await _poll_for_message(pubsub_listener, timeout=15.0)

    assert payload is not None, (
        "No reanalysis_requested signal received on channel "
        f"{channel!r} within 15s. The Celery worker either did not "
        "execute reanalyze_finding or did not publish the signal."
    )

    # ----- 4. Assert payload shape: the worker's publish must carry
    # the per-finding metadata the downstream orchestrator subscribes for.
    assert payload["type"] == "reanalysis_requested", payload
    assert payload["finding_id"] == stale_finding["finding_id"], payload
    assert payload["brand"] == brand, payload
    assert payload["triggered_by"] == "sentinel:staleness", payload
    assert "requested_at" in payload, payload
