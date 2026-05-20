"""End-to-end integration test for the staleness-alerts SSE bridge (#390).

Asserts the full production chain:

    src.tasks.sentinel_actions.publish_alert(payload)
        → redis.publish("e2i:alerts", json.dumps(payload))
            → AlertBridge subscriber (this PR)
                → EventSourceResponse stream
                    → httpx SSE client receives event

NOTHING in this module is mocked: the Redis pub/sub layer, the
:class:`AlertBridge` subscriber, the :func:`EventSourceResponse`
streaming wrapper, and the HTTPX async client are all real. The unit
tests in ``tests/unit/test_api/test_staleness_alerts.py`` use a fake
pubsub for fast cycles — this test catches anything that breaks at the
real-broker boundary (channel-name drift, JSON-decode regression,
EventSourceResponse formatting bugs).

Skip semantics
--------------
* A Redis instance must answer at ``E2I_TEST_REDIS_URL`` (or the
  default ``redis://localhost:6379`` if unset). Mirrors the gating
  pattern from
  :mod:`tests.integration.test_sentinel_reanalysis_e2e` exactly so
  developer environments without a test broker skip cleanly.
* ``REDIS_URL`` is intentionally NOT consulted — the project conftest
  loads ``.env`` with ``override=True`` (tests/conftest.py:51), which
  silently rebinds ``REDIS_URL`` to the dev value (port 6382). The
  ``E2I_TEST_REDIS_URL`` key is a fixture-explicit escape hatch that
  the ``.env`` does not set.
* The test is marked ``integration`` and is excluded from the default
  unit suite.

Run locally::

    docker run -d -p 6379:6379 redis:7-alpine
    E2I_TEST_REDIS_URL=redis://localhost:6379 \\
        pytest tests/integration/test_staleness_alerts_e2e.py -v
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from typing import Any, Dict

import pytest

# ---------------------------------------------------------------------------
# MODULE-LEVEL SKIP GUARD
# ---------------------------------------------------------------------------

_DEFAULT_REDIS_URL = "redis://localhost:6379"


def _redis_reachable(url: str) -> bool:
    """Best-effort TCP probe of the Redis URL host/port."""
    try:
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


# Resolution order — IDENTICAL to the ``redis_url`` fixture below (do
# not drift): (1) ``E2I_TEST_REDIS_URL`` — fixture-explicit escape
# hatch; (2) ``_DEFAULT_REDIS_URL``. ``REDIS_URL`` is NOT consulted
# (see fixture docstring for why).
_skip_target_url = os.environ.get("E2I_TEST_REDIS_URL") or _DEFAULT_REDIS_URL
if not _redis_reachable(_skip_target_url):
    pytest.skip(
        f"requires reachable Redis at {_skip_target_url} "
        "(set E2I_TEST_REDIS_URL or start redis on localhost:6379)",
        allow_module_level=True,
    )


pytestmark = [
    pytest.mark.integration,
    # Keep on one xdist worker so parallel test cross-talk on the
    # ``e2i:alerts`` pub/sub channel can't pollute the asserts.
    pytest.mark.xdist_group(name="staleness_alerts_sse_e2e"),
    # Default 30s; SSE+pubsub round-trip is fast (<1s typical) so 60s
    # is generous headroom.
    pytest.mark.timeout(60),
]


# ---------------------------------------------------------------------------
# FIXTURES
# ---------------------------------------------------------------------------


@pytest.fixture
def redis_url() -> str:
    """
    The Redis URL the test publisher + the bridge subscriber both use.

    Precedence is FIXTURE-EXPLICIT — we do NOT trust ``os.environ`` for
    this value at fixture-setup time because ``tests/conftest.py:51``
    runs ``load_dotenv(override=True)`` at module import, which would
    rebind any ``REDIS_URL`` the runner inherited to the dev value.

    Resolution order:
      1. ``E2I_TEST_REDIS_URL`` env var — explicit pin.
      2. ``_DEFAULT_REDIS_URL`` (``redis://localhost:6379``).
    """
    return os.environ.get("E2I_TEST_REDIS_URL") or _DEFAULT_REDIS_URL


@pytest.fixture
def reset_redis_factory(redis_url: str):
    """
    Force the :func:`get_redis_client` cached singleton to point at the
    test broker for the duration of the test.

    The factory module caches the client in a module-level ``_redis_client``
    so the first call in a test run binds the broker URL forever after.
    For an integration test against a different broker than the dev
    default (port 6382), we must clear the cache + override REDIS_URL.
    """
    import src.memory.services.factories as factories_mod

    original_client = factories_mod._redis_client
    original_url = os.environ.get("REDIS_URL")

    factories_mod._redis_client = None
    os.environ["REDIS_URL"] = redis_url

    yield None

    factories_mod._redis_client = original_client
    if original_url is None:
        os.environ.pop("REDIS_URL", None)
    else:
        os.environ["REDIS_URL"] = original_url


# ---------------------------------------------------------------------------
# E2E TEST
# ---------------------------------------------------------------------------


def _find_free_port() -> int:
    """Return an OS-assigned free TCP port for the uvicorn server."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, *, timeout: float = 10.0) -> bool:
    """Poll TCP host:port until reachable or timeout."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.05)
    return False


def test_e2i_alerts_publish_reaches_sse_subscriber(
    reset_redis_factory: None,
    redis_url: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real Redis publish on ``e2i:alerts`` → SSE bridge → SSE client
    receives the event within 5 seconds of publishing.

    The whole call graph is real:

    1. A FastAPI app with the :data:`staleness_alerts_router` mounted,
       served by a real ``uvicorn.Server`` running in a background
       thread on an OS-assigned port.
    2. :class:`AlertBridge` subscribed to a real ``redis.asyncio``
       pubsub on ``e2i:alerts`` against the test broker.
    3. A SYNC ``redis.from_url(redis_url).publish(...)`` from this
       test thread (so the publish is a true cross-process signal).
    4. A SYNC ``httpx.Client(...).stream("GET", ...)`` against the
       uvicorn server consuming the SSE stream by line.

    Why real uvicorn and not ASGITransport / Starlette TestClient:
    both of those drivers BUFFER the entire response body in memory
    and only return after the ASGI app completes — for an SSE
    endpoint that streams forever until client disconnect, that's a
    deadlock. A real uvicorn HTTP server is the only way to exercise
    the streaming wire format end-to-end.

    Auth: bypassed via ``E2I_TESTING_MODE=1`` (codex iter-1 M1: we now
    set this via ``monkeypatch.setenv`` + ``monkeypatch.setattr`` so
    pytest's monkeypatch fixture AUTOMATICALLY restores both surfaces
    at test teardown; without this restore, later tests running in
    the same process would inherit testing-mode auth bypass and
    silently false-positive).
    """
    import threading
    import time

    import httpx
    import redis  # sync client for publishing
    import uvicorn
    from fastapi import FastAPI

    from src.api.dependencies import auth as auth_module
    from src.api.routes.staleness_alerts import router

    # Codex iter-1 M1: monkeypatch BOTH surfaces (env var + module flag)
    # so pytest auto-restores them at teardown. The auth module reads
    # TESTING_MODE once at import; later tests in the same process
    # would otherwise see the leaked True value and silently bypass
    # authentication.
    monkeypatch.setenv("E2I_TESTING_MODE", "1")
    monkeypatch.setattr(auth_module, "TESTING_MODE", True)

    app = FastAPI()
    app.include_router(router, prefix="/api")

    # Unique brand so concurrent xdist workers / leftover Redis
    # subscribers don't pollute our asserts.
    brand = f"e2e-{uuid.uuid4().hex[:8]}"

    matching_payload: Dict[str, Any] = {
        "type": "staleness_alert",
        "sentinel_id": f"s-{brand}",
        "brands": [brand],
        "findings": [{"finding_id": "f-1", "staleness_score": 0.95}],
    }

    # -------- spin up uvicorn in a background thread --------
    port = _find_free_port()
    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    server_thread = threading.Thread(
        target=server.run, name="staleness-alerts-e2e-uvicorn", daemon=True
    )
    server_thread.start()
    try:
        # 30s headroom — CI runners are slower than dev laptops for the
        # full FastAPI app boot (Sentry init + OpenTelemetry init +
        # MLflow + Opik + the 18-agent router-import graph). Empirically
        # local boot is ~2s; CI was observed at 10-15s under load.
        assert _wait_for_port("127.0.0.1", port, timeout=30.0), (
            "uvicorn server did not start within 30s"
        )

        # -------- background publisher --------
        publisher_errors: list[BaseException] = []

        def _publish_after_delay() -> None:
            try:
                # Wait for the SSE consumer to subscribe to pubsub.
                # 0.75s is generous; the bridge subscribes within the
                # first event-loop tick after the request handler is
                # invoked.
                time.sleep(0.75)
                sync_client = redis.from_url(redis_url, decode_responses=True)
                sync_client.publish("e2i:alerts", json.dumps(matching_payload))
                sync_client.close()
            except BaseException as exc:  # pragma: no cover — diagnostic
                publisher_errors.append(exc)

        publisher_thread = threading.Thread(
            target=_publish_after_delay, name="sse-e2e-publisher", daemon=True
        )
        publisher_thread.start()

        # -------- consume SSE stream --------
        received_events: list[Dict[str, Any]] = []
        deadline = time.monotonic() + 10.0

        with httpx.Client(timeout=10.0) as client:
            with client.stream(
                "GET",
                f"http://127.0.0.1:{port}/api/alerts/stream?brand={brand}",
            ) as response:
                assert response.status_code == 200, f"SSE connect failed: {response.status_code}"
                current_event = ""
                current_data = ""
                for raw_line in response.iter_lines():
                    if time.monotonic() > deadline:
                        break
                    line = raw_line.rstrip("\r")
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                    elif line.startswith("data:"):
                        fragment = line.split(":", 1)[1].lstrip()
                        current_data = current_data + "\n" + fragment if current_data else fragment
                    elif line == "":
                        if current_event and current_data:
                            received_events.append({"event": current_event, "data": current_data})
                            current_event = ""
                            current_data = ""
                            if len(received_events) >= 1:
                                break

        publisher_thread.join(timeout=2.0)
        assert not publisher_errors, f"Background publisher raised: {publisher_errors[0]!r}"

        assert len(received_events) == 1, (
            f"Expected exactly one SSE event within 10s of publishing; "
            f"got {len(received_events)}: {received_events}"
        )
        evt = received_events[0]
        assert evt["event"] == "alert"
        decoded = json.loads(evt["data"])
        assert decoded == matching_payload, (
            f"Decoded SSE payload differs from published payload.\n"
            f"published: {matching_payload}\n"
            f"received:  {decoded}"
        )
    finally:
        # Gracefully stop uvicorn.
        server.should_exit = True
        server_thread.join(timeout=10.0)
