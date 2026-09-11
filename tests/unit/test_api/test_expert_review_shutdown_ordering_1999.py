"""#1999 item 2: on worker shutdown, a shielded assessment build finishes and
persists BEFORE the lifespan tears dependencies down, even when its client left.

#1998 documented the opposite as a limitation ("the lifespan closes Redis/Supabase
while such a task may still be building; loop teardown then cancels it before
its persist"). Measured on this stack (uvicorn 0.34 ``UvicornWorker``, which sets
no ``timeout_graceful_shutdown``; the real app's seven BaseHTTPMiddleware
layers): a client disconnect does not cancel the request task, and uvicorn's
``Server.shutdown()`` waits for every request task before it sends
``lifespan.shutdown``. The request task awaits the shielded build, so the build
and its persist complete first, and ``_INFLIGHT_BUILDS`` is empty when the
lifespan runs. A real gunicorn run agreed (SIGTERM mid-request: the request
finished, then the lifespan).

This pins that ordering through the REAL app and the REAL assessment route
under a REAL uvicorn server, and a positive control shows the test can see the
failure: with ``timeout_graceful_shutdown`` set, uvicorn cancels the request
task, and the lifespan starts while the orphaned build is still in flight and
unpersisted. That is the configuration a lifespan drain would be needed for.

The lifespan scope is intercepted to snapshot state when uvicorn sends it (the
real lifespan would connect to every backing service). The LLM build is a
gated stub and the store an in-memory repo, as in
test_expert_review_assessment_lock_1993.py; the lock runs in its local mode.
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
from typing import Any, Dict, List, Optional

import pytest
import uvicorn

# Module-level on purpose: importing src.api.main is paid once at collection,
# outside pytest-timeout's per-test budget (see test_gaps_time_period_1834.py).
import src.api.routes.expert_review as route_mod
from src.api.dependencies.auth import require_operator
from src.api.dependencies.inflight_lock import InflightLock
from src.api.main import app

RID = "5d0c7b1e-3f2a-4c68-9b1d-7e4a2c9f0b36"
URL = f"/api/expert-reviews/{RID}/assessment?force=true"
GATE_TIMEOUT = 5.0
ROW: Dict[str, Any] = {
    "review_id": RID,
    "review_type": "dag_approval",
    "dag_version_hash": "h" * 64,
    "brand": "Kisqali",
    "approval_status": "pending",
    "dag_structure_json": json.dumps({"nodes": ["t", "y"], "edges": [["t", "y"]]}),
    "related_validation_ids": [],
}


class _Repo:
    def __init__(self) -> None:
        self.row = dict(ROW)
        self.writes: List[Dict[str, Any]] = []

    async def get_by_id(self, review_id: str) -> Optional[Dict[str, Any]]:
        return dict(self.row) if review_id == RID else None

    async def update_agent_assessment(self, review_id: str, assessment: Dict[str, Any]) -> bool:
        self.writes.append(dict(assessment))
        self.row["agent_assessment_json"] = json.dumps(assessment)
        return True


async def _redis_down() -> Any:
    raise RuntimeError("Redis client not initialised (startup degraded mode)")


async def _until(pred, what: str, timeout: float = GATE_TIMEOUT) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while not pred():
        if loop.time() >= deadline:
            raise AssertionError(f"timed out waiting until: {what}")
        await asyncio.sleep(0.01)


async def _shutdown_with_a_disconnected_build(monkeypatch, **config_kwargs: Any) -> Dict[str, Any]:
    """Start a build, drop its client, stop the server while the build is
    gated, open the gate once the server is shutting down, and report what the
    lifespan saw when uvicorn sent ``lifespan.shutdown``."""
    repo = _Repo()
    started, gate = threading.Event(), threading.Event()

    async def _repo_factory():
        return repo

    async def _no_validation_rows(ids):
        return []

    def _gated_build(review, validations):
        started.set()
        if not gate.wait(timeout=GATE_TIMEOUT):
            raise RuntimeError("test gate never opened")
        return {"items": [{"id": "q1", "verdict": "supports"}], "is_fallback": False}

    monkeypatch.setattr(route_mod, "_get_expert_review_repo", _repo_factory)
    monkeypatch.setattr(route_mod, "_get_validation_rows", _no_validation_rows)
    monkeypatch.setattr(route_mod, "_build_assessment", _gated_build)
    monkeypatch.setattr(
        route_mod,
        "_ASSESSMENT_LOCK",
        InflightLock("test:1999:shutdown", redis_factory=_redis_down),
    )
    monkeypatch.setitem(app.dependency_overrides, require_operator, lambda: {"id": "op-1"})

    seen: Dict[str, Any] = {}

    async def asgi(scope, receive, send):
        if scope["type"] != "lifespan":
            await app(scope, receive, send)
            return
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                seen["inflight_builds"] = len(route_mod._INFLIGHT_BUILDS)
                seen["persisted_writes"] = len(repo.writes)
                await send({"type": "lifespan.shutdown.complete"})
                return

    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(asgi, lifespan="on", log_level="warning", **config_kwargs)
    )
    serve = asyncio.create_task(server.serve(sockets=[sock]))
    try:
        await _until(lambda: server.started, "the server started")
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(f"POST {URL} HTTP/1.1\r\nHost: t\r\nContent-Length: 0\r\n\r\n".encode())
        await writer.drain()
        await asyncio.wait_for(asyncio.to_thread(started.wait, GATE_TIMEOUT), GATE_TIMEOUT)
        assert len(route_mod._INFLIGHT_BUILDS) == 1

        writer.close()  # the client leaves mid-build (nginx 504, closed tab)
        await _until(lambda: not server.server_state.connections, "uvicorn saw the disconnect")
        server.should_exit = True
        # Let shutdown reach its wait (or its graceful timeout) before the build ends.
        await asyncio.sleep(0.5)
        gate.set()
        await asyncio.wait_for(serve, timeout=GATE_TIMEOUT)
        seen["writes_after_exit"] = len(repo.writes)
        return seen
    finally:
        gate.set()
        if not serve.done():
            server.force_exit = True
            await asyncio.gather(serve, return_exceptions=True)
        # An orphaned build (positive control) still finishes on this loop.
        await asyncio.gather(*list(route_mod._INFLIGHT_BUILDS), return_exceptions=True)


@pytest.mark.unit
async def test_worker_shutdown_lets_a_disconnected_requests_build_persist_before_the_lifespan(
    monkeypatch,
):
    # The UvicornWorker configuration: no graceful-shutdown timeout.
    seen = await _shutdown_with_a_disconnected_build(monkeypatch)

    assert seen["inflight_builds"] == 0, seen
    assert seen["persisted_writes"] == 1, seen


@pytest.mark.unit
async def test_positive_control_a_graceful_shutdown_timeout_orphans_the_build(monkeypatch):
    """The test can see the defect #1998 described: when uvicorn cancels the
    request task, the lifespan starts with the build unfinished and unpersisted."""
    seen = await _shutdown_with_a_disconnected_build(monkeypatch, timeout_graceful_shutdown=0.1)

    assert seen["inflight_builds"] == 1, seen
    assert seen["persisted_writes"] == 0, seen
