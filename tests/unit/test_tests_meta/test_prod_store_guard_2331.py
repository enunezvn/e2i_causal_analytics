"""The unit-tree prod-store guard refuses connections to the droplet's prod stores (#2331).

On the droplet PROD == DEV: the repo-root ``.env`` points FalkorDB, Redis, MLflow and
Supabase at the live containers on localhost, and on 2026-09-29 the model_selector
unit suite wrote ``SUITED_FOR`` edges into prod FalkorDB twice. CI has none of those
services, so the same tests pass there on ``ECONNREFUSED``.

Two layers are tested here:

* **Mechanism** -- a :class:`ProdStoreGuard` configured with a throwaway listener's
  port as its "prod" target. Every client family the unit tree uses makes a REAL
  connect attempt at that listener; the test asserts the attempt was refused, was
  recorded, and never reached the listener's accept queue. This half is independent
  of the environment and runs identically in CI.
* **Wiring** -- the conftest's guard is live during unit tests on the droplet: a bare
  TCP connect to each real prod port is refused by the guard. In CI the guard is off
  (some unit lanes run their OWN throwaway Redis/MLflow on these ports), so these
  probes skip there.
"""

from __future__ import annotations

import asyncio
import errno
import os
import socket
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Bound at collection time, as a module doing ``from psycopg2 import connect``
# would be. The unit conftest installs the wrappers before collection.
from psycopg2 import connect as _psycopg2_connect_alias

from tests.prod_store_guard import (
    GUARD_TAG,
    PROD_STORE_PORTS,
    STRICT_ENV_VAR,
    ProdStoreGuard,
    active_guards,
    guard_enabled,
)

try:  # psycopg 3 is used by src/ but is not in requirements.txt, so CI lacks it
    from psycopg import connect as _psycopg3_connect_alias
except ImportError:
    _psycopg3_connect_alias = None

REPO_ROOT = Path(__file__).resolve().parents[3]

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def listener() -> Iterator[socket.socket]:
    """A real TCP listener on an ephemeral loopback port that never accepts.

    A successful connect lands in its backlog without an ``accept`` call, so
    ``_backlog_empty`` is a direct observation of whether a client got through."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))
    srv.listen(16)
    try:
        yield srv
    finally:
        srv.close()


def _backlog_empty(srv: socket.socket) -> bool:
    srv.setblocking(False)
    try:
        conn, _ = srv.accept()
    except BlockingIOError:
        return True
    conn.close()
    return False


def _guard_for(port: int) -> ProdStoreGuard:
    return ProdStoreGuard(ports={port: "test-store"}, unix_paths={})


def _assert_refused_and_recorded(guard: ProdStoreGuard, srv: socket.socket, via: str) -> None:
    port = srv.getsockname()[1]
    assert guard.attempts, "the guard recorded no attempt"
    assert all(a.port == port for a in guard.attempts), guard.attempts
    assert {a.via for a in guard.attempts} == {via}, guard.attempts
    assert _backlog_empty(srv), "the client REACHED the listener -- the guard did not refuse"


# ---------------------------------------------------------------------------
# mechanism: raw sockets
# ---------------------------------------------------------------------------


def test_control_unguarded_port_reaches_the_listener(listener: socket.socket) -> None:
    """Control: a port outside the guard's set is untouched (the throwaway-store opt-in)."""
    guard = ProdStoreGuard(ports={1: "not-this-one"}, unix_paths={})
    with guard.active():
        socket.create_connection(listener.getsockname(), timeout=2).close()
    assert guard.attempts == []
    assert not _backlog_empty(listener)


def test_sync_create_connection_is_refused(listener: socket.socket) -> None:
    guard = _guard_for(listener.getsockname()[1])
    with guard.active(), pytest.raises(ConnectionRefusedError, match=r"#2331"):
        socket.create_connection(listener.getsockname(), timeout=2)
    _assert_refused_and_recorded(guard, listener, "socket")


def test_connect_ex_returns_econnrefused(listener: socket.socket) -> None:
    guard = _guard_for(listener.getsockname()[1])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with guard.active():
            assert s.connect_ex(listener.getsockname()) == errno.ECONNREFUSED
    finally:
        s.close()
    _assert_refused_and_recorded(guard, listener, "socket")


def test_localhost_name_is_resolved_and_refused(listener: socket.socket) -> None:
    guard = _guard_for(listener.getsockname()[1])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with guard.active(), pytest.raises(ConnectionRefusedError):
            s.connect(("localhost", listener.getsockname()[1]))
    finally:
        s.close()
    _assert_refused_and_recorded(guard, listener, "socket")


def test_non_loopback_local_interface_address_is_refused() -> None:
    """``.env`` reaches Supabase through the docker bridge (``172.17.0.1``), and the
    pooler/db ports are bound on ``0.0.0.0`` -- a host's own interface address is
    as "prod" as loopback."""
    import psutil

    addrs = [
        a.address
        for nic in psutil.net_if_addrs().values()
        for a in nic
        if a.family == socket.AF_INET and not a.address.startswith("127.")
    ]
    if not addrs:
        pytest.skip("host has no non-loopback IPv4 interface")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((addrs[0], 0))
    srv.listen(4)
    try:
        guard = _guard_for(srv.getsockname()[1])
        with guard.active(), pytest.raises(ConnectionRefusedError):
            socket.create_connection(srv.getsockname(), timeout=2)
        _assert_refused_and_recorded(guard, srv, "socket")
    finally:
        srv.close()


def test_remote_host_on_a_guarded_port_is_not_the_guards_business() -> None:
    guard = ProdStoreGuard(ports={6381: "FalkorDB"}, unix_paths={})
    assert guard.match_inet("127.0.0.1", 6381) == "FalkorDB"
    assert guard.match_inet("::1", 6381) == "FalkorDB"
    assert guard.match_inet("0.0.0.0", 6381) == "FalkorDB"
    assert guard.match_inet("::ffff:127.0.0.1", 6381) == "FalkorDB"
    assert guard.match_inet("8.8.8.8", 6381) is None
    assert guard.match_inet("127.0.0.1", 6399) is None


def test_unix_socket_path_is_refused() -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "store.sock")
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        srv.bind(path)
        srv.listen(4)
        try:
            guard = ProdStoreGuard(ports={}, unix_paths={path: "unix-store"})
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                with guard.active(), pytest.raises(ConnectionRefusedError):
                    s.connect(path)
            finally:
                s.close()
            assert [a.target for a in guard.attempts] == [path]
            assert _backlog_empty(srv)
        finally:
            srv.close()


# ---------------------------------------------------------------------------
# mechanism: asyncio
# ---------------------------------------------------------------------------


def test_asyncio_open_connection_is_refused(listener: socket.socket) -> None:
    host, port = listener.getsockname()
    guard = _guard_for(port)

    async def go() -> None:
        await asyncio.open_connection(host, port)

    with guard.active(), pytest.raises(ConnectionRefusedError):
        asyncio.run(go())
    _assert_refused_and_recorded(guard, listener, "socket")


def test_asyncio_loop_sock_connect_is_refused(listener: socket.socket) -> None:
    guard = _guard_for(listener.getsockname()[1])

    async def go() -> None:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setblocking(False)
        try:
            await asyncio.get_running_loop().sock_connect(s, listener.getsockname())
        finally:
            s.close()

    with guard.active(), pytest.raises(ConnectionRefusedError):
        asyncio.run(go())
    _assert_refused_and_recorded(guard, listener, "socket")


def test_asyncio_create_connection_is_refused(listener: socket.socket) -> None:
    host, port = listener.getsockname()
    guard = _guard_for(port)

    async def go() -> None:
        await asyncio.get_running_loop().create_connection(asyncio.Protocol, host, port)

    with guard.active(), pytest.raises(ConnectionRefusedError):
        asyncio.run(go())
    _assert_refused_and_recorded(guard, listener, "socket")


# ---------------------------------------------------------------------------
# mechanism: the store clients the unit tree actually uses
# ---------------------------------------------------------------------------


def test_redis_sync_client_is_refused(listener: socket.socket) -> None:
    import redis

    host, port = listener.getsockname()
    guard = _guard_for(port)
    client = redis.Redis(
        host=host, port=port, socket_connect_timeout=2, socket_timeout=2, retry=None
    )
    with guard.active(), pytest.raises(redis.exceptions.ConnectionError):
        client.ping()
    _assert_refused_and_recorded(guard, listener, "socket")


def test_redis_async_client_is_refused(listener: socket.socket) -> None:
    import redis.asyncio as aioredis
    from redis.exceptions import ConnectionError as RedisConnectionError

    host, port = listener.getsockname()
    guard = _guard_for(port)

    async def go() -> None:
        client = aioredis.Redis(
            host=host, port=port, socket_connect_timeout=2, socket_timeout=2, retry=None
        )
        try:
            await client.ping()
        finally:
            await client.aclose()

    with guard.active(), pytest.raises(RedisConnectionError):
        asyncio.run(go())
    _assert_refused_and_recorded(guard, listener, "socket")


def test_falkordb_client_is_refused(listener: socket.socket) -> None:
    from falkordb import FalkorDB
    from redis.exceptions import ConnectionError as RedisConnectionError

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(RedisConnectionError):
        FalkorDB(host=host, port=port, socket_connect_timeout=2, socket_timeout=2).select_graph(
            "g"
        ).query("RETURN 1")
    _assert_refused_and_recorded(guard, listener, "socket")


def test_requests_is_refused(listener: socket.socket) -> None:
    import requests

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(requests.exceptions.ConnectionError):
        requests.get(f"http://{host}:{port}/", timeout=2)
    _assert_refused_and_recorded(guard, listener, "socket")


def test_httpx_sync_and_async_are_refused(listener: socket.socket) -> None:
    import httpx

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(httpx.ConnectError):
        httpx.get(f"http://{host}:{port}/", timeout=2)

    async def go() -> None:
        async with httpx.AsyncClient(timeout=2) as client:
            await client.get(f"http://{host}:{port}/")

    with guard.active(), pytest.raises(httpx.ConnectError):
        asyncio.run(go())
    assert len(guard.attempts) >= 2
    _assert_refused_and_recorded(guard, listener, "socket")


def test_mlflow_client_is_refused(listener: socket.socket, monkeypatch: pytest.MonkeyPatch) -> None:
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "2")
    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(MlflowException):
        MlflowClient(tracking_uri=f"http://{host}:{port}").get_registered_model("m")
    _assert_refused_and_recorded(guard, listener, "socket")


def test_supabase_client_is_refused(listener: socket.socket) -> None:
    import httpx
    from supabase import ClientOptions, create_client

    host, port = listener.getsockname()
    guard = _guard_for(port)
    client = create_client(
        f"http://{host}:{port}", "test-key", ClientOptions(postgrest_client_timeout=2)
    )
    with guard.active(), pytest.raises(httpx.ConnectError):
        client.table("t").select("*").execute()
    _assert_refused_and_recorded(guard, listener, "socket")


def test_psycopg2_is_refused_before_libpq(listener: socket.socket) -> None:
    """libpq opens its socket in C, below ``socket.socket`` -- the guard has to sit
    on the driver entry point, and must still raise the driver's own error."""
    import psycopg2

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(psycopg2.OperationalError, match=r"#2331"):
        psycopg2.connect(host=host, port=port, dbname="x", user="x", connect_timeout=2)
    with guard.active(), pytest.raises(psycopg2.OperationalError, match=r"#2331"):
        psycopg2.connect(f"postgresql://x:x@{host}:{port}/x?connect_timeout=2")
    assert len(guard.attempts) == 2
    _assert_refused_and_recorded(guard, listener, "psycopg2")


def test_psycopg3_sync_and_async_are_refused_before_libpq(listener: socket.socket) -> None:
    psycopg = pytest.importorskip("psycopg")

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(psycopg.OperationalError, match=r"#2331"):
        psycopg.connect(f"host={host} port={port} dbname=x user=x connect_timeout=2")
    with guard.active(), pytest.raises(psycopg.OperationalError, match=r"#2331"):
        psycopg.Connection.connect(host=host, port=port, dbname="x", connect_timeout=2)

    async def go() -> None:
        await psycopg.AsyncConnection.connect(f"postgresql://x:x@{host}:{port}/x?connect_timeout=2")

    with guard.active(), pytest.raises(psycopg.OperationalError, match=r"#2331"):
        asyncio.run(go())
    assert len(guard.attempts) == 3
    _assert_refused_and_recorded(guard, listener, "psycopg")


def test_psycopg2_alias_bound_at_collection_is_still_guarded(listener: socket.socket) -> None:
    import psycopg2

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(psycopg2.OperationalError, match=r"#2331"):
        _psycopg2_connect_alias(host=host, port=port, dbname="x", connect_timeout=2)
    _assert_refused_and_recorded(guard, listener, "psycopg2")


def test_psycopg3_alias_bound_at_collection_is_still_guarded(listener: socket.socket) -> None:
    psycopg = pytest.importorskip("psycopg")
    assert _psycopg3_connect_alias is not None

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(psycopg.OperationalError, match=r"#2331"):
        _psycopg3_connect_alias(f"host={host} port={port} dbname=x connect_timeout=2")
    _assert_refused_and_recorded(guard, listener, "psycopg")


def test_pg_hostaddr_env_behind_a_remote_host_name_is_refused(
    listener: socket.socket, monkeypatch: pytest.MonkeyPatch
) -> None:
    """libpq connects to ``hostaddr`` and only uses ``host`` for auth/SSL."""
    import psycopg2

    host, port = listener.getsockname()
    monkeypatch.setenv("PGHOSTADDR", host)
    guard = _guard_for(port)
    with guard.active(), pytest.raises(psycopg2.OperationalError, match=r"#2331"):
        psycopg2.connect(host="remote.invalid", port=port, dbname="x", connect_timeout=2)
    _assert_refused_and_recorded(guard, listener, "psycopg2")


def test_pg_multi_host_list_is_refused_on_its_local_member(listener: socket.socket) -> None:
    psycopg = pytest.importorskip("psycopg")

    host, port = listener.getsockname()
    guard = _guard_for(port)
    with guard.active(), pytest.raises(psycopg.OperationalError, match=r"#2331"):
        psycopg.connect(f"host=remote.invalid,{host} port=5,{port} dbname=x connect_timeout=2")
    _assert_refused_and_recorded(guard, listener, "psycopg")


@pytest.mark.parametrize(
    ("env", "key"),
    [
        ({"PGPORT": "1"}, "port"),  # explicit empty port -> 5432, not PGPORT
        ({"PGHOST": "remote.invalid"}, "host"),  # explicit empty host -> local socket
        ({"PGSERVICE": "l2331-unknown-service"}, "service"),  # empty service masks env
    ],
)
def test_pg_explicit_empty_conninfo_values_mask_the_environment(
    monkeypatch: pytest.MonkeyPatch, env: dict[str, str], key: str
) -> None:
    """libpq takes a key present in the conninfo even when empty; the environment
    only fills absent keys. Each case here ends at local 5432."""
    from tests.prod_store_guard import _pg_param, _pg_targets

    for k, v in env.items():
        monkeypatch.setenv(k, v)
    params = {"host": "127.0.0.1", "port": "5432", key: ""}
    if key == "service":
        assert _pg_param(params, "service", "PGSERVICE") == ""
    else:
        assert ("127.0.0.1", 5432) in _pg_targets(params)


def test_pg_service_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """A pg_service.conf entry hides its endpoint; the guard refuses rather than guess."""
    import psycopg2

    monkeypatch.setenv("PGSERVICE", "l2331-unknown-service")
    guard = ProdStoreGuard(ports={5432: "Postgres"}, unix_paths={})
    with guard.active(), pytest.raises(psycopg2.OperationalError, match=r"#2331"):
        psycopg2.connect(dbname="x", connect_timeout=2)
    assert [a.target for a in guard.attempts] == ["service=l2331-unknown-service"]


def test_linux_abstract_unix_socket_passes_through() -> None:
    if not sys.platform.startswith("linux"):
        pytest.skip("abstract unix sockets are Linux-only")
    name = f"\0l2331-abstract-{os.getpid()}"
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(name)
    srv.listen(4)
    try:
        guard = ProdStoreGuard(ports={}, unix_paths={"/var/run/docker.sock": "docker"})
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            with guard.active():
                s.connect(name)
        finally:
            s.close()
        assert guard.attempts == []
    finally:
        srv.close()


def test_attempt_names_the_src_frame_that_made_it(listener: socket.socket) -> None:
    """The census needs to know WHICH production code reached out, not just the test."""
    host, port = listener.getsockname()
    fake_src = os.path.join(os.sep, "repo", "src", "memory", "fake_client.py")
    code = compile(
        "import socket\ndef open_store(addr):\n    socket.create_connection(addr, timeout=2)\n",
        fake_src,
        "exec",
    )
    namespace: dict = {}
    exec(code, namespace)
    guard = _guard_for(port)
    with guard.active(), pytest.raises(ConnectionRefusedError):
        namespace["open_store"]((host, port))
    (attempt,) = guard.attempts
    assert attempt.origin == os.path.join("src", "memory", "fake_client.py") + ":3 open_store"
    assert attempt.describe().endswith("from " + attempt.origin)


def test_innermost_active_guard_owns_the_attempt(listener: socket.socket) -> None:
    """The conftest's per-item guard sits inside any outer guard; its report must
    not lose the item's attempts to the outer one."""
    outer = _guard_for(listener.getsockname()[1])
    inner = _guard_for(listener.getsockname()[1])
    with outer.active(), inner.active(), pytest.raises(ConnectionRefusedError):
        socket.create_connection(listener.getsockname(), timeout=2)
    assert outer.attempts == []
    _assert_refused_and_recorded(inner, listener, "socket")


def test_deactivated_guard_restores_normal_connects(listener: socket.socket) -> None:
    guard = _guard_for(listener.getsockname()[1])
    with guard.active():
        pass
    socket.create_connection(listener.getsockname(), timeout=2).close()
    assert guard.attempts == []
    assert not _backlog_empty(listener)


# ---------------------------------------------------------------------------
# wiring: the unit-tree conftest's guard is live on the droplet
# ---------------------------------------------------------------------------


def test_default_targets_cover_the_prod_stores_named_in_2331() -> None:
    for port in (6381, 6382, 5000, 54321, 4443):
        assert port in PROD_STORE_PORTS, port


@pytest.mark.parametrize("port", sorted(PROD_STORE_PORTS))
def test_unit_test_connect_to_prod_port_is_refused(port: int) -> None:
    """A bare TCP connect (no bytes sent) to each prod port from inside a unit test.

    Before the guard existed this CONNECTED on the droplet (the red run). With the
    guard it is refused by the guard, tagged, before any packet leaves. In CI the
    guard is off (a CI lane may run its own MLflow on 5000), so this is skipped there;
    the listener tests above prove the mechanism in every environment."""
    if not guard_enabled():
        pytest.skip("the unit-tree guard is off in GitHub Actions")
    with pytest.raises(ConnectionRefusedError) as exc_info:
        socket.create_connection(("127.0.0.1", port), timeout=2).close()
    assert GUARD_TAG in str(exc_info.value)
    # The conftest's guard recorded it. Taking the record also keeps this
    # deliberate probe out of the summary and out of strict mode.
    unit_guard = active_guards()[-1]  # innermost: an outer guard may also be active
    assert [a.port for a in unit_guard.take_new_attempts()] == [port]


# ---------------------------------------------------------------------------
# reporting: an attempt is never silent; strict mode fails the test
# ---------------------------------------------------------------------------

_INNER_TEST = """
import socket

import pytest


def _reach():
    with pytest.raises(ConnectionRefusedError):
        socket.create_connection(("127.0.0.1", 6381), timeout=2)


def test_reaches_for_prod_falkordb():
    _reach()


@pytest.mark.xfail(reason="expected failure that also reaches for prod")
def test_xfail_reaching():
    _reach()
    assert False


@pytest.mark.xfail(reason="unexpected pass that also reaches for prod")
def test_xpass_reaching():
    _reach()
"""

_INNER_CONFTEST = """
from tests.unit.conftest import (  # noqa: F401
    pytest_runtest_logreport,
    pytest_runtest_makereport,
    pytest_runtest_protocol,
    pytest_terminal_summary,
)
"""


def _run_inner_session(tmp_path, strict: bool) -> subprocess.CompletedProcess[str]:
    """A separate pytest session wired with the unit conftest's guard hooks only
    (its own rootdir and ini, so neither the repo's addopts nor the root conftest
    load). The guard is forced on, as on the droplet, wherever this runs."""
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "conftest.py").write_text(_INNER_CONFTEST)
    (tmp_path / "test_inner.py").write_text(_INNER_TEST)
    env = {k: v for k, v in os.environ.items() if k not in {"GITHUB_ACTIONS", STRICT_ENV_VAR}}
    env["PYTHONPATH"] = str(REPO_ROOT)
    # The venv's entry-point plugins (opik, langsmith, ...) cost ~20 s to load and
    # are irrelevant here.
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if strict:
        env[STRICT_ENV_VAR] = "1"
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "."],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_refused_attempt_is_reported_in_the_terminal_summary(tmp_path) -> None:
    proc = _run_inner_session(tmp_path, strict=False)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "1 passed, 1 xfailed, 1 xpassed" in proc.stdout
    assert "3 refused connection attempt(s) to PRODUCTION stores from 3 test(s)" in proc.stdout
    assert "test_inner.py::test_reaches_for_prod_falkordb [call] -> 127.0.0.1:6381" in proc.stdout


def test_strict_mode_fails_the_test_that_made_the_attempt(tmp_path) -> None:
    """Including xfail/xpass: their ``wasxfail`` would otherwise let pytest count the
    promoted failure as expected and exit 0."""
    proc = _run_inner_session(tmp_path, strict=True)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "3 failed" in proc.stdout
    assert f"{STRICT_ENV_VAR}=1 and this test tried to reach a production store" in proc.stdout
