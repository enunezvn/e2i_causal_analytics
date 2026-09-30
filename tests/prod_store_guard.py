"""Refuse unit-test connections to the droplet's production stores (#2331).

Why this exists
---------------
On the droplet PROD == DEV. The repo-root ``.env`` (loaded with ``override=True`` by
``tests/conftest.py``) points every client at the live containers: FalkorDB
``localhost:6381``, Redis ``localhost:6382``, MLflow ``localhost:5000``, Supabase
``172.17.0.1:54321`` and Postgres ``127.0.0.1:5432``. A unit test that builds a real
client therefore talks to production. On 2026-09-29 the model_selector unit suite
wrote ``SUITED_FOR`` edges into prod FalkorDB twice (18:12Z and 21:13Z), clobbering
the ``updated_at`` of a genuine edge. CI has none of these services, so the same
tests pass there on ``ECONNREFUSED`` and nothing ever looked red.

The env pins in ``tests/unit/conftest.py`` (#1420 Supabase, #2207 Postgres DSN) close
the leak one variable at a time. This guard is the backstop underneath them: it
refuses the *connection*, whatever env var, default or cached singleton produced it.

What it does
------------
While a :class:`ProdStoreGuard` is active, a connect to a guarded port on a LOCAL
address (loopback, unspecified, or any of this host's own interface addresses --
``172.17.0.1`` is how ``.env`` reaches Supabase) is refused:

* ``socket.socket.connect`` raises :class:`ProdStoreRefused` (a
  ``ConnectionRefusedError`` with ``errno.ECONNREFUSED``); ``connect_ex`` returns
  ``ECONNREFUSED``. That one seam covers sync sockets, ``socket.create_connection``,
  asyncio's selector loop (``sock_connect`` / ``create_connection`` /
  ``open_connection`` all end in ``sock.connect``), redis-py and falkordb (sync and
  asyncio), requests/urllib3 (MLflow) and httpx/httpcore/anyio (Supabase).
* libpq opens its socket in C, below ``socket.socket``, so ``psycopg2.connect`` and
  psycopg 3's ``Connection.connect`` / ``AsyncConnection.connect`` are wrapped at the
  driver entry point and raise the driver's own ``OperationalError``.
* ``AF_UNIX`` connects to a guarded path (the docker socket) are refused the same way.

Failure semantics: refuse, record, report
-----------------------------------------
The refusal is exactly what CI sees, so production code paths (which mostly
log-and-continue on a failed connect) behave on the droplet as they do in CI, and a
test's outcome is the same in both places. Every refusal is recorded against the
test that made it and printed in a ``prod-store guard`` terminal-summary section,
so an attempt is never silent. ``E2I_PROD_STORE_GUARD_STRICT=1`` additionally fails
each test that made an attempt -- the census / regression-hunting mode.

Not covered: uvloop (its connects happen inside libuv; nothing in the unit tree
installs it), and a background thread still connecting after its test finished (the
guard is active per unit test item, see ``tests/unit/conftest.py``).

Opting in to a real store: run a throwaway one on a non-guarded port and point the
test's env at it. Guarded ports are never exempt, except the Supabase REST ports for
the pre-existing ``@pytest.mark.real_supabase`` read-only checks (#1420).
"""

from __future__ import annotations

import contextlib
import errno
import functools
import ipaddress
import os
import socket
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

GUARD_TAG = "[prod-store guard #2331]"

STRICT_ENV_VAR = "E2I_PROD_STORE_GUARD_STRICT"

# Every port a prod container (or the host front door) publishes on this droplet,
# measured 2026-09-29 with ``docker ps`` + ``ss -ltn``. Nothing on the unit tree
# may reach any of them: in CI they are all dead (the unit jobs' own throwaway
# Redis is on 6379 and their MLflow on 5000 -- which is why the guard is off in CI).
PROD_STORE_PORTS: dict[int, str] = {
    6381: "FalkorDB (e2i_falkordb)",
    6382: "Redis (e2i_redis, also the Celery broker)",
    5000: "MLflow (e2i_mlflow)",
    54321: "Supabase REST via Kong (supabase-kong)",
    8443: "Supabase Kong TLS (supabase-kong)",
    5432: "Postgres session pooler (supabase-pooler; .env SUPABASE_DB_URL)",
    6543: "Postgres transaction pooler (supabase-pooler)",
    5433: "Postgres direct (supabase-db)",
    4000: "Supabase analytics (supabase-analytics)",
    3001: "Supabase Studio (supabase-studio)",
    6567: "Feast feature server (e2i_feast)",
    3000: "BentoML model server (e2i_bentoml)",
    8000: "e2i API (e2i_api)",
    3030: "FalkorDB browser (e2i_falkordb_browser)",
    80: "host front door (nginx -> e2i_api)",
    443: "host front door TLS (nginx -> e2i_api)",
}

# The docker socket controls every prod container. No unit test uses the docker
# SDK today (grep, 2026-09-29); refusing it costs nothing.
PROD_UNIX_SOCKETS: dict[str, str] = {
    "/var/run/docker.sock": "Docker daemon (controls every prod container)",
    "/run/docker.sock": "Docker daemon (controls every prod container)",
}

# Reached by the pre-existing ``@pytest.mark.real_supabase`` read-only checks.
SUPABASE_REST_PORTS: frozenset[int] = frozenset({54321, 8443})

# Connection kwargs of psycopg 3's ``connect`` that are NOT libpq parameters.
_PSYCOPG_NON_CONNINFO_KWARGS = frozenset(
    {"autocommit", "prepare_threshold", "context", "row_factory", "cursor_factory"}
)


def guard_enabled() -> bool:
    """Guard the unit tree everywhere except GitHub Actions.

    CI's unit jobs stand up their OWN throwaway Redis/MLflow on some of these
    ports, and CI is where "connection refused" is already the baseline; the
    droplet is the only place these ports are production. Keyed on "not CI"
    rather than "is the droplet" so an unknown environment fails safe."""
    return os.environ.get("GITHUB_ACTIONS", "").lower() != "true"


def strict_mode() -> bool:
    return os.environ.get(STRICT_ENV_VAR, "").lower() in {"1", "true", "yes"}


class ProdStoreRefused(ConnectionRefusedError):
    """The guard's refusal. A ``ConnectionRefusedError`` so callers behave as in CI."""


@dataclass(frozen=True)
class Attempt:
    test: str
    target: str
    port: int | None
    store: str
    via: str

    def describe(self) -> str:
        return f"{self.target} {self.store} via {self.via}"


@functools.lru_cache(maxsize=1)
def _interface_addresses() -> frozenset[str]:
    addrs: set[str] = set()
    try:
        import psutil

        for nic in psutil.net_if_addrs().values():
            for a in nic:
                if a.family in (socket.AF_INET, socket.AF_INET6):
                    addrs.add(str(ipaddress.ip_address(a.address.split("%")[0])))
    except Exception:  # pragma: no cover - psutil is a hard dependency here
        pass
    return frozenset(addrs)


def _ip_is_local(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if ip.is_loopback or ip.is_unspecified:
        return True
    return str(ip) in _interface_addresses()


def is_local_host(host: str) -> bool:
    """True when ``host`` names this machine (loopback, 0.0.0.0/::, own interfaces)."""
    try:
        return _ip_is_local(ipaddress.ip_address(host.split("%")[0]))
    except ValueError:
        pass
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return False
    for info in infos:
        try:
            if _ip_is_local(ipaddress.ip_address(str(info[4][0]).split("%")[0])):
                return True
        except ValueError:
            continue
    return False


@dataclass
class ProdStoreGuard:
    ports: Mapping[int, str] = field(default_factory=lambda: dict(PROD_STORE_PORTS))
    unix_paths: Mapping[str, str] = field(default_factory=lambda: dict(PROD_UNIX_SOCKETS))
    exempt_ports: frozenset[int] = frozenset()
    current_test: str | None = None
    attempts: list[Attempt] = field(default_factory=list)
    _reported: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def match_inet(self, host: str, port: int) -> str | None:
        store = self.ports.get(port)
        if store is None or port in self.exempt_ports:
            return None
        return store if is_local_host(host) else None

    def match_unix(self, path: str) -> str | None:
        if path in self.unix_paths:
            return self.unix_paths[path]
        if not path or path.startswith("\0"):
            return None  # Linux abstract socket: no filesystem path to canonicalise
        return self.unix_paths.get(os.path.realpath(path))

    def refuse(self, target: str, port: int | None, store: str, via: str) -> str:
        test = self.current_test or "<outside a unit test>"
        with self._lock:
            self.attempts.append(Attempt(test, target, port, store, via))
        return (
            f"{GUARD_TAG} refused {via} connect to {target} -- {store}. On the droplet "
            f"this is PRODUCTION (test: {test}). Patch the store client at the test "
            "boundary, or point the test at a throwaway store on a non-prod port."
        )

    def take_new_attempts(self) -> list[Attempt]:
        """Attempts recorded since the previous call (one report phase's worth)."""
        with self._lock:
            new = self.attempts[self._reported :]
            self._reported = len(self.attempts)
        return new

    @contextlib.contextmanager
    def active(self) -> Iterator[ProdStoreGuard]:
        install_patches()
        _ACTIVE.append(self)
        try:
            yield self
        finally:
            _ACTIVE.remove(self)


# ---------------------------------------------------------------------------
# patches -- installed once per process, a pass-through while no guard is active
# ---------------------------------------------------------------------------

_ACTIVE: list[ProdStoreGuard] = []
_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


def active_guards() -> tuple[ProdStoreGuard, ...]:
    return tuple(_ACTIVE)


def _check_socket(sock: socket.socket, address: Any) -> None:
    if not _ACTIVE:
        return
    if sock.family in (socket.AF_INET, socket.AF_INET6):
        if not (isinstance(address, tuple) and len(address) >= 2):
            return
        host, port = str(address[0]), address[1]
        for guard in reversed(tuple(_ACTIVE)):  # the innermost guard owns it
            store = guard.match_inet(host, port)
            if store is not None:
                target = f"[{host}]:{port}" if ":" in host else f"{host}:{port}"
                msg = guard.refuse(target, port, store, "socket")
                raise ProdStoreRefused(errno.ECONNREFUSED, msg)
    elif sock.family == getattr(socket, "AF_UNIX", None) and isinstance(address, (str, bytes)):
        path = os.fsdecode(address)
        for guard in reversed(tuple(_ACTIVE)):  # the innermost guard owns it
            store = guard.match_unix(path)
            if store is not None:
                msg = guard.refuse(path, None, store, "socket")
                raise ProdStoreRefused(errno.ECONNREFUSED, msg)


def _pg_targets(params: Mapping[str, Any]) -> list[tuple[str, int]]:
    """(host, port) pairs libpq could connect to, from the conninfo and PG* env.

    libpq connects to ``hostaddr`` when given and uses ``host`` for auth/SSL; both
    are checked, so a local ``hostaddr`` behind a remote ``host`` name is still
    caught. An empty host or a unix-socket directory means this machine."""
    port_spec = str(params.get("port") or os.environ.get("PGPORT", "") or "5432")
    ports = [p.strip() for p in port_spec.split(",")]
    targets = []
    for key, env_var in (("hostaddr", "PGHOSTADDR"), ("host", "PGHOST")):
        hosts = str(params.get(key) or os.environ.get(env_var, "")).split(",")
        for i, h in enumerate(hosts):
            h = h.strip()
            if key == "hostaddr" and not h:
                continue
            if not h or h.startswith("/") or h.startswith("@"):
                h = "127.0.0.1"
            p = ports[i] if i < len(ports) else ports[0]
            try:
                targets.append((h, int(p or 5432)))
            except ValueError:
                continue  # libpq rejects it itself
    return targets


def _check_pg(params: Mapping[str, Any], via: str, error: type[Exception]) -> None:
    guards = tuple(reversed(_ACTIVE))  # the innermost guard owns it
    service = params.get("service") or os.environ.get("PGSERVICE")
    if service and guards:
        # A pg_service.conf entry hides its endpoint from the guard: fail closed.
        store = "Postgres service (endpoint unresolvable by the guard)"
        raise error(guards[0].refuse(f"service={service}", None, store, via))
    for host, port in _pg_targets(params):
        for guard in guards:
            store = guard.match_inet(host, port)
            if store is not None:
                raise error(guard.refuse(f"{host}:{port}", port, store, via))


def _patch_psycopg2() -> None:
    try:
        import psycopg2
        from psycopg2.extensions import make_dsn, parse_dsn
    except ImportError:  # pragma: no cover - installed in every env that runs tests
        return
    real_connect = psycopg2.connect

    @functools.wraps(real_connect)
    def connect(
        dsn: Any = None, connection_factory: Any = None, cursor_factory: Any = None, **kwargs: Any
    ) -> Any:
        if _ACTIVE:
            conn_kwargs = {k: v for k, v in kwargs.items() if k not in {"async", "async_"}}
            try:
                params = parse_dsn(make_dsn(dsn, **conn_kwargs))
            except Exception:
                params = None  # unparseable: let psycopg2 raise its own error
            if params is not None:
                _check_pg(params, "psycopg2", psycopg2.OperationalError)
        return real_connect(dsn, connection_factory, cursor_factory, **kwargs)

    psycopg2.connect = connect


def _patch_psycopg3() -> None:
    try:
        import psycopg
        from psycopg.conninfo import conninfo_to_dict, make_conninfo
    except ImportError:  # pragma: no cover
        return

    def params_of(conninfo: str, kwargs: Mapping[str, Any]) -> Mapping[str, Any] | None:
        conn_kwargs = {k: v for k, v in kwargs.items() if k not in _PSYCOPG_NON_CONNINFO_KWARGS}
        try:
            return conninfo_to_dict(make_conninfo(conninfo, **conn_kwargs))
        except Exception:
            return None

    real_sync = psycopg.Connection.__dict__["connect"].__func__
    real_async = psycopg.AsyncConnection.__dict__["connect"].__func__

    @functools.wraps(real_sync)
    def sync_connect(cls: Any, conninfo: str = "", **kwargs: Any) -> Any:
        if _ACTIVE and (params := params_of(conninfo, kwargs)) is not None:
            _check_pg(params, "psycopg", psycopg.OperationalError)
        return real_sync(cls, conninfo, **kwargs)

    @functools.wraps(real_async)
    async def async_connect(cls: Any, conninfo: str = "", **kwargs: Any) -> Any:
        if _ACTIVE and (params := params_of(conninfo, kwargs)) is not None:
            _check_pg(params, "psycopg", psycopg.OperationalError)
        return await real_async(cls, conninfo, **kwargs)

    psycopg.Connection.connect = classmethod(sync_connect)  # type: ignore[method-assign,assignment]
    psycopg.AsyncConnection.connect = classmethod(async_connect)  # type: ignore[method-assign,assignment]
    psycopg.connect = psycopg.Connection.connect  # the module alias is bound at import


def install_patches() -> None:
    """Install the pass-through wrappers (idempotent). Call it before test modules
    are collected, so a ``from psycopg2 import connect`` alias binds the wrapper."""
    global _INSTALLED
    with _INSTALL_LOCK:
        if _INSTALLED:
            return
        real_connect = socket.socket.connect
        real_connect_ex = socket.socket.connect_ex

        def connect(self: socket.socket, address: Any) -> None:
            _check_socket(self, address)
            return real_connect(self, address)

        def connect_ex(self: socket.socket, address: Any) -> int:
            try:
                _check_socket(self, address)
            except ProdStoreRefused:
                return errno.ECONNREFUSED
            return real_connect_ex(self, address)

        socket.socket.connect = connect  # type: ignore[method-assign]
        socket.socket.connect_ex = connect_ex  # type: ignore[method-assign]
        _patch_psycopg2()
        _patch_psycopg3()
        _INSTALLED = True
