"""Fixtures for the learning-loop real-DB tests (opt-in: ``E2I_DB_INTEGRATION=1``).

WHERE THIS SUITE RUNS, AND WHERE IT DOES NOT
--------------------------------------------
* **Not in CI.** GitHub-hosted runners cannot reach the droplet's database, so every test that
  takes a fixture below SKIPS in ``backend-tests.yml``. Only the pure tests in this directory
  (``test_fixture_modes.py``, ``test_lane_migration_files.py``, the prod-access refusals) run
  there. A skipped real-DB test is not coverage.
* **In every deploy**, blocking, before migrations and before any container flip:
  ``scripts/deploy/realdb_suite_gate.sh`` from ``deploy.yml`` (#2065).
* **On the droplet** by hand, with ``E2I_DB_INTEGRATION=1`` and ``-n 0``.

MODES (#2065; ``test_fixture_modes.py`` pins the decisions)
-----------------------------------------------------------
``base_db`` (session) is prod as it is NOW: schema, ledger, and registry rows synced from code.
``pending`` is derived: runner keys prod's ledger does not hold. So the base is exactly "prod
before this deploy's migrations".

* Behaviour tests use ``clone_db`` / ``module_db``: copies of ``deployed_db``, the base with every
  pending migration applied (the base itself when nothing is pending). They run on every run.
* Upgrade-path tests carry ``@pytest.mark.realdb_upgrade(*keys)`` and use ``base_clone_db``:
  copies of the base BEFORE the pending migrations. They run only while the keys they upgrade
  through are pending (any pending key when none are named), and otherwise skip naming them.

Tests never write to a template; they write to clones, and ``PgConn.rolled_back()`` isolates
per test.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Iterator, List, Tuple

import pytest

from tests.unit.test_database.learning_loop import _pg

UPGRADE_MARKER = "realdb_upgrade"

#: Filled by the session fixtures, printed in the terminal summary so a deploy log says which
#: mode ran and against what.
_RUN: Dict[str, object] = {}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{UPGRADE_MARKER}(*keys): real-DB upgrade-path test; runs on prod's current schema "
        "BEFORE the pending migrations, only while the named ledger keys (any, if none named) "
        "are pending. See tests/unit/test_database/learning_loop/conftest.py.",
    )


def pytest_terminal_summary(terminalreporter) -> None:  # type: ignore[no-untyped-def]
    if not _RUN:
        return
    terminalreporter.write_line(
        "learning-loop real-DB: prod ledger {ledger} keys; pending {pending}; simulated {sim}; "
        "deployed template {deployed}; registry sync {sync}".format(
            ledger=_RUN.get("ledger"),
            pending=_RUN.get("pending"),
            sim=_RUN.get("simulated"),
            deployed=_RUN.get("deployed"),
            sync=_RUN.get("registry_sync"),
        )
    )


def _require_opt_in() -> None:
    if not _pg.db_integration_enabled():
        pytest.skip(_pg.OPT_IN_SKIP_REASON)


@pytest.fixture(scope="session")
def prod_readonly() -> _pg.ProdReadOnly:
    _require_opt_in()
    return _pg.ProdReadOnly()


@pytest.fixture(scope="session")
def pg_container(prod_readonly: _pg.ProdReadOnly) -> Iterator[_pg.ThrowawayPg]:
    _pg.reap_orphans()
    pg = _pg.ThrowawayPg(image=prod_readonly.image())
    try:
        pg.start()
        yield pg
    finally:
        pg.stop()


class BaseDb(_pg.PgConn):
    def __init__(self, pg: _pg.ThrowawayPg, db: str, build: _pg.BaseBuild):
        super().__init__(pg, db)
        self.build = build
        self.restore_log = build.restore_log


@pytest.fixture(scope="session")
def base_db(pg_container: _pg.ThrowawayPg, prod_readonly: _pg.ProdReadOnly) -> BaseDb:
    build = _pg.build_base(
        pg_container, prod_readonly, simulate=os.getenv(_pg.SIMULATE_PENDING_ENV)
    )
    _RUN.update(
        ledger=len(build.prod_ledger),
        pending=build.pending,
        simulated=build.simulated,
        registry_sync=build.registry_sync,
    )
    return BaseDb(pg_container, _pg.BASE_DB, build)


@pytest.fixture(scope="session")
def pending_migrations(base_db: BaseDb) -> List[str]:
    return list(base_db.build.pending)


@pytest.fixture(scope="session")
def deployed_db(base_db: BaseDb) -> _pg.PgConn:
    """The post-deploy schema: the base with every pending migration applied."""
    db, build = _pg.build_deployed(base_db.pg, base_db.build.pending)
    _RUN.update(deployed=db, deployed_applied=build.applied)
    if build.registry_sync is not None:
        _RUN["registry_sync"] = build.registry_sync
    return _pg.PgConn(base_db.pg, db)


@pytest.fixture(autouse=True)
def _upgrade_path_gate(request: pytest.FixtureRequest) -> None:
    marker = request.node.get_closest_marker(UPGRADE_MARKER)
    if marker is None:
        return
    pending = request.getfixturevalue("pending_migrations")
    reason = _pg.upgrade_skip_reason(tuple(marker.args), pending)
    if reason is not None:
        pytest.skip(reason)


def _clones(pg: _pg.ThrowawayPg, template: str) -> Iterator[Callable[[str], _pg.PgConn]]:
    made: List[_pg.PgConn] = []

    def make(label: str) -> _pg.PgConn:
        conn = _pg.clone(pg, label, template=template)
        made.append(conn)
        return conn

    yield make
    for conn in made:
        _pg.drop(conn)


@pytest.fixture
def clone_db(deployed_db: _pg.PgConn) -> Iterator[Callable[[str], _pg.PgConn]]:
    """Behaviour mode: independent copies of the post-deploy schema; only these are dropped."""
    yield from _clones(deployed_db.pg, deployed_db.db)


@pytest.fixture
def base_clone_db(base_db: BaseDb) -> Iterator[Callable[[str], _pg.PgConn]]:
    """Upgrade-path mode: copies of prod's current schema, BEFORE the pending migrations."""
    yield from _clones(base_db.pg, base_db.db)


@pytest.fixture(scope="module")
def module_db(request: pytest.FixtureRequest, deployed_db: _pg.PgConn) -> Iterator[_pg.PgConn]:
    """Behaviour mode: one copy of the post-deploy schema per module."""
    conn = _pg.clone(deployed_db.pg, request.module.__name__.rsplit(".", 1)[-1], deployed_db.db)
    try:
        yield conn
    finally:
        _pg.drop(conn)


__all__: Tuple[str, ...] = ("BaseDb",)
