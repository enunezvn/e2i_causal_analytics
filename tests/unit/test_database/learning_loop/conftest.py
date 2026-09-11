"""Fixtures for the learning-loop real-DB tests (opt-in: ``E2I_DB_INTEGRATION=1``).

``pg_container`` → ``base_db`` (session) is the prod-faithful copy. Tests never write to it:
they use ``clone_db`` (function scope) or ``module_db`` (one clone per module, with the lane's
migrations applied through ``upto``), and ``PgConn.rolled_back()`` for per-test isolation.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterator, List, Optional, Tuple

import pytest

from tests.unit.test_database.learning_loop import _pg


def _require_opt_in() -> None:
    if not _pg.db_integration_enabled():
        pytest.skip("real-DB integration; set E2I_DB_INTEGRATION=1 on the droplet")


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
    def __init__(self, pg: _pg.ThrowawayPg, db: str, restore_log: _pg.RestoreLog):
        super().__init__(pg, db)
        self.restore_log = restore_log


@pytest.fixture(scope="session")
def base_db(pg_container: _pg.ThrowawayPg, prod_readonly: _pg.ProdReadOnly) -> BaseDb:
    already = _pg.lane_migrations_already_in_prod(prod_readonly)
    if already:
        # The copy is "prod before this lane"; once the lane is deployed that base is gone.
        pytest.skip(
            f"prod already carries the lane's migrations {already}; the upgrade fixture no longer applies"
        )
    log = _pg.build_base(pg_container, prod_readonly)
    return BaseDb(pg_container, _pg.BASE_DB, log)


@pytest.fixture
def clone_db(base_db: BaseDb) -> Iterator[Callable[[str], _pg.PgConn]]:
    """Independent copies of the base database; only the copies made here are dropped."""
    made: List[_pg.PgConn] = []

    def make(label: str) -> _pg.PgConn:
        conn = _pg.clone(base_db.pg, label)
        made.append(conn)
        return conn

    yield make
    for conn in made:
        _pg.drop(conn)


@pytest.fixture(scope="module")
def module_db(
    request: pytest.FixtureRequest, base_db: BaseDb
) -> Iterator[Callable[[Optional[str]], _pg.PgConn]]:
    """One clone per (module, upto), with the lane's migrations applied through ``upto``."""
    made: Dict[Optional[str], _pg.PgConn] = {}

    def get(upto: Optional[str]) -> _pg.PgConn:
        if upto not in made:
            conn = _pg.clone(base_db.pg, f"{request.module.__name__.rsplit('.', 1)[-1]}")
            _pg.migrate(conn, upto)
            made[upto] = conn
        return made[upto]

    yield get
    for conn in made.values():
        _pg.drop(conn)


__all__: Tuple[str, ...] = ("BaseDb",)
