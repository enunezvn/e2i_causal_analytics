"""Session fixtures for the learning-loop real-DB tests (opt-in: ``E2I_DB_INTEGRATION=1``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List

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
    pg = _pg.ThrowawayPg(image=prod_readonly.image())
    pg.start()
    try:
        yield pg
    finally:
        pg.stop()


@dataclass
class BaseDb(_pg.PgConn):
    restore_log: _pg.RestoreLog = None  # type: ignore[assignment]

    def __init__(self, pg: _pg.ThrowawayPg, db: str, restore_log: _pg.RestoreLog):
        super().__init__(pg, db)
        self.restore_log = restore_log


@pytest.fixture(scope="session")
def base_db(pg_container: _pg.ThrowawayPg, prod_readonly: _pg.ProdReadOnly) -> BaseDb:
    log = _pg.build_base(pg_container, prod_readonly)
    return BaseDb(pg_container, "learning_loop_base", log)


@pytest.fixture
def clone_db(base_db: BaseDb) -> Iterator[Callable[[str], _pg.PgConn]]:
    """Independent copies of the base database; each is dropped after the test."""
    made: List[_pg.PgConn] = []

    def make(name: str) -> _pg.PgConn:
        conn = _pg.clone(base_db.pg, name)
        made.append(conn)
        return conn

    yield make
    for conn in made:
        conn.pg.rows("template1", f"drop database if exists {conn.db} with (force)")
