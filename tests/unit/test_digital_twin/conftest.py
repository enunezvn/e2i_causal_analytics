"""Shared fakes for digital-twin unit tests (#549).

``FakeSupabaseClient`` is a faithful in-memory stand-in for the supabase-py
async fluent client — exactly the chain ``BaseRepository`` /
``TwinModelRepository`` / ``TwinRetrainingJobRepository`` use:

    await client.table(name).insert(data).execute()
    await client.table(name).select("*").eq("id", id).execute()
    await client.table(name).update(data).eq("id", id).execute()

``.execute()`` is async (the repos await it) and returns an object with a
``.data`` list of rows. A single instance is a shared store (one rows list per
table name), so injecting ONE client into TWO repository/service instances
reproduces the Celery worker boundary — separate instances, SAME database.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest


class _Result:
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data


class _FakeQuery:
    def __init__(self, table: "_FakeTable"):
        self._table = table
        self._op: str | None = None
        self._payload: Dict[str, Any] | None = None
        self._filters: Dict[str, Any] = {}

    def select(self, *_args: Any) -> "_FakeQuery":
        self._op = "select"
        return self

    def insert(self, data: Dict[str, Any]) -> "_FakeQuery":
        self._op = "insert"
        self._payload = data
        return self

    def update(self, data: Dict[str, Any]) -> "_FakeQuery":
        self._op = "update"
        self._payload = data
        return self

    def delete(self) -> "_FakeQuery":
        self._op = "delete"
        return self

    def eq(self, column: str, value: Any) -> "_FakeQuery":
        self._filters[column] = value
        return self

    def limit(self, _n: int) -> "_FakeQuery":
        return self

    def offset(self, _n: int) -> "_FakeQuery":
        return self

    def order(self, *_a: Any, **_k: Any) -> "_FakeQuery":
        return self

    def _matches(self, row: Dict[str, Any]) -> bool:
        return all(row.get(k) == v for k, v in self._filters.items())

    async def execute(self) -> _Result:
        if self._op == "insert":
            assert self._payload is not None
            self._table.rows.append(dict(self._payload))
            return _Result([dict(self._payload)])
        if self._op == "select":
            return _Result([dict(r) for r in self._table.rows if self._matches(r)])
        if self._op == "update":
            assert self._payload is not None
            changed = []
            for row in self._table.rows:
                if self._matches(row):
                    row.update(self._payload)
                    changed.append(dict(row))
            return _Result(changed)
        if self._op == "delete":
            keep = [r for r in self._table.rows if not self._matches(r)]
            removed = len(self._table.rows) - len(keep)
            self._table.rows[:] = keep
            return _Result([{}] * removed)
        return _Result([])


class _FakeTable:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []


class FakeSupabaseClient:
    """A process-shared store: one rows list per table name, like a real DB."""

    def __init__(self) -> None:
        self._tables: Dict[str, _FakeTable] = {}

    def table(self, name: str) -> _FakeQuery:
        tbl = self._tables.setdefault(name, _FakeTable())
        return _FakeQuery(tbl)


@pytest.fixture
def fake_supabase() -> FakeSupabaseClient:
    """A fresh shared in-memory Supabase store for one test."""
    return FakeSupabaseClient()
