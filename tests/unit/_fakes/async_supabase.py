"""An in-memory fake of the ASYNC supabase client (``await client.table(t)...execute()``).

Drives the real repository facades (BaseRepository / MLExperimentRepository /
MLModelRegistryRepository / RetrainingHistoryRepository) so a test asserts on the rows
that LANDED, not on mock call shapes. Supports the builder chain those repositories use:
select / eq / neq / in_ / is_ / not_ / like / order / limit / range / insert / update /
upsert / delete. Unknown columns in an ``eq`` filter do not match (like PostgREST);
``order`` sorts like Postgres (chained keys, NULLS LAST ascending / FIRST descending).
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


class _AsyncQuery:
    def __init__(self, store: Dict[str, List[Dict[str, Any]]], table: str):
        self._store = store
        self._table = table
        self._op = "select"
        self._payload: Any = None
        self._filters: List[tuple] = []
        self._limit: Optional[int] = None
        self._offset = 0
        self._order: List[tuple] = []
        self._negate_next = False

    # builders -------------------------------------------------------------
    def select(self, *_a, **_k):
        self._op = "select"
        return self

    def insert(self, data):
        self._op = "insert"
        self._payload = data
        return self

    def upsert(self, data, **_k):
        self._op = "upsert"
        self._payload = data
        return self

    def update(self, data):
        self._op = "update"
        self._payload = data
        return self

    def delete(self):
        self._op = "delete"
        return self

    @property
    def not_(self):
        self._negate_next = True
        return self

    def _add(self, kind, col, val):
        self._filters.append((kind, col, val, self._negate_next))
        self._negate_next = False
        return self

    def eq(self, col, val):
        return self._add("eq", col, val)

    def neq(self, col, val):
        return self._add("neq", col, val)

    def in_(self, col, vals):
        return self._add("in", col, list(vals))

    def is_(self, col, val):
        return self._add("is", col, val)

    def like(self, col, pattern):
        return self._add("like", col, pattern)

    def order(self, column, *, desc=False, **_k):
        # PostgREST semantics: chained calls append sort keys; ASC puts NULLs last,
        # DESC puts them first (Postgres defaults).
        self._order.append((column, desc))
        return self

    def limit(self, n, *_a, **_k):
        self._limit = n
        return self

    def offset(self, n, *_a, **_k):
        self._offset = int(n)
        return self

    def range(self, start, end, *_a, **_k):
        self._limit = end - start + 1
        return self

    # execution ------------------------------------------------------------
    def _match(self, row: Dict[str, Any]) -> bool:
        for kind, col, val, negate in self._filters:
            present = col in row
            if kind == "eq":
                ok = present and str(row.get(col)).lower() == str(val).lower()
            elif kind == "neq":
                ok = present and str(row.get(col)).lower() != str(val).lower()
            elif kind == "in":
                ok = present and str(row.get(col)) in {str(v) for v in val}
            elif kind == "is":
                ok = (row.get(col) is None) if val in ("null", None) else (row.get(col) == val)
            elif kind == "like":
                import fnmatch

                ok = present and fnmatch.fnmatch(str(row.get(col)), str(val).replace("%", "*"))
            else:
                ok = True
            if negate:
                ok = not ok
            if not ok:
                return False
        return True

    async def execute(self):
        rows = self._store.setdefault(self._table, [])
        if self._op in ("insert", "upsert"):
            payload = self._payload if isinstance(self._payload, list) else [self._payload]
            landed = []
            for p in payload:
                row = dict(p)
                if not row.get("id"):
                    row["id"] = str(uuid.uuid4())
                # schema default (migration 069): provenance-tagged tables default false
                row.setdefault("is_synthetic", False)
                rows.append(row)
                landed.append(dict(row))
            return SimpleNamespace(data=landed, count=len(landed))
        if self._op == "update":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                r.update(self._payload)
            return SimpleNamespace(data=[dict(r) for r in hit], count=len(hit))
        if self._op == "delete":
            hit = [r for r in rows if self._match(r)]
            for r in hit:
                rows.remove(r)
            return SimpleNamespace(data=[dict(r) for r in hit], count=len(hit))
        out = [dict(r) for r in rows if self._match(r)]
        for column, desc in reversed(self._order):  # stable sort, last key first
            present = [r for r in out if r.get(column) is not None]
            nulls = [r for r in out if r.get(column) is None]
            present.sort(
                key=lambda r: r[column] if isinstance(r[column], (int, float)) else str(r[column]),
                reverse=desc,
            )
            out = nulls + present if desc else present + nulls
        out = out[self._offset :]
        if self._limit is not None:
            out = out[: self._limit]
        return SimpleNamespace(data=out, count=len(out))


class FakeAsyncSupabase:
    """``store[table] -> list[row]``; every ``table()`` call starts a fresh chain."""

    def __init__(self, store: Optional[Dict[str, List[Dict[str, Any]]]] = None):
        self.store: Dict[str, List[Dict[str, Any]]] = store or {}
        self.tables_touched: List[str] = []

    def table(self, name: str) -> _AsyncQuery:
        self.tables_touched.append(name)
        return _AsyncQuery(self.store, name)

    def rows(self, table: str) -> List[Dict[str, Any]]:
        return self.store.get(table, [])
