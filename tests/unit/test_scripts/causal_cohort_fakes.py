"""Recording PostgREST fakes shared by the causal-cohort loader tests.

Nothing happens until ``.execute()`` is called, and ``.execute().count`` is only
a real integer when the select asked for ``count="exact"`` -- otherwise it is
``None``, exactly like a real client, so ``int(...)`` on a dropped
``count="exact"`` fails loudly instead of silently returning a filtered row
count. The Optum test module carries its own byte-identical copy (pinned there
before this helper existed); this module serves the CSU sibling.
"""

from __future__ import annotations


class FakeQuery:
    def __init__(self, table: "FakeTable", op: str, batch=None, on_conflict=None):
        self._t, self._op, self._filters, self._count = table, op, [], None
        self._batch, self._on_conflict = batch, on_conflict
        self._range = None
        self._order = None

    def select(self, cols, count=None):
        self._count = count
        return self

    def order(self, col, desc=False):
        self._order = (col, desc)
        return self

    def eq(self, col, val):
        self._filters.append((col, val))
        return self

    def range(self, start, end):
        self._range = (start, end)
        return self

    def execute(self):
        if self._op == "upsert":
            for rec in self._batch:
                self._t.rows[rec["patient_id"]] = rec
            self._t.upserts.append((list(self._batch), self._on_conflict))
            return type("R", (), {"data": list(self._batch), "count": None})()
        rows = [r for r in self._t.rows.values() if all(r.get(c) == v for c, v in self._filters)]
        total = len(rows)
        if self._order is not None:
            col, desc = self._order
            rows = sorted(rows, key=lambda r: str(r.get(col)), reverse=desc)
        if self._range is not None:
            start, end = self._range
            rows = rows[start : end + 1]
        count = total if self._count == "exact" else None
        return type("R", (), {"data": rows, "count": count})()


class FakeTable:
    def __init__(self, missing: bool = False, name: str = "table"):
        self.rows: dict = {}
        self.upserts: list = []
        self.missing = missing
        self.name = name

    def select(self, cols, count=None):
        if self.missing:
            raise RuntimeError(f'relation "{self.name}" does not exist')
        return FakeQuery(self, "select").select(cols, count=count)

    def upsert(self, batch, on_conflict=None):
        # Building the query must not write anything -- only .execute() on the
        # returned object may (mutation-proof for a dropped .execute() call).
        return FakeQuery(self, "upsert", batch=batch, on_conflict=on_conflict)


class FakeClient:
    def __init__(self, missing: bool = False):
        self.t = FakeTable(missing=missing)
        self.tables: list = []

    def table(self, name):
        self.tables.append(name)
        self.t.name = name
        return self.t
