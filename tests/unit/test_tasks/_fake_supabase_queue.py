"""Test-scoped fake of the async supabase client for the #1515 drainer tests.

Models ONLY the ``chatbot_optimization_requests`` queue surface the drainer
touches: the three 035 lifecycle RPCs it calls and the small set of PostgREST
table operations (conditional UPDATE for the compare-and-set claim / zombie
recovery, SELECT for the idle check and the table-backed
``get_pending_requests``).

This is an EXPLICIT substitution at the DB boundary, used by unit tests so the
drain-cycle logic can be exercised without a database. Fidelity of the real 035
SQL functions (claim semantics, status transitions, improvement computation) is
covered separately by tests/integration/test_chatbot_optimization_queue_db.py,
which runs the REAL functions against the live local Supabase under
``E2I_DB_INTEGRATION=1``.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FakeResult:
    """Mimics the postgrest APIResponse: only ``.data`` is consumed."""

    def __init__(self, data: Any) -> None:
        self.data = data


class _FakeRpcCall:
    def __init__(self, db: "FakeQueueDB", fn: str, params: Dict[str, Any]) -> None:
        self._db = db
        self._fn = fn
        self._params = params

    async def execute(self) -> FakeResult:
        return self._db._run_rpc(self._fn, self._params)


class _FakeTableQuery:
    """Chainable subset of the postgrest query builder used by the drainer."""

    def __init__(self, db: "FakeQueueDB", table: str) -> None:
        self._db = db
        self._table = table
        self._mode: Optional[str] = None  # "select" | "update" | "delete"
        self._payload: Dict[str, Any] = {}
        self._filters: List[Tuple[str, str, Any]] = []  # (op, column, value)
        self._order: List[Tuple[str, bool]] = []  # (column, desc)
        self._limit: Optional[int] = None

    # -- verbs ---------------------------------------------------------------
    def select(self, _cols: str = "*") -> "_FakeTableQuery":
        self._mode = "select"
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeTableQuery":
        self._mode = "update"
        self._payload = dict(payload)
        return self

    def delete(self) -> "_FakeTableQuery":
        self._mode = "delete"
        return self

    # -- filters -------------------------------------------------------------
    def eq(self, column: str, value: Any) -> "_FakeTableQuery":
        self._filters.append(("eq", column, value))
        return self

    def lt(self, column: str, value: Any) -> "_FakeTableQuery":
        self._filters.append(("lt", column, value))
        return self

    def in_(self, column: str, values: List[Any]) -> "_FakeTableQuery":
        self._filters.append(("in", column, list(values)))
        return self

    def like(self, column: str, pattern: str) -> "_FakeTableQuery":
        self._filters.append(("like", column, pattern))
        return self

    # -- modifiers -----------------------------------------------------------
    def order(self, column: str, desc: bool = False) -> "_FakeTableQuery":
        self._order.append((column, desc))
        return self

    def limit(self, n: int) -> "_FakeTableQuery":
        self._limit = n
        return self

    # -- execution -----------------------------------------------------------
    def _matches(self, row: Dict[str, Any]) -> bool:
        for op, column, value in self._filters:
            actual = row.get(column)
            if op == "eq":
                if actual != value:
                    return False
            elif op == "lt":
                # ISO-8601 strings in one tz-offset format compare correctly
                # as strings; None never matches (SQL NULL semantics).
                if actual is None or not (actual < value):
                    return False
            elif op == "in":
                if actual not in value:
                    return False
            elif op == "like":
                prefix = value.rstrip("%")
                if not isinstance(actual, str) or not actual.startswith(prefix):
                    return False
        return True

    async def execute(self) -> FakeResult:
        db = self._db
        db.table_ops.append(
            {
                "table": self._table,
                "mode": self._mode,
                "payload": dict(self._payload),
                "filters": list(self._filters),
                "order": list(self._order),
                "limit": self._limit,
            }
        )
        rows = [r for r in db.rows if self._matches(r)]
        if self._mode == "update":
            for row in rows:
                row.update(self._payload)
                row["updated_at"] = _now_iso()  # set_updated_at trigger
            return FakeResult([copy.deepcopy(r) for r in rows])
        if self._mode == "delete":
            db.rows = [r for r in db.rows if r not in rows]
            return FakeResult([copy.deepcopy(r) for r in rows])
        # select
        for column, desc in reversed(self._order):
            rows.sort(key=lambda r: r.get(column), reverse=desc)
        if self._limit is not None:
            rows = rows[: self._limit]
        return FakeResult([copy.deepcopy(r) for r in rows])


class FakeQueueDB:
    """Stateful in-memory stand-in for the 035 queue table + its RPCs.

    ``race_first_peek=True`` simulates a competing claimer landing between the
    drainer's peek and its claim: the FIRST ``get_next_optimization_request``
    call returns a snapshot of the top pending row while flipping the stored
    row to 'processing', so the drainer's compare-and-set claim must lose.
    """

    def __init__(
        self,
        rows: Optional[List[Dict[str, Any]]] = None,
        race_first_peek: bool = False,
    ) -> None:
        self.rows: List[Dict[str, Any]] = [dict(r) for r in (rows or [])]
        self.rpc_calls: List[Tuple[str, Dict[str, Any]]] = []
        self.table_ops: List[Dict[str, Any]] = []
        self._race_first_peek = race_first_peek
        self._peeked = False

    # -- client surface ------------------------------------------------------
    def rpc(self, fn: str, params: Optional[Dict[str, Any]] = None) -> _FakeRpcCall:
        return _FakeRpcCall(self, fn, dict(params or {}))

    def table(self, name: str) -> _FakeTableQuery:
        return _FakeTableQuery(self, name)

    # -- helpers -------------------------------------------------------------
    def row(self, request_id: str) -> Optional[Dict[str, Any]]:
        for r in self.rows:
            if r.get("request_id") == request_id:
                return r
        return None

    def _pending_sorted(self, module_name: Optional[str]) -> List[Dict[str, Any]]:
        pending = [
            r
            for r in self.rows
            if r.get("status") == "pending"
            and (module_name is None or r.get("module_name") == module_name)
        ]
        pending.sort(key=lambda r: (-int(r.get("priority", 1)), r.get("created_at") or ""))
        return pending

    # -- RPC semantics (mirrors database/chat/035 functions) -----------------
    def _run_rpc(self, fn: str, params: Dict[str, Any]) -> FakeResult:
        self.rpc_calls.append((fn, dict(params)))

        if fn == "get_next_optimization_request":
            pending = self._pending_sorted(params.get("p_module_name"))
            if not pending:
                return FakeResult([])
            top = pending[0]
            snapshot = {
                k: copy.deepcopy(top.get(k))
                for k in (
                    "id",
                    "request_id",
                    "module_name",
                    "signal_count",
                    "min_reward",
                    "budget",
                    "priority",
                    "created_at",
                )
            }
            if self._race_first_peek and not self._peeked:
                self._peeked = True
                top["status"] = "processing"  # the competing claimer won
                top["started_at"] = _now_iso()
            return FakeResult([snapshot])

        if fn == "update_optimization_request_status":
            row = self.row(params["p_request_id"])
            if row is None:
                return FakeResult(False)
            status = params["p_status"]
            row["status"] = status
            if status == "processing" and not row.get("started_at"):
                row["started_at"] = _now_iso()
            if status in ("completed", "failed", "cancelled"):
                row["completed_at"] = _now_iso()
            for src_key, col in (
                ("p_baseline_score", "baseline_score"),
                ("p_optimized_score", "optimized_score"),
                ("p_error_message", "error_message"),
                ("p_optimization_run_id", "optimization_run_id"),
            ):
                if params.get(src_key) is not None:
                    row[col] = params[src_key]
            baseline = row.get("baseline_score")
            optimized = row.get("optimized_score")
            if baseline and optimized is not None and baseline > 0:
                row["improvement_percent"] = ((optimized - baseline) / baseline) * 100
            row["updated_at"] = _now_iso()
            return FakeResult(True)

        if fn == "cancel_stale_optimization_requests":
            hours = int(params.get("p_max_age_hours", 24))
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
            count = 0
            for row in self.rows:
                created = row.get("created_at") or ""
                if row.get("status") == "pending" and created < cutoff:
                    row["status"] = "cancelled"
                    row["error_message"] = "Cancelled: request exceeded maximum pending age"
                    row["completed_at"] = _now_iso()
                    count += 1
            return FakeResult(count)

        raise AssertionError(f"FakeQueueDB: unexpected rpc {fn!r}")


def make_request_row(
    request_id: str,
    module_name: str = "intent_classifier",
    *,
    status: str = "pending",
    priority: int = 1,
    budget: str = "light",
    min_reward: float = 0.5,
    signal_count: int = 60,
    created_at: Optional[str] = None,
    started_at: Optional[str] = None,
) -> Dict[str, Any]:
    """A row shaped like the 035 table."""
    return {
        "id": abs(hash(request_id)) % 100000,
        "request_id": request_id,
        "module_name": module_name,
        "signal_count": signal_count,
        "min_reward": min_reward,
        "budget": budget,
        "priority": priority,
        "status": status,
        "started_at": started_at,
        "completed_at": None,
        "baseline_score": None,
        "optimized_score": None,
        "improvement_percent": None,
        "error_message": None,
        "optimization_run_id": None,
        "created_at": created_at or _now_iso(),
        "updated_at": _now_iso(),
        "metadata": {},
    }
