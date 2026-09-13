"""#1991 debt 3 -- the review is keyed on the ESTIMAND, its structures are a timeline.

Migration 140 makes ``expert_reviews.estimand_key`` a STORED GENERATED column
(``lower(brand):treatment:outcome``) and moves pending uniqueness onto it, so:

* writers must NEVER send ``estimand_key`` (PostgREST rejects an insert that
  supplies a generated column) -- Python only ever computes it for a LOOKUP,
  and that computation must mirror the SQL expression operand for operand or
  the recovery lookup reads a different row than the index keyed;
* the 23505 recovery lookup keys on the estimand, not on (hash, brand).

Migration 141 makes ``expert_review_versions`` a TIMELINE (no UNIQUE on
(review_id, hash)): a revert A -> B -> A appends a third row, and same-hash
idempotence is the caller's job, not a constraint's. ``service_role`` holds
SELECT+INSERT only there, so the writer must use ``insert()`` -- ``upsert()``
would need UPDATE and fail 42501.

The double below is a small PostgREST-shaped recorder: it answers what the repo
actually SENT (rows inserted, payloads updated, filters and ordering applied),
which is the thing these invariants are about.
"""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from src.repositories.expert_review import ExpertReviewRepository, estimand_key_for

REPO = Path(__file__).resolve().parents[3]


class _FakeQuery:
    """One PostgREST call chain; records every step, resolves on execute()."""

    def __init__(self, client: "_FakeClient", table: str) -> None:
        self._client = client
        self._table = table
        self._op: Optional[str] = None
        self._payload: Any = None
        self._filters: List[Tuple[str, str, Any]] = []
        self._orders: List[Tuple[str, bool]] = []
        self._limit: Optional[int] = None

    def _rec(self, method: str, args: Tuple[Any, ...]) -> "_FakeQuery":
        self._client._calls.setdefault(self._table, []).append((method, args))
        return self

    def select(self, *cols: str) -> "_FakeQuery":
        self._op = "select"
        return self._rec("select", tuple(cols))

    def insert(self, row: Dict[str, Any]) -> "_FakeQuery":
        self._op, self._payload = "insert", row
        return self._rec("insert", (row,))

    def upsert(self, row: Dict[str, Any], **kwargs: Any) -> "_FakeQuery":
        # Supported ONLY so a writer that wrongly used it is caught by the
        # assertion rather than by an AttributeError from the double.
        self._op, self._payload = "upsert", row
        return self._rec("upsert", (row,))

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._op, self._payload = "update", payload
        return self._rec("update", (payload,))

    def eq(self, key: str, value: Any) -> "_FakeQuery":
        self._filters.append(("eq", key, value))
        return self._rec("eq", (key, value))

    def is_(self, key: str, value: Any) -> "_FakeQuery":
        self._filters.append(("is", key, value))
        return self._rec("is_", (key, value))

    def or_(self, expr: str) -> "_FakeQuery":
        self._filters.append(("or", expr, None))
        return self._rec("or_", (expr,))

    def order(self, column: str, desc: bool = False) -> "_FakeQuery":
        self._orders.append((column, desc))
        return self._rec("order", (column, desc))

    def limit(self, n: int) -> "_FakeQuery":
        self._limit = n
        return self._rec("limit", (n,))

    def _matches(self, row: Dict[str, Any]) -> bool:
        for kind, key, value in self._filters:
            if kind == "or":
                raise NotImplementedError(
                    "this double does not evaluate PostgREST or= filters; the faithful "
                    "one lives in test_approval_expiry_readers_1972.py"
                )
            if kind == "eq" and row.get(key) != value:
                return False
            if kind == "is" and row.get(key) is not None:
                return False
        return True

    async def execute(self) -> SimpleNamespace:
        if self._op in ("insert", "upsert"):
            pending = self._client._fail_insert.pop(self._table, None)
            if pending is not None:
                raise pending
            self._client._inserted.setdefault(self._table, []).append(self._payload)
            if self._table == "expert_reviews":
                return SimpleNamespace(data=[{**self._payload, "review_id": "new-id"}])
            return SimpleNamespace(data=[self._payload])
        if self._op == "update":
            # What was SENT (recorded even when it matches nothing) and what it
            # CHANGED are different questions: an UPDATE answers with the rows it
            # actually touched, so a filter that matches no seeded row returns no
            # data -- which is exactly how a writer learns it changed nothing.
            self._client._updated.setdefault(self._table, []).append(self._payload)
            matched = [r for r in self._client._rows.get(self._table, []) if self._matches(r)]
            for row in matched:
                row.update(self._payload)
            return SimpleNamespace(data=[dict(r) for r in matched])
        rows = [dict(r) for r in self._client._rows.get(self._table, []) if self._matches(r)]
        for column, desc in reversed(self._orders):  # last key first => stable
            rows.sort(key=lambda r: (r.get(column) is None, r.get(column) or ""), reverse=desc)
        return SimpleNamespace(data=rows[: self._limit] if self._limit else rows)


class _FakeClient:
    def __init__(self) -> None:
        self._rows: Dict[str, List[Dict[str, Any]]] = {}
        self._inserted: Dict[str, List[Any]] = {}
        self._updated: Dict[str, List[Any]] = {}
        self._calls: Dict[str, List[Tuple[str, Tuple[Any, ...]]]] = {}
        self._fail_insert: Dict[str, BaseException] = {}

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)

    def seed(self, table: str, rows: List[Dict[str, Any]]) -> None:
        self._rows[table] = [dict(r) for r in rows]

    def fail_next_insert(self, table: str, exc: BaseException) -> None:
        self._fail_insert[table] = exc

    def inserted(self, table: str) -> List[Any]:
        return self._inserted.get(table, [])

    def updated(self, table: str) -> List[Any]:
        return self._updated.get(table, [])

    def calls(self, table: str) -> List[Tuple[str, Tuple[Any, ...]]]:
        return self._calls.get(table, [])

    def rows(self, table: str) -> List[Dict[str, Any]]:
        """The seeded rows AS THEY STAND -- an applied update is visible here."""
        return self._rows.get(table, [])


@pytest.fixture
def fake_client() -> _FakeClient:
    return _FakeClient()


# --------------------------------------------------------------------------
# The key itself
# --------------------------------------------------------------------------


@pytest.mark.unit
def test_estimand_key_is_lowercase_and_null_safe():
    assert (
        estimand_key_for("Remibrutinib", "treatment_arm", "persistent_180d")
        == "remibrutinib:treatment_arm:persistent_180d"
    )
    assert estimand_key_for(None, "T", "Y") == ":t:y"
    assert estimand_key_for("B", None, None) == "b::"


@pytest.mark.unit
def test_estimand_key_matches_migration_140_expression():
    """The Python key must equal the SQL generated column, operand for operand."""
    sql = (REPO / "database/migrations/140_expert_reviews_estimand_key.sql").read_text()
    i_b = sql.index("lower(COALESCE(brand, ''))")
    i_t = sql.index("lower(COALESCE(treatment_variable, ''))")
    i_o = sql.index("lower(COALESCE(outcome_variable, ''))")
    assert i_b < i_t < i_o and "GENERATED ALWAYS AS" in sql
    # ... and the ':' separator between them, twice -- a key joined on anything
    # else would look right operand for operand and still miss every row.
    expression = sql[i_b : i_o + len("lower(COALESCE(outcome_variable, ''))")]
    assert expression.count("|| ':' ||") == 2, expression
    assert estimand_key_for("Kisqali", "HCP_Engagement", "TRx") == "kisqali:hcp_engagement:trx"


# --------------------------------------------------------------------------
# create_review
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_create_review_never_sends_the_generated_estimand_key(fake_client):
    repo = ExpertReviewRepository(supabase_client=fake_client)
    rid = await repo.create_review(
        reviewer_id="q1",
        review_type="dag_approval",
        dag_version_hash="h1",
        brand="B",
        treatment_variable="T",
        outcome_variable="Y",
    )
    row = fake_client.inserted("expert_reviews")[0]
    assert "estimand_key" not in row and row["brand"] == "B" and rid == "new-id"


@pytest.mark.unit
async def test_create_review_recovers_pending_by_estimand_on_unique_violation(fake_client):
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r-existing", "estimand_key": "b:t:y", "approval_status": "pending"}],
    )
    fake_client.fail_next_insert(
        "expert_reviews",
        Exception('duplicate key value violates unique constraint "uq_er_pending_estimand"'),
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    rid = await repo.create_review(
        reviewer_id="q1",
        review_type="dag_approval",
        dag_version_hash="h9",
        brand="B",
        treatment_variable="T",
        outcome_variable="Y",
    )
    assert rid == "r-existing"
    # the lookup keyed on the estimand, not on the hash/brand
    assert ("eq", ("estimand_key", "b:t:y")) in fake_client.calls("expert_reviews")
    assert not any(
        method == "eq" and args[0] == "dag_version_hash"
        for method, args in fake_client.calls("expert_reviews")
    )


@pytest.mark.unit
async def test_create_review_carries_supersedes_review_id(fake_client):
    repo = ExpertReviewRepository(supabase_client=fake_client)
    await repo.create_review(
        reviewer_id="q",
        review_type="dag_approval",
        dag_version_hash="h2",
        brand="B",
        treatment_variable="T",
        outcome_variable="Y",
        supersedes_review_id="r0",
    )
    assert fake_client.inserted("expert_reviews")[0]["supersedes_review_id"] == "r0"


# --------------------------------------------------------------------------
# History on the estimand
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_reviews_for_estimand_newest_first(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {"review_id": "r1", "estimand_key": "b:t:y", "created_at": "2026-09-01T00:00:00+00:00"},
            {"review_id": "r2", "estimand_key": "b:t:y", "created_at": "2026-09-02T00:00:00+00:00"},
            {"review_id": "rx", "estimand_key": "b:t:z", "created_at": "2026-09-03T00:00:00+00:00"},
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    rows = await repo.get_reviews_for_estimand("b:t:y")
    assert [r["review_id"] for r in rows] == ["r2", "r1"]


# --------------------------------------------------------------------------
# The structure timeline
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_append_version_inserts_then_updates_current_hash_and_snapshot(fake_client):
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure={"nodes": ["a"], "edges": []},
        adjustment_set_hash="a2",
        query_id="q2",
    )
    assert ok is True
    v = fake_client.inserted("expert_review_versions")[0]
    assert v == {
        "review_id": "r1",
        "dag_version_hash": "h2",
        "dag_structure_json": {"nodes": ["a"], "edges": []},
        "adjustment_set_hash": "a2",
        "query_id": "q2",
    }
    upd = fake_client.updated("expert_reviews")[0]
    assert upd == {"dag_version_hash": "h2", "dag_structure_json": {"nodes": ["a"], "edges": []}}
    assert ("eq", ("approval_status", "pending")) in fake_client.calls("expert_reviews")
    # insert(), never upsert() -- service_role has no UPDATE on the versions table
    assert not any(method == "upsert" for method, _ in fake_client.calls("expert_review_versions"))


@pytest.mark.unit
async def test_append_version_returns_false_when_insert_fails(fake_client):
    fake_client.fail_next_insert("expert_review_versions", Exception("42501 permission denied"))
    repo = ExpertReviewRepository(supabase_client=fake_client)
    assert (
        await repo.append_version(
            "r1",
            dag_version_hash="h2",
            dag_structure=None,
            adjustment_set_hash=None,
            query_id=None,
        )
        is False
    )
    assert fake_client.updated("expert_reviews") == []  # no update when the append failed


@pytest.mark.unit
async def test_append_version_clears_the_snapshot_when_no_structure_is_given(fake_client):
    """A hash with no structure to show CLEARS the review's snapshot -- deliberate.

    ``dag_structure_json`` is what the review UI renders. Leaving the previous
    structure in place under a NEW ``dag_version_hash`` would render a DAG the
    review no longer covers: a plausible-wrong picture a reviewer cannot tell
    from the real one. This is the opposite of ``update_dag_structure``, which
    refuses an empty structure precisely because it is a BACKFILL of the hash
    already on the row.
    """
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "dag_structure_json": {"nodes": ["stale"], "edges": []},
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h3",
        dag_structure=None,
        adjustment_set_hash=None,
        query_id=None,
    )
    assert ok is True
    assert fake_client.updated("expert_reviews")[0]["dag_structure_json"] is None


@pytest.mark.unit
async def test_append_version_returns_false_when_review_is_not_pending(fake_client, caplog):
    """A RESOLVED review is never advanced: the version row is appended (the
    timeline is honest about the structure the run produced), but the review
    keeps the hash it was resolved on, and False says so."""
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "approved", "dag_version_hash": "h1"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.WARNING):
        ok = await repo.append_version(
            "r1",
            dag_version_hash="h2",
            dag_structure=None,
            adjustment_set_hash=None,
            query_id=None,
        )
    assert ok is False
    assert fake_client.inserted("expert_review_versions")[0]["dag_version_hash"] == "h2"
    # the resolved row is untouched -- the pending-only filter matched nothing
    assert fake_client.rows("expert_reviews")[0]["dag_version_hash"] == "h1"
    assert any("no PENDING review" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
async def test_get_versions_ordered_by_created_then_version_id(fake_client):
    fake_client.seed(
        "expert_review_versions",
        [
            {"version_id": "v2", "review_id": "r1", "created_at": "2026-09-01T00:00:00+00:00"},
            {"version_id": "v1", "review_id": "r1", "created_at": "2026-09-01T00:00:00+00:00"},
            {"version_id": "v3", "review_id": "r1", "created_at": "2026-09-02T00:00:00+00:00"},
            {"version_id": "vz", "review_id": "r9", "created_at": "2026-08-01T00:00:00+00:00"},
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    assert [v["version_id"] for v in await repo.get_versions("r1")] == ["v1", "v2", "v3"]


# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_summary_counts_superseded(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {"approval_status": "pending", "valid_until": None},
            {"approval_status": "superseded", "valid_until": None},
            {"approval_status": "superseded", "valid_until": None},
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    s = await repo.get_review_summary()
    assert s["pending"] == 1 and s["superseded"] == 2


@pytest.mark.unit
async def test_summary_zero_dict_includes_superseded():
    repo = ExpertReviewRepository(supabase_client=None)
    s = await repo.get_review_summary()
    assert s["superseded"] == 0
