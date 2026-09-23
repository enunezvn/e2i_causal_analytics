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

    def neq(self, key: str, value: Any) -> "_FakeQuery":
        self._filters.append(("neq", key, value))
        return self._rec("neq", (key, value))

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
            if kind == "neq" and row.get(key) == value:
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
        [
            {
                "review_id": "r-existing",
                "estimand_key": "b:t:y",
                "approval_status": "pending",
                "review_type": "dag_approval",
            }
        ],
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
    # ... and scoped to the RUNTIME queue (migration 153, #2244): a dag_approval
    # consult may only adopt a row the runtime index could have rejected it for.
    assert ("neq", ("review_type", "initial_dag")) in fake_client.calls("expert_reviews")


# --------------------------------------------------------------------------
# Two review queues (#2244, migration 153): the 23505 recovery never crosses them
# --------------------------------------------------------------------------


def _seed_pending(fake_client, review_id: str, review_type: str) -> None:
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": review_id,
                "estimand_key": "b:t:y",
                "approval_status": "pending",
                "review_type": review_type,
                "brand": "B",
                "treatment_variable": "T",
                "outcome_variable": "Y",
            }
        ],
    )


_UNIQUE = 'duplicate key value violates unique constraint "uq_er_pending_estimand_runtime"'


@pytest.mark.unit
def test_structural_author_review_type_is_the_loaders_type():
    """One constant names the structural-author queue on both sides of the split:
    the repository (recovery + the gate's queue filter) and the loader."""
    from src.data.kg.structural_prior_loader import REVIEW_TYPE
    from src.repositories.expert_review import STRUCTURAL_AUTHOR_REVIEW_TYPE

    assert STRUCTURAL_AUTHOR_REVIEW_TYPE == REVIEW_TYPE == "initial_dag"


@pytest.mark.unit
async def test_gate_consult_recovery_never_adopts_a_pending_structural_author_review(
    fake_client,
):
    """A pending Lane B ``initial_dag`` review on the estimand is NOT the gate's
    consult: a 23505 on the dag_approval insert (a concurrent runtime mint) must
    recover a runtime row or nothing -- never the authored review, whose
    snapshot the gate would then advance to its own structure."""
    _seed_pending(fake_client, "r-author", "initial_dag")
    fake_client.fail_next_insert("expert_reviews", Exception(_UNIQUE))
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.create_review(
        reviewer_id="q1",
        review_type="dag_approval",
        dag_version_hash="h9",
        brand="B",
        treatment_variable="T",
        outcome_variable="Y",
    )

    assert rid is None
    assert ("neq", ("review_type", "initial_dag")) in fake_client.calls("expert_reviews")


@pytest.mark.unit
async def test_structural_author_recovery_never_adopts_a_pending_gate_consult(fake_client):
    """The reverse order: Lane B's ``initial_dag`` insert loses a race in ITS queue
    and must not come back holding the gate's dag_approval row (its evidence
    write would then match zero rows, or worse, land on the consult)."""
    _seed_pending(fake_client, "r-gate", "dag_approval")
    fake_client.fail_next_insert(
        "expert_reviews",
        Exception(
            'duplicate key value violates unique constraint "uq_er_pending_estimand_structural"'
        ),
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.create_review(
        reviewer_id="structural_author",
        review_type="initial_dag",
        dag_version_hash="authored",
        brand="B",
        treatment_variable="T",
        outcome_variable="Y",
    )

    assert rid is None
    assert ("eq", ("review_type", "initial_dag")) in fake_client.calls("expert_reviews")
    assert ("neq", ("review_type", "initial_dag")) not in fake_client.calls("expert_reviews")


@pytest.mark.unit
async def test_structural_author_recovery_returns_its_own_queues_pending_row(fake_client):
    _seed_pending(fake_client, "r-author", "initial_dag")
    fake_client.fail_next_insert(
        "expert_reviews",
        Exception(
            'duplicate key value violates unique constraint "uq_er_pending_estimand_structural"'
        ),
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.create_review(
        reviewer_id="structural_author",
        review_type="initial_dag",
        dag_version_hash="authored-2",
        brand="B",
        treatment_variable="T",
        outcome_variable="Y",
    )

    assert rid == "r-author"


@pytest.mark.unit
async def test_renew_review_refuses_a_structural_author_original(fake_client):
    """A renewal is always a runtime-queue ``quarterly_audit`` row; renewing a Lane
    B ``initial_dag`` review would convert an authored review into a gate
    consult (codex r1 MED). Fail closed: nothing inserted, None returned."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r-author",
                "estimand_key": "b:t:y",
                "approval_status": "approved",
                "review_type": "initial_dag",
                "dag_version_hash": "authored",
                "brand": "B",
                "treatment_variable": "T",
                "outcome_variable": "Y",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.renew_review(original_review_id="r-author", reviewer_id="q2")

    assert rid is None
    assert fake_client.inserted("expert_reviews") == []


@pytest.mark.unit
async def test_review_summary_can_count_the_runtime_queue_alone(fake_client):
    """The operator summary counts both queues (the Expert Reviews page shows
    both); the gate's health read asks for the runtime queue only."""
    fake_client.seed(
        "expert_reviews",
        [
            {"review_id": "1", "approval_status": "pending", "review_type": "initial_dag"},
            {"review_id": "2", "approval_status": "pending", "review_type": "dag_approval"},
            {
                "review_id": "3",
                "approval_status": "approved",
                "review_type": "initial_dag",
                "valid_until": None,
            },
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    both = await repo.get_review_summary()
    runtime = await repo.get_review_summary(runtime_only=True)

    assert (both["pending"], both["approved"]) == (2, 1)
    assert (runtime["pending"], runtime["approved"]) == (1, 0)


@pytest.mark.unit
async def test_renewal_recovery_is_the_runtime_queue(fake_client):
    """A renewal (quarterly_audit, #2090) shares the gate consult's slot, so its
    recovery reads the runtime queue -- and never the structural one."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r-original",
                "estimand_key": "b:t:y",
                "approval_status": "approved",
                "review_type": "dag_approval",
                "dag_version_hash": "h1",
                "brand": "B",
                "treatment_variable": "T",
                "outcome_variable": "Y",
            },
            {
                "review_id": "r-author",
                "estimand_key": "b:t:y",
                "approval_status": "pending",
                "review_type": "initial_dag",
            },
        ],
    )
    fake_client.fail_next_insert("expert_reviews", Exception(_UNIQUE))
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.renew_review(original_review_id="r-original", reviewer_id="q2")

    assert rid is None
    assert ("neq", ("review_type", "initial_dag")) in fake_client.calls("expert_reviews")


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
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
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
    # The cached assessment graded the PREVIOUS structure, so the advance clears
    # it (see test_append_version_clears_the_agent_assessment).
    assert upd == {
        "dag_version_hash": "h2",
        # Migration 142: the review row carries the adjustment half of its own
        # current identity, so the guards keyed on the row see a WHOLE version.
        "adjustment_set_hash": "a2",
        "dag_structure_json": {"nodes": ["a"], "edges": []},
        "agent_assessment_json": None,
    }
    assert ("eq", ("approval_status", "pending")) in fake_client.calls("expert_reviews")
    # Not given -> the column is left alone (the key is absent from the payload),
    # never overwritten with NULL.
    assert "related_validation_ids" not in upd
    # insert(), never upsert() -- service_role has no UPDATE on the versions table
    assert not any(method == "upsert" for method, _ in fake_client.calls("expert_review_versions"))


@pytest.mark.unit
async def test_append_version_repoints_the_evidence_at_the_new_versions_run(fake_client):
    """The advanced review must cite the evidence of the run that advanced it.

    ``related_validation_ids`` is the column the review detail route renders
    evidence from (src/api/routes/expert_review.py). After an append the review
    carries a NEW ``dag_version_hash``; leaving the previous run's validation
    ids beside it shows a reviewer statistics computed on a structure the review
    no longer covers. The versions table has no such column -- the ids belong to
    the review's current state, so only the review UPDATE carries them.
    """
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "related_validation_ids": ["v-old"],
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure=None,
        adjustment_set_hash=None,
        query_id="q2",
        related_validation_ids=["v1"],
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )

    assert ok is True
    assert fake_client.updated("expert_reviews")[0] == {
        "dag_version_hash": "h2",
        "adjustment_set_hash": None,
        "dag_structure_json": None,
        "agent_assessment_json": None,
        "related_validation_ids": ["v1"],
    }
    # The versions row is unchanged -- no such column there.
    assert "related_validation_ids" not in fake_client.inserted("expert_review_versions")[0]


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
            expected_current_hash="h1",
            expected_current_adjustment_hash=None,
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
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
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
            expected_current_hash="h1",
            expected_current_adjustment_hash=None,
        )
    assert ok is False
    assert fake_client.inserted("expert_review_versions")[0]["dag_version_hash"] == "h2"
    # the resolved row is untouched -- the pending-only filter matched nothing
    assert fake_client.rows("expert_reviews")[0]["dag_version_hash"] == "h1"
    # The advance is now ``advance_review``'s, so the warning is its one: "no
    # longer pending on <pair>" covers the resolved row and the lost race alike,
    # which is the same fact from the UPDATE's point of view -- the filters that
    # ARE the precondition matched nothing.
    assert any("no longer pending on" in r.getMessage() for r in caplog.records)


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


# --------------------------------------------------------------------------
# H1 -- the resolution is bound to the version the reviewer saw
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_submit_review_binds_the_resolution_to_the_expected_hash(fake_client):
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.submit_review(
        review_id="r1",
        approval_status="approved",
        checklist={"confounders_complete": True},
        expected_dag_version_hash="h1",
        expected_adjustment_set_hash=None,
    )
    assert ok is True
    calls = fake_client.calls("expert_reviews")
    assert ("eq", ("dag_version_hash", "h1")) in calls
    assert ("eq", ("approval_status", "pending")) in calls
    assert fake_client.rows("expert_reviews")[0]["approval_status"] == "approved"


@pytest.mark.unit
async def test_submit_review_refuses_a_stale_hash_and_leaves_the_row_pending(fake_client, caplog):
    """The review advanced to h2; the form was opened on h1. The UPDATE matches
    zero rows, so the approval covers nothing -- never h2 by accident."""
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h2"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.WARNING):
        ok = await repo.submit_review(
            review_id="r1",
            approval_status="approved",
            checklist={},
            expected_dag_version_hash="h1",
            expected_adjustment_set_hash=None,
        )
    assert ok is False
    row = fake_client.rows("expert_reviews")[0]
    assert row["approval_status"] == "pending" and row["dag_version_hash"] == "h2"
    assert "valid_until" not in row and "resolved_at" not in row
    assert any("matched no rows" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------
# H2 -- the latest recorded version, and a compare-and-set advance
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_get_latest_version_reads_the_last_row_by_created_then_version_id(fake_client):
    """The mirror of ``get_versions``' order, read from the OTHER end in ONE query:
    the caller needs only the newest row, so the whole timeline must not be
    fetched to take its last element."""
    fake_client.seed(
        "expert_review_versions",
        [
            {"version_id": "v1", "review_id": "r1", "created_at": "2026-09-01T00:00:00+00:00"},
            {"version_id": "v3", "review_id": "r1", "created_at": "2026-09-02T00:00:00+00:00"},
            {"version_id": "v4", "review_id": "r1", "created_at": "2026-09-02T00:00:00+00:00"},
            {"version_id": "vz", "review_id": "r9", "created_at": "2099-01-01T00:00:00+00:00"},
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    latest = await repo.get_latest_version("r1")
    assert latest is not None and latest["version_id"] == "v4"
    calls = fake_client.calls("expert_review_versions")
    assert ("order", ("created_at", True)) in calls
    assert ("order", ("version_id", True)) in calls
    assert ("limit", (1,)) in calls
    # ... and the oldest-first reader is unchanged, so the two agree on the order
    assert [v["version_id"] for v in await repo.get_versions("r1")] == ["v1", "v3", "v4"]


@pytest.mark.unit
async def test_get_latest_version_is_none_when_the_review_has_no_timeline(fake_client):
    repo = ExpertReviewRepository(supabase_client=fake_client)
    assert await repo.get_latest_version("r1") is None
    assert await ExpertReviewRepository(supabase_client=None).get_latest_version("r1") is None


@pytest.mark.unit
async def test_get_latest_version_reraises_a_read_failure(fake_client, caplog):
    """R1/R3: 'no version recorded' is what the caller reads as 'nothing to
    compare, append', so an outage must never look like it."""

    class _Boom(_FakeClient):
        def table(self, name: str):
            raise RuntimeError("connection refused")

    repo = ExpertReviewRepository(supabase_client=_Boom())
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError):
        await repo.get_latest_version("r1")
    assert any("latest version" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
async def test_append_version_advance_is_a_compare_and_set_on_the_current_hash(fake_client):
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure=None,
        adjustment_set_hash=None,
        query_id=None,
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )
    assert ok is True
    assert ("eq", ("dag_version_hash", "h1")) in fake_client.calls("expert_reviews")
    assert fake_client.rows("expert_reviews")[0]["dag_version_hash"] == "h2"


@pytest.mark.unit
async def test_append_version_cas_mismatch_leaves_the_review_on_the_winners_hash(
    fake_client, caplog
):
    """Two concurrent appends: the other one advanced the review to h3 first, so
    this one must not drag it back to h2. The version row it already inserted
    stays as a timeline fact."""
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h3"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.WARNING):
        ok = await repo.append_version(
            "r1",
            dag_version_hash="h2",
            dag_structure=None,
            adjustment_set_hash=None,
            query_id=None,
            expected_current_hash="h1",
            expected_current_adjustment_hash=None,
        )
    assert ok is False
    assert fake_client.rows("expert_reviews")[0]["dag_version_hash"] == "h3"
    assert fake_client.inserted("expert_review_versions")[0]["dag_version_hash"] == "h2"
    assert any("concurrent" in r.getMessage().lower() for r in caplog.records)


@pytest.mark.unit
async def test_append_version_requires_the_whole_expected_pair(fake_client):
    """The compare-and-set is no longer opt-in (codex round 2).

    It was optional because a caller might not have read the current hash. That
    is no longer true of any caller: the gate reads the pending row before it
    decides anything, and the mint knows the pair ``create_review`` just stored.
    An optional CAS is a path where the advance carries no precondition at all --
    the exact shape of the strand round 2 found -- so both halves are required
    keyword-only and a caller that omits either fails loudly here rather than
    quietly overwriting a winner in production.
    """
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with pytest.raises(TypeError, match="expected_current_hash"):
        await repo.append_version(
            "r1",
            dag_version_hash="h2",
            dag_structure=None,
            adjustment_set_hash=None,
            query_id=None,
        )
    with pytest.raises(TypeError, match="expected_current_adjustment_hash"):
        await repo.append_version(
            "r1",
            dag_version_hash="h2",
            dag_structure=None,
            adjustment_set_hash=None,
            query_id=None,
            expected_current_hash="h1",
        )
    # Nothing was written on either refusal: the signature is checked before the
    # timeline insert, so a mis-called append cannot leave an orphan row.
    assert fake_client.inserted("expert_review_versions") == []
    assert fake_client.updated("expert_reviews") == []


# --------------------------------------------------------------------------
# H3 -- a new structure never carries the old structure's assessment
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_append_version_clears_the_agent_assessment(fake_client):
    """The cached assessment grades the DAG and the evidence of the version that
    was current when it was built. Advancing the review without clearing it
    leaves the review UI showing a grading of a structure the review no longer
    covers -- and the request-level cache would serve it beside the new one.

    The key must be PRESENT with a literal None (PostgREST writes SQL NULL);
    ``append_version`` builds its payload explicitly, so it is not dropped by the
    "remove None values" pattern ``submit_review`` uses.
    """
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "agent_assessment_json": {"items": [{"id": "q1", "verdict": "supports"}]},
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure={"nodes": ["a"], "edges": []},
        adjustment_set_hash="a2",
        query_id="q2",
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )
    assert ok is True
    upd = fake_client.updated("expert_reviews")[0]
    assert "agent_assessment_json" in upd and upd["agent_assessment_json"] is None
    assert fake_client.rows("expert_reviews")[0]["agent_assessment_json"] is None


@pytest.mark.unit
async def test_update_agent_assessment_can_be_bound_to_one_structure(fake_client):
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    assert (
        await repo.update_agent_assessment("r1", {"items": []}, for_dag_version_hash="h1") is True
    )
    assert ("eq", ("dag_version_hash", "h1")) in fake_client.calls("expert_reviews")
    assert fake_client.rows("expert_reviews")[0]["agent_assessment_json"] == {"items": []}


@pytest.mark.unit
async def test_update_agent_assessment_refuses_a_structure_that_moved(fake_client, caplog):
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h2"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.WARNING):
        ok = await repo.update_agent_assessment("r1", {"items": []}, for_dag_version_hash="h1")
    assert ok is False
    assert "agent_assessment_json" not in fake_client.rows("expert_reviews")[0]
    # The refusal must be OBSERVABLE, and say which version was asked for: this
    # is the path the route reports as persisted=False, so an operator reading
    # the logs has to be able to tell it from a nonexistent review.
    assert any(
        "matched no rows" in r.getMessage() and "h1" in r.getMessage() for r in caplog.records
    )


@pytest.mark.unit
async def test_update_agent_assessment_without_a_hash_is_unchanged(fake_client):
    """The filter is opt-in: a caller with no version in hand writes as before."""
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h2"}],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    assert await repo.update_agent_assessment("r1", {"items": []}) is True
    assert not any(
        m == "eq" and a[0] == "dag_version_hash" for m, a in fake_client.calls("expert_reviews")
    )


@pytest.mark.unit
async def test_append_version_advances_the_snapshot_when_only_the_adjustment_set_moved(
    fake_client,
):
    """H4 at the WRITE end: the advance is not conditional on the hash CHANGING.

    ``compute_dag_hash`` excludes adjustment sets, so a covariate-only change
    appends a version whose ``dag_version_hash`` equals the one the review
    already carries -- and the compare-and-set then compares that hash with
    itself. Everything about this call is a no-op on the hash column, which is
    exactly why the OTHER columns have to move: the review's snapshot must show
    the new covariates, and the cached assessment (which graded the old ones)
    must go, or the reviewer reads the covariate change nowhere.
    """
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "dag_structure_json": {"nodes": ["T", "Y"], "adjustment_sets": [["W"]]},
                "agent_assessment_json": {"items": [{"id": "q1", "verdict": "supports"}]},
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    new_structure = {"nodes": ["T", "Y"], "adjustment_sets": [["W", "X"]]}
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h1",  # UNCHANGED -- and the CAS expects the same value
        dag_structure=new_structure,
        adjustment_set_hash="adj-WX",
        query_id="q2",
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )

    assert ok is True
    row = fake_client.rows("expert_reviews")[0]
    assert row["dag_structure_json"] == new_structure
    assert row["agent_assessment_json"] is None
    version = fake_client.inserted("expert_review_versions")[0]
    assert version["dag_version_hash"] == "h1"
    assert version["adjustment_set_hash"] == "adj-WX"
    assert version["dag_structure_json"] == new_structure


# --------------------------------------------------------------------------
# Codex round 2 -- the review ROW carries its FULL version identity
#
# ``compute_dag_hash`` excludes adjustment sets, so every guard keyed on the row
# alone matched half an identity and an ADJUSTMENT-ONLY advance walked past all
# three of them. Migration 142 adds ``expert_reviews.adjustment_set_hash``; these
# pin that each guard now binds to the PAIR, with NULL matched EXPLICITLY (an
# IS NULL filter) rather than ignored -- an omitted filter would match every row
# and silently restore the defect these tests exist to prevent.
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_submit_review_refuses_an_adjustment_only_advance(fake_client):
    """Codex round-2 finding 1: the form was opened on (h1, adj-W); a run advanced
    the review to (h1, adj-Z). The DAG hash is UNCHANGED, so the hash filter alone
    matched and the reviewer signed off covariates they were never shown."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-Z",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.submit_review(
        review_id="r1",
        approval_status="approved",
        checklist={},
        expected_dag_version_hash="h1",
        expected_adjustment_set_hash="adj-W",
    )
    assert ok is False
    row = fake_client.rows("expert_reviews")[0]
    assert row["approval_status"] == "pending" and row["adjustment_set_hash"] == "adj-Z"
    assert "valid_until" not in row and "resolved_at" not in row


@pytest.mark.unit
async def test_submit_review_matches_an_unknown_adjustment_set_with_is_null(fake_client):
    """A row whose adjustment set is UNKNOWN (NULL: pre-142, or minted by the old
    image) is resolvable by a form that echoes that same unknown -- and the filter
    that does it is ``is_``, not an omission. PostgREST renders ``adjustment_set_hash
    =eq.None`` as a literal string comparison, so eq can never express this."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": None,
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.submit_review(
        review_id="r1",
        approval_status="approved",
        checklist={},
        expected_dag_version_hash="h1",
        expected_adjustment_set_hash=None,
    )
    assert ok is True
    calls = fake_client.calls("expert_reviews")
    assert ("is_", ("adjustment_set_hash", "null")) in calls
    assert not any(m == "eq" and a[0] == "adjustment_set_hash" for m, a in calls)
    assert fake_client.rows("expert_reviews")[0]["approval_status"] == "approved"


@pytest.mark.unit
async def test_submit_review_refuses_unknown_against_a_row_that_learned_its_hash(fake_client):
    """The other direction of the same asymmetry: the form carried no adjustment
    hash, but the row has since learned one. That IS a version change, so the
    resolution must not land."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-W",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.submit_review(
        review_id="r1",
        approval_status="approved",
        checklist={},
        expected_dag_version_hash="h1",
        expected_adjustment_set_hash=None,
    )
    assert ok is False
    assert fake_client.rows("expert_reviews")[0]["approval_status"] == "pending"


@pytest.mark.unit
async def test_update_agent_assessment_refuses_an_adjustment_only_advance(fake_client, caplog):
    """Codex round-2 finding 2: the build graded (h1, adj-W); the advance to
    (h1, adj-Z) cleared the cache; the old build then re-filled it under the hash
    guard, because the hash never moved."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-Z",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.WARNING):
        ok = await repo.update_agent_assessment(
            "r1",
            {"items": []},
            for_dag_version_hash="h1",
            for_adjustment_set_hash="adj-W",
        )
    assert ok is False
    assert "agent_assessment_json" not in fake_client.rows("expert_reviews")[0]
    # The warning must name BOTH halves, or an operator cannot tell an
    # adjustment-only refusal from a hash one.
    assert any("adj-W" in r.getMessage() and "h1" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
async def test_update_agent_assessment_binds_to_an_unknown_adjustment_set(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": None,
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.update_agent_assessment(
        "r1", {"items": []}, for_dag_version_hash="h1", for_adjustment_set_hash=None
    )
    assert ok is True
    assert ("is_", ("adjustment_set_hash", "null")) in fake_client.calls("expert_reviews")
    assert fake_client.rows("expert_reviews")[0]["agent_assessment_json"] == {"items": []}


@pytest.mark.unit
async def test_update_agent_assessment_applies_no_version_filter_without_a_hash(fake_client):
    """A pre-141 row that never carried a hash has no version to bind to, so the
    guard is OMITTED entirely -- both halves, not just the one. Filtering the
    adjustment half alone would refuse every write instead of guarding one."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h2",
                "adjustment_set_hash": "adj-Z",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    assert await repo.update_agent_assessment("r1", {"items": []}) is True
    calls = fake_client.calls("expert_reviews")
    assert not any(m == "eq" and a[0] == "dag_version_hash" for m, a in calls)
    assert not any(a[0] == "adjustment_set_hash" for m, a in calls)


@pytest.mark.unit
async def test_advance_review_is_a_compare_and_set_on_the_pair(fake_client):
    """Codex round-2 finding 3, the write end: an advance that changes only the
    adjustment set must still be refused when another advance got there first."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-C",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h1",
        dag_structure={"nodes": ["T"], "adjustment_sets": [["B"]]},
        adjustment_set_hash="adj-B",
        expected_current_hash="h1",
        expected_current_adjustment_hash="adj-A",
    )
    assert ok is False
    row = fake_client.rows("expert_reviews")[0]
    assert row["adjustment_set_hash"] == "adj-C", "the winner's advance must stand"
    # Nothing was inserted: advance_review is the review UPDATE alone.
    assert fake_client.inserted("expert_review_versions") == []


@pytest.mark.unit
async def test_advance_review_writes_the_pair_and_clears_the_assessment(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-A",
                "agent_assessment_json": {"items": [{"id": "q1"}]},
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    structure = {"nodes": ["T", "Y"], "adjustment_sets": [["B"]]}
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h1",
        dag_structure=structure,
        adjustment_set_hash="adj-B",
        expected_current_hash="h1",
        expected_current_adjustment_hash="adj-A",
    )
    assert ok is True
    row = fake_client.rows("expert_reviews")[0]
    assert row["adjustment_set_hash"] == "adj-B"
    assert row["dag_structure_json"] == structure
    assert row["agent_assessment_json"] is None
    calls = fake_client.calls("expert_reviews")
    assert ("eq", ("dag_version_hash", "h1")) in calls
    assert ("eq", ("adjustment_set_hash", "adj-A")) in calls
    assert ("eq", ("approval_status", "pending")) in calls


@pytest.mark.unit
async def test_advance_review_sends_a_literal_none_when_the_adjustment_set_is_unknown(fake_client):
    """A run with no structure in scope knows no adjustment set. The payload must
    carry a LITERAL None -- not omit the key -- or the row keeps a stale hash that
    no longer describes the structure it now carries, which is precisely the
    plausible-wrong value the pair exists to prevent."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-A",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h2",
        dag_structure=None,
        adjustment_set_hash=None,
        expected_current_hash="h1",
        expected_current_adjustment_hash="adj-A",
    )
    assert ok is True
    payload = fake_client.updated("expert_reviews")[0]
    assert "adjustment_set_hash" in payload and payload["adjustment_set_hash"] is None
    assert fake_client.rows("expert_reviews")[0]["adjustment_set_hash"] is None


@pytest.mark.unit
async def test_advance_review_matches_an_unknown_current_adjustment_with_is_null(fake_client):
    """A row that has not learned its adjustment set yet (NULL) is advanced by a
    run that carries one -- the row LEARNS it. The CAS expresses "still unknown"
    as IS NULL, never as an omitted filter."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": None,
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h1",
        dag_structure={"nodes": ["T"], "adjustment_sets": [["W"]]},
        adjustment_set_hash="adj-W",
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )
    assert ok is True
    assert ("is_", ("adjustment_set_hash", "null")) in fake_client.calls("expert_reviews")
    assert fake_client.rows("expert_reviews")[0]["adjustment_set_hash"] == "adj-W"


@pytest.mark.unit
async def test_advance_review_warns_naming_both_expected_halves(fake_client, caplog):
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h9",
                "adjustment_set_hash": "adj-Z",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.WARNING):
        ok = await repo.advance_review(
            "r1",
            dag_version_hash="h2",
            dag_structure=None,
            adjustment_set_hash=None,
            expected_current_hash="h1",
            expected_current_adjustment_hash="adj-A",
        )
    assert ok is False
    assert any("h1" in r.getMessage() and "adj-A" in r.getMessage() for r in caplog.records), (
        "the warning must name BOTH expected halves, or the loser cannot be diagnosed"
    )


@pytest.mark.unit
async def test_append_version_is_the_insert_plus_the_advance(fake_client):
    """``append_version`` composes the two: the timeline row goes in first, then
    the same compare-and-set advance. A failed advance leaves the version row --
    the timeline records what runs produced, the review records what it is on."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-A",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure={"nodes": ["T"]},
        adjustment_set_hash="adj-B",
        query_id="q2",
        expected_current_hash="h1",
        expected_current_adjustment_hash="adj-A",
    )
    assert ok is True
    version = fake_client.inserted("expert_review_versions")[0]
    assert version["dag_version_hash"] == "h2" and version["adjustment_set_hash"] == "adj-B"
    row = fake_client.rows("expert_reviews")[0]
    assert row["dag_version_hash"] == "h2" and row["adjustment_set_hash"] == "adj-B"


@pytest.mark.unit
async def test_append_version_keeps_the_version_row_when_the_pair_cas_loses(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-C",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h1",
        dag_structure=None,
        adjustment_set_hash="adj-B",
        query_id=None,
        expected_current_hash="h1",
        expected_current_adjustment_hash="adj-A",
    )
    assert ok is False
    assert fake_client.inserted("expert_review_versions")[0]["adjustment_set_hash"] == "adj-B"
    assert fake_client.rows("expert_reviews")[0]["adjustment_set_hash"] == "adj-C"


@pytest.mark.unit
async def test_create_review_stores_the_adjustment_set_hash(fake_client):
    """The mint gives the row BOTH halves from the start, so the version-1 append's
    compare-and-set has a pair to expect."""
    repo = ExpertReviewRepository(supabase_client=fake_client)
    await repo.create_review(
        reviewer_id="q1",
        review_type="dag_approval",
        dag_version_hash="h1",
        brand="Remi",
        treatment_variable="treatment_arm",
        outcome_variable="persistent_180d",
        adjustment_set_hash="adj-W",
    )
    row = fake_client.inserted("expert_reviews")[0]
    assert row["adjustment_set_hash"] == "adj-W"


@pytest.mark.unit
async def test_create_review_omits_an_unknown_adjustment_set_hash(fake_client):
    """None is stripped like every other unset column: an omitted key lets the
    DB default (NULL = unknown) stand, and keeps the payload minimal."""
    repo = ExpertReviewRepository(supabase_client=fake_client)
    await repo.create_review(
        reviewer_id="q1",
        review_type="dag_approval",
        dag_version_hash="h1",
        brand="Remi",
        treatment_variable="treatment_arm",
        outcome_variable="persistent_180d",
    )
    assert "adjustment_set_hash" not in fake_client.inserted("expert_reviews")[0]


# --------------------------------------------------------------------------
# record_version -- the timeline write WITHOUT the row advance
#
# Recording a structure and moving the review onto it are two decisions (the
# gate makes them separately), so the repository offers them separately. A
# review already ON the pair it is recording needs the timeline row and nothing
# else: routing that through append_version re-wrote the row with the pair it
# already had, which cleared the advisory assessment for no reason.
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_record_version_inserts_the_timeline_row_and_touches_nothing_else(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-W",
                "agent_assessment_json": {"items": [{"id": "q1"}]},
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.record_version(
        "r1",
        dag_version_hash="h1",
        dag_structure={"nodes": ["T", "Y"], "adjustment_sets": [["W"]]},
        adjustment_set_hash="adj-W",
        query_id="q2",
    )
    assert ok is True
    assert fake_client.inserted("expert_review_versions")[0] == {
        "review_id": "r1",
        "dag_version_hash": "h1",
        "dag_structure_json": {"nodes": ["T", "Y"], "adjustment_sets": [["W"]]},
        "adjustment_set_hash": "adj-W",
        "query_id": "q2",
    }
    # The REVIEW is untouched -- no UPDATE at all, so the cached advisory
    # grading of this very structure survives.
    assert fake_client.updated("expert_reviews") == []
    assert fake_client.rows("expert_reviews")[0]["agent_assessment_json"] == {
        "items": [{"id": "q1"}]
    }
    # insert(), never upsert() -- service_role has no UPDATE on the versions table.
    assert not any(m == "upsert" for m, _ in fake_client.calls("expert_review_versions"))


@pytest.mark.unit
async def test_record_version_returns_false_and_logs_when_the_insert_fails(fake_client, caplog):
    fake_client.fail_next_insert("expert_review_versions", Exception("42501 permission denied"))
    repo = ExpertReviewRepository(supabase_client=fake_client)
    with caplog.at_level(logging.ERROR):
        ok = await repo.record_version(
            "r1",
            dag_version_hash="h1",
            dag_structure=None,
            adjustment_set_hash=None,
            query_id=None,
        )
    assert ok is False
    assert any("r1" in r.getMessage() for r in caplog.records)


@pytest.mark.unit
async def test_append_version_composes_record_version_and_advance_review(fake_client, monkeypatch):
    """``append_version`` must not keep its own copy of the insert: one
    definition of "write the timeline row", one of "move the review"."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": "adj-A",
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    seen: Dict[str, Any] = {}

    async def _spy_record(review_id: str, **kwargs: Any) -> bool:
        seen["record"] = {"review_id": review_id, **kwargs}
        return True

    monkeypatch.setattr(repo, "record_version", _spy_record)

    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure={"nodes": ["T"]},
        adjustment_set_hash="adj-B",
        query_id="q2",
        expected_current_hash="h1",
        expected_current_adjustment_hash="adj-A",
    )
    assert ok is True
    assert seen["record"] == {
        "review_id": "r1",
        "dag_version_hash": "h2",
        "dag_structure": {"nodes": ["T"]},
        "adjustment_set_hash": "adj-B",
        "query_id": "q2",
    }
    # ... and the advance still ran, on the real repository method.
    row = fake_client.rows("expert_reviews")[0]
    assert row["dag_version_hash"] == "h2" and row["adjustment_set_hash"] == "adj-B"


@pytest.mark.unit
async def test_append_version_does_not_advance_when_the_record_fails(fake_client):
    """Unchanged contract, now expressed through the split: a failed timeline
    write leaves the review exactly where it was."""
    fake_client.seed(
        "expert_reviews",
        [{"review_id": "r1", "approval_status": "pending", "dag_version_hash": "h1"}],
    )
    fake_client.fail_next_insert("expert_review_versions", Exception("boom"))
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.append_version(
        "r1",
        dag_version_hash="h2",
        dag_structure=None,
        adjustment_set_hash=None,
        query_id=None,
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )
    assert ok is False
    assert fake_client.updated("expert_reviews") == []
    assert fake_client.rows("expert_reviews")[0]["dag_version_hash"] == "h1"


# --------------------------------------------------------------------------
# The compare-and-set matches a NOT-RECORDED half explicitly (#1991 debt 3)
#
# ``expert_reviews.dag_version_hash`` is nullable -- migration 141's backfill
# filters ``IS NOT NULL``, which is only needed because NULLs are possible -- so
# the pair CAS must be able to name "this row records no hash". An ``eq`` on
# None cannot: PostgREST would compare against the literal text "None" and match
# nothing, so a NULL-hash pending row could take an appended version row and
# then never advance onto it. Both halves route through
# ``match_nullable_column`` for the one reason its docstring gives.
# --------------------------------------------------------------------------


@pytest.mark.unit
async def test_advance_review_matches_an_unrecorded_hash_with_is_null(fake_client):
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": None,
                "adjustment_set_hash": None,
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h2",
        dag_structure={"nodes": ["T", "Y"]},
        adjustment_set_hash="adj-B",
        expected_current_hash=None,
        expected_current_adjustment_hash=None,
    )

    assert ok is True, "a pending row that records no hash is still advanceable"
    calls = fake_client.calls("expert_reviews")
    assert ("is_", ("dag_version_hash", "null")) in calls
    assert not any(m == "eq" and a[0] == "dag_version_hash" for m, a in calls)
    row = fake_client.rows("expert_reviews")[0]
    assert row["dag_version_hash"] == "h2" and row["adjustment_set_hash"] == "adj-B"


@pytest.mark.unit
async def test_advance_review_still_matches_a_known_hash_with_eq(fake_client):
    """The other half of the same filter: a recorded hash is an ``eq``, so the
    NULL case above is a narrowing and not a loosening."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": None,
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h2",
        dag_structure=None,
        adjustment_set_hash=None,
        expected_current_hash="h1",
        expected_current_adjustment_hash=None,
    )

    assert ok is True
    calls = fake_client.calls("expert_reviews")
    assert ("eq", ("dag_version_hash", "h1")) in calls
    assert not any(m == "is_" and a[0] == "dag_version_hash" for m, a in calls)


@pytest.mark.unit
async def test_advance_review_refuses_an_unrecorded_hash_against_a_row_that_has_one(fake_client):
    """ "Not recorded" is a PRECONDITION, not a wildcard: a row that has since
    learned its hash must not be advanced by a caller that read no hash."""
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r1",
                "approval_status": "pending",
                "dag_version_hash": "h1",
                "adjustment_set_hash": None,
            }
        ],
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)
    ok = await repo.advance_review(
        "r1",
        dag_version_hash="h2",
        dag_structure=None,
        adjustment_set_hash=None,
        expected_current_hash=None,
        expected_current_adjustment_hash=None,
    )

    assert ok is False
    assert fake_client.rows("expert_reviews")[0]["dag_version_hash"] == "h1"


# --------------------------------------------------------------------------
# renew_review (#2090): the renewal insert races uq_er_pending_estimand too
# --------------------------------------------------------------------------


def _seed_original_and_pending(fake_client: _FakeClient) -> None:
    fake_client.seed(
        "expert_reviews",
        [
            {
                "review_id": "r-original",
                "estimand_key": "b:t:y",
                "approval_status": "approved",
                "dag_version_hash": "h1",
                "brand": "B",
                "treatment_variable": "T",
                "outcome_variable": "Y",
            },
            {
                "review_id": "r-pending",
                "estimand_key": "b:t:y",
                "approval_status": "pending",
                "review_type": "dag_approval",
            },
        ],
    )


@pytest.mark.unit
async def test_renew_review_recovers_pending_by_estimand_on_unique_violation(fake_client):
    _seed_original_and_pending(fake_client)
    fake_client.fail_next_insert(
        "expert_reviews",
        Exception('duplicate key value violates unique constraint "uq_er_pending_estimand"'),
    )
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.renew_review(original_review_id="r-original", reviewer_id="q2")

    assert rid == "r-pending"
    # the same by-estimand lookup create_review uses, keyed from the ORIGINAL's estimand
    assert ("eq", ("estimand_key", "b:t:y")) in fake_client.calls("expert_reviews")
    assert fake_client.inserted("expert_reviews") == []


@pytest.mark.unit
async def test_renew_review_does_not_mask_a_non_unique_failure(fake_client):
    """Only a 23505 is recovered; any other insert failure still returns None."""
    _seed_original_and_pending(fake_client)
    fake_client.fail_next_insert("expert_reviews", Exception("connection reset by peer"))
    repo = ExpertReviewRepository(supabase_client=fake_client)

    rid = await repo.renew_review(original_review_id="r-original", reviewer_id="q2")

    assert rid is None
    assert ("eq", ("estimand_key", "b:t:y")) not in fake_client.calls("expert_reviews")
