"""Unit tests for cleanup_orphan_models.decommission — mock client, no real DB.

Covers:
1. Dry-run (execute=False): NO delete/update called; returns "would_archive" with
   the counted metrics_rows.
2. Execute=True: ml_performance_metrics DELETE is called BEFORE the
   ml_model_registry UPDATE; the UPDATE sets stage='archived' and filters by the
   resolved model_id.
3. A handle that resolves to None ("absent") triggers NO mutation.
4. Only ORPHAN_MODELS handles are ever acted on (the hardcoded tuple guard).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.mlops.gold_standard_eval.cleanup_orphan_models import (
    ORPHAN_MODELS,
    decommission,
)

# ---------------------------------------------------------------------------
# Fake async Supabase client
# ---------------------------------------------------------------------------
# We build the mock using a MagicMock chain that mirrors the PostgREST
# builder pattern:
#
#   client.table("x").select(...).eq(...).execute()
#   client.table("x").delete().eq(...).execute()
#   client.table("x").update({...}).eq(...).execute()
#
# Each builder method returns an object with the next builder; `.execute()`
# is an AsyncMock so it can be awaited.


def _make_count_response(n: int) -> MagicMock:
    """Return a fake PostgREST response with .count = n and .data = []."""
    resp = MagicMock()
    resp.count = n
    resp.data = []
    return resp


def _make_resolve_response(model_id: str | None) -> MagicMock:
    """Return a fake registry lookup response."""
    resp = MagicMock()
    resp.data = [{"id": model_id}] if model_id else []
    return resp


class _FakeBuilder:
    """Minimal chainable builder that records calls and returns itself."""

    def __init__(self) -> None:
        self._calls: list[tuple] = []
        self._execute_return: MagicMock = MagicMock()
        self._execute_return.data = []
        self._execute_return.count = 0

    def set_execute_return(self, value: MagicMock) -> "_FakeBuilder":
        self._execute_return = value
        return self

    def select(self, *args: object, **kwargs: object) -> "_FakeBuilder":
        self._calls.append(("select", args, kwargs))
        return self

    def delete(self) -> "_FakeBuilder":
        self._calls.append(("delete",))
        return self

    def update(self, payload: dict) -> "_FakeBuilder":
        self._calls.append(("update", payload))
        return self

    def eq(self, col: str, val: object) -> "_FakeBuilder":
        self._calls.append(("eq", col, val))
        return self

    def limit(self, n: int) -> "_FakeBuilder":
        self._calls.append(("limit", n))
        return self

    async def execute(self) -> MagicMock:
        self._calls.append(("execute",))
        return self._execute_return


class FakeClient:
    """
    Fake async Supabase client.

    ``_resolve_model_id`` iterates two columns ('model_version', 'model_name')
    and calls .table().select().eq().limit().execute() for each.

    This client allows per-table, per-operation configuration so tests can
    control exactly what each query returns.

    Attribute ``ops`` is a flat list of (table, operation, *args) tuples in
    the order the calls arrived — used to assert DELETE-before-UPDATE ordering.
    """

    def __init__(
        self,
        *,
        # map handle -> uuid (or None when absent)
        model_ids: dict[str, str | None],
        # map model_id -> row count for ml_performance_metrics
        metrics_counts: dict[str, int],
    ) -> None:
        self._model_ids = model_ids
        self._metrics_counts = metrics_counts
        # All mutation calls recorded as (table, op, eq_col, eq_val)
        self.mutation_calls: list[tuple] = []

    def table(self, name: str) -> "_TableProxy":
        return _TableProxy(name, self)


class _TableProxy:
    """Records calls on a per-table basis and routes to the right response."""

    def __init__(self, name: str, client: FakeClient) -> None:
        self._name = name
        self._client = client
        self._pending: list[tuple] = []

    # ---- chainable builder methods ----------------------------------------

    def select(self, *_args: object, **_kw: object) -> "_TableProxy":
        self._pending.append(("select",))
        return self

    def delete(self) -> "_TableProxy":
        self._pending.append(("delete",))
        return self

    def update(self, payload: dict) -> "_TableProxy":
        self._pending.append(("update", payload))
        return self

    def eq(self, col: str, val: object) -> "_TableProxy":
        self._pending.append(("eq", col, val))
        return self

    def limit(self, n: int) -> "_TableProxy":
        self._pending.append(("limit", n))
        return self

    async def execute(self) -> MagicMock:
        self._pending.append(("execute",))

        # Classify the chain.
        ops = [p[0] for p in self._pending]
        has_delete = "delete" in ops
        has_update = "update" in ops

        if has_delete:
            # Record the mutation so tests can assert call order.
            eq_col, eq_val = self._get_eq()
            self._client.mutation_calls.append(("delete", self._name, eq_col, eq_val))
            resp = MagicMock()
            resp.data = []
            resp.count = 0
            return resp

        if has_update:
            update_payload = next(p[1] for p in self._pending if p[0] == "update")
            eq_col, eq_val = self._get_eq()
            self._client.mutation_calls.append(
                ("update", self._name, update_payload, eq_col, eq_val)
            )
            resp = MagicMock()
            resp.data = []
            resp.count = 0
            return resp

        # SELECT path — distinguish registry lookup vs metrics count.
        if self._name == "ml_model_registry":
            # _resolve_model_id filters by model_version or model_name.
            eq_col, eq_val = self._get_eq()
            model_id = self._client._model_ids.get(str(eq_val))
            resp = MagicMock()
            resp.data = [{"id": model_id}] if model_id else []
            return resp

        if self._name == "ml_performance_metrics":
            eq_col, eq_val = self._get_eq()
            count = self._client._metrics_counts.get(str(eq_val), 0)
            resp = MagicMock()
            resp.count = count
            resp.data = []
            return resp

        resp = MagicMock()
        resp.data = []
        resp.count = 0
        return resp

    def _get_eq(self) -> tuple[str, object]:
        """Return the (col, val) from the first .eq() call in the chain."""
        for p in self._pending:
            if p[0] == "eq":
                return p[1], p[2]
        return ("?", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HANDLE_INIT = "csu_initiation_goldstd_lr_v1"
HANDLE_PERS = "pnh_persistence_goldstd_lr_v1"
HANDLE_DISC = "pnh_discontinuation_goldstd_lr_v1"

UUID_INIT = "aaaa0000-0000-0000-0000-000000000001"
UUID_PERS = "bbbb0000-0000-0000-0000-000000000002"
UUID_DISC = "cccc0000-0000-0000-0000-000000000003"

_ALL_MODEL_IDS = {
    HANDLE_INIT: UUID_INIT,
    HANDLE_PERS: UUID_PERS,
    HANDLE_DISC: UUID_DISC,
}

_ALL_METRICS = {
    UUID_INIT: 3,
    UUID_PERS: 5,
    UUID_DISC: 2,
}


def _make_client(
    model_ids: dict | None = None,
    metrics_counts: dict | None = None,
) -> FakeClient:
    return FakeClient(
        model_ids=model_ids if model_ids is not None else _ALL_MODEL_IDS,
        metrics_counts=metrics_counts if metrics_counts is not None else _ALL_METRICS,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dry_run_no_mutations() -> None:
    """execute=False: DELETE and UPDATE must NEVER be called; status=would_archive."""
    client = _make_client()
    report = await decommission(db=client, execute=False)

    assert report["executed"] is False
    assert report["summary"]["would_archive"] == 3
    assert report["summary"]["archived"] == 0
    assert report["summary"]["absent"] == 0

    # No mutations should have been recorded.
    assert client.mutation_calls == [], (
        f"Dry-run should make NO mutations but got: {client.mutation_calls}"
    )

    for result in report["results"]:
        assert result["status"] == "would_archive"
        assert result["model_id"] is not None


@pytest.mark.asyncio
async def test_dry_run_reports_metrics_count() -> None:
    """Dry-run still counts metrics rows and surfaces them in the report."""
    client = _make_client()
    report = await decommission(db=client, execute=False)

    by_handle = {r["handle"]: r for r in report["results"]}
    assert by_handle[HANDLE_INIT]["metrics_rows"] == 3
    assert by_handle[HANDLE_PERS]["metrics_rows"] == 5
    assert by_handle[HANDLE_DISC]["metrics_rows"] == 2


@pytest.mark.asyncio
async def test_execute_delete_before_update() -> None:
    """execute=True: DELETE ml_performance_metrics BEFORE UPDATE ml_model_registry."""
    client = _make_client()
    report = await decommission(db=client, execute=True)

    assert report["executed"] is True
    assert report["summary"]["archived"] == 3

    # All mutation calls must exist.
    assert len(client.mutation_calls) > 0

    # For EACH handle, the delete must appear before the corresponding update.
    for handle, model_id in [
        (HANDLE_INIT, UUID_INIT),
        (HANDLE_PERS, UUID_PERS),
        (HANDLE_DISC, UUID_DISC),
    ]:
        calls_for_model: list[tuple] = [
            c
            for c in client.mutation_calls
            if (c[0] == "delete" and c[3] == model_id) or (c[0] == "update" and c[4] == model_id)
        ]
        ops_order = [c[0] for c in calls_for_model]
        assert ops_order == ["delete", "update"], (
            f"For handle={handle!r} expected ['delete', 'update'] but got {ops_order}"
        )


@pytest.mark.asyncio
async def test_execute_sets_archived_stage() -> None:
    """execute=True: the registry UPDATE payload must be stage='archived'."""
    client = _make_client()
    await decommission(db=client, execute=True)

    update_calls = [c for c in client.mutation_calls if c[0] == "update"]
    assert len(update_calls) == 3, f"Expected 3 UPDATE calls, got {len(update_calls)}"

    for call_record in update_calls:
        _op, _table, payload, eq_col, eq_val = call_record
        assert payload == {"stage": "archived"}, (
            f"UPDATE payload must be {{'stage': 'archived'}} but got {payload!r}"
        )
        assert eq_col == "id", f"UPDATE must filter by 'id' but got eq_col={eq_col!r}"
        assert eq_val in (UUID_INIT, UUID_PERS, UUID_DISC), (
            f"Unexpected model_id in UPDATE: {eq_val!r}"
        )


@pytest.mark.asyncio
async def test_absent_handle_skipped() -> None:
    """A handle that resolves to None is reported 'absent' with no mutations."""
    # Make HANDLE_INIT absent.
    model_ids = dict(_ALL_MODEL_IDS)
    model_ids[HANDLE_INIT] = None

    client = _make_client(model_ids=model_ids)
    report = await decommission(db=client, execute=True)

    by_handle = {r["handle"]: r for r in report["results"]}
    assert by_handle[HANDLE_INIT]["status"] == "absent"
    assert by_handle[HANDLE_INIT]["model_id"] is None
    assert by_handle[HANDLE_INIT]["metrics_rows"] == 0

    # No delete/update calls should reference UUID_INIT (since it's absent).
    for mut in client.mutation_calls:
        if mut[0] == "delete":
            assert mut[3] != UUID_INIT
        if mut[0] == "update":
            assert mut[4] != UUID_INIT

    # The other two still get archived.
    assert by_handle[HANDLE_PERS]["status"] == "archived"
    assert by_handle[HANDLE_DISC]["status"] == "archived"

    assert report["summary"]["absent"] == 1
    assert report["summary"]["archived"] == 2


@pytest.mark.asyncio
async def test_only_orphan_handles_are_touched() -> None:
    """The script never constructs a non-orphan handle — ORPHAN_MODELS is the full list."""
    # Verify the constant itself: exactly 3 entries, all well-known.
    assert len(ORPHAN_MODELS) == 3
    assert HANDLE_INIT in ORPHAN_MODELS
    assert HANDLE_PERS in ORPHAN_MODELS
    assert HANDLE_DISC in ORPHAN_MODELS

    # Ensure no live per-brand handles appear in ORPHAN_MODELS.
    live_handles = [
        "initiation_remibrutinib_goldstd_lr_v1",
        "persistence_remibrutinib_goldstd_lr_v1",
        "discontinuation_kisqali_goldstd_lr_v1",
        "hcp_adoption_kisqali_goldstd_lr_v1",
    ]
    for live in live_handles:
        assert live not in ORPHAN_MODELS, f"Live handle {live!r} must NOT be in ORPHAN_MODELS"

    # On a real execute run, the mutations target only the resolved UUIDs of ORPHAN_MODELS.
    client = _make_client()
    await decommission(db=client, execute=True)

    for mut in client.mutation_calls:
        mutated_id = mut[3] if mut[0] == "delete" else mut[4]
        assert mutated_id in (UUID_INIT, UUID_PERS, UUID_DISC), (
            f"Mutation targeted unexpected id={mutated_id!r}"
        )
