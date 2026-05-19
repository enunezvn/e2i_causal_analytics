"""Unit tests for the dispatcher's dispatch_agent → Celery bridge (#375 iter-1 H1).

Codex iter-0 H1: ``src/memory/sentinels/registry.py::_fire_action`` only
publishes an ``InsightSignalBus`` event for ``dispatch_agent`` actions; it
never calls ``celery_app.send_task(...)`` for the four plan-specced action
handlers. Without that bridge, YAML-loaded sentinels using
``action: rerun_all_active_cohorts`` (etc.) fire the bus event but never
enqueue the corresponding Celery task — operator intent silently lost.

This test set verifies:

1. When ``agent_name`` is one of the 4 plan-specced names, the dispatcher
   calls ``celery_app.send_task`` with the full task path
   ``src.tasks.sentinel_actions.<agent_name>`` and a structured args/kwargs
   payload.
2. The complementary ``InsightSignalBus.publish`` call is still made —
   the bus event is additive, not a replacement.
3. When ``agent_name`` is NOT a plan-specced name, ``send_task`` is NOT
   called (back-compat with arbitrary agent dispatch).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.memory.sentinels.registry import dispatch_sentinels

# ---------------------------------------------------------------------------
# FakeSupabase fixture (mirrors the shape used by the existing cooldown +
# plan-specced tests; kept local so this test file is self-contained)
# ---------------------------------------------------------------------------


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._lt: Dict[str, Any] = {}
        self._update_payload: Dict[str, Any] = {}
        self._insert_payload: Any = None

    def select(self, cols: str, count: Optional[str] = None) -> "_FakeQuery":
        self._mode = "select"
        return self

    def update(self, payload: Dict[str, Any]) -> "_FakeQuery":
        self._mode = "update"
        self._update_payload = payload
        return self

    def insert(self, payload: Any) -> "_FakeQuery":
        self._mode = "insert"
        self._insert_payload = payload
        return self

    def eq(self, col: str, val: Any) -> "_FakeQuery":
        self._filters[col] = val
        return self

    def lt(self, col: str, val: Any) -> "_FakeQuery":
        self._lt[col] = val
        return self

    def gt(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def gte(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def lte(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def neq(self, col: str, val: Any) -> "_FakeQuery":
        return self

    def is_(self, col: str, val: str) -> "_FakeQuery":
        return self

    def order(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def limit(self, n: int) -> "_FakeQuery":
        return self

    def in_(self, col: str, vals: List[Any]) -> "_FakeQuery":
        return self

    def execute(self) -> MagicMock:
        mock = MagicMock()
        if self._mode == "insert":
            payload = self._insert_payload
            rows_to_insert = payload if isinstance(payload, list) else [payload]
            inserted = []
            for r in rows_to_insert:
                row = dict(r)
                if self.table_name == "sentinels":
                    row.setdefault(
                        "sentinel_id",
                        f"fake-{len(self.store.rows[self.table_name]) + 1}",
                    )
                self.store.rows.setdefault(self.table_name, []).append(row)
                inserted.append(row)
            mock.data = inserted
            mock.count = None
            return mock
        rows = list(self.store.rows.get(self.table_name, []))
        for col, want in self._filters.items():
            rows = [r for r in rows if r.get(col) == want]
        for col, threshold in self._lt.items():
            rows = [r for r in rows if (r.get(col) or 0) < threshold]
        if self._mode == "update":
            for r in rows:
                for orig in self.store.rows[self.table_name]:
                    if orig is r:
                        orig.update(self._update_payload)
                        break
        mock.data = rows
        mock.count = None
        return mock


class FakeSupabase:
    def __init__(self) -> None:
        self.rows: Dict[str, List[Dict[str, Any]]] = {
            "sentinels": [],
            "causal_paths": [],
            "triggers": [],
            "insight_edges": [],
        }

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self, name)


@pytest.fixture
def fake_supabase() -> FakeSupabase:
    return FakeSupabase()


@pytest.fixture(autouse=True)
def patch_clients(fake_supabase):
    """Stub Supabase, the bus's Redis, and the bus itself.

    The default ``get_insight_signal_bus`` returns a singleton tied to a
    real Redis (xadd) — across parameterized cases this leaks event-loop
    state. We patch the bus factory to return a per-test MagicMock so
    tests stay isolated (the bus event is asserted on the mock).
    """
    fake_redis = AsyncMock()
    fake_redis.publish = AsyncMock(return_value=1)
    fake_redis.xadd = AsyncMock(return_value="0-0")
    fake_bus = MagicMock()
    fake_bus.publish = AsyncMock()
    with (
        patch("src.memory.sentinels.registry.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.sentinels.registry.get_insight_signal_bus", return_value=fake_bus),
        patch("src.memory.lifecycle.invalidator.get_supabase_client", return_value=fake_supabase),
        patch("src.memory.lifecycle.invalidator.get_redis_client", return_value=fake_redis),
        patch("src.memory.services.factories.get_redis_client", return_value=fake_redis),
    ):
        yield fake_supabase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_dispatch_agent_sentinel(
    fake_supabase: FakeSupabase,
    *,
    agent_name: str,
    sentinel_id: str = "s-bridge",
    brand: str = "Kisqali",
    action_input: Optional[Dict[str, Any]] = None,
) -> None:
    """Seed a single causal_paths row + a dispatch_agent threshold_breach sentinel
    whose action_config.agent_name is the param. Causes one match → one fire."""
    fake_supabase.rows["sentinels"].append(
        {
            "sentinel_id": sentinel_id,
            "name": f"test-{agent_name}",
            "pattern_type": "threshold_breach",
            "pattern_config": {
                "table": "causal_paths",
                "column": "causal_effect_size",
                "op": "<",
                "value": 0.05,
            },
            "action_type": "dispatch_agent",
            "action_config": {
                "agent_name": agent_name,
                "input": action_input or {},
            },
            "brand": brand,
            "enabled": True,
            "fire_count": 0,
            "last_fired_at": None,
            "cooldown_minutes": None,
        }
    )
    fake_supabase.rows["causal_paths"].append(
        {"path_id": "cp-1", "brand": brand, "causal_effect_size": 0.01}
    )


# ---------------------------------------------------------------------------
# H1 — dispatch_agent must call celery_app.send_task for plan-specced names
# ---------------------------------------------------------------------------


PLAN_SPECCED_ACTION_NAMES = [
    "rerun_all_active_cohorts",
    "notify_and_queue_reanalysis",
    "flag_for_review",
    "run_full_consolidation",
]


@pytest.mark.parametrize("agent_name", PLAN_SPECCED_ACTION_NAMES)
@pytest.mark.asyncio
async def test_dispatch_agent_enqueues_celery_task_for_plan_specced_name(
    fake_supabase: FakeSupabase,
    agent_name: str,
):
    """When agent_name is one of the 4 plan-specced names, the dispatcher
    MUST call ``celery_app.send_task`` with the full task path so a Celery
    worker can run the action handler.

    Codex iter-0 H1: pre-fix this assertion fails — only the bus event
    fires.
    """
    _seed_dispatch_agent_sentinel(
        fake_supabase,
        agent_name=agent_name,
        sentinel_id=f"s-{agent_name}",
    )
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        result = await dispatch_sentinels()
    assert result.fired == 1, f"expected sentinel to fire, got {result}"
    # The mapping from plan-action-name → Celery task path is the explicit
    # contract from src/tasks/sentinel_actions.py: each task is registered
    # under "src.tasks.sentinel_actions.<name>".
    expected_task = f"src.tasks.sentinel_actions.{agent_name}"
    celery_mock.send_task.assert_called_once()
    call_args = celery_mock.send_task.call_args
    # Accept either positional or kwargs form.
    actual_task = call_args.args[0] if call_args.args else call_args.kwargs["name"]
    assert actual_task == expected_task, (
        f"dispatcher must enqueue Celery task {expected_task!r}; got {actual_task!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_agent_celery_task_args_include_sentinel_id_brands_trigger_data(
    fake_supabase: FakeSupabase,
):
    """The Celery task expects ``(sentinel_id, brands, trigger_data)`` per plan
    §3.8 — verify the dispatcher passes them in a shape the handler can consume."""
    _seed_dispatch_agent_sentinel(
        fake_supabase,
        agent_name="rerun_all_active_cohorts",
        sentinel_id="s-args",
        brand="Pluvicto",
        action_input={"refreshed_at": "2026-05-19T00:00:00Z"},
    )
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        await dispatch_sentinels()
    call_args = celery_mock.send_task.call_args
    kwargs_passed = call_args.kwargs
    # The contract is keyword args (so positional ordering can't drift).
    # Either kwargs= or args= form is acceptable.
    inner_kwargs = kwargs_passed.get("kwargs") or {}
    inner_args = kwargs_passed.get("args") or ()
    # Sentinel id present in either form.
    assert "s-args" in str(inner_kwargs.values()) or "s-args" in str(inner_args), (
        f"sentinel_id must be in Celery task args; got kwargs={inner_kwargs!r} args={inner_args!r}"
    )
    # Brand present (single 'Pluvicto').
    assert "Pluvicto" in str(inner_kwargs.values()) or "Pluvicto" in str(inner_args), (
        f"brand must be in Celery task args; got {call_args!r}"
    )


@pytest.mark.asyncio
async def test_dispatch_agent_still_publishes_bus_event(
    fake_supabase: FakeSupabase,
):
    """The Celery bridge is ADDITIVE: the InsightSignalBus.publish call must
    still fire so any non-Celery subscriber (e.g. local orchestrator) gets
    the event too. Codex iter-0 H1 explicitly: "Existing event-bus publish
    should remain — it's complementary, not a replacement."

    The autouse ``patch_clients`` fixture provides a fake_bus on
    ``src.memory.sentinels.registry.get_insight_signal_bus``. We retrieve
    it here so the assertion can pin it.
    """
    _seed_dispatch_agent_sentinel(
        fake_supabase,
        agent_name="flag_for_review",
        sentinel_id="s-bus",
    )
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        # Pull the patched factory's return so we can assert against it.
        from src.memory.sentinels.registry import get_insight_signal_bus

        bus_under_test = get_insight_signal_bus()
        await dispatch_sentinels()
    # Both bridges must fire — the bus event is non-negotiable.
    bus_under_test.publish.assert_called_once()
    celery_mock.send_task.assert_called_once()


@pytest.mark.asyncio
async def test_dispatch_agent_unknown_name_does_not_call_celery(
    fake_supabase: FakeSupabase,
):
    """An agent_name that ISN'T one of the 4 plan-specced names must NOT
    enqueue a Celery task — the dispatcher leaves it to the bus subscriber
    (back-compat with arbitrary agent dispatch from PR #250)."""
    _seed_dispatch_agent_sentinel(
        fake_supabase,
        agent_name="drift_monitor",  # legacy / generic name — not in plan set
        sentinel_id="s-legacy",
    )
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        await dispatch_sentinels()
    celery_mock.send_task.assert_not_called()


# ---------------------------------------------------------------------------
# Invariant lock — #375 codex iter-1 M1
# ---------------------------------------------------------------------------


def test_plan_action_constants_are_in_lockstep():
    """``PLAN_ACTION_TASK_NAMES`` (the frozenset used by the YAML loader for
    validation) MUST stay aligned with ``PLAN_ACTION_TO_CELERY_TASK`` (the dict
    used by the dispatcher to enqueue Celery tasks). If they diverge silently,
    either:

    * the loader rejects an action that the dispatcher knows how to enqueue, or
    * the loader accepts an action the dispatcher can't route → bus-only fire.

    Post-consolidation (codex iter-1 M1) the frozenset is derived from the dict
    via ``frozenset(PLAN_ACTION_TO_CELERY_TASK)``; this test is the invariant
    lock for any future hand-edit drift.
    """
    from src.memory.sentinels.config_loader import PLAN_ACTION_TASK_NAMES
    from src.memory.sentinels.registry import PLAN_ACTION_TO_CELERY_TASK

    assert set(PLAN_ACTION_TASK_NAMES) == set(PLAN_ACTION_TO_CELERY_TASK), (
        "PLAN_ACTION_TASK_NAMES (config_loader) and PLAN_ACTION_TO_CELERY_TASK "
        "(registry) must reference the same set of plan-specced action names. "
        f"loader-only: {set(PLAN_ACTION_TASK_NAMES) - set(PLAN_ACTION_TO_CELERY_TASK)!r}; "
        f"registry-only: {set(PLAN_ACTION_TO_CELERY_TASK) - set(PLAN_ACTION_TASK_NAMES)!r}"
    )
