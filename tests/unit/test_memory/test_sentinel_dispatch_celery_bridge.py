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


class _NotProxy:
    """Stand-in for the Supabase ``.not_`` accessor.

    M2 (#381): the invalidation_count evaluator uses
    ``query.not_.is_("invalidated_at", "null")`` to enumerate invalidated rows.
    The proxy forwards the next call into the underlying ``_FakeQuery`` with
    the negation flag set.
    """

    def __init__(self, query: "_FakeQuery") -> None:
        self._query = query

    def is_(self, col: str, val: str) -> "_FakeQuery":
        return self._query._apply_is(col, val, negated=True)


class _FakeQuery:
    def __init__(self, store: "FakeSupabase", table: str) -> None:
        self.store = store
        self.table_name = table
        self._mode: Optional[str] = None
        self._filters: Dict[str, Any] = {}
        self._lt: Dict[str, Any] = {}
        # Null-shaped filters: (col -> "null") with negation flag tracked
        # separately so we can model both ``is_(col, "null")`` and
        # ``not_.is_(col, "null")``.
        self._is_null: Dict[str, bool] = {}  # col -> True if filter requires NULL
        self._is_not_null: Dict[str, bool] = {}  # col -> True if filter requires NOT NULL
        self._update_payload: Dict[str, Any] = {}
        self._insert_payload: Any = None

    @property
    def not_(self) -> _NotProxy:
        return _NotProxy(self)

    def _apply_is(self, col: str, val: str, *, negated: bool) -> "_FakeQuery":
        # Supabase serializes IS NULL via the string "null" / "NULL".
        normalized = (val or "").lower()
        if normalized in {"null", "none"}:
            if negated:
                self._is_not_null[col] = True
            else:
                self._is_null[col] = True
        return self

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
        return self._apply_is(col, val, negated=False)

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
            # Model the live `is_synthetic NOT NULL DEFAULT false` column: a
            # planted row without the key is a REAL row (#894 sentinel reads
            # default-exclude synthetic via .eq('is_synthetic', False)).
            default = False if col == "is_synthetic" else None
            rows = [r for r in rows if r.get(col, default) == want]
        for col, threshold in self._lt.items():
            rows = [r for r in rows if (r.get(col) or 0) < threshold]
        # NULL filters: row.get(col) is None
        for col in self._is_null:
            rows = [r for r in rows if r.get(col) is None]
        # NOT NULL filters: row.get(col) is not None
        for col in self._is_not_null:
            rows = [r for r in rows if r.get(col) is not None]
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
            "executive_insights": [],
            "ml_predictions": [],
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


# ---------------------------------------------------------------------------
# M2 (#381) — staleness_threshold sentinel must populate trigger_data["stale_findings"]
# ---------------------------------------------------------------------------


def _seed_invalidation_count_sentinel(
    fake_supabase: FakeSupabase,
    *,
    table: str = "executive_insights",
    brand: str = "Kisqali",
    sentinel_id: str = "s-staleness",
) -> None:
    """Seed a sentinel whose internal pattern_type is invalidation_count (the
    shipped analog of the plan's staleness_threshold trigger), wired to
    dispatch the notify_and_queue_reanalysis Celery task.

    M2 (#381): the bug pre-fix was that the YAML's ``sentinel_staleness_alert``
    used ``trigger_type: staleness_threshold`` (plan vocab) which the loader
    translated to ``threshold_breach`` against ``causal_effect_size`` — never
    populating ``stale_findings`` in trigger_data. The fix introduces an
    ``invalidation_count`` internal pattern_type that queries
    ``invalidated_at IS NOT NULL`` (Decision 3 = KEEP BINARY semantics) and
    a dispatcher special-case that packages the matches into
    ``trigger_data['stale_findings']`` before enqueuing the Celery task.
    """
    fake_supabase.rows["sentinels"].append(
        {
            "sentinel_id": sentinel_id,
            "name": "High staleness alert",
            "pattern_type": "invalidation_count",
            "pattern_config": {
                "table": table,
                "tier": "semantic",
            },
            "action_type": "dispatch_agent",
            "action_config": {
                "agent_name": "notify_and_queue_reanalysis",
                "input": {},
            },
            "brand": brand,
            "enabled": True,
            "fire_count": 0,
            "last_fired_at": None,
            "cooldown_minutes": None,
        }
    )


@pytest.mark.asyncio
async def test_staleness_threshold_dispatcher_populates_stale_findings(
    fake_supabase: FakeSupabase,
):
    """M2 (#381): when an invalidation_count sentinel matches invalidated rows
    and the action is ``notify_and_queue_reanalysis``, the dispatcher MUST
    package the matches into ``trigger_data['stale_findings']`` so the
    Celery handler can iterate them. Without this packaging, the handler at
    src/tasks/sentinel_actions.py:172 sees ``trigger_data.get('stale_findings')
    or []`` → notifies 0 findings.

    Pre-fix this test FAILS for the right reason: trigger_data only carries
    ``match`` + ``action_input``, not ``stale_findings``.
    """
    _seed_invalidation_count_sentinel(fake_supabase)
    # Two invalidated rows in executive_insights with brand=Kisqali.
    fake_supabase.rows["executive_insights"] = [
        {
            "insight_id": "ei-stale-1",
            "brand": "Kisqali",
            "invalidated_at": "2026-05-18T12:00:00+00:00",
            "invalidation_reason": "cascade",
        },
        {
            "insight_id": "ei-stale-2",
            "brand": "Kisqali",
            "invalidated_at": "2026-05-19T01:00:00+00:00",
            "invalidation_reason": "cascade",
        },
        # One non-invalidated row that MUST NOT be packaged into stale_findings.
        {
            "insight_id": "ei-fresh",
            "brand": "Kisqali",
            "invalidated_at": None,
        },
    ]
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        result = await dispatch_sentinels()
    assert result.fired == 1, f"expected staleness sentinel to fire, got {result}"
    celery_mock.send_task.assert_called()
    # Pull the FIRST send_task call (single-fire-with-list semantics for
    # notify_and_queue_reanalysis; the handler caps at top-5 internally).
    call_args = celery_mock.send_task.call_args_list[0]
    inner_kwargs = call_args.kwargs.get("kwargs") or {}
    trigger_data = inner_kwargs.get("trigger_data") or {}
    stale_findings = trigger_data.get("stale_findings")
    assert stale_findings is not None, (
        "M2: trigger_data MUST carry 'stale_findings' for notify_and_queue_reanalysis; "
        f"got trigger_data={trigger_data!r}"
    )
    assert isinstance(stale_findings, list), (
        f"M2: stale_findings must be a list, got {type(stale_findings).__name__}"
    )
    finding_ids = {f.get("finding_id") for f in stale_findings}
    assert finding_ids == {"ei-stale-1", "ei-stale-2"}, (
        f"M2: stale_findings must enumerate only invalidated rows; got finding_ids={finding_ids}"
    )
    # Decision 3 = KEEP BINARY: every finding's staleness_score is 1.0.
    for finding in stale_findings:
        assert finding.get("staleness_score") == 1.0, (
            f"M2: under Decision 3 = KEEP BINARY, staleness_score MUST be 1.0; "
            f"got finding={finding!r}"
        )


@pytest.mark.asyncio
async def test_staleness_threshold_brand_scoping(fake_supabase: FakeSupabase):
    """M2 (#381) brand scoping: an invalidation_count sentinel scoped to brand X
    MUST only include invalidated rows whose brand matches X.

    Cross-brand bleed would be a security regression — sentinels are
    brand-scoped at every layer per the registry docstring (lines 51-58).
    """
    _seed_invalidation_count_sentinel(
        fake_supabase,
        table="executive_insights",
        brand="Kisqali",
        sentinel_id="s-stale-brand",
    )
    fake_supabase.rows["executive_insights"] = [
        {
            "insight_id": "ei-kisqali",
            "brand": "Kisqali",
            "invalidated_at": "2026-05-18T12:00:00+00:00",
        },
        # Pluvicto invalidation MUST NOT leak into the Kisqali sentinel's
        # stale_findings.
        {
            "insight_id": "ei-pluvicto",
            "brand": "Pluvicto",
            "invalidated_at": "2026-05-18T12:00:00+00:00",
        },
    ]
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        await dispatch_sentinels()
    call_args = celery_mock.send_task.call_args_list[0]
    inner_kwargs = call_args.kwargs.get("kwargs") or {}
    trigger_data = inner_kwargs.get("trigger_data") or {}
    finding_ids = {f.get("finding_id") for f in (trigger_data.get("stale_findings") or [])}
    assert finding_ids == {"ei-kisqali"}, (
        f"M2: cross-brand bleed — Kisqali sentinel must not include Pluvicto rows; "
        f"got finding_ids={finding_ids}"
    )


@pytest.mark.asyncio
async def test_staleness_threshold_no_invalidated_rows_no_fire(
    fake_supabase: FakeSupabase,
):
    """M2 (#381): when no rows are invalidated, the sentinel MUST NOT fire.

    Edge case: if the evaluator returns [] (no matches), the dispatcher
    should not fire (per ``if not matches: continue`` at registry.py:401-402).
    """
    _seed_invalidation_count_sentinel(
        fake_supabase,
        table="executive_insights",
        brand="Kisqali",
        sentinel_id="s-stale-empty",
    )
    fake_supabase.rows["executive_insights"] = [
        {
            "insight_id": "ei-fresh-1",
            "brand": "Kisqali",
            "invalidated_at": None,
        },
        {
            "insight_id": "ei-fresh-2",
            "brand": "Kisqali",
            "invalidated_at": None,
        },
    ]
    with patch("src.memory.sentinels.registry.celery_app") as celery_mock:
        celery_mock.send_task = MagicMock()
        result = await dispatch_sentinels()
    assert result.fired == 0, (
        f"M2: with no invalidated rows, staleness sentinel MUST NOT fire; got {result}"
    )
    celery_mock.send_task.assert_not_called()
