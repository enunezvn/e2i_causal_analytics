"""#2076: a session-less ``contribute_to_memory`` records NULL, never a minted uuid.

Every agent memory hook used to answer an absent session id with
``str(uuid.uuid4())``. The episodic row then carried a uuid that belongs to no
conversation and to no audit chain — the same "invented identity" defect
#2068 / #2062 / #2064 removed from the pre-hashing hooks. The episodic writer
already coerces an unusable id to ``None`` (``_coerce_session_id``, #1404) and
``episodic_memories.session_id`` is nullable, so handing the writer ``None``
records an honest NULL.

The mint also produced a *second* silent defect: the session-keyed Redis writes
(``{agent}:cache:{session_id}``, ``session:{session_id}:messages``) were keyed on
a uuid nobody could ever ask for again — a write-only key burning its TTL. With
no session there is nothing to key on, so those writes are skipped and counted
as not cached.

These tests double only the hooks' own storage methods; the ``contribute_to_memory``
control flow under test is real.
"""

from __future__ import annotations

import importlib
import inspect
from contextlib import ExitStack
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, patch

import pytest


@dataclass(frozen=True)
class HookCase:
    """One agent's ``contribute_to_memory`` and the writes it drives."""

    #: Dotted module path of the agent's ``memory_hooks`` module.
    module: str
    #: Name of the hooks class ``contribute_to_memory`` accepts.
    hooks_cls: str
    #: The episodic writer method that must receive ``session_id=None``.
    episodic: str
    #: ``result`` payload that clears the hook's skip gate.
    result: Dict[str, Any]
    #: ``state`` payload that clears the hook's skip gate.
    state: Dict[str, Any] = field(default_factory=dict)
    #: Session-KEYED writes that must be skipped entirely when there is no session.
    skipped: tuple[str, ...] = ()
    #: Other hook methods to double so the test stays hermetic.
    others: tuple[str, ...] = ()
    #: Cache writes whose key does NOT embed the session. They must still happen
    #: when there is no session — guarding them would break a live read/write pair.
    unkeyed_cache: tuple[str, ...] = ()
    #: Counter keys that must stay 0 because their write was skipped.
    zero_counts: tuple[str, ...] = ()
    #: Writes NOT keyed on the session: they must still run and record a null
    #: session field rather than being skipped.
    null_passthrough: tuple[str, ...] = ()
    #: True when the hook also calls the module-level ``persist_agent_activity``.
    activity: bool = False

    @property
    def agent(self) -> str:
        return self.module.split(".")[-2]


_AGENTS = "src.agents"
_ML = "src.agents.ml_foundation"

# One row per site found by ``grep -rn "uuid4()" src/agents --include=memory_hooks.py``
# minus feedback_learner's two rows, which mint an ``example_id`` / ``signal_id``
# rather than a session id.
CASES: List[HookCase] = [
    HookCase(
        module=f"{_AGENTS}.causal_impact.memory_hooks",
        hooks_cls="CausalImpactMemoryHooks",
        episodic="store_causal_analysis",
        result={"status": "success"},
        skipped=("cache_causal_analysis",),
        others=("store_causal_path",),
        zero_counts=("working_cached",),
        activity=True,
    ),
    HookCase(
        module=f"{_AGENTS}.cohort_constructor.memory_hooks",
        hooks_cls="CohortConstructorMemoryHooks",
        episodic="store_cohort_result",
        result={"status": "success", "cohort_id": "c1"},
        skipped=("cache_cohort_result",),
        others=("store_cohort_pattern", "store_eligibility_rule"),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_AGENTS}.experiment_monitor.memory_hooks",
        hooks_cls="ExperimentMonitorMemoryHooks",
        episodic="store_monitoring_check",
        result={"alerts": []},
        # ``cache_monitoring_status`` keys on ``experiment_monitor:status:{ids}``,
        # not on the session, so it must still run without one.
        unkeyed_cache=("cache_monitoring_status",),
        others=("store_alert", "cache_alert"),
    ),
    HookCase(
        module=f"{_AGENTS}.explainer.memory_hooks",
        hooks_cls="ExplanationMemoryHooks",
        episodic="store_explanation",
        result={},
        skipped=("cache_explanation",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_AGENTS}.gap_analyzer.memory_hooks",
        hooks_cls="GapAnalyzerMemoryHooks",
        episodic="store_gap_analysis",
        result={},
        skipped=("cache_gap_analysis",),
        zero_counts=("working_cached",),
        activity=True,
    ),
    HookCase(
        module=f"{_AGENTS}.health_score.memory_hooks",
        hooks_cls="HealthScoreMemoryHooks",
        episodic="store_health_check",
        result={},
        # ``cache_health_check`` keys on ``health_score:cache:{check_scope}`` — the
        # very key ``_get_cached_health`` reads back by scope on every run — so it
        # must still run without a session.
        unkeyed_cache=("cache_health_check",),
    ),
    HookCase(
        module=f"{_AGENTS}.heterogeneous_optimizer.memory_hooks",
        hooks_cls="HeterogeneousOptimizerMemoryHooks",
        episodic="store_cate_analysis",
        result={"status": "success"},
        skipped=("cache_cate_analysis",),
        others=("store_segment_profiles",),
        zero_counts=("working_cached",),
        activity=True,
    ),
    HookCase(
        module=f"{_AGENTS}.orchestrator.memory_hooks",
        hooks_cls="OrchestratorMemoryHooks",
        episodic="store_orchestration",
        result={"status": "completed", "query_id": "q1", "response_text": "r"},
        state={"query": "what drives trx?"},
        # Both are session-KEYED Redis writes: the cache key embeds the session and
        # ``store_conversation_turn`` writes ``session:{session_id}:messages``.
        skipped=("cache_orchestration_result", "store_conversation_turn"),
        # ``track_routing_decision`` is NOT session-keyed: it lpushes onto the
        # global ``orchestrator:routing_decisions`` list and carries the session
        # only as a payload field, where a null is honest. It still runs.
        null_passthrough=("track_routing_decision",),
        zero_counts=("working_cached", "conversation_stored"),
    ),
    HookCase(
        module=f"{_AGENTS}.prediction_synthesizer.memory_hooks",
        hooks_cls="PredictionSynthesizerMemoryHooks",
        episodic="store_prediction",
        result={},
        state={"entity_id": "e1", "entity_type": "hcp", "prediction_target": "trx"},
        # ``cache_prediction`` is HALF-keyed: an entity key with no session in it
        # (read back by ``_get_cached_predictions``) plus a session-keyed copy. It
        # must still be called; it guards only the session copy internally. The
        # key-level pin lives in ``test_cache_prediction_*`` below.
        unkeyed_cache=("cache_prediction",),
    ),
    HookCase(
        module=f"{_AGENTS}.resource_optimizer.memory_hooks",
        hooks_cls="ResourceOptimizerMemoryHooks",
        episodic="store_optimization",
        result={},
        skipped=("cache_optimization",),
        others=("store_optimization_pattern",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_AGENTS}.tool_composer.memory_hooks",
        hooks_cls="ToolComposerMemoryHooks",
        episodic="store_composition",
        result={"composition_id": "c1", "success": False},
        skipped=("cache_composition_result",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.data_preparer.memory_hooks",
        hooks_cls="DataPreparerMemoryHooks",
        episodic="store_qc_report",
        result={},
        skipped=("cache_qc_report",),
        others=("store_data_quality_pattern",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.feature_analyzer.memory_hooks",
        hooks_cls="FeatureAnalyzerMemoryHooks",
        episodic="store_feature_analysis",
        result={},
        skipped=("cache_feature_analysis",),
        others=("store_feature_importance_patterns",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.model_deployer.memory_hooks",
        hooks_cls="ModelDeployerMemoryHooks",
        episodic="store_deployment",
        result={"status": "completed"},
        skipped=("cache_deployment_manifest",),
        others=("store_deployment_pattern",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.model_selector.memory_hooks",
        hooks_cls="ModelSelectorMemoryHooks",
        episodic="store_model_selection",
        result={},
        skipped=("cache_model_selection",),
        others=("store_algorithm_pattern",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.model_trainer.memory_hooks",
        hooks_cls="ModelTrainerMemoryHooks",
        episodic="store_training_result",
        result={},
        skipped=("cache_training_result",),
        others=("store_model_pattern",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.observability_connector.memory_hooks",
        hooks_cls="ObservabilityConnectorMemoryHooks",
        episodic="store_observability_event",
        result={},
        skipped=("cache_metrics",),
        others=("store_health_snapshot",),
        zero_counts=("working_cached",),
    ),
    HookCase(
        module=f"{_ML}.scope_definer.memory_hooks",
        hooks_cls="ScopeDefinerMemoryHooks",
        episodic="store_scope_definition",
        result={},
        state={"validation_passed": True},
        skipped=("cache_scope_definition",),
        others=("store_experiment_pattern",),
        zero_counts=("working_cached",),
    ),
]

_IDS = [c.agent for c in CASES]


async def _contribute(case: HookCase, session_id: Optional[str] = None) -> tuple:
    """Run ``case``'s hook with every storage method doubled.

    Returns ``(counts, mocks)``. ``session_id`` is omitted from the call entirely
    when None, which is the session-less shape under test.
    """
    mod = importlib.import_module(case.module)
    hooks = getattr(mod, case.hooks_cls)()

    mocks: Dict[str, AsyncMock] = {case.episodic: AsyncMock(return_value="mem-2076")}
    # A skipped write returns True so a MISSING guard would show up as a non-zero
    # counter, not merely as an extra call.
    for name in case.skipped:
        mocks[name] = AsyncMock(return_value=True)
    for name in case.others:
        mocks[name] = AsyncMock(return_value=None)
    for name in case.null_passthrough:
        mocks[name] = AsyncMock(return_value=True)
    # Returns True so a WRONGLY guarded write shows up as working_cached == 0.
    for name in case.unkeyed_cache:
        mocks[name] = AsyncMock(return_value=True)

    kwargs: Dict[str, Any] = {"result": case.result, "memory_hooks": hooks}
    params = inspect.signature(mod.contribute_to_memory).parameters
    if "state" in params:
        kwargs["state"] = case.state
    if session_id is not None:
        kwargs["session_id"] = session_id

    with ExitStack() as stack:
        for name, mock in mocks.items():
            stack.enter_context(patch.object(hooks, name, mock))
        if case.activity:
            mocks["persist_agent_activity"] = stack.enter_context(
                patch.object(mod, "persist_agent_activity", return_value=None)
            )
        counts = await mod.contribute_to_memory(**kwargs)

    return counts, mocks


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_IDS)
async def test_absent_session_reaches_episodic_writer_as_none(case: HookCase):
    """No session id in, ``session_id=None`` out — never a minted uuid."""
    _, mocks = await _contribute(case)

    episodic = mocks[case.episodic]
    episodic.assert_awaited_once()
    assert "session_id" in episodic.await_args.kwargs, (
        f"{case.agent}: the episodic writer must receive session_id by keyword"
    )
    assert episodic.await_args.kwargs["session_id"] is None, (
        f"{case.agent}: episodic write received "
        f"{episodic.await_args.kwargs['session_id']!r} — a session id was invented"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "case", [c for c in CASES if c.skipped], ids=[c.agent for c in CASES if c.skipped]
)
async def test_absent_session_skips_session_keyed_writes(case: HookCase):
    """A Redis key built from the session is not written when there is no session."""
    counts, mocks = await _contribute(case)

    for name in case.skipped:
        mocks[name].assert_not_awaited()
    for key in case.zero_counts:
        assert counts[key] == 0, f"{case.agent}: {key} counted a write that was skipped"


@pytest.mark.asyncio
@pytest.mark.parametrize("case", CASES, ids=_IDS)
async def test_present_session_is_passed_through_unchanged(case: HookCase):
    """The change is scoped to the session-less path: a real id still flows through."""
    session_id = "46d40f52-39ac-4b79-b3a4-1f1292059a00~eeba22e7-4d9d-49ea-977b-b9e9d1549c53"
    counts, mocks = await _contribute(case, session_id=session_id)

    assert mocks[case.episodic].await_args.kwargs["session_id"] == session_id
    for name in case.skipped:
        mocks[name].assert_awaited()
    for key in case.zero_counts:
        assert counts[key] == 1, f"{case.agent}: {key} must still count a real session's write"


_UNKEYED = [c for c in CASES if c.unkeyed_cache]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _UNKEYED, ids=[c.agent for c in _UNKEYED])
async def test_absent_session_still_writes_unkeyed_caches(case: HookCase):
    """A cache whose key does not embed the session is NOT collateral of #2076.

    ``health_score`` is the live example: the route calls ``check_health(scope=...)``
    with no session at all, and ``_get_cached_health`` reads the same
    ``health_score:cache:{check_scope}`` key back on the next run. Guarding that
    write on the session would silently stop every production run from filling a
    cache it still reads.
    """
    counts, mocks = await _contribute(case)

    for name in case.unkeyed_cache:
        mocks[name].assert_awaited_once()
    assert counts["working_cached"] == 1, (
        f"{case.agent}: a cache not keyed on the session must still be written"
    )


_NULL_THROUGH = [c for c in CASES if c.null_passthrough or c.activity]


@pytest.mark.asyncio
@pytest.mark.parametrize("case", _NULL_THROUGH, ids=[c.agent for c in _NULL_THROUGH])
async def test_absent_session_still_runs_unkeyed_writes_with_a_null_field(case: HookCase):
    """A write that carries the session as a PAYLOAD field keeps running.

    ``track_routing_decision`` lpushes onto a global list and the activity writers
    put the id in ``input_data``; neither builds a key from it. Skipping them would
    throw away real signal, so they run and record an honest null.
    """
    _, mocks = await _contribute(case)

    for name in case.null_passthrough:
        mocks[name].assert_awaited_once()
        assert mocks[name].await_args.kwargs["session_id"] is None

    if case.activity:
        activity = mocks["persist_agent_activity"]
        activity.assert_called_once()
        assert activity.call_args.kwargs["input_data"]["session_id"] is None


# ===========================================================================
# prediction_synthesizer: the HALF-keyed cache, pinned at the Redis-key level
# ===========================================================================


class _FakeRedis:
    """Records every setex key so the two halves can be told apart."""

    def __init__(self) -> None:
        self.keys: List[str] = []

    async def setex(self, key: str, ttl: int, value: str) -> None:
        self.keys.append(key)


def _hooks_with_fake_redis():
    from src.agents.prediction_synthesizer.memory_hooks import PredictionSynthesizerMemoryHooks

    redis = _FakeRedis()
    working = AsyncMock()
    working.get_client = AsyncMock(return_value=redis)
    hooks = PredictionSynthesizerMemoryHooks()
    hooks._working_memory = working
    return hooks, redis


_ENTITY_KEY = "prediction_synthesizer:entity:hcp:e1:trx"


@pytest.mark.asyncio
async def test_cache_prediction_without_session_still_writes_the_entity_key():
    """The entity cache carries no session, so it must not be collateral of #2076."""
    hooks, redis = _hooks_with_fake_redis()

    ok = await hooks.cache_prediction(
        session_id=None,
        entity_id="e1",
        entity_type="hcp",
        prediction_target="trx",
        prediction_result={"p": 1},
    )

    assert ok is True
    assert _ENTITY_KEY in redis.keys, "the session-less run starved the entity cache"
    assert not [k for k in redis.keys if k.startswith("prediction_synthesizer:session:")], (
        f"a session-keyed copy was written without a session: {redis.keys}"
    )


@pytest.mark.asyncio
async def test_cache_prediction_with_session_writes_both_keys():
    """With a session, both halves are written exactly as before."""
    hooks, redis = _hooks_with_fake_redis()

    await hooks.cache_prediction(
        session_id="s-2076",
        entity_id="e1",
        entity_type="hcp",
        prediction_target="trx",
        prediction_result={"p": 1},
    )

    assert redis.keys == [_ENTITY_KEY, "prediction_synthesizer:session:s-2076"]
