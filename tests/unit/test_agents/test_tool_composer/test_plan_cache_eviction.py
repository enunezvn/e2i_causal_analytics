"""A plan that failed is not reused as if it had worked (spec §7.3).

The in-process plan cache hands a structurally similar decomposition (Jaccard of intents and
entities plus dependency similarity >= 0.8) the steps of an earlier plan and skips the LLM
planner, for 15 minutes. It cached every plan at planning time, before execution, so a plan whose
composition then failed was served again. Now:

- ``ExecutionPlan.plan_source`` says which path built the plan (``llm`` / ``plan_cache`` /
  ``kpi_deterministic``) and ``plan_cache_key`` the signature it was stored or matched under;
- after execution the composer evicts that key when every executed tool failed, or when any step
  was a plan defect or named an unregistered tool;
- a successful plan, and a partial one without a defect, stay cached (G6's intent: skip planning
  for similar work);
- the deterministic KPI plan never touches the cache.

Real ToolPlanner, real ToolComposerCacheManager, real DecompositionResult objects that meet the
cache's eligibility; no LLM call on the cached path.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd
import pytest

from src.agents.tool_composer.cache import get_cache_manager
from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    ToolInput,
    ToolMapping,
    ToolOutput,
)
from src.agents.tool_composer.planner import ToolPlanner


def _decomposition(entities_first: List[str], extra: Optional[str] = None) -> DecompositionResult:
    second = ["TRx"] + ([extra] if extra else [])
    return DecompositionResult(
        original_query="what drove Kisqali TRx and which segments respond",
        sub_questions=[
            SubQuestion(id="sq_1", question="drivers?", intent="CAUSAL", entities=entities_first),
            SubQuestion(
                id="sq_2",
                question="segments?",
                intent="COMPARATIVE",
                entities=second,
                depends_on=["sq_1"],
            ),
        ],
        decomposition_reasoning="t",
    )


D1 = _decomposition(["Kisqali"])
D2 = _decomposition(["Kisqali"], extra="Q2")


def _plan(decomposition: DecompositionResult) -> ExecutionPlan:
    steps = [
        ExecutionStep(
            step_id="step_1",
            sub_question_id="sq_1",
            tool_name="causal_effect_estimator",
            source_agent="causal_impact",
            input_mapping={"treatment": "accepted", "outcome": "converted"},
        ),
        ExecutionStep(
            step_id="step_2",
            sub_question_id="sq_2",
            tool_name="cate_analyzer",
            source_agent="heterogeneous_optimizer",
            input_mapping={"effect": "$step_1.effect", "dimension": "region"},
            depends_on_steps=["step_1"],
        ),
    ]
    return ExecutionPlan(
        decomposition=decomposition,
        steps=steps,
        tool_mappings=[
            ToolMapping(
                sub_question_id=s.sub_question_id,
                tool_name=s.tool_name,
                source_agent=s.source_agent,
                confidence=0.9,
                reasoning="t",
            )
            for s in steps
        ],
        parallel_groups=[["step_1"], ["step_2"]],
        planning_reasoning="cached",
    )


def _trace(plan: ExecutionPlan, classes: List[str]) -> ExecutionTrace:
    trace = ExecutionTrace(plan_id=plan.plan_id)
    started = datetime.now(timezone.utc)
    for step, cls in zip(plan.steps, classes, strict=True):
        ok = cls in ("succeeded", "cache_hit")
        trace.add_result(
            StepResult(
                step_id=step.step_id,
                sub_question_id=step.sub_question_id,
                tool_name=step.tool_name,
                input=ToolInput(tool_name=step.tool_name, parameters={}),
                output=ToolOutput(
                    tool_name=step.tool_name, success=ok, result={"ok": True} if ok else None
                ),
                status=ExecutionStatus.COMPLETED if ok else ExecutionStatus.FAILED,
                started_at=started,
                completed_at=started + timedelta(milliseconds=5),
                outcome_class=cls,
            )
        )
    return trace


@pytest.fixture
def planner(mock_llm_client, mock_tool_registry) -> ToolPlanner:
    return ToolPlanner(
        llm_client=mock_llm_client, tool_registry=mock_tool_registry, use_episodic_memory=False
    )


@pytest.fixture
def composer(mock_llm_client, mock_tool_registry) -> ToolComposer:
    return ToolComposer(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        enable_memory_contribution=False,
    )


def _key(decomposition: DecompositionResult) -> str:
    cache = get_cache_manager().plan_cache
    return cache._hash_signature(cache._extract_signature(decomposition))


def test_fixture_decompositions_meet_the_cache_eligibility():
    # Precondition: without it every eviction assertion below would be vacuous.
    cache = get_cache_manager().plan_cache
    sig1, sig2 = cache._extract_signature(D1), cache._extract_signature(D2)
    assert cache._compute_similarity(sig1, sig2) >= cache.similarity_threshold
    assert _key(D1) != _key(D2)
    assert D1.question_count == D2.question_count == len(_plan(D1).steps)


def test_cached_adaptation_carries_matched_key(planner):
    p1 = _plan(D1)
    get_cache_manager().cache_plan(D1, p1)
    adapted = planner._try_cached_plan(D2, None, None, None)
    assert adapted is not None
    assert adapted.plan_source == "plan_cache"
    assert adapted.plan_cache_key == _key(D1)
    assert [s.model_dump() for s in adapted.steps] == [s.model_dump() for s in p1.steps]


def test_failed_composition_evicts_and_next_lookup_misses(planner, composer):
    get_cache_manager().cache_plan(D1, _plan(D1))
    adapted = planner._try_cached_plan(D2, None, None, None)
    assert adapted is not None
    composer._after_execution(adapted, _trace(adapted, ["error", "timeout"]))
    assert planner._try_cached_plan(D2, None, None, None) is None
    assert get_cache_manager().get_similar_plan(D1) is None


@pytest.mark.parametrize("defect", ["not_registered", "plan_defect"])
def test_a_defect_step_evicts_even_when_another_step_succeeded(planner, composer, defect):
    get_cache_manager().cache_plan(D1, _plan(D1))
    adapted = planner._try_cached_plan(D2, None, None, None)
    composer._after_execution(adapted, _trace(adapted, ["succeeded", defect]))
    assert planner._try_cached_plan(D2, None, None, None) is None


@pytest.mark.parametrize(
    "classes",
    [["succeeded", "succeeded"], ["succeeded", "refused"], ["cache_hit", "dependency_unmet"]],
)
def test_success_or_partial_without_defect_keeps_cache(planner, composer, classes):
    get_cache_manager().cache_plan(D1, _plan(D1))
    adapted = planner._try_cached_plan(D2, None, None, None)
    composer._after_execution(adapted, _trace(adapted, classes))
    kept = planner._try_cached_plan(D2, None, None, None)
    assert kept is not None and kept.plan_cache_key == _key(D1)


def test_eviction_is_exact_and_idempotent(composer):
    other = _decomposition(["Fabhalta"])
    get_cache_manager().cache_plan(D1, _plan(D1))
    get_cache_manager().cache_plan(other, _plan(other))
    failed = _plan(D1).model_copy(update={"plan_source": "plan_cache", "plan_cache_key": _key(D1)})
    composer._after_execution(failed, _trace(failed, ["error", "error"]))
    composer._after_execution(failed, _trace(failed, ["error", "error"]))
    assert get_cache_manager().get_similar_plan(D1) is None
    assert get_cache_manager().get_similar_plan(other) is not None


async def test_llm_plan_is_cached_under_its_key(planner, mock_llm_client, sample_decomposition):
    plan = await planner.plan(sample_decomposition)
    assert plan.plan_source == "llm"
    assert plan.plan_cache_key == _key(sample_decomposition)
    matched = get_cache_manager().get_similar_plan(sample_decomposition)
    assert matched is not None and matched[0].plan_id == plan.plan_id


async def test_uncached_planner_leaves_no_key(
    mock_llm_client, mock_tool_registry, sample_decomposition
):
    planner = ToolPlanner(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        use_episodic_memory=False,
        enable_caching=False,
    )
    plan = await planner.plan(sample_decomposition)
    assert (plan.plan_source, plan.plan_cache_key) == ("llm", None)


def _kpi_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "converted": [0, 1] * 20,
            "accepted": [1, 0] * 20,
            "confidence_score": [0.5 + 0.01 * i for i in range(40)],
            "delivery_channel": ["email", "crm", "phone", "portal"] * 10,
        }
    )


def test_kpi_plan_is_deterministic_and_never_touches_the_cache(composer):
    context = {"estimation_data": _kpi_frame(), "kpi_outcome": "converted"}
    plan = composer._build_kpi_causal_plan(D1, context, "converted")
    assert plan is not None
    assert (plan.plan_source, plan.plan_cache_key) == ("kpi_deterministic", None)
    assert get_cache_manager().get_similar_plan(D1) is None

    get_cache_manager().cache_plan(D2, _plan(D2))
    composer._after_execution(plan, _trace(_plan(D1), ["error", "error"]))  # nothing to evict
    assert get_cache_manager().get_similar_plan(D2) is not None
