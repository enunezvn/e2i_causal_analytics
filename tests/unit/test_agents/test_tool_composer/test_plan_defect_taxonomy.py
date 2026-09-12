"""#2045 + #2024 — the executor must discover a bad PLAN cheaply, not expensively.

Both issues are the same defect class: the planner emits a malformed plan and the
executor pays tool-invocation cost to find out.

* **#2045** — the planner omits a tool's REQUIRED arguments. The call raises a plain
  ``TypeError``, which falls into the generic retry arm: 3 identical doomed
  invocations (each holding a bounded heavy-compute slot for a sync tool), then the
  circuit breaker opens and blocks OTHER, valid steps that use the same tool. A
  missing required argument is deterministic over the step's resolved inputs — no
  retry can fix it — and it is a PLAN defect, not a signal about the tool's health.

* **#2024** — the planner attaches ``depends_on_steps`` that the step never binds.
  F5 skips the step with "dependency unmet" even though it consumed nothing from the
  failed upstream. Measured: ``counterfactual_simulator``'s registry pairs with
  ``causal_effect_estimator`` / ``cate_analyzer`` / ``gap_calculator`` are
  ordering-only — none of its parameters (``brand``, ``intervention``,
  ``target_entities``) is an output field of any of the three.

The taxonomy both land in: the executor could not legitimately CALL the tool because
the PLAN is wrong -> ``plan_defect`` (joining the #1573 unresolvable-reference case),
never retried, never charged to the circuit breaker. A genuine unmet DATA dependency
stays ``dependency_unmet``. A tool that RAN and rejected its input stays
``input_rejected``.

Falsifiability: every test below names the exact behaviour it fails on today.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer.executor import CircuitState, PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    DependencyType,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionStep,
    SubQuestion,
    ToolMapping,
)
from src.tool_registry.registry import ToolParameter, ToolRegistry, ToolSchema

# ---------------------------------------------------------------------------
# Fixtures: a registry of tools whose SIGNATURES are the thing under test.
# ---------------------------------------------------------------------------


def _register(registry: ToolRegistry, name: str, fn: Any, params: List[ToolParameter]) -> None:
    registry.register(
        schema=ToolSchema(
            name=name,
            description=f"{name} (test)",
            source_agent="gap_analyzer",
            tier=2,
            input_parameters=params,
            output_schema="Dict[str, Any]",
            avg_execution_ms=10,
        ),
        callable=fn,
    )


def _make_registry(invoked: Dict[str, int]) -> ToolRegistry:
    """Mirrors the real shapes measured in the registry (2026-09-12, 20 tools).

    ``strict_tool`` mirrors ``gap_calculator(metric, entity_type, entities, **kwargs)``
    — three required positional params, no defaults, plus a ``**kwargs`` catch-all.
    """
    registry = ToolRegistry()
    registry.clear()

    def strict_tool(
        metric: str, entity_type: str, entities: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        invoked["strict_tool"] = invoked.get("strict_tool", 0) + 1
        return {"gap": 1.0, "metric": metric, "entity_type": entity_type, "entities": entities}

    def failing_tool(**kwargs: Any) -> Dict[str, Any]:
        invoked["failing_tool"] = invoked.get("failing_tool", 0) + 1
        raise RuntimeError("upstream boom")

    def ordering_only_tool(
        intervention: str, brand: str, target_entities: Optional[List[str]] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Mirrors ``counterfactual_simulator`` after #2015: needs only its own args."""
        invoked["ordering_only_tool"] = invoked.get("ordering_only_tool", 0) + 1
        return {"simulated_lift": 0.05, "brand": brand, "intervention": intervention}

    _register(
        registry,
        "strict_tool",
        strict_tool,
        [
            ToolParameter("metric", "str", "metric", True),
            ToolParameter("entity_type", "str", "entity type", True),
            ToolParameter("entities", "List[str]", "entities", True),
        ],
    )
    _register(registry, "failing_tool", failing_tool, [])
    _register(
        registry,
        "ordering_only_tool",
        ordering_only_tool,
        [
            ToolParameter("intervention", "str", "intervention", True),
            ToolParameter("brand", "str", "brand", True),
        ],
    )
    return registry


def _plan(steps: List[ExecutionStep], groups: List[List[str]]) -> ExecutionPlan:
    sub_ids = sorted({s.sub_question_id for s in steps})
    return ExecutionPlan(
        decomposition=DecompositionResult(
            original_query="plan-defect taxonomy?",
            sub_questions=[
                SubQuestion(id=i, question=i, intent="GAP", entities=[], depends_on=[])
                for i in sub_ids
            ],
            decomposition_reasoning="t",
            timestamp=datetime.now(timezone.utc),
        ),
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
        estimated_duration_ms=100,
        parallel_groups=groups,
        planning_reasoning="t",
        timestamp=datetime.now(timezone.utc),
    )


def _step(
    step_id: str,
    tool: str,
    mapping: Dict[str, Any],
    depends: Optional[List[str]] = None,
    sq: str = "sq_1",
) -> ExecutionStep:
    return ExecutionStep(
        step_id=step_id,
        sub_question_id=sq,
        tool_name=tool,
        source_agent="gap_analyzer",
        input_mapping=mapping,
        dependency_type=DependencyType.SEQUENTIAL,
        depends_on_steps=depends or [],
    )


def _executor(registry: ToolRegistry, **kw: Any) -> PlanExecutor:
    # backoff_base_delay=0: the retry sleeps are not what these tests measure.
    return PlanExecutor(tool_registry=registry, enable_caching=False, backoff_base_delay=0.0, **kw)


def _count_dispatches(executor: PlanExecutor) -> Dict[str, int]:
    """Count hand-offs of a SYNC tool call to the bounded heavy-compute pool.

    Counting calls INSIDE the tool body is vacuous for this defect: a missing
    required argument raises ``TypeError`` during argument binding, so the body
    never runs and a body counter stays 0 even across three dispatches. What
    #2045 actually costs is the dispatch — a bounded-pool slot and a retry cycle
    per attempt — so that is what these tests measure.
    """
    counts: Dict[str, int] = {}
    original = executor._run_sync_tool

    async def counting(tool_callable: Any, resolved_inputs: Dict[str, Any], tool_name: str) -> Any:
        counts[tool_name] = counts.get(tool_name, 0) + 1
        return await original(tool_callable, resolved_inputs, tool_name)

    executor._run_sync_tool = counting  # type: ignore[method-assign]
    return counts


# ---------------------------------------------------------------------------
# #2045 — a missing required argument is a PLAN defect, discovered before the call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_required_arguments_never_invoke_the_tool() -> None:
    """RED today: ``strict_tool()`` raises TypeError INSIDE the retry loop, so the
    tool is invoked max_retries+1 = 3 times and the step is classed ``error``.
    The fix must detect the arity mismatch BEFORE invocation: 0 invocations,
    ``attempts == 0``, ``outcome_class == 'plan_defect'``."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=2)
    dispatched = _count_dispatches(executor)

    plan = _plan([_step("step_1", "strict_tool", {})], [["step_1"]])
    trace = await executor.execute(plan, context={})
    result = trace.get_result("step_1")

    assert dispatched.get("strict_tool", 0) == 0, (
        "a step missing required arguments must NEVER be dispatched to the tool; "
        f"dispatched {dispatched.get('strict_tool', 0)}x"
    )
    assert result.status == ExecutionStatus.FAILED
    assert result.outcome_class == "plan_defect", (
        f"a planner-omitted argument is a plan defect, got {result.outcome_class!r}"
    )
    assert result.attempts == 0, f"no attempt was made against the tool, got {result.attempts}"


@pytest.mark.asyncio
async def test_missing_required_arguments_are_named_in_the_user_visible_reason() -> None:
    """The synthesis-visible reason must name the tool and EVERY missing parameter —
    not a bare Python ``TypeError`` repr. RED today: the message is whatever
    CPython produced inside the generic arm, recorded after 3 retries."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=2)
    dispatched = _count_dispatches(executor)

    # The planner supplied ONE of the three required arguments.
    plan = _plan([_step("step_1", "strict_tool", {"metric": "trx"})], [["step_1"]])
    trace = await executor.execute(plan, context={})
    error = trace.get_result("step_1").output.error or ""

    # RED today: CPython's TypeError happens to name them too — but only after the
    # call was dispatched 3 times. The reason must come from the SIGNATURE instead.
    assert dispatched.get("strict_tool", 0) == 0, (
        f"the reason must be derived from the signature, not from a dispatched call; "
        f"dispatched {dispatched.get('strict_tool', 0)}x"
    )
    assert "strict_tool" in error, f"reason must name the tool; got {error!r}"
    for missing in ("entity_type", "entities"):
        assert missing in error, f"reason must name missing argument {missing!r}; got {error!r}"
    assert "metric" not in error.split("supplied")[0], (
        f"a SUPPLIED argument must not be reported missing; got {error!r}"
    )


@pytest.mark.asyncio
async def test_missing_required_arguments_do_not_open_the_circuit_breaker() -> None:
    """The #2045 live symptom: three defective ``gap_calculator`` steps opened the
    breaker, so a LATER valid step using the same tool was skipped without running.

    RED today: 3 defective steps x 3 attempts trips the failure threshold and
    step_4 comes back ``circuit_open`` with the tool never invoked."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=2, circuit_failure_threshold=3)
    dispatched = _count_dispatches(executor)

    valid = {"metric": "trx", "entity_type": "territory", "entities": ["T1"]}
    plan = _plan(
        [
            _step("step_1", "strict_tool", {}, sq="sq_1"),
            _step("step_2", "strict_tool", {}, sq="sq_2"),
            _step("step_3", "strict_tool", {}, sq="sq_3"),
            _step("step_4", "strict_tool", valid, sq="sq_4"),
        ],
        [["step_1"], ["step_2"], ["step_3"], ["step_4"]],
    )
    trace = await executor.execute(plan, context={})

    stats = executor.failure_tracker.get_stats("strict_tool")
    assert stats is None or stats.circuit_breaker.state == CircuitState.CLOSED, (
        "a plan defect says nothing about the tool's health and must not open its circuit"
    )
    step_4 = trace.get_result("step_4")
    assert step_4.status == ExecutionStatus.COMPLETED, (
        f"a VALID step using the same tool must still run; got {step_4.status!r} / "
        f"{step_4.outcome_class!r}: {step_4.output.error!r}"
    )
    assert dispatched.get("strict_tool", 0) == 1, (
        f"only the one valid step may reach the tool; the three defective steps cost "
        f"nine doomed dispatches today. Got {dispatched.get('strict_tool', 0)}"
    )
    assert invoked.get("strict_tool", 0) == 1


@pytest.mark.asyncio
async def test_a_complete_argument_set_still_runs_including_extra_kwargs() -> None:
    """False-positive guard: the arity check must not reject calls that work today.
    ``strict_tool`` has ``**kwargs``, so surplus keys are legal."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=0)

    mapping = {
        "metric": "trx",
        "entity_type": "territory",
        "entities": ["T1", "T2"],
        "surplus": "absorbed by **kwargs",
    }
    trace = await executor.execute(
        _plan([_step("step_1", "strict_tool", mapping)], [["step_1"]]), context={}
    )
    result = trace.get_result("step_1")

    assert result.status == ExecutionStatus.COMPLETED, (
        f"a complete argument set must still execute; got {result.output.error!r}"
    )
    assert result.outcome_class == "succeeded"
    assert invoked.get("strict_tool", 0) == 1


@pytest.mark.asyncio
async def test_a_tool_that_runs_and_rejects_its_input_is_still_input_rejected() -> None:
    """Taxonomy boundary: the arity guard must not swallow the #1573 case. A tool
    whose arguments BIND but whose values it rejects still RUNS and reports
    ``input_rejected`` — the tool's own verdict, not a plan defect."""
    from src.agents.tool_composer.errors import ToolInputError

    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)

    def picky_tool(value: Any, **kwargs: Any) -> Dict[str, Any]:
        invoked["picky_tool"] = invoked.get("picky_tool", 0) + 1
        raise ToolInputError("value must not be None")

    _register(registry, "picky_tool", picky_tool, [ToolParameter("value", "str", "v", True)])
    executor = _executor(registry, max_retries=2)

    trace = await executor.execute(
        _plan([_step("step_1", "picky_tool", {"value": None})], [["step_1"]]), context={}
    )
    result = trace.get_result("step_1")

    assert invoked.get("picky_tool", 0) == 1, "the tool must RUN — its arguments bind"
    assert result.outcome_class == "input_rejected", (
        f"a tool's own input verdict is not a plan defect; got {result.outcome_class!r}"
    )


# ---------------------------------------------------------------------------
# #2024 — F5 skips only a dependent that actually BINDS the failed step's output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ordering_only_dependency_proceeds_when_the_upstream_fails() -> None:
    """RED today: F5 keys off ``depends_on_steps`` alone, so step_2 is SKIPPED with
    "dependency unmet" though its ``input_mapping`` binds nothing from step_1 —
    the #2024 live shape (``counterfactual_simulator`` after #2015)."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=0)

    plan = _plan(
        [
            _step("step_1", "failing_tool", {}, sq="sq_1"),
            _step(
                "step_2",
                "ordering_only_tool",
                {"intervention": "add_reps", "brand": "Kisqali"},
                depends=["step_1"],
                sq="sq_2",
            ),
        ],
        [["step_1"], ["step_2"]],
    )
    trace = await executor.execute(plan, context={})

    assert trace.get_result("step_1").status == ExecutionStatus.FAILED
    step_2 = trace.get_result("step_2")
    assert step_2.status == ExecutionStatus.COMPLETED, (
        "a step that binds NOTHING from the failed upstream needed nothing from it "
        f"and must run; got {step_2.status!r}: {step_2.output.error!r}"
    )
    assert invoked.get("ordering_only_tool", 0) == 1
    # The upstream failure is not hidden: it stays in the trace, where the
    # synthesizer reads it into `failed_components`.
    assert "upstream boom" in (trace.get_result("step_1").output.error or "")


@pytest.mark.asyncio
async def test_a_bound_dependency_is_still_skipped() -> None:
    """Regression guard for the fix's conservatism: a TOP-LEVEL ``$step_1.field``
    reference IS a data dependency, so F5 must still skip — the #1573/F5 contract
    (a dependent must never run on a silently-None upstream)."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=0)

    plan = _plan(
        [
            _step("step_1", "failing_tool", {}, sq="sq_1"),
            _step(
                "step_2",
                "ordering_only_tool",
                {"intervention": "add_reps", "brand": "$step_1.brand"},
                depends=["step_1"],
                sq="sq_2",
            ),
        ],
        [["step_1"], ["step_2"]],
    )
    trace = await executor.execute(plan, context={})
    step_2 = trace.get_result("step_2")

    assert step_2.status == ExecutionStatus.SKIPPED
    assert step_2.outcome_class == "dependency_unmet"
    assert "step_1" in (step_2.output.error or "")
    assert invoked.get("ordering_only_tool", 0) == 0


@pytest.mark.asyncio
async def test_a_nested_binding_is_still_skipped() -> None:
    """A reference NESTED in a dict/list is still a binding: ``_resolve_inputs``
    degrades it to None (lenient contract), which is exactly the silently-None
    upstream F5 exists to prevent. The fix must walk the whole mapping, not just
    its top level — otherwise it reopens the #1573 crash on a nested construction."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=0)

    plan = _plan(
        [
            _step("step_1", "failing_tool", {}, sq="sq_1"),
            _step(
                "step_2",
                "ordering_only_tool",
                {
                    "intervention": "add_reps",
                    "brand": "Kisqali",
                    "target_entities": ["T1", "$step_1.top_performer"],
                },
                depends=["step_1"],
                sq="sq_2",
            ),
        ],
        [["step_1"], ["step_2"]],
    )
    trace = await executor.execute(plan, context={})
    step_2 = trace.get_result("step_2")

    assert step_2.status == ExecutionStatus.SKIPPED, (
        f"a reference nested in a list still binds the failed step; got {step_2.status!r}"
    )
    assert invoked.get("ordering_only_tool", 0) == 0


@pytest.mark.asyncio
async def test_a_mixed_dependency_skips_on_the_bound_one_only() -> None:
    """Two unmet upstreams, one bound and one ordering-only: the step is skipped,
    and the reason names ONLY the dependency it actually needed. Reporting the
    ordering-only one as the cause would send the learning loop after the wrong
    step."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=0)

    plan = _plan(
        [
            _step("step_1", "failing_tool", {}, sq="sq_1"),
            _step("step_2", "failing_tool", {}, sq="sq_2"),
            _step(
                "step_3",
                "ordering_only_tool",
                {"intervention": "add_reps", "brand": "$step_2.brand"},
                depends=["step_1", "step_2"],
                sq="sq_3",
            ),
        ],
        [["step_1", "step_2"], ["step_3"]],
    )
    trace = await executor.execute(plan, context={})
    step_3 = trace.get_result("step_3")

    assert step_3.status == ExecutionStatus.SKIPPED
    error = step_3.output.error or ""
    assert "step_2" in error, f"the BOUND unmet dependency must be named; got {error!r}"
    assert "step_1" not in error, (
        f"an ordering-only dependency is not why the step could not run; got {error!r}"
    )


@pytest.mark.asyncio
async def test_the_executor_never_injects_a_frame_a_tool_cannot_accept() -> None:
    """Attribution guard for the #2045 arity check.

    ``_maybe_autopopulate_dataframe`` used to inject ``estimation_data`` into every
    tool, resting on a docstring claim measured FALSE on 2026-09-12: 2 of the 20
    registered tools (``detect_structural_drift``, ``model_inference``) declare no
    ``**kwargs``. Without Gate 0 the arity check would report the executor's OWN
    injection as a plan defect — blaming the planner for the executor's kwarg and
    evicting a perfectly good cached plan. The tool must simply run."""
    import pandas as pd

    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)

    def no_kwargs_tool(baseline: str) -> Dict[str, Any]:
        """Mirrors detect_structural_drift / model_inference: no **kwargs."""
        invoked["no_kwargs_tool"] = invoked.get("no_kwargs_tool", 0) + 1
        return {"drift": 0.0, "baseline": baseline}

    _register(
        registry,
        "no_kwargs_tool",
        no_kwargs_tool,
        [ToolParameter("baseline", "str", "baseline", True)],
    )
    executor = _executor(registry, max_retries=2)

    trace = await executor.execute(
        _plan([_step("step_1", "no_kwargs_tool", {"baseline": "2026-01"})], [["step_1"]]),
        context={"estimation_data": pd.DataFrame({"a": [1, 2]})},
    )
    result = trace.get_result("step_1")

    assert result.status == ExecutionStatus.COMPLETED, (
        f"a tool without **kwargs must not be handed an injected frame; "
        f"got {result.outcome_class!r}: {result.output.error!r}"
    )
    assert invoked.get("no_kwargs_tool", 0) == 1
    assert "estimation_data" not in result.input.parameters


@pytest.mark.asyncio
async def test_the_two_defects_compose_into_one_taxonomy() -> None:
    """A step with an ordering-only unmet dependency AND missing required arguments
    proceeds past F5 (#2024) and is then caught by the arity guard (#2045) — one
    ``plan_defect``, zero invocations, no breaker charge. Without the #2045 half,
    the #2024 half would hand the retry storm a step F5 used to absorb."""
    invoked: Dict[str, int] = {}
    registry = _make_registry(invoked)
    executor = _executor(registry, max_retries=2)
    dispatched = _count_dispatches(executor)

    plan = _plan(
        [
            _step("step_1", "failing_tool", {}, sq="sq_1"),
            _step("step_2", "strict_tool", {"metric": "trx"}, depends=["step_1"], sq="sq_2"),
        ],
        [["step_1"], ["step_2"]],
    )
    trace = await executor.execute(plan, context={})
    step_2 = trace.get_result("step_2")

    assert step_2.outcome_class == "plan_defect", (
        f"the real defect is the missing arguments, got {step_2.outcome_class!r}"
    )
    assert dispatched.get("strict_tool", 0) == 0
    stats = executor.failure_tracker.get_stats("strict_tool")
    assert stats is None or stats.circuit_breaker.state == CircuitState.CLOSED
