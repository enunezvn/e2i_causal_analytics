"""#2020: the fail-closed answer keeps honest refusals and drops library internals.

Calls ``_create_total_failure_result`` directly with REAL decomposition and plan objects (both
are required fields of ``CompositionResult``), built the way ``test_fail_closed_zero_tools_f6``
builds them. No LLM call is made on this path.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.models.composition_models import (
    CompositionResult,
    DecompositionResult,
    ExecutionPlan,
    ExecutionStatus,
    ExecutionTrace,
    StepResult,
    ToolInput,
    ToolOutput,
)

_COMPOSER_LOGGER = "src.agents.tool_composer.composer"

# The shape #2020 measured on the deployed image: causal_effect_estimator's RuntimeError carrying
# the DoWhy pipeline's own error text for an all-null treatment column.
_DOWHY_INTERNALS = (
    "causal_effect_estimator: DoWhy pipeline failed errors=[{'library': 'dowhy', 'error': "
    "\"DoWhy estimate_effect failed for method_name='backdoor.linear_regression': Found array "
    'with 0 sample(s) (shape=(0,)) while a minimum of 1 is required."}]'
)
_LEAKS = ("DoWhy", "dowhy", "backdoor.linear_regression", "shape=(0,)", "array with 0 sample")
_REFUSAL = (
    "gap_calculator: the estimation data covers only brand 'Kisqali'; a brand-vs-brand gap "
    "needs at least two brands. Refusing to report a gap against a single brand."
)


def _step(
    tool: str, error: str, outcome_class: Optional[str], reason_code: Optional[str]
) -> StepResult:
    now = datetime.now(timezone.utc)
    return StepResult(
        step_id=f"s_{tool}",
        sub_question_id="sq_1",
        tool_name=tool,
        input=ToolInput(tool_name=tool, parameters={}),
        output=ToolOutput(tool_name=tool, success=False, error=error),
        status=ExecutionStatus.FAILED,
        started_at=now,
        completed_at=now,
        outcome_class=outcome_class,
        attempts=1,
        error_type="ToolRefusalError" if outcome_class == "refused" else "RuntimeError",
        reason_code=reason_code,
    )


def _fail_closed(*steps: StepResult) -> CompositionResult:
    decomposition = DecompositionResult(
        original_query="q", sub_questions=[], decomposition_reasoning="r"
    )
    plan = ExecutionPlan(
        decomposition=decomposition, steps=[], tool_mappings=[], planning_reasoning="r"
    )
    trace = ExecutionTrace(plan_id=plan.plan_id)
    for step in steps:
        trace.add_result(step)
    composer = ToolComposer(llm_client=object(), enable_memory_contribution=False)
    return composer._create_total_failure_result(
        "q", decomposition, plan, trace, datetime.now(timezone.utc), {}
    )


def _user_visible_text(result: CompositionResult) -> str:
    return " || ".join([result.response.answer, *result.response.caveats, *result.errors])


def test_a_tool_authored_refusal_reaches_the_answer_verbatim():
    result = _fail_closed(_step("gap_calculator", _REFUSAL, "refused", "coverage_gap"))
    assert _REFUSAL in result.response.answer


def test_an_input_rejection_reaches_the_answer_verbatim():
    text = "input contract violation: counterfactual_simulator: expected_effect is None"
    result = _fail_closed(
        _step("counterfactual_simulator", text, "input_rejected", "missing_required_input")
    )
    assert text in result.response.answer


def test_library_internals_reach_no_user_visible_field_but_do_reach_the_log(caplog):
    with caplog.at_level(logging.WARNING, logger=_COMPOSER_LOGGER):
        result = _fail_closed(
            _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error")
        )
    visible = _user_visible_text(result)
    for leak in _LEAKS:
        assert leak not in visible, f"{leak!r} leaked into a user-visible field"
    assert (
        "causal_effect_estimator: the tool failed to complete [tool_error]"
        in result.response.answer
    )
    assert _DOWHY_INTERNALS in caplog.text, "the raw text must still be logged"


def test_a_timeout_renders_its_own_sentence():
    result = _fail_closed(
        _step("refutation_runner", "exceeded the 120s step budget", "timeout", "tool_timeout")
    )
    assert (
        "refutation_runner: the tool exceeded its time budget [tool_timeout]"
        in result.response.answer
    )
    assert "120s" not in result.response.answer


def test_a_mixed_composition_keeps_the_refusal_and_drops_the_internals():
    result = _fail_closed(
        _step("gap_calculator", _REFUSAL, "refused", "coverage_gap"),
        _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"),
    )
    assert _REFUSAL in result.response.answer
    visible = _user_visible_text(result)
    for leak in _LEAKS:
        assert leak not in visible


def test_a_refused_step_without_a_code_fails_closed():
    """After Task 3 the refusal arm always sets a code, so a refused step without one did not
    come from that arm and its text is not trusted."""
    result = _fail_closed(_step("mystery_tool", "raw text from somewhere", "refused", None))
    assert "raw text from somewhere" not in _user_visible_text(result)
    assert "mystery_tool: the tool failed to complete [tool_error]" in result.response.answer


def test_the_failed_components_are_unchanged():
    result = _fail_closed(
        _step("gap_calculator", _REFUSAL, "refused", "coverage_gap"),
        _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"),
    )
    assert result.response.failed_components == ["gap_calculator", "causal_effect_estimator"]


def test_a_failed_step_with_neither_text_nor_a_code_adds_no_reason_fragment():
    """Nothing to withhold and nothing to say: no canonical sentence is invented for a step
    that carried no text and no code, mirroring #1574's no-reason contract."""
    result = _fail_closed(_step("mystery_tool", "", None, None))
    assert result.response.answer == (
        "Unable to complete analysis: All 1 tool(s) failed; no analysis could be "
        "completed. Returning a failed result rather than a fabricated answer."
    )
    assert "[tool_error]" not in _user_visible_text(result)
    assert "mystery_tool:" not in _user_visible_text(result)
    assert result.response.caveats == [result.error]
    assert result.response.failed_components == ["mystery_tool"]
