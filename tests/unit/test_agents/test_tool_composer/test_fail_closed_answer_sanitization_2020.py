"""#2020: the fail-closed answer keeps honest refusals and drops library internals.

Calls ``_create_total_failure_result`` directly with REAL decomposition and plan objects (both
are required fields of ``CompositionResult``), built the way ``test_fail_closed_zero_tools_f6``
builds them. No LLM call is made on this path.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from src.agents.tool_composer import tool_registrations as tr
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
from src.agents.tool_composer.reason_codes import ReasonCode, canonical_sentence

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
_ERROR_TYPES = {"refused": "ToolRefusalError", "input_rejected": "ToolInputError"}


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
        error_type=_ERROR_TYPES.get(outcome_class or "", "RuntimeError"),
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


def _rendered(tool: str, code: ReasonCode) -> str:
    return f"{tool}: {canonical_sentence(code)} [{code.value}]"


def _user_visible_text(result: CompositionResult) -> str:
    # Every field a consumer can surface. ``execution.step_results[].output.error`` is deliberately
    # excluded: it keeps the raw text, and no consumer surfaces it.
    response = result.response
    return " || ".join(
        [
            response.answer,
            *response.caveats,
            *response.citations,
            str(response.supporting_data),
            *result.errors,
            str(result.error),
        ]
    )


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
    assert _rendered("causal_effect_estimator", ReasonCode.TOOL_ERROR) in result.response.answer
    assert _DOWHY_INTERNALS in caplog.text, "the raw text must still be logged"


def test_a_timeout_renders_its_own_sentence():
    result = _fail_closed(
        _step("refutation_runner", "exceeded the 120s step budget", "timeout", "tool_timeout")
    )
    assert _rendered("refutation_runner", ReasonCode.TOOL_TIMEOUT) in result.response.answer
    assert "120s" not in _user_visible_text(result)


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
    assert _rendered("mystery_tool", ReasonCode.TOOL_ERROR) in result.response.answer


def test_the_failed_components_are_unchanged():
    result = _fail_closed(
        _step("gap_calculator", _REFUSAL, "refused", "coverage_gap"),
        _step("causal_effect_estimator", _DOWHY_INTERNALS, "error", "tool_error"),
    )
    assert result.response.failed_components == ["gap_calculator", "causal_effect_estimator"]


def test_a_failed_step_with_neither_text_nor_a_code_adds_no_reason_fragment():
    """Nothing to withhold and nothing to say: no canonical sentence is invented for a step
    that carried no text and no code."""
    result = _fail_closed(_step("mystery_tool", "", None, None))
    assert "Reason(s):" not in result.response.answer
    assert result.response.failed_components == ["mystery_tool"]


def test_verbatim_refusals_are_joined_before_canonical_fragments_so_truncation_spares_them():
    """Canonical fragments are often LONGER than the raw executor text they replace, and #1574's
    pathological refusal sits close to the 2000-char carry budget. The answer is truncated from
    the END, so a refusal joined after the canonical fragments loses its estimation_data_scope."""
    reason = tr._gap_comparability_reason(
        entity_type="territory",
        group_col="territory",
        groups_present=[f"territory_{i}_" + "x" * 200 for i in range(400)],
        groups_matched=[],
        entities=[f"requested_{i}_" + "y" * 200 for i in range(400)],
        row_count=10**9,
    )
    refusal = f"gap_calculator: {reason}"
    result = _fail_closed(
        _step("causal_effect_estimator", "x", "error", "tool_error"),
        _step("power_calculator", "x", "error", "tool_error"),
        _step(
            "sensitivity_analyzer", "upstream step failed", "dependency_unmet", "dependency_unmet"
        ),
        _step("gap_calculator", reason, "refused", "coverage_gap"),
    )
    answer = result.response.answer
    # The budget really is exceeded, so this exercises truncation rather than passing under it.
    assert answer.endswith("…(truncated)")
    assert "estimation_data_scope=" in answer
    # The WHOLE refusal survives: the scope marker alone can survive a cut inside the scope.
    assert refusal in answer, "truncation cut into the tool-authored refusal"
    # caveats and errors carry the reasons in the same order as the answer.
    expected_order = [
        "gap_calculator",
        "causal_effect_estimator",
        "power_calculator",
        "sensitivity_analyzer",
    ]
    assert [c.split(":", 1)[0] for c in result.response.caveats[1:]] == expected_order
    assert [e.split(":", 1)[0] for e in result.errors[1:]] == expected_order


def test_a_trusted_step_with_empty_text_renders_its_codes_sentence():
    """One rule: verbatim only when trusted AND non-empty, otherwise the code's sentence."""
    result = _fail_closed(_step("gap_calculator", "", "refused", "coverage_gap"))
    assert _rendered("gap_calculator", ReasonCode.COVERAGE_GAP) in result.response.answer


def test_an_unknown_reason_code_renders_as_tool_error_in_sentence_and_tag():
    """``StepResult.reason_code`` is a plain string, so a value outside the closed set must not
    print raw: both the sentence and the tag fall back to tool_error."""
    result = _fail_closed(_step("mystery_tool", "boom", "error", "<script>x"))
    assert _rendered("mystery_tool", ReasonCode.TOOL_ERROR) in result.response.answer
    assert "<script>" not in _user_visible_text(result)


def test_a_refused_step_carrying_tool_error_stays_verbatim():
    """The coded-error constructor fails soft to tool_error on a code outside the closed set; the
    text is still tool-authored, so the pairing keeps the refusal verbatim."""
    result = _fail_closed(_step("gap_calculator", _REFUSAL, "refused", "tool_error"))
    assert _REFUSAL in result.response.answer


def test_an_input_rejection_without_a_code_fails_closed():
    result = _fail_closed(
        _step("counterfactual_simulator", "raw input text", "input_rejected", None)
    )
    assert "raw input text" not in _user_visible_text(result)
    assert _rendered("counterfactual_simulator", ReasonCode.TOOL_ERROR) in result.response.answer


def test_an_uncoded_error_with_text_renders_tool_error():
    result = _fail_closed(_step("causal_effect_estimator", _DOWHY_INTERNALS, "error", None))
    visible = _user_visible_text(result)
    for leak in _LEAKS:
        assert leak not in visible
    assert _rendered("causal_effect_estimator", ReasonCode.TOOL_ERROR) in result.response.answer
