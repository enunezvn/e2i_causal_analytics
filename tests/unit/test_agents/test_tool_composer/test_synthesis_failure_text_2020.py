"""#2020 / D6: the synthesis prompt applies the same refusal-text rule as the fail-closed answer.

A PARTIAL success does not fail closed: it is synthesized, and ``_format_results`` wrote
``Error: {result.output.error}`` for every failed step. Raw DoWhy / sklearn / driver text therefore
reached the synthesis LLM and, through it, the answer — the leak Task 4 closed on the fail-closed
path. Rendered here through the REAL ``_format_results`` (the #2019 seam); no LLM call is made.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pytest

from src.agents.tool_composer.models.composition_models import (
    DecompositionResult,
    ExecutionStatus,
    ExecutionTrace,
    StepResult,
    SubQuestion,
    SynthesisInput,
    ToolInput,
    ToolOutput,
)
from src.agents.tool_composer.reason_codes import (
    TOOL_AUTHORED_CLASSES,
    ReasonCode,
    canonical_sentence,
    user_safe_failure_text,
)
from src.agents.tool_composer.synthesizer import ResponseSynthesizer
from tests.unit.test_agents.test_tool_composer.test_fail_closed_answer_sanitization_2020 import (
    _DOWHY_INTERNALS,
    _LEAKS,
    _REFUSAL,
    _fail_closed,
)

_SYNTHESIZER_LOGGER = "src.agents.tool_composer.synthesizer"
_UNCODED = "SENTINEL refusal text that no refusal arm produced"


def _step(
    tool: str,
    *,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    outcome_class: Optional[str] = None,
    reason_code: Optional[str] = None,
) -> StepResult:
    now = datetime.now(timezone.utc)
    succeeded = result is not None
    return StepResult(
        step_id=f"s_{tool}",
        sub_question_id=f"sq_{tool}",
        tool_name=tool,
        input=ToolInput(tool_name=tool, parameters={}),
        output=ToolOutput(tool_name=tool, success=succeeded, result=result, error=error),
        status=ExecutionStatus.COMPLETED if succeeded else ExecutionStatus.FAILED,
        started_at=now,
        completed_at=now,
        outcome_class=outcome_class,
        reason_code=reason_code,
    )


def _prompt(*steps: StepResult) -> str:
    trace = ExecutionTrace(plan_id="plan_2020_synthesis")
    sub_questions: List[SubQuestion] = []
    for step in steps:
        sub_questions.append(
            SubQuestion(
                id=step.sub_question_id,
                question=f"What does {step.tool_name} report?",
                intent="CAUSAL",
            )
        )
        trace.add_result(step)
    query = "Which payer tiers respond to copay support?"
    synthesis_input = SynthesisInput(
        original_query=query,
        decomposition=DecompositionResult(
            original_query=query, sub_questions=sub_questions, decomposition_reasoning="test"
        ),
        execution_trace=trace,
    )
    return ResponseSynthesizer(llm_client=object())._format_results(synthesis_input)


def _block(prompt: str, tool: str) -> str:
    """The prompt text for one tool's step."""
    for block in prompt.split("## Sub-Question:"):
        if f"\nTool: {tool}\n" in block:
            return block
    raise AssertionError(f"no block for {tool!r} in the prompt")


def _sentence(code: str) -> str:
    return f"{canonical_sentence(code)} [{code}]"


def _canonical_line(code: ReasonCode) -> str:
    return f"Error: {_sentence(code.value)}"


# ---------------------------------------------------------------------------
# The synthesis prompt
# ---------------------------------------------------------------------------


def test_a_partial_success_prompt_keeps_the_refusal_and_withholds_library_text(caplog):
    with caplog.at_level(logging.WARNING, logger=_SYNTHESIZER_LOGGER):
        prompt = _prompt(
            _step("segment_ranker", result={"recommended_targets": ["payer_tier_2"]}),
            _step(
                "causal_effect_estimator",
                error=_DOWHY_INTERNALS,
                outcome_class="error",
                reason_code="tool_error",
            ),
            _step(
                "gap_calculator",
                error=_REFUSAL,
                outcome_class="refused",
                reason_code="coverage_gap",
            ),
            _step("mystery_tool", error=_UNCODED, outcome_class="refused", reason_code=None),
        )

    for leak in _LEAKS:
        assert leak not in prompt, f"{leak!r} reached the synthesis prompt"
    assert "SENTINEL" not in prompt, "an uncoded refusal's text is not trusted"
    # The tool-authored refusal is a finding, not a crash: it stays verbatim.
    assert f"Error: {_REFUSAL}" in _block(prompt, "gap_calculator")
    # Both untrusted steps render the canonical sentence and its code.
    assert _canonical_line(ReasonCode.TOOL_ERROR) in _block(prompt, "causal_effect_estimator")
    assert _canonical_line(ReasonCode.TOOL_ERROR) in _block(prompt, "mystery_tool")
    # The successful step is untouched.
    assert "payer_tier_2" in _block(prompt, "segment_ranker")
    # The raw text is withheld from the prompt, not lost.
    assert _DOWHY_INTERNALS in caplog.text
    assert _UNCODED in caplog.text


def test_the_withheld_text_is_logged_exactly_once(caplog):
    """The helper is pure and the caller logs, so the synthesizer path must not log twice."""
    with caplog.at_level(logging.DEBUG):
        _prompt(
            _step(
                "causal_effect_estimator",
                error=_DOWHY_INTERNALS,
                outcome_class="error",
                reason_code="tool_error",
            )
        )
    hits = [r for r in caplog.records if _DOWHY_INTERNALS in r.getMessage()]
    assert len(hits) == 1, f"expected one log record carrying the raw text, got {len(hits)}"
    assert hits[0].name == _SYNTHESIZER_LOGGER


def test_a_coded_step_without_text_renders_its_codes_sentence():
    prompt = _prompt(
        _step(
            "sensitivity_analyzer", outcome_class="dependency_unmet", reason_code="dependency_unmet"
        )
    )
    assert _canonical_line(ReasonCode.DEPENDENCY_UNMET) in _block(prompt, "sensitivity_analyzer")


def test_a_failed_step_with_neither_text_nor_a_code_adds_no_error_line():
    prompt = _prompt(_step("mystery_tool", error="", outcome_class=None, reason_code=None))
    assert "Error:" not in prompt
    assert "Status: FAILED" in prompt


# ---------------------------------------------------------------------------
# The shared rule
# ---------------------------------------------------------------------------

_TIMEOUT_TEXT = "exceeded the 120s step budget"
_INPUT_TEXT = "input contract violation: counterfactual_simulator: expected_effect is None"

# (outcome_class, reason_code, raw, expected fragment, expected withheld raw)
_RULE = [
    pytest.param("refused", "coverage_gap", _REFUSAL, _REFUSAL, None, id="coded-refusal-verbatim"),
    # The signature accepts the enum member as well as its string value.
    pytest.param(
        "refused", ReasonCode.COVERAGE_GAP, _REFUSAL, _REFUSAL, None, id="coded-refusal-enum-member"
    ),
    pytest.param(
        "input_rejected",
        "missing_required_input",
        _INPUT_TEXT,
        _INPUT_TEXT,
        None,
        id="coded-input-rejection-verbatim",
    ),
    # The coded-error constructor fails soft to tool_error; the text is still tool-authored.
    pytest.param("refused", "tool_error", _REFUSAL, _REFUSAL, None, id="refusal-with-tool_error"),
    pytest.param(
        "error",
        "tool_error",
        _DOWHY_INTERNALS,
        _sentence("tool_error"),
        _DOWHY_INTERNALS,
        id="library-text-withheld",
    ),
    pytest.param(
        "timeout",
        "tool_timeout",
        _TIMEOUT_TEXT,
        _sentence("tool_timeout"),
        _TIMEOUT_TEXT,
        id="timeout-own-sentence",
    ),
    pytest.param(
        "refused", None, _UNCODED, _sentence("tool_error"), _UNCODED, id="uncoded-refusal"
    ),
    pytest.param(
        "refused",
        "not_a_real_code",
        _UNCODED,
        _sentence("tool_error"),
        _UNCODED,
        id="refusal-with-unknown-code",
    ),
    pytest.param(
        "error", "<script>x", "boom", _sentence("tool_error"), "boom", id="unknown-code-not-echoed"
    ),
    pytest.param(
        "refused", "coverage_gap", "", _sentence("coverage_gap"), None, id="authored-empty-text"
    ),
    pytest.param(
        "refused", "coverage_gap", "   ", _sentence("coverage_gap"), None, id="whitespace-is-empty"
    ),
    pytest.param(None, None, "", None, None, id="no-text-no-code"),
    pytest.param(None, None, None, None, None, id="none-text-no-code"),
]


@pytest.mark.parametrize("outcome_class,reason_code,raw,fragment,withheld", _RULE)
def test_the_rule(outcome_class, reason_code, raw, fragment, withheld):
    assert user_safe_failure_text(outcome_class, reason_code, raw) == (fragment, withheld)


def test_the_rule_is_pure(caplog):
    """No prefix and no logging: each caller prefixes and logs, so nothing is logged twice."""
    with caplog.at_level(logging.DEBUG):
        for case in _RULE:
            user_safe_failure_text(*case.values[:3])
    assert caplog.records == []


def test_only_refused_and_input_rejected_are_tool_authored():
    assert TOOL_AUTHORED_CLASSES == frozenset({"refused", "input_rejected"})


@pytest.mark.parametrize("outcome_class,reason_code,raw,fragment,withheld", _RULE)
def test_the_prompt_and_the_fail_closed_answer_cannot_drift(
    outcome_class, reason_code, raw, fragment, withheld
):
    """Both surfaces say the same thing about the same step."""
    step = _step("probe_tool", error=raw, outcome_class=outcome_class, reason_code=reason_code)
    block = _block(_prompt(step), "probe_tool")
    answer = _fail_closed(step).response.answer
    if fragment is None:
        assert "Error:" not in block
        assert "Reason(s):" not in answer
    else:
        assert f"Error: {fragment}" in block
        assert f"probe_tool: {fragment}" in answer
    if withheld is not None:
        # Withheld sentinels must stay distinctive, never substrings of a canonical sentence.
        # The fragment REPLACES the raw text: a surface that appended it would pass the checks above.
        assert withheld not in block
        assert withheld not in answer
