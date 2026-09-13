"""#2020 Task 4c: a composition phase that throws must not put library exception text in the answer.

``ToolComposer.compose`` turns a raised phase error into ``answer=f"Unable to complete analysis:
{error}"`` and ``caveats=[error]``. The three phase classes used to be built at their catch-alls as
``f"Failed to decompose query: {e}"`` and alike, so whatever an LLM client, pydantic or a driver
raised reached the user verbatim; the composer's own last arm did the same with ``f"Unexpected
error: {e}"``. Library text now goes to the log with the exception, and the error carries a fixed
authored sentence with the original kept as ``__cause__``.

An authored error raised deeper in the same ``try`` passes through its catch-all unchanged: "Too
few sub-questions", "Unknown tool in plan", "unbound column" and the plan-graph problems are
findings about the query or the plan, and wrapping them in the generic sentence would erase them.
"""

from __future__ import annotations

import ast
import json
import logging
import types
from pathlib import Path
from typing import List
from unittest.mock import AsyncMock

import pytest

import src.agents.tool_composer.decomposer as decomposer_module
import src.agents.tool_composer.planner as planner_module
from src.agents.tool_composer.agent import ToolComposerAgent
from src.agents.tool_composer.composer import ToolComposer
from src.agents.tool_composer.decomposer import DecompositionError, QueryDecomposer
from src.agents.tool_composer.executor import ExecutionError, PlanExecutor
from src.agents.tool_composer.models.composition_models import DecompositionResult, SubQuestion
from src.agents.tool_composer.planner import PlanningError, ToolPlanner
from tests.unit.ast_guards import caught_exception_interpolations

SENTINEL = "LIBTEXT_SENTINEL"
REPO_ROOT = Path(__file__).resolve().parents[4]

DECOMPOSE_FAILED = "the query could not be broken into sub-questions because of an internal error"
PLAN_FAILED = "no execution plan could be built because of an internal error"
EXECUTE_FAILED = "the plan stopped on an internal error"
INVALID_JSON = "Invalid JSON in LLM response"
UNEXPECTED = "Unexpected error: the analysis could not be completed."
AGENT_FAILED = "Tool composition failed: the analysis could not be completed."

PHASE_ERRORS = frozenset({"DecompositionError", "PlanningError", "ExecutionError"})
PHASE_MODULES = ("decomposer.py", "planner.py", "executor.py")


def _one_question_decomposition() -> DecompositionResult:
    return DecompositionResult(
        original_query="what drove it?",
        sub_questions=[
            SubQuestion(id="sq_1", question="q1", intent="CAUSAL", entities=[], depends_on=[])
        ],
        decomposition_reasoning="t",
    )


def _planner(mock_llm_client, mock_tool_registry) -> ToolPlanner:
    return ToolPlanner(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        use_episodic_memory=False,
        enable_caching=False,
    )


def _json_decode_raiser(*_args, **_kwargs):
    raise json.JSONDecodeError(SENTINEL, "doc", 0)


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------


async def test_decomposer_library_exception_is_logged_not_raised(mock_llm_client, caplog):
    original = RuntimeError(SENTINEL)
    mock_llm_client.set_error(original)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(DecompositionError) as exc_info:
            await QueryDecomposer(llm_client=mock_llm_client).decompose("q")
    assert str(exc_info.value) == DECOMPOSE_FAILED
    assert SENTINEL in caplog.text
    assert exc_info.value.__cause__ is original


async def test_decomposer_invalid_json_keeps_prefix_only(mock_llm_client, monkeypatch, caplog):
    fake_json = types.SimpleNamespace(
        loads=_json_decode_raiser, JSONDecodeError=json.JSONDecodeError
    )
    monkeypatch.setattr(decomposer_module, "json", fake_json)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(DecompositionError) as exc_info:
            await QueryDecomposer(llm_client=mock_llm_client).decompose("q")
    assert str(exc_info.value) == INVALID_JSON
    assert SENTINEL in caplog.text
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


async def test_decomposer_authored_error_passes_through_unwrapped(mock_llm_client):
    mock_llm_client.set_decomposition_response(
        json.dumps({"reasoning": "t", "sub_questions": [{"id": "sq_1", "question": "q1"}]})
    )
    with pytest.raises(DecompositionError) as exc_info:
        await QueryDecomposer(llm_client=mock_llm_client).decompose("q")
    assert str(exc_info.value) == "Too few sub-questions: 1 < 2"


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


async def test_planner_library_exception_is_logged_not_raised(
    mock_llm_client, mock_tool_registry, sample_decomposition, caplog
):
    original = RuntimeError(SENTINEL)
    mock_llm_client.set_error(original)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(PlanningError) as exc_info:
            await _planner(mock_llm_client, mock_tool_registry).plan(sample_decomposition)
    assert str(exc_info.value) == PLAN_FAILED
    assert SENTINEL in caplog.text
    assert exc_info.value.__cause__ is original


async def test_planner_invalid_json_keeps_prefix_only(
    mock_llm_client, mock_tool_registry, sample_decomposition, monkeypatch, caplog
):
    monkeypatch.setattr(planner_module, "parse_llm_json", _json_decode_raiser)
    with caplog.at_level(logging.WARNING):
        with pytest.raises(PlanningError) as exc_info:
            await _planner(mock_llm_client, mock_tool_registry).plan(sample_decomposition)
    assert str(exc_info.value) == INVALID_JSON
    assert SENTINEL in caplog.text
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


async def test_planner_authored_error_passes_through_unwrapped(mock_llm_client, mock_tool_registry):
    mock_llm_client.set_planning_response(
        json.dumps(
            {
                "reasoning": "t",
                "tool_mappings": [],
                "execution_steps": [
                    {"step_id": "step_1", "sub_question_id": "sq_1", "tool_name": "ghost_tool"}
                ],
            }
        )
    )
    with pytest.raises(PlanningError) as exc_info:
        await _planner(mock_llm_client, mock_tool_registry).plan(_one_question_decomposition())
    assert str(exc_info.value) == "Unknown tool in plan: ghost_tool"


async def test_planner_duplicate_step_ids_named_without_pydantic_text(
    mock_llm_client, mock_tool_registry
):
    # Without the planner's own check this reached the user as pydantic's ValidationError text,
    # including ``input_value=`` with the whole LLM plan and a pydantic docs URL.
    duplicate = {
        "step_id": "step_1",
        "sub_question_id": "sq_1",
        "tool_name": "causal_effect_estimator",
        "input_mapping": {},
        "depends_on_steps": [],
    }
    mock_llm_client.set_planning_response(
        json.dumps(
            {
                "reasoning": "t",
                "tool_mappings": [
                    {"sub_question_id": "sq_1", "tool_name": "causal_effect_estimator"}
                ],
                "execution_steps": [duplicate, dict(duplicate)],
                "parallel_groups": [["step_1"]],
            }
        )
    )
    with pytest.raises(PlanningError) as exc_info:
        await _planner(mock_llm_client, mock_tool_registry).plan(_one_question_decomposition())
    message = str(exc_info.value)
    assert message == "duplicate step id(s) in plan: ['step_1']"
    assert "validation error" not in message and "input_value" not in message


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


async def test_executor_library_exception_is_logged_not_raised(
    mock_tool_registry, sample_execution_plan, monkeypatch, caplog
):
    original = RuntimeError(SENTINEL)
    executor = PlanExecutor(tool_registry=mock_tool_registry, enable_caching=False, max_retries=0)
    monkeypatch.setattr(executor, "_execute_step", AsyncMock(side_effect=original))
    with caplog.at_level(logging.WARNING):
        with pytest.raises(ExecutionError) as exc_info:
            await executor.execute(sample_execution_plan)
    assert str(exc_info.value) == EXECUTE_FAILED
    assert SENTINEL in caplog.text
    assert exc_info.value.__cause__ is original


async def test_executor_plan_graph_problem_named_and_no_step_runs(
    mock_tool_registry, sample_execution_plan, monkeypatch
):
    sample_execution_plan.steps[0].depends_on_steps.append("step_2")
    executor = PlanExecutor(tool_registry=mock_tool_registry, enable_caching=False, max_retries=0)
    step = AsyncMock()
    monkeypatch.setattr(executor, "_execute_step", step)
    with pytest.raises(ExecutionError) as exc_info:
        await executor.execute(sample_execution_plan)
    assert str(exc_info.value) == "dependency cycle among plan steps"
    assert step.await_count == 0


# ---------------------------------------------------------------------------
# Composer and agent: every surface a failed result carries
# ---------------------------------------------------------------------------


def _surfaces(result) -> List[str]:
    return [
        result.response.answer,
        *result.response.caveats,
        *result.errors,
        result.error or "",
        result.decomposition.decomposition_reasoning,
        result.plan.planning_reasoning,
    ]


def _composer(mock_llm_client, mock_tool_registry) -> ToolComposer:
    return ToolComposer(
        llm_client=mock_llm_client,
        tool_registry=mock_tool_registry,
        enable_memory_contribution=False,
    )


async def test_composer_non_phase_exception_uses_fixed_sentence(
    mock_llm_client, mock_tool_registry, monkeypatch, caplog
):
    composer = _composer(mock_llm_client, mock_tool_registry)
    monkeypatch.setattr(
        composer.decomposer, "decompose", AsyncMock(side_effect=RuntimeError(SENTINEL))
    )
    with caplog.at_level(logging.WARNING):
        result = await composer.compose("q")
    assert result.success is False
    assert result.error == UNEXPECTED
    assert result.response.answer == f"Unable to complete analysis: {UNEXPECTED}"
    assert not [s for s in _surfaces(result) if SENTINEL in s]
    assert SENTINEL in caplog.text


def _fail_decompose(composer: ToolComposer, llm_client, monkeypatch) -> None:
    llm_client.set_error(RuntimeError(SENTINEL))


def _fail_plan(composer: ToolComposer, llm_client, monkeypatch) -> None:
    # Not set_error: that would fail decompose first.
    monkeypatch.setattr(
        composer.planner, "_call_llm", AsyncMock(side_effect=RuntimeError(SENTINEL))
    )


def _fail_execute(composer: ToolComposer, llm_client, monkeypatch) -> None:
    monkeypatch.setattr(
        composer.executor, "_execute_step", AsyncMock(side_effect=RuntimeError(SENTINEL))
    )


@pytest.mark.parametrize(
    ("inject", "expected_error"),
    [
        (_fail_decompose, f"Decomposition failed: {DECOMPOSE_FAILED}"),
        (_fail_plan, f"Planning failed: {PLAN_FAILED}"),
        (_fail_execute, f"Execution failed: {EXECUTE_FAILED}"),
    ],
    ids=["decompose", "plan", "execute"],
)
async def test_composer_library_text_inside_a_phase_stays_out(
    mock_llm_client, mock_tool_registry, monkeypatch, caplog, inject, expected_error
):
    composer = _composer(mock_llm_client, mock_tool_registry)
    inject(composer, mock_llm_client, monkeypatch)
    with caplog.at_level(logging.WARNING):
        result = await composer.compose("q")
    assert result.error == expected_error
    assert not [s for s in _surfaces(result) if SENTINEL in s]
    assert SENTINEL in caplog.text


async def test_composer_authored_phase_error_keeps_its_text(mock_llm_client, mock_tool_registry):
    mock_llm_client.set_decomposition_response(
        json.dumps({"reasoning": "t", "sub_questions": [{"id": "sq_1", "question": "q1"}]})
    )
    result = await _composer(mock_llm_client, mock_tool_registry).compose("q")
    assert result.error == "Decomposition failed: Too few sub-questions: 1 < 2"
    assert result.response.answer == f"Unable to complete analysis: {result.error}"


async def test_agent_output_error_is_a_fixed_sentence(mock_llm_client, monkeypatch, caplog):
    # ToolComposerOutput.error reaches the chat answer: the orchestrator's dispatcher does not
    # fail tool_composer closed on status, and its synthesizer stringifies an output whose
    # ``response`` is empty (dispatcher.py ``_agent_failed``; synthesizer.py ``_extract_response``).
    agent = ToolComposerAgent(llm_client=mock_llm_client)

    def _raise() -> None:
        raise RuntimeError(SENTINEL)

    monkeypatch.setattr(agent, "_ensure_composer", _raise)
    with caplog.at_level(logging.WARNING):
        output = await agent.run({"query": "q"})
    assert output.success is False
    assert output.error == AGENT_FAILED
    assert SENTINEL not in repr(output.to_dict())
    assert SENTINEL in caplog.text


# ---------------------------------------------------------------------------
# AST guard: no phase error interpolates the exception its handler caught
# (the helper's own self-tests are in tests/unit/test_ast_guards.py)
# ---------------------------------------------------------------------------


def test_no_phase_error_interpolates_a_caught_exception():
    package = REPO_ROOT / "src" / "agents" / "tool_composer"
    violations = {
        name: caught_exception_interpolations(ast.parse((package / name).read_text()), PHASE_ERRORS)
        for name in PHASE_MODULES
    }
    assert violations == {name: [] for name in PHASE_MODULES}
