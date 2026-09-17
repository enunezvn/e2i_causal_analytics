"""#2061 — the executor never hands a tool an argument it cannot bind, and a tool
requires every argument it tells the planner is required.

Two invariants over the LIVE registry (every registered tool, so a newly registered
tool is covered without editing this file):

1. **Auto-population binds.** ``PlanExecutor._execute_step`` runs the
   ``_maybe_autopopulate_*`` hooks and merges what they return into the call's kwargs.
   Before Gate 0 (#2045) the DataFrame hook injected ``estimation_data`` into every tool,
   resting on the claim "all composable tools accept ``**kwargs``" — false for
   ``detect_structural_drift`` and ``model_inference``. The call then raised ``TypeError``
   from the executor's own doing. The hooks and their keywords are DERIVED from the
   dispatch source, and the kwargs checked are the ones the executor actually validates
   (and would call with) under a context that satisfies every hook's gates.

2. **Planner-visible means signature-visible.** The #2045 pre-dispatch guard judges a
   call by the callable's signature. Required schema parameters must therefore be
   required by the signature. The optional parameters repaired in #2078 are explicit too,
   so their declared names and defaults cannot drift independently inside ``**kwargs``.

Positive controls guard against a vacuous pass: a first measurement of this registry read
the wrong schema attribute and reported all-zeros, so the registry size, the declared
parameters, the derived hooks and each hook actually firing are all asserted.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import textwrap
from typing import Any, Dict, List, Tuple

import pandas as pd
import pytest

from src.agents.tool_composer import executor as executor_module
from src.agents.tool_composer.executor import PlanExecutor
from src.agents.tool_composer.models.composition_models import (
    DependencyType,
    ExecutionStatus,
    ExecutionStep,
)
from src.tool_registry.registry import ToolParameter, ToolSchema, get_registry
from src.tool_registry.tools.causal_discovery import register_all_discovery_tools
from src.tool_registry.tools.model_inference import register_model_inference_tool
from src.tool_registry.tools.structural_drift import register_structural_drift_tool

_HOOK_PREFIX = "_maybe_autopopulate_"


def _registry():
    registry = get_registry()
    # Idempotent: the discovery / inference / drift tools register outside the composer
    # module, so make sure they are present whatever the import order.
    register_all_discovery_tools()
    register_model_inference_tool()
    register_structural_drift_tool()
    return registry


LIVE_TOOLS = sorted(_registry().list_tools())


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"treatment": [0, 1, 0, 1], "outcome": [1.0, 2.0, 1.5, 2.5], "x1": [1, 2, 3, 4]}
    )


def _hook_keywords() -> Dict[str, str]:
    """``{hook method: keyword it injects}``, read from ``_execute_step`` itself.

    Matches ``autopop = self._maybe_autopopulate_X(...)`` followed by
    ``resolved_inputs = {**resolved_inputs, "<keyword>": autopop}``.
    """
    source = textwrap.dedent(inspect.getsource(PlanExecutor._execute_step))
    tree = ast.parse(source)
    result_of: Dict[str, str] = {}
    keywords: Dict[str, str] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        target, value = node.targets[0], node.value
        if not isinstance(target, ast.Name):
            continue
        if (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and isinstance(value.func.value, ast.Name)
            and value.func.value.id == "self"
            and value.func.attr.startswith(_HOOK_PREFIX)
        ):
            result_of[target.id] = value.func.attr
        elif target.id == "resolved_inputs" and isinstance(value, ast.Dict):
            for key, item in zip(value.keys, value.values, strict=True):
                if (
                    isinstance(key, ast.Constant)
                    and isinstance(key.value, str)
                    and isinstance(item, ast.Name)
                    and item.id in result_of
                ):
                    keywords[result_of[item.id]] = key.value
    return keywords


class _StopBeforeDispatch(Exception):
    pass


class _ProbeRegistry:
    """One tool that declares every hook keyword and accepts any keyword: every hook's
    tool-side gate passes, so whatever does not fire failed on the CONTEXT."""

    def __init__(self, keywords: List[str]) -> None:
        self._schema = ToolSchema(
            name="probe_2061",
            description="probe",
            source_agent="test",
            tier=2,
            input_parameters=[ToolParameter(k, "Any", k, False) for k in keywords],
            output_schema="Dict[str, Any]",
        )

    def get_schema(self, name: str) -> ToolSchema:
        return self._schema

    def get_callable(self, name: str) -> Any:
        return lambda **kwargs: kwargs


def _kwargs_the_executor_would_call(
    tool_name: str, registry: Any = None
) -> Tuple[Any, Dict[str, Any]]:
    """Run the real ``_execute_step`` up to the pre-dispatch validation and capture the
    kwargs it validates — by construction the kwargs the tool would be called with.

    The step binds NOTHING, so every key captured was injected by the executor. The
    context is meant to satisfy every hook's gates (an ``experiment_id``, a DataFrame, a
    role-attribution row that passes the trust gate); the probe test proves it does.
    """
    executor = PlanExecutor(tool_registry=registry or _registry(), enable_caching=False)
    captured: Dict[str, Any] = {}

    def spy(tool_callable: Any, resolved_inputs: Dict[str, Any], name: str) -> None:
        captured["callable"] = tool_callable
        captured["kwargs"] = dict(resolved_inputs)
        raise _StopBeforeDispatch

    executor._validate_call_arguments = spy  # type: ignore[method-assign]
    step = ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name=tool_name,
        source_agent="test",
        input_mapping={},
        dependency_type=DependencyType.SEQUENTIAL,
    )
    context = {"experiment_id": "exp-2061", "estimation_data": _frame()}
    row = {"causal_role": "confounder", "feature": "x1", "source": "manifest"}
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(executor_module, "query_active_role_attributions", lambda _id: [row])
        with pytest.raises(_StopBeforeDispatch):
            asyncio.run(executor._execute_step(step, {}, context))
    return captured["callable"], captured["kwargs"]


@pytest.fixture(scope="module")
def injected_by_tool() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    return {name: _kwargs_the_executor_would_call(name) for name in LIVE_TOOLS}


# ---------------------------------------------------------------------------
# Positive controls: none of the checks below may pass on an empty measurement.
# ---------------------------------------------------------------------------


def test_the_live_registry_is_the_real_one() -> None:
    registry = _registry()
    with_params = [n for n in LIVE_TOOLS if registry.get_schema(n).input_parameters]
    assert len(LIVE_TOOLS) >= 20, LIVE_TOOLS
    assert with_params == LIVE_TOOLS, sorted(set(LIVE_TOOLS) - set(with_params))


def test_every_autopopulate_hook_is_derived_with_its_keyword() -> None:
    hooks = {name for name in vars(PlanExecutor) if name.startswith(_HOOK_PREFIX)}
    assert hooks, "no autopopulate hooks found on PlanExecutor"
    assert set(_hook_keywords()) == hooks, (
        f"hooks whose injection site in _execute_step was not recognised: "
        f"{sorted(hooks - set(_hook_keywords()))}"
    )


def test_the_capture_context_fires_every_hook() -> None:
    """Without this the per-tool check could pass because a hook never fired.

    The probe declares every currently injected keyword, so a failure means the
    all-gates-open context no longer exercises one of the generic hooks."""
    keywords = sorted(set(_hook_keywords().values()))
    _, kwargs = _kwargs_the_executor_would_call("probe_2061", _ProbeRegistry(keywords))
    assert sorted(kwargs) == keywords


def test_nothing_but_a_hook_injects(injected_by_tool) -> None:
    derived = set(_hook_keywords().values())
    observed = {key for _, kwargs in injected_by_tool.values() for key in kwargs}
    assert observed, "no keyword was injected into any live tool"
    assert observed <= derived, (
        f"injected by something other than a hook: {sorted(observed - derived)}"
    )


# ---------------------------------------------------------------------------
# 1. Auto-population never injects a keyword the tool cannot bind.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LIVE_TOOLS)
def test_every_injected_keyword_binds(name, injected_by_tool) -> None:
    tool_callable, kwargs = injected_by_tool[name]
    # The dispatch boundary, independently of the executor's own ``_accepts_keyword``.
    signature = inspect.signature(tool_callable, follow_wrapped=False)
    unbindable: List[str] = []
    for key, value in kwargs.items():
        try:
            signature.bind_partial(**{key: value})
        except TypeError:
            unbindable.append(key)
    assert not unbindable, (
        f"{name}{signature}: the executor injects {unbindable} but the tool cannot bind them"
    )


# ---------------------------------------------------------------------------
# 2. Every parameter declared required is required by the signature.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", LIVE_TOOLS)
def test_declared_required_parameters_are_signature_required(name) -> None:
    registered = _registry().get(name)
    parameters = inspect.signature(registered.callable, follow_wrapped=False).parameters
    by_keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    unenforced = [
        p.name
        for p in registered.schema.input_parameters
        if p.required
        and not (
            p.name in parameters
            and parameters[p.name].kind in by_keyword
            and parameters[p.name].default is inspect.Parameter.empty
        )
    ]
    assert not unenforced, (
        f"{name}: declared required to the planner but not required by the signature, so "
        f"the pre-dispatch guard cannot enforce them: {unenforced}"
    )


@pytest.mark.parametrize(
    ("name", "parameter", "default"),
    [
        ("gap_calculator", "group_by", None),
        ("risk_scorer", "id_column", "patient_id"),
        ("risk_scorer", "outcome", "discontinuation_flag"),
        ("roi_estimator", "value_per_unit", 1.0),
    ],
)
def test_issue_2078_planner_inputs_are_explicit_with_declared_defaults(
    name: str, parameter: str, default: Any
) -> None:
    """The four #2078 planner inputs live in the dispatch signature, not ``**kwargs``."""
    registered = _registry().get(name)
    parameters = inspect.signature(registered.callable, follow_wrapped=False).parameters
    by_keyword = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    schema_parameter = next(p for p in registered.schema.input_parameters if p.name == parameter)

    assert parameter in parameters
    assert parameters[parameter].kind in by_keyword
    assert parameters[parameter].default == default == schema_parameter.default


@pytest.mark.parametrize(
    ("name", "mapping"),
    [
        ("refutation_runner", {"confounders": ["x1"]}),
        ("sensitivity_analyzer", {"ate": 0.3, "ci_lower": 0.1, "ci_upper": 0.5}),
    ],
)
def test_a_plan_omitting_treatment_and_outcome_is_a_plan_defect(name, mapping) -> None:
    """The consequence of invariant 2 on the two tools it flagged: the omission is
    caught before dispatch, not refused by the tool after it."""
    executor = PlanExecutor(tool_registry=_registry(), enable_caching=False, backoff_base_delay=0.0)
    dispatched: List[str] = []
    original = executor._run_sync_tool

    async def counting(tool_callable: Any, resolved_inputs: Dict[str, Any], tool_name: str) -> Any:
        dispatched.append(tool_name)
        return await original(tool_callable, resolved_inputs, tool_name)

    executor._run_sync_tool = counting  # type: ignore[method-assign]
    step = ExecutionStep(
        step_id="step_1",
        sub_question_id="sq_1",
        tool_name=name,
        source_agent="causal_impact",
        input_mapping=mapping,
        dependency_type=DependencyType.SEQUENTIAL,
    )
    result = asyncio.run(executor._execute_step(step, {}, {"estimation_data": _frame()}))

    assert result.outcome_class == "plan_defect", (
        f"got {result.outcome_class!r} after {result.attempts} attempt(s): {result.output.error!r}"
    )
    assert result.status == ExecutionStatus.FAILED
    assert result.attempts == 0 and dispatched == []
    assert "treatment" in result.output.error and "outcome" in result.output.error
