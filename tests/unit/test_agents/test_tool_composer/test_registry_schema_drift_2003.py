"""#2003 — every surface that describes a composable tool states the inputs it accepts
and the keys it returns.

Three surfaces describe a tool's inputs and outputs (measured 2026-09-11):

1. The live registry (``src.tool_registry.registry``), filled by ``@composable_tool`` and
   ``registry.register``. It is the ONLY one read at runtime: the planner prompt
   (``ToolPlanner._format_tools_for_prompt``), the DSPy planning signature and the
   executor's auto-population gates all call ``get_schemas_for_planning`` /
   ``get_schema`` on it. A declared input the callable does not take, an accepted input
   the planner is never told about, or an output field list that is missing (the planner
   then invents ``$step_N.<field>`` references, #1573) are defects in the planning path.
2. ``create_default_tools()`` (``src/agents/tool_composer/tool_registry.py``), the JSON
   Schema copy that also carries each tool's category and dependency metadata. Nothing
   imports it at runtime; its hand-written schemas had drifted on 15 of 16 tools, which
   is how #2003 came to be filed against a copy the planner never reads.
3. The ``tool_registry`` / ``tool_dependencies`` rows seeded by ``database/ml/013`` and
   ``027``. No code reads them either; they drifted the same way.

The live registry is the single source of truth. The callable itself is the ground truth
the registry is checked against: its signature (and the literal ``kwargs`` keys its body
reads) for inputs, and the keys it really returns — each tool is CALLED on real minimal
inputs — for outputs. The copy and the DB rows are then checked against the registry.
"""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import re
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import pytest

from src.agents.tool_composer import composer as _composer  # noqa: F401 - registers every tool
from src.tool_registry.registry import get_registry
from src.tool_registry.tools.causal_discovery import register_all_discovery_tools
from src.tool_registry.tools.model_inference import register_model_inference_tool
from src.tool_registry.tools.structural_drift import register_structural_drift_tool

REPO_ROOT = Path(__file__).resolve().parents[4]

# The tools the composer registers. Pinned so a registry that silently lost its tools
# cannot make every per-tool check below pass vacuously.
LIVE_TOOLS = frozenset(
    {
        "cate_analyzer",
        "causal_effect_estimator",
        "cohort_builder",
        "cohort_statistics",
        "cohort_validator",
        "counterfactual_simulator",
        "detect_structural_drift",
        "discover_dag",
        "distribution_comparator",
        "gap_calculator",
        "model_inference",
        "power_calculator",
        "propensity_estimator",
        "psi_calculator",
        "rank_drivers",
        "refutation_runner",
        "risk_scorer",
        "roi_estimator",
        "segment_ranker",
        "sensitivity_analyzer",
    }
)

# Accepted inputs that are deliberately NOT planner-visible, with the reason.
INTERNAL_INPUTS: Dict[str, Dict[str, str]] = {
    "discover_dag": {"trace_context": "Opik trace plumbing for agent callers"},
    "rank_drivers": {"trace_context": "Opik trace plumbing for agent callers"},
    "detect_structural_drift": {"trace_context": "Opik trace plumbing for agent callers"},
    "model_inference": {"trace_context": "Opik trace plumbing for agent callers"},
    "causal_effect_estimator": {
        "data_source": "provenance label for the pipeline run",
        "query": "provenance text for the pipeline run",
    },
    "refutation_runner": {
        "treatment_var": "alias of treatment",
        "outcome_var": "alias of outcome",
        "covariates": "alias of confounders",
        "common_causes": "alias of confounders",
    },
}


def _registry():
    registry = get_registry()
    # Idempotent (register() skips a known name): guarantees the discovery / inference /
    # drift tools are present even if module import order skipped their registration.
    register_all_discovery_tools()
    register_model_inference_tool()
    register_structural_drift_tool()
    return registry


def _function_def(fn: Any) -> ast.FunctionDef | ast.AsyncFunctionDef:
    source = textwrap.dedent(inspect.getsource(inspect.unwrap(fn)))
    node = ast.parse(source).body[0]
    assert isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)), node
    return node


def _kwargs_literal_keys(fn: Any) -> Set[str]:
    """Literal keys the tool body reads from ``**kwargs``.

    Covers ``kwargs.get("key")`` and ``_first_kwarg(kwargs, ("a", "b"))``, the two shapes
    the tool bodies use. A key read inside a helper by a non-literal expression (the
    DataFrame keys) is invisible here by construction — those are injected by the
    executor, not planned.
    """
    keys: Set[str] = set()
    for node in ast.walk(_function_def(fn)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "get"
            and isinstance(func.value, ast.Name)
            and func.value.id == "kwargs"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            keys.add(node.args[0].value)
        if (
            isinstance(func, ast.Name)
            and func.id == "_first_kwarg"
            and len(node.args) == 2
            and isinstance(node.args[1], ast.Tuple)
        ):
            keys.update(
                elt.value
                for elt in node.args[1].elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            )
    return keys


def _signature(fn: Any):
    sig = inspect.signature(inspect.unwrap(fn))
    explicit = {
        p.name: p
        for p in sig.parameters.values()
        if p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
    }
    has_var_kwargs = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
    return explicit, has_var_kwargs


# ---------------------------------------------------------------------------
# Real minimal inputs: every tool is CALLED so its returned keys are measured.
# ---------------------------------------------------------------------------


def _frame(n: int = 400, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    treatment = (x1 + rng.normal(size=n) > 0).astype(int)
    outcome = 0.5 * treatment + 0.8 * x1 + 0.3 * x2 + rng.normal(size=n)
    logit = -0.2 + 0.9 * treatment + 0.6 * x2
    flag = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "patient_id": [f"pt-{i:04d}" for i in range(n)],
            "geographic_region": rng.choice(["northeast", "south", "midwest"], size=n),
            "period": np.where(np.arange(n) < n // 2, "p1", "p2"),
            "treatment": treatment,
            "outcome": outcome,
            "x1": x1,
            "x2": x2,
            "discontinuation_flag": flag,
        }
    )


def _dump(result: Any) -> Dict[str, Any]:
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    return result.model_dump() if hasattr(result, "model_dump") else result


def _call_every_tool() -> Dict[str, Dict[str, Any]]:
    """Call each live tool on real inputs, chaining real upstream outputs where a tool
    consumes one. ``model_inference`` is excluded: its only computation is a network call
    to a BentoML endpoint (checked structurally instead)."""
    registry = _registry()
    df = _frame()
    call = lambda name, **kw: _dump(registry.get_callable(name)(**kw))  # noqa: E731
    out: Dict[str, Dict[str, Any]] = {}
    out["cohort_builder"] = call("cohort_builder", brand="Kisqali", estimation_data=df)
    out["cohort_validator"] = call(
        "cohort_validator", cohort_result=out["cohort_builder"], estimation_data=df
    )
    out["cohort_statistics"] = call(
        "cohort_statistics", cohort_result=out["cohort_builder"], estimation_data=df
    )
    out["causal_effect_estimator"] = call(
        "causal_effect_estimator",
        treatment="treatment",
        outcome="outcome",
        confounders=["x1", "x2"],
        estimation_data=df,
    )
    estimate = out["causal_effect_estimator"]
    out["refutation_runner"] = call(
        "refutation_runner",
        estimate_id="est-2003",
        treatment="treatment",
        outcome="outcome",
        confounders=["x1", "x2"],
        estimation_data=df,
    )
    out["sensitivity_analyzer"] = call(
        "sensitivity_analyzer",
        ate=0.5,
        ci_lower=0.2,
        ci_upper=0.8,
    )
    out["cate_analyzer"] = call(
        "cate_analyzer",
        treatment="treatment",
        outcome="outcome",
        segments=["geographic_region"],
        estimation_data=df,
    )
    out["segment_ranker"] = call("segment_ranker", cate_results=out["cate_analyzer"])
    out["gap_calculator"] = call(
        "gap_calculator",
        metric="outcome",
        entity_type="region",
        entities=[],
        estimation_data=df,
        group_by="geographic_region",
    )
    out["roi_estimator"] = call(
        "roi_estimator", gap_analysis=out["gap_calculator"], investment=1000.0
    )
    out["power_calculator"] = call("power_calculator", effect_size=0.3)
    out["counterfactual_simulator"] = call(
        "counterfactual_simulator",
        intervention="rep call frequency",
        target_entities=list(out["cate_analyzer"]["high_responders"]) or ["south"],
        expected_effect=float(estimate["ate"]),
    )
    out["psi_calculator"] = call(
        "psi_calculator",
        feature="x1",
        baseline_period="p1",
        current_period="p2",
        estimation_data=df,
    )
    out["distribution_comparator"] = call(
        "distribution_comparator",
        features=["x1", "x2"],
        period_1="p1",
        period_2="p2",
        estimation_data=df,
    )
    out["risk_scorer"] = call(
        "risk_scorer", entity_type="patient", risk_type="discontinuation", estimation_data=df
    )
    out["propensity_estimator"] = call(
        "propensity_estimator", treatment="treatment", covariates=["x1", "x2"], estimation_data=df
    )
    out["discover_dag"] = call(
        "discover_dag", data=df[["treatment", "outcome", "x1", "x2"]].to_dict("list")
    )
    out["rank_drivers"] = call(
        "rank_drivers",
        dag_edge_list=out["discover_dag"]["edge_list"],
        target="outcome",
        shap_values=np.random.default_rng(5).normal(size=(60, 2)).tolist(),
        feature_names=["x1", "x2"],
    )
    out["detect_structural_drift"] = call(
        "detect_structural_drift",
        baseline_dag_adjacency=[[0, 1, 0], [0, 0, 1], [0, 0, 0]],
        current_dag_adjacency=[[0, 0, 0], [1, 0, 1], [0, 0, 0]],
        dag_nodes=["x1", "x2", "outcome"],
    )
    return out


@pytest.fixture(scope="module")
def returned_by_tool() -> Dict[str, Dict[str, Any]]:
    return _call_every_tool()


# ---------------------------------------------------------------------------
# 1. The live registry against the callables
# ---------------------------------------------------------------------------


def test_live_registry_holds_exactly_the_composer_tools():
    assert set(_registry().list_tools()) == LIVE_TOOLS


@pytest.mark.parametrize("name", sorted(LIVE_TOOLS))
def test_declared_inputs_are_inputs_the_callable_accepts(name):
    registered = _registry().get(name)
    explicit, has_var_kwargs = _signature(registered.callable)
    accepted = set(explicit) | (
        _kwargs_literal_keys(registered.callable) if has_var_kwargs else set()
    )
    declared = [p.name for p in registered.schema.input_parameters]
    assert len(declared) == len(set(declared)), f"{name}: duplicate declared inputs {declared}"
    assert set(declared) <= accepted, (
        f"{name}: declares inputs the callable never reads: {sorted(set(declared) - accepted)}"
    )


@pytest.mark.parametrize("name", sorted(LIVE_TOOLS))
def test_every_accepted_input_is_declared_or_documented_internal(name):
    registered = _registry().get(name)
    explicit, has_var_kwargs = _signature(registered.callable)
    accepted = set(explicit) | (
        _kwargs_literal_keys(registered.callable) if has_var_kwargs else set()
    )
    internal = set(INTERNAL_INPUTS.get(name, {}))
    declared = {p.name for p in registered.schema.input_parameters}
    assert internal <= accepted, f"{name}: stale INTERNAL_INPUTS {sorted(internal - accepted)}"
    assert accepted - internal == declared, (
        f"{name}: accepted-but-undeclared={sorted(accepted - internal - declared)} "
        f"declared-but-not-accepted={sorted(declared - (accepted - internal))}"
    )


@pytest.mark.parametrize("name", sorted(LIVE_TOOLS))
def test_required_flags_match_the_signature(name):
    registered = _registry().get(name)
    explicit, _ = _signature(registered.callable)
    wrong = {
        p.name: {
            "declared_required": p.required,
            "signature_required": explicit[p.name].default is inspect.Parameter.empty,
        }
        for p in registered.schema.input_parameters
        if p.name in explicit
        and p.required != (explicit[p.name].default is inspect.Parameter.empty)
    }
    assert not wrong, f"{name}: required flags disagree with the signature: {wrong}"


@pytest.mark.parametrize("name", sorted(LIVE_TOOLS))
def test_every_tool_registers_its_output_model(name):
    registered = _registry().get(name)
    model = registered.pydantic_output_model
    assert model is not None, (
        f"{name}: no output model registered, so the planner prompt lists no output fields "
        f"(declared output_schema={registered.schema.output_schema!r})"
    )
    assert model.__name__ == registered.schema.output_schema


def test_every_tool_but_model_inference_is_called(returned_by_tool):
    assert set(returned_by_tool) == LIVE_TOOLS - {"model_inference"}


@pytest.mark.parametrize("name", sorted(LIVE_TOOLS - {"model_inference"}))
def test_returned_keys_are_the_output_model_fields(name, returned_by_tool):
    model = _registry().get(name).pydantic_output_model
    assert model is not None, f"{name}: no output model registered"
    returned = set(returned_by_tool[name])
    fields = set(model.model_fields)
    assert returned == fields, (
        f"{name}: returned-not-declared={sorted(returned - fields)} "
        f"declared-not-returned={sorted(fields - returned)}"
    )


def test_model_inference_returns_its_output_model_dump():
    """Structural check for the one tool whose computation is a network call.

    ``model_inference`` must return ``<x>.model_dump()`` where ``x`` is the result of
    ``ModelInferenceTool.invoke``, which is annotated to return the registered model.
    """
    from typing import get_type_hints

    from src.tool_registry.tools.model_inference import ModelInferenceTool

    registered = _registry().get("model_inference")
    assert get_type_hints(ModelInferenceTool.invoke)["return"] is registered.pydantic_output_model
    node = _function_def(registered.callable)
    returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
    assert len(returns) == 1
    value = returns[0].value
    assert isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)
    assert value.func.attr == "model_dump" and isinstance(value.func.value, ast.Name)
    assigned = [
        n.value
        for n in ast.walk(node)
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == value.func.value.id for t in n.targets)
    ]
    assert len(assigned) == 1 and isinstance(assigned[0], ast.Await)
    call = assigned[0].value
    assert isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
    assert call.func.attr == "invoke"


# ---------------------------------------------------------------------------
# 2. create_default_tools() against the live registry
# ---------------------------------------------------------------------------


def _static_tools():
    from src.agents.tool_composer.tool_registry import create_default_tools

    return {tool.name: tool for tool in create_default_tools()}


def test_default_tools_cover_exactly_the_live_tools():
    assert set(_static_tools()) == LIVE_TOOLS


@pytest.mark.parametrize("name", sorted(LIVE_TOOLS))
def test_default_tool_schema_matches_the_live_registry(name):
    static = _static_tools().get(name)
    assert static is not None, f"{name}: absent from create_default_tools()"
    registered = _registry().get(name)
    declared = {p.name for p in registered.schema.input_parameters}
    required = {p.name for p in registered.schema.input_parameters if p.required}
    assert set(static.input_schema.get("properties", {})) == declared
    assert set(static.input_schema.get("required", [])) == required
    model = registered.pydantic_output_model
    assert model is not None, f"{name}: no output model registered"
    assert set(static.output_schema.get("properties", {})) == set(model.model_fields)
    assert static.fn is registered.callable


def test_default_tool_dependencies_name_real_tools_and_real_fields():
    from src.agents.tool_composer import tool_registry as static_module

    mappings = getattr(static_module, "DEPENDENCY_FIELD_MAPPINGS", None)
    assert mappings is not None, "DEPENDENCY_FIELD_MAPPINGS is not defined"
    registry = _registry()
    tools = _static_tools()
    pairs = {(name, producer) for name, tool in tools.items() for producer in tool.can_consume_from}
    assert set(mappings) == pairs
    for (consumer, producer), (output_field, input_field) in mappings.items():
        assert producer in LIVE_TOOLS, (consumer, producer)
        if output_field is not None:
            fields = set(registry.get(producer).pydantic_output_model.model_fields)
            assert output_field in fields, (consumer, producer, output_field, sorted(fields))
        if input_field is not None:
            params = {p.name for p in registry.get(consumer).schema.input_parameters}
            assert input_field in params, (consumer, producer, input_field, sorted(params))


# ---------------------------------------------------------------------------
# 3. The DB rows (tool_registry / tool_dependencies) against the live registry
# ---------------------------------------------------------------------------

SYNC_MARKER = "tool-registry-schema-sync"
_PAYLOAD_RE = re.compile(r"\$(tool_registry_sync|tool_dependencies_sync)\$(.*?)\$\1\$", re.DOTALL)


def _latest_sync_migration() -> Path:
    candidates = sorted(
        path
        for path in (REPO_ROOT / "database" / "ml").glob("[0-9][0-9][0-9]_*.sql")
        if SYNC_MARKER in path.read_text()
    )
    assert candidates, f"no database/ml migration carries the {SYNC_MARKER!r} payload"
    return candidates[-1]


def _sync_payloads() -> Dict[str, List[Dict[str, Any]]]:
    text = _latest_sync_migration().read_text()
    payloads = {tag: json.loads(body) for tag, body in _PAYLOAD_RE.findall(text)}
    assert set(payloads) == {"tool_registry_sync", "tool_dependencies_sync"}, set(payloads)
    return payloads


def test_db_sync_covers_every_seeded_tool():
    from scripts.generate_tool_registry_sync_migration import NOT_SEEDED_IN_DB

    rows = _sync_payloads()["tool_registry_sync"]
    names = [row["name"] for row in rows]
    assert len(names) == len(set(names))
    assert set(names) == LIVE_TOOLS - set(NOT_SEEDED_IN_DB)


def test_db_sync_rows_match_the_live_registry():
    registry = _registry()
    for row in _sync_payloads()["tool_registry_sync"]:
        registered = registry.get(row["name"])
        declared = {p.name for p in registered.schema.input_parameters}
        required = {p.name for p in registered.schema.input_parameters if p.required}
        assert set(row["input_schema"]["properties"]) == declared, row["name"]
        assert set(row["input_schema"].get("required", [])) == required, row["name"]
        assert set(row["output_schema"]["properties"]) == set(
            registered.pydantic_output_model.model_fields
        ), row["name"]


def test_db_sync_dependencies_match_the_default_tool_mappings():
    from scripts.generate_tool_registry_sync_migration import NOT_SEEDED_IN_DB
    from src.agents.tool_composer.tool_registry import DEPENDENCY_FIELD_MAPPINGS

    rows = _sync_payloads()["tool_dependencies_sync"]
    got = {(r["consumer"], r["producer"]): (r["output_field"], r["input_field"]) for r in rows}
    expected = {
        pair: fields
        for pair, fields in DEPENDENCY_FIELD_MAPPINGS.items()
        if not set(pair) & set(NOT_SEEDED_IN_DB)
    }
    assert got == expected


def test_generator_output_is_what_the_drift_parser_reads():
    from scripts.generate_tool_registry_sync_migration import SYNC_MARKER as GENERATOR_MARKER
    from scripts.generate_tool_registry_sync_migration import render_sql

    payloads = _sync_payloads()
    sql = render_sql(
        payloads["tool_registry_sync"], payloads["tool_dependencies_sync"], "999_x.sql"
    )
    assert GENERATOR_MARKER == SYNC_MARKER and SYNC_MARKER in sql
    reparsed = {tag: json.loads(body) for tag, body in _PAYLOAD_RE.findall(sql)}
    assert reparsed == payloads
    # The row-count guards are sized to the payload the migration carries.
    assert f"v_expected := {len(payloads['tool_registry_sync'])};" in sql
    assert f"v_expected := {len(payloads['tool_dependencies_sync'])};" in sql
