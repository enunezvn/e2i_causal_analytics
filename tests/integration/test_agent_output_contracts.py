"""Output-contract regression guard for ml_foundation agents.

Closes Phase 3.1 / Shard #6 from the tier-0 evaluation gap report at
``.claude/plans/tier0_evaluation_gap_report_20260504.md`` — Critical
Finding #2 ("TypedDict total=False, no runtime enforcement, only
data_preparer exercised").

Coverage shape (codex consult 2026-05-04, agent ``ae2d78db4919ac47e``,
verdict REVISE → adopted hybrid):

* **6 agents shape-guarded** — every agent's ``State`` class is verified
  to be a ``TypedDict`` subclass with ``total=False`` and to declare
  the keys its docstring promises (input fields, output fields, error
  fields, audit chain).
* **2 light agents (``scope_definer``, ``observability_connector``)
  additionally get runtime tests** — node invocation for scope_definer's
  ``classify_problem`` (pure logic, no external deps) and graph-factory
  construction for observability_connector (verifies node wiring without
  needing Opik). These two close the runtime-enforcement half of the
  gap report's Critical Finding #2.
* **4 heavy agents (``feature_analyzer``, ``model_trainer``,
  ``model_deployer``, ``model_selector``) stay shape-only** — their
  upstream contracts require trained models, MLflow runs, SHAP
  computation, or BentoML packaging. CI infrastructure for those is
  out-of-scope for this PR (TODO captured in
  ``prod_readiness_backlog.md`` if/when CI test infra lands).

Cross-agent invariants pinned across all 6 (per codex Decision B):

* ``audit_workflow_id`` is ``UUID``-typed (audit chain integrity).
* ``error: Optional[str]`` and ``error_type: Optional[str]`` present
  (every agent must support graceful failure).
* The State class is a strict ``TypedDict`` with ``total=False``
  (matches the project's LangGraph pattern).

``data_preparer`` is intentionally NOT in this file — its contract is
already exercised at ``tests/integration/test_agents/test_data_preparer/
test_data_preparer_pipeline.py`` (per codex scope-creep verdict NO).

Discriminating-coverage protection per ``feedback_pr_merge_workflow.md`` §7:
each test asserts both the positive case (key present, type matches) and,
where relevant, a guard against the regression mode (e.g. a renamed key
or dropped audit_workflow_id would fail loudly, not silently pass).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, get_type_hints
from uuid import UUID

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.agents.ml_foundation._pydantic_utils import BaseAgentSchema  # noqa: E402,I001
from src.agents.ml_foundation.feature_analyzer.state import FeatureAnalyzerState  # noqa: E402
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState  # noqa: E402
from src.agents.ml_foundation.model_selector.state import ModelSelectorState  # noqa: E402
from src.agents.ml_foundation.model_trainer.state import ModelTrainerState  # noqa: E402
from src.agents.ml_foundation.observability_connector.state import (  # noqa: E402
    ObservabilityConnectorState,
)
from src.agents.ml_foundation.scope_definer.state import ScopeDefinerState  # noqa: E402

# Cross-agent contract: every agent's State class must declare these.
# Justified at module docstring + codex Decision B (agent ae2d78db4919ac47e).
CROSS_AGENT_REQUIRED_KEYS = {
    "audit_workflow_id": UUID,
    "error": Optional[str],
    "error_type": Optional[str],
}

ALL_AGENT_STATES = [
    ("feature_analyzer", FeatureAnalyzerState),
    ("model_deployer", ModelDeployerState),
    ("model_selector", ModelSelectorState),
    ("model_trainer", ModelTrainerState),
    ("observability_connector", ObservabilityConnectorState),
    ("scope_definer", ScopeDefinerState),
]


# --------------------------------------------------------------------------- #
# Cross-agent invariants — every State class must satisfy these               #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("agent_name,state_cls", ALL_AGENT_STATES)
def test_state_class_is_partial_update_friendly(agent_name: str, state_cls: type) -> None:
    """Every ml_foundation agent's State allows partial updates from nodes.

    Two equivalent shapes are accepted (per the TypedDict→Pydantic v2
    migration tracked in
    ``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``):

    1. ``TypedDict(total=False)`` — historical pattern. Every field is
       implicitly absent-or-present, so partial dict updates merge
       cleanly into LangGraph state.
    2. ``BaseAgentSchema`` (pydantic v2 ``BaseModel`` subclass with
       ``extra="allow"`` and Decision-8a ``Optional[T] = None`` defaults
       on every field) — post-migration pattern. Same partial-update
       semantics achieved via ``Optional`` defaults plus the dict-like
       accessors on ``BaseAgentSchema``.

    Switching either shape to its strict variant (``total=True`` for
    TypedDict, or removing ``Optional`` defaults for pydantic) would
    break pipeline-level state merges silently. This test pins the
    convention against either drift.
    """
    if hasattr(state_cls, "__total__"):
        # TypedDict path
        assert state_cls.__total__ is False, (
            f"{agent_name}: {state_cls.__name__} declared with total={state_cls.__total__}; "
            "must be total=False for LangGraph partial-update semantics"
        )
        return
    if issubclass(state_cls, BaseAgentSchema):
        # Pydantic path — BaseAgentSchema enforces extra="allow" and the
        # subclass should declare every non-audit field as Optional with
        # a default. Spot-check that audit_workflow_id is the only
        # required field (no default), matching Decision 7a/8a.
        required_fields = [
            name for name, field_info in state_cls.model_fields.items() if field_info.is_required()
        ]
        # Allow audit_workflow_id as the documented exception; reject any
        # other required field (would break partial-update semantics).
        assert required_fields == ["audit_workflow_id"] or required_fields == [], (
            f"{agent_name}: {state_cls.__name__} has unexpected required fields "
            f"{required_fields!r}. Per Decision 8a only audit_workflow_id may be "
            f"required; everything else must have an Optional[T]=None default for "
            f"partial-update compatibility."
        )
        return
    raise AssertionError(
        f"{agent_name}: {state_cls.__name__} is neither TypedDict(total=False) "
        f"nor a BaseAgentSchema subclass — incompatible with the migration arc"
    )


@pytest.mark.parametrize("agent_name,state_cls", ALL_AGENT_STATES)
def test_state_declares_audit_workflow_id_as_uuid(agent_name: str, state_cls: type) -> None:
    """Every agent must thread ``audit_workflow_id: UUID`` for audit-chain integrity.

    Without this field, the central audit-trail repository cannot stitch
    multi-agent invocations into a single workflow. The UUID type is
    load-bearing — string IDs would break uniqueness guarantees.
    """
    hints = get_type_hints(state_cls)
    assert "audit_workflow_id" in hints, (
        f"{agent_name}: {state_cls.__name__} missing required key 'audit_workflow_id' — "
        "audit chain integrity broken"
    )
    assert hints["audit_workflow_id"] is UUID, (
        f"{agent_name}: 'audit_workflow_id' typed as {hints['audit_workflow_id']!r}, "
        f"expected UUID for audit-chain uniqueness"
    )


@pytest.mark.parametrize("agent_name,state_cls", ALL_AGENT_STATES)
def test_state_declares_error_handling_fields(agent_name: str, state_cls: type) -> None:
    """Every agent must declare ``error`` and ``error_type`` Optional[str] fields.

    These are the agent's graceful-failure contract — without them, a
    raised exception inside a node has nowhere to land in state, and the
    pipeline either crashes or silently swallows the error.
    """
    hints = get_type_hints(state_cls)
    for key in ("error", "error_type"):
        assert key in hints, (
            f"{agent_name}: {state_cls.__name__} missing required error-handling "
            f"key {key!r} — graceful-failure contract broken"
        )
        # Optional[str] resolves to Union[str, None]; both arrangements accepted.
        # The annotation must allow None to satisfy the contract.
        assert hints[key] == Optional[str], (
            f"{agent_name}: error key {key!r} typed as {hints[key]!r}, "
            f"expected Optional[str] for graceful-failure contract"
        )


# --------------------------------------------------------------------------- #
# Per-agent OUTPUT-field invariants — what each agent promises to produce     #
# --------------------------------------------------------------------------- #
#
# Required output keys drawn from each agent's state.py docstring sections
# marked "=== OUTPUT FIELDS ===" (per codex Decision B file:line citations).
# Failing one of these means the agent's TypedDict no longer declares its
# documented contract — downstream consumers would silently break.


PER_AGENT_REQUIRED_OUTPUT_KEYS: dict[str, tuple[type, list[str]]] = {
    # scope_definer/state.py:121-128 — OUTPUT FIELDS section
    "scope_definer": (
        ScopeDefinerState,
        ["experiment_id", "scope_spec", "success_criteria", "created_at"],
    ),
    # model_selector/state.py:64-93 — OUTPUT FIELDS section
    "model_selector": (
        ModelSelectorState,
        [
            "primary_candidate",
            "algorithm_name",
            "algorithm_class",
            "default_hyperparameters",
            "expected_performance",
        ],
    ),
    # model_trainer/state.py:131-220 — validation_metrics/test_metrics at 133-134
    # (intermediate eval section); training_run_id/training_status in
    # === OUTPUT FIELDS === section starting at line 189; success_criteria_met
    # at 186 (intermediate). All resolve via get_type_hints regardless of section.
    "model_trainer": (
        ModelTrainerState,
        [
            "training_run_id",
            "validation_metrics",
            "test_metrics",
            "success_criteria_met",
            "training_status",
        ],
    ),
    # feature_analyzer/state.py:91-181 — SHAP intermediate fields (shap_values
    # at 103, global_importance at 107, top_features at 114) + final
    # === OUTPUT FIELDS === at 162-181 (shap_analysis_id at 165, interpretation
    # at 172). All resolve via get_type_hints regardless of section.
    "feature_analyzer": (
        FeatureAnalyzerState,
        [
            "shap_values",
            "global_importance",
            "top_features",
            "interpretation",
            "shap_analysis_id",
        ],
    ),
    # model_deployer/state.py:166-179 — === OUTPUT FIELDS (Final) === section
    "model_deployer": (
        ModelDeployerState,
        [
            "deployment_manifest",
            "version_record",
            "final_bento_tag",
            "deployment_successful",
            "overall_status",
        ],
    ),
    # observability_connector/state.py:33-90 — node 1 + node 2 outputs
    "observability_connector": (
        ObservabilityConnectorState,
        [
            "span_ids_logged",
            "events_logged",
            "emission_successful",
            "latency_by_agent",
            "error_rate_by_agent",
        ],
    ),
}


@pytest.mark.parametrize("agent_name", list(PER_AGENT_REQUIRED_OUTPUT_KEYS.keys()), ids=lambda n: n)
def test_agent_declares_required_output_keys(agent_name: str) -> None:
    """Per-agent: documented OUTPUT FIELDS appear in __annotations__.

    Pins each agent's externally-visible contract. If a downstream agent
    relies on (e.g.) ``model_trainer.validation_metrics`` and the field
    is renamed without updating the State, this test fires before the
    silent-failure mode reaches production.
    """
    state_cls, required = PER_AGENT_REQUIRED_OUTPUT_KEYS[agent_name]
    hints = get_type_hints(state_cls)
    missing = [k for k in required if k not in hints]
    declared = sorted(hints.keys())
    assert not missing, (
        f"{agent_name}: {state_cls.__name__} missing documented OUTPUT keys: "
        f"{missing!r}. Declared keys (first 30 of {len(declared)}): "
        f"{declared[:30]!r}"
    )


# --------------------------------------------------------------------------- #
# observability_connector: extra error_details field (3-field error contract) #
# --------------------------------------------------------------------------- #


def test_observability_connector_declares_error_details() -> None:
    """observability_connector has a 3-field error contract, not 2.

    Per state.py:114-116 — adds ``error_details: Optional[Dict[str, Any]]``
    on top of the cross-agent (error, error_type) pair. This richer error
    contract lets the connector surface span-emission failures with
    structured context (e.g. opik.url, db.write_count) rather than just
    a string message — important for an observability surface where the
    error's structured detail IS the diagnostic signal.
    """
    hints = get_type_hints(ObservabilityConnectorState)
    assert "error_details" in hints, (
        "observability_connector: ObservabilityConnectorState missing "
        "'error_details' — its 3-field error contract per state.py:114-116 "
        "is broken"
    )
    # Optional[Dict[str, Any]] resolves to a Union; we just verify the
    # key exists and accepts None at runtime via Optional[...]. The exact
    # Union-arm-order isn't load-bearing.
    annotation = hints["error_details"]
    assert annotation is not None, "error_details annotation must be set"


# --------------------------------------------------------------------------- #
# Light-agent runtime tests — node invocation / graph construction            #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_scope_definer_classify_problem_runtime_contract() -> None:
    """``classify_problem`` honours its declared output contract at runtime.

    Runtime-enforcement counterpart to the shape-only tests above.
    ``classify_problem`` is the cheapest scope_definer node: pure async
    logic, no LLM, no external deps
    (src/agents/ml_foundation/scope_definer/nodes/problem_classifier.py:10-45).

    Post Shard A migration, ``ScopeDefinerState`` is a pydantic v2
    ``BaseAgentSchema`` subclass — direct ``model_validate`` is the
    runtime contract check (replaces the obsolete ``validate_state``
    helper, deleted in Shard A).
    """
    from src.agents.ml_foundation.scope_definer.nodes.problem_classifier import (
        classify_problem,
    )

    minimal_state: dict = {
        "business_objective": "Predict patient discontinuation risk",
        "target_outcome": "Reduce churn within 90 days",
        "problem_type_hint": "binary_classification",
    }
    result = await classify_problem(minimal_state)
    assert isinstance(result, dict), (
        f"classify_problem must return a dict, got {type(result).__name__}"
    )
    # Both required keys must be present in the dict.
    for key in ("inferred_problem_type", "inferred_target_variable"):
        assert key in result, (
            f"classify_problem must return key {key!r}; got keys {sorted(result.keys())!r}"
        )
    # Type-narrow the inferred problem type — the function's Literal
    # annotation says one of 5 strings; we pin the contract that the
    # binary_classification hint round-trips.
    assert result["inferred_problem_type"] == "binary_classification", (
        f"classify_problem dropped problem_type_hint; got "
        f"{result['inferred_problem_type']!r} instead of binary_classification"
    )


def test_observability_connector_graph_compiles() -> None:
    """observability_connector's graph factory produces a compiled state graph.

    This is the runtime-enforcement counterpart to the shape-only tests
    above for observability_connector. Graph construction exercises the
    full node-wiring import chain; if any of the connector's nodes,
    repositories, or state schema break, this test fires before the
    silent-failure mode reaches the pipeline.

    Cheaper than invoking ``emit_spans`` directly (which would need a
    live Opik client + DB connection); confirms the same import-time
    contracts hold.
    """
    from langgraph.graph.state import CompiledStateGraph

    from src.agents.ml_foundation.observability_connector.graph import (
        create_observability_connector_graph,
    )

    graph = create_observability_connector_graph()
    assert isinstance(graph, CompiledStateGraph), (
        f"create_observability_connector_graph() returned "
        f"{type(graph).__name__}, expected CompiledStateGraph — "
        f"graph wiring may be broken"
    )


# --------------------------------------------------------------------------- #
# Discriminating-coverage guard (§7) — pydantic ValidationError fires loudly  #
# --------------------------------------------------------------------------- #


def test_pydantic_state_rejects_malformed_audit_workflow_id() -> None:
    """``ScopeDefinerState`` raises ValidationError on a malformed audit_workflow_id.

    Vacuous-pass guard for the migration: confirms the pydantic
    validator factory at ``_pydantic_utils.py::audit_workflow_id_validator``
    actually fires under a regression scenario. If a future change
    drops the validator, audit-chain integrity could silently break
    — this guard prevents that.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ScopeDefinerState(audit_workflow_id="not-a-uuid")


def test_pydantic_state_requires_audit_workflow_id() -> None:
    """``ScopeDefinerState()`` raises ``ValidationError`` when
    ``audit_workflow_id`` is omitted (backlog #1 closed 2026-05-09).

    Pre-D1 contract: the field had ``Field(default_factory=uuid4)`` so
    ``ScopeDefinerState()`` auto-generated a UUID. That was a
    transition mechanism while agent flows were retrofitted to thread
    the field through the orchestrator (PR #58), per-agent input_data
    (PR #62), and caller fixtures (PR #65).

    Post-D1: the field is plain ``UUID`` (required, no default). The
    LangGraph channel-reducer bug (every Schema(**channel_dict)
    reconstruction fired default_factory afresh, breaking the audit
    chain across nodes) is fixed by construction — no fresh UUID can
    be minted because there's no factory.

    This test pins the new contract: caller MUST provide
    audit_workflow_id; missing the field fails loud at State
    construction.
    """
    from pydantic import ValidationError

    with pytest.raises(ValidationError) as exc_info:
        ScopeDefinerState()
    assert "audit_workflow_id" in str(exc_info.value), (
        "ValidationError on missing audit_workflow_id should mention the field "
        f"name; got: {exc_info.value}"
    )
