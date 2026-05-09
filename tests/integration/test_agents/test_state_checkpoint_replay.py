"""Checkpoint-replay smoke tests for migrated ml_foundation State classes.

Pins the contract that pre-migration TypedDict-shape JSON checkpoints
(stored in Redis / FalkorDB / Postgres by the project's RedisSaver at
``src/memory/working_memory.py``) deserialize cleanly into the new
pydantic v2 ``BaseAgentSchema`` subclasses.

Migration plan reference:
``.claude/plans/typeddict_to_pydantic_migration_plan_20260504.md``
section "LangGraph compatibility — Checkpointer" specifies that
JSON-shape compatibility is preserved provided every field has either
``Optional[T] = None`` or an explicit default. This test exercises
that contract for the agents migrated in Shard A (``scope_definer``,
``data_preparer``).

Decision 7a in the plan requires that ``audit_workflow_id`` accept
both ``UUID`` instances (in-process construction) and ``str`` values
(JSON checkpoint restore). The tests here verify both representations
round-trip cleanly.

Each test follows the same shape:

1. Build a representative state dict that mimics what a
   pre-migration TypedDict-state checkpoint would have stored.
2. Serialize to JSON (``json.dumps``) — emulates the on-the-wire
   format used by RedisSaver.
3. Deserialize via ``Schema.model_validate_json`` — emulates what a
   restored pydantic state looks like in a post-migration replay.
4. Assert that critical fields round-tripped cleanly.

These are SMOKE tests — not exhaustive field-level fixtures. The
intent is to catch the migration-breaking class of regression where
a required field is added or a serialization rule changes
incompatibly. Field-by-field round-trip coverage lives in
``tests/unit/test_ml_foundation_schemas.py``.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

from src.agents.ml_foundation.data_preparer.state import DataPreparerState
from src.agents.ml_foundation.feature_analyzer.state import FeatureAnalyzerState
from src.agents.ml_foundation.model_deployer.state import ModelDeployerState
from src.agents.ml_foundation.model_selector.state import ModelSelectorState
from src.agents.ml_foundation.model_trainer.state import ModelTrainerState
from src.agents.ml_foundation.observability_connector.state import (
    ObservabilityConnectorState,
)
from src.agents.ml_foundation.scope_definer.state import ScopeDefinerState


def test_scope_definer_state_replays_typed_dict_checkpoint() -> None:
    """A pre-migration TypedDict-shape JSON checkpoint deserializes
    cleanly into the post-migration pydantic ScopeDefinerState.

    The fixture mimics a real scope_definer mid-pipeline state with
    populated business-context fields, an inferred problem type, and
    a complete scope_spec dict — typical of what RedisSaver would
    persist mid-graph.
    """
    audit_id = uuid4()
    pre_migration_state = {
        "audit_workflow_id": str(audit_id),  # JSON form: stringified UUID
        "experiment_id": "exp_remi_us_20260505_test01",
        "problem_description": "HCP engagement classification",
        "business_objective": "Predict HCP prescription intent",
        "target_outcome": "Increase Remibrutinib uptake",
        "problem_type_hint": "binary_classification",
        "brand": "Remibrutinib",
        "region": "US",
        "use_case": "commercial_targeting",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "scope_spec": {
            "experiment_id": "exp_remi_us_20260505_test01",
            "problem_type": "binary_classification",
            "minimum_samples": 500,
        },
        "success_criteria": {
            "minimum_auc": 0.75,
            "baseline_model": "logistic_regression",
        },
        "validation_passed": True,
        "validation_warnings": [],
        "created_at": "2026-05-05T00:00:00",
        "created_by": "scope_definer",
    }

    json_payload = json.dumps(pre_migration_state)
    restored = ScopeDefinerState.model_validate_json(json_payload)

    # audit_workflow_id round-tripped through str → UUID (Decision 7a).
    assert isinstance(restored.audit_workflow_id, UUID)
    assert restored.audit_workflow_id == audit_id

    # Business fields preserved.
    assert restored.experiment_id == "exp_remi_us_20260505_test01"
    assert restored.brand == "Remibrutinib"
    assert restored.problem_type_hint == "binary_classification"
    assert restored.inferred_problem_type == "binary_classification"

    # Nested dicts preserved (scope_spec is Optional[Dict[str, Any]]).
    assert restored.scope_spec is not None
    assert restored.scope_spec["minimum_samples"] == 500
    assert restored.success_criteria is not None
    assert restored.success_criteria["minimum_auc"] == 0.75

    # Dict-like access via the BaseAgentSchema shim works on the restored
    # instance (the actual call sites in nodes use this access pattern).
    assert restored["brand"] == "Remibrutinib"
    assert restored.get("nonexistent_field", "default") == "default"


def test_scope_definer_state_replays_minimal_checkpoint() -> None:
    """A minimal-keys checkpoint (just audit_workflow_id) deserializes
    successfully — every other field defaults to None per Decision 8a.

    This is the lower-bound smoke test: if the partial-update
    semantics break for any reason (e.g., a field accidentally loses
    its default), this test fires.
    """
    audit_id = uuid4()
    minimal = {"audit_workflow_id": str(audit_id)}
    restored = ScopeDefinerState.model_validate_json(json.dumps(minimal))
    assert restored.audit_workflow_id == audit_id
    # Spot-check Decision-8a defaults.
    assert restored.brand is None
    assert restored.scope_spec is None
    assert restored.experiment_id is None


def test_scope_definer_state_round_trips_through_json() -> None:
    """Construct → JSON → reconstruct preserves equality.

    Discriminating-coverage guard: validates that
    ``model_dump_json`` and ``model_validate_json`` are inverses for
    the migrated state class. If a future serializer change breaks
    the round-trip (e.g., UUID emitted as bytes), this fires.
    """
    audit_id = uuid4()
    original = ScopeDefinerState(
        audit_workflow_id=audit_id,
        experiment_id="exp_test",
        brand="Kisqali",
        validation_passed=True,
    )
    json_str = original.model_dump_json()
    restored = ScopeDefinerState.model_validate_json(json_str)
    assert restored == original


def test_data_preparer_state_replays_typed_dict_checkpoint() -> None:
    """A pre-migration TypedDict-shape data_preparer checkpoint
    deserializes cleanly into the post-migration pydantic state.

    The fixture mimics a mid-pipeline state with the QC gate decided
    and feast registration complete — a state that would persist to
    Redis between pipeline steps.
    """
    audit_id = uuid4()
    pre_migration_state = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_remi_us_20260505_test01",
        "scope_spec": {"problem_type": "binary_classification"},
        "data_source": "business_metrics",
        "qc_status": "passed",
        "completeness_score": 0.95,
        "validity_score": 0.92,
        "consistency_score": 0.93,
        "uniqueness_score": 0.99,
        "timeliness_score": 0.88,
        "overall_score": 0.93,
        "leakage_detected": False,
        "leakage_severity": "none",
        "feast_registration_status": "completed",
        "feast_features_registered": 9,
        "feast_blocked": False,
        "feast_fallback_used": False,
        "row_count": 10000,
        "column_count": 42,
        "validated_at": "2026-05-05T00:00:00",
        "is_ready": True,
        "qc_passed": True,
        "qc_score": 0.93,
        "blockers": [],
        "gate_passed": True,
        "remediation_status": "not_needed",
        "remediation_attempts": 0,
    }

    json_payload = json.dumps(pre_migration_state)
    restored = DataPreparerState.model_validate_json(json_payload)

    # audit_workflow_id round-tripped (Decision 7a).
    assert isinstance(restored.audit_workflow_id, UUID)
    assert restored.audit_workflow_id == audit_id

    # QC gate decision preserved (load-bearing for downstream model_trainer).
    assert restored.qc_status == "passed"
    assert restored.gate_passed is True
    assert restored.overall_score == 0.93

    # Feast fields preserved.
    assert restored.feast_registration_status == "completed"
    assert restored.feast_features_registered == 9
    assert restored.feast_blocked is False

    # Dict-like access works on the restored instance.
    assert restored["qc_passed"] is True
    assert restored.get("validation_suite") is None  # not set in fixture


def test_data_preparer_state_replays_with_blocking_issues() -> None:
    """A failed-gate checkpoint deserializes preserving the blocking
    information that downstream model_trainer.check_qc_gate reads.
    """
    audit_id = uuid4()
    failed_state = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "qc_status": "failed",
        "gate_passed": False,
        "blocking_issues": [
            "completeness_below_threshold:0.65",
            "leakage_detected:critical:future_prescription_data",
        ],
        "leakage_detected": True,
        "leakage_severity": "critical",
        "leaked_features": ["future_prescription_data"],
    }
    restored = DataPreparerState.model_validate_json(json.dumps(failed_state))

    assert restored.gate_passed is False
    assert restored.blocking_issues is not None
    assert len(restored.blocking_issues) == 2
    assert restored.leakage_severity == "critical"
    assert restored.leaked_features == ["future_prescription_data"]


def test_uuid_audit_workflow_id_accepts_both_string_and_uuid() -> None:
    """Cross-cutting Decision 7a contract: both representations
    construct valid state.

    Discrimination guard: the validator factory at
    ``_pydantic_utils.py::audit_workflow_id_validator`` MUST accept
    both forms. If a future refactor narrows the type to UUID-only,
    every checkpoint replay would fail at restore time.
    """
    audit_id = uuid4()
    via_uuid = ScopeDefinerState(audit_workflow_id=audit_id)
    via_str = ScopeDefinerState(audit_workflow_id=str(audit_id))
    assert via_uuid.audit_workflow_id == via_str.audit_workflow_id == audit_id


# --------------------------------------------------------------------------- #
# Shard C — leaf agents                                                       #
# --------------------------------------------------------------------------- #


def test_observability_connector_state_replays_checkpoint() -> None:
    """ObservabilityConnectorState round-trips a representative metrics snapshot."""
    audit_id = uuid4()
    pre_migration = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "events_logged": 42,
        "emission_successful": True,
        "latency_by_agent": {"scope_definer": {"p50": 2.1, "p95": 4.5, "p99": 8.2, "avg": 3.2}},
        "error_rate_by_agent": {"scope_definer": 0.02, "data_preparer": 0.05},
        "overall_success_rate": 0.97,
        "quality_score": 0.92,
    }
    restored = ObservabilityConnectorState.model_validate_json(json.dumps(pre_migration))
    assert restored.audit_workflow_id == audit_id
    assert restored.events_logged == 42
    assert restored.emission_successful is True
    assert restored.quality_score == 0.92
    assert restored["overall_success_rate"] == 0.97


def test_model_selector_state_replays_checkpoint() -> None:
    """ModelSelectorState round-trips a representative selection-result snapshot."""
    audit_id = uuid4()
    pre_migration = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "scope_spec": {"problem_type": "binary_classification"},
        "primary_candidate": {"name": "LogisticRegression", "version": "v1"},
        "algorithm_name": "LogisticRegression",
        "algorithm_class": "sklearn.linear_model.LogisticRegression",
        "default_hyperparameters": {"C": 1.0, "max_iter": 1000},
        "hyperparameter_search_space": {
            "C": {"type": "float", "low": 0.01, "high": 10.0, "log": True}
        },
        "expected_performance": {"auc_roc": 0.78, "f1_score": 0.65},
        "interpretability_score": 0.85,
        "selection_score": 0.82,
        "stage": "development",
    }
    restored = ModelSelectorState.model_validate_json(json.dumps(pre_migration))
    assert restored.audit_workflow_id == audit_id
    assert restored.algorithm_name == "LogisticRegression"
    assert restored.interpretability_score == 0.85
    assert restored.expected_performance is not None
    assert restored.expected_performance["auc_roc"] == 0.78


def test_model_deployer_state_replays_checkpoint() -> None:
    """ModelDeployerState round-trips a representative deployment-result snapshot."""
    audit_id = uuid4()
    pre_migration = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "model_uri": "runs:/abc123/model",
        "validation_metrics": {"auc_roc": 0.85, "chosen_threshold_source": "validation"},
        "success_criteria_met": True,
        "target_environment": "staging",
        "deployment_action": "promote",
        "model_version": 2,
        "promotion_successful": True,
        "deployment_status": "healthy",
        "health_check_passed": True,
        "deployment_successful": True,
        "overall_status": "completed",
    }
    restored = ModelDeployerState.model_validate_json(json.dumps(pre_migration))
    assert restored.audit_workflow_id == audit_id
    assert restored.target_environment == "staging"
    assert restored.deployment_action == "promote"
    assert restored.deployment_status == "healthy"
    assert restored.overall_status == "completed"
    assert restored.validation_metrics is not None
    assert restored.validation_metrics["chosen_threshold_source"] == "validation"


def test_feature_analyzer_state_replays_checkpoint() -> None:
    """FeatureAnalyzerState round-trips a representative SHAP-analysis snapshot.

    Note: ``shap_values: np.ndarray`` is excluded from the JSON fixture —
    np.ndarray serialization to JSON requires explicit handling
    (``.tolist()``); a follow-up sub-shard adds a ``field_serializer`` if
    SHAP analysis is checkpointed in production. Today's persistence path
    stores SHAP via the semantic-memory entries, not JSON checkpoints.
    """
    audit_id = uuid4()
    pre_migration = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "training_run_id": "run_abc",
        "problem_type": "classification",
        "feature_columns": ["age", "weight", "history"],
        "global_importance": {"age": 0.45, "weight": 0.32, "history": 0.23},
        "top_features": ["age", "weight", "history"],
        "explainer_type": "TreeExplainer",
        "shap_analysis_id": "shap_xyz",
        "interpretation": "Age is the dominant feature.",
        "status": "completed",
        "shap_skipped": False,
    }
    restored = FeatureAnalyzerState.model_validate_json(json.dumps(pre_migration))
    assert restored.audit_workflow_id == audit_id
    assert restored.problem_type == "classification"
    assert restored.feature_columns == ["age", "weight", "history"]
    assert restored.shap_analysis_id == "shap_xyz"
    assert restored.global_importance is not None
    assert restored.global_importance["age"] == 0.45


def test_feature_analyzer_state_holds_numpy_shap_values() -> None:
    """``shap_values: Optional[np.ndarray]`` works at construction time.

    arbitrary_types_allowed is inherited from BaseAgentSchema. A regression
    that drops it (or that constrains shap_values to a non-arbitrary type)
    would break the SHAP-computation node's state writes.
    """
    import numpy as np

    arr = np.array([[0.1, 0.2], [0.3, 0.4]])
    state = FeatureAnalyzerState(audit_workflow_id=uuid4(), shap_values=arr)
    assert state.shap_values is not None
    assert state.shap_values.shape == (2, 2)
    np.testing.assert_array_equal(state.shap_values, arr)


def test_model_trainer_state_replays_checkpoint() -> None:
    """ModelTrainerState round-trips a representative training-result
    snapshot through JSON, including mixed-type metric bags.

    Added per codex review I3 (2026-05-05). The migration plan's
    Hotspot #2 calls out ``validation_metrics`` / ``test_metrics`` as
    the largest cross-agent contract risk; Shard B caught a
    ``Dict[str, float]`` strict-validation regression at CI time
    when the evaluator emitted ``chosen_threshold_source: str`` and
    ``net_benefit_grid: dict`` values inside the metric bags. This
    test pins the post-fix contract: mixed-type values (str, dict,
    float) all round-trip cleanly through ``Dict[str, Any]``-typed
    metric fields.

    Non-serializable runtime objects (``trained_model``, ``preprocessor``,
    ``X_train_resampled``) are excluded from the JSON fixture — they
    do not round-trip through a JSON checkpointer (RedisSaver
    persistence is JSON-shape) and are reconstructed at runtime from
    MLflow / artifact stores.
    """
    audit_id = uuid4()
    pre_migration = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "algorithm_name": "LogisticRegression",
        "algorithm_class": "sklearn.linear_model.LogisticRegression",
        "default_hyperparameters": {"C": 1.0, "max_iter": 1000},
        "problem_type": "binary_classification",
        "qc_gate_passed": True,
        "qc_gate_message": "QC passed",
        "evaluation_mode": "single",
        "fold_random_state": 42,
        "fold_idx": 0,
        # Metric bags with MIXED-TYPE values — the post-Shard-B widening to
        # Dict[str, Any] is what makes this round-trip work.
        "validation_metrics": {
            "auc_roc": 0.85,
            "f1_score": 0.72,
            "chosen_threshold_source": "validation",  # str inside metric bag
            "net_benefit_grid": {  # nested dict inside metric bag
                "p_t=0.05": 0.5228,
                "p_t=0.50": 0.5333,
            },
        },
        "test_metrics": {
            "auc_roc": 0.83,
            "f1_score": 0.70,
            "chosen_threshold_source": "validation",
            "net_benefit_grid": {"p_t=0.05": 0.5128, "p_t=0.50": 0.5233},
        },
        "test_metrics_at_05": {
            "auc_roc": 0.83,
            "net_benefit_grid": {"p_t=0.05": 0.5128, "p_t=0.50": 0.5233},
        },
        "success_criteria_met": True,
        "training_status": "completed",
        "training_run_id": "run_abc123",
        "model_id": "model_xyz",
        "mlflow_run_id": "mlflow_run_001",
        "mlflow_status": "success",
    }

    json_payload = json.dumps(pre_migration)
    restored = ModelTrainerState.model_validate_json(json_payload)

    # audit_workflow_id round-tripped through str → UUID (Decision 7a).
    assert isinstance(restored.audit_workflow_id, UUID)
    assert restored.audit_workflow_id == audit_id

    # Top-line training fields preserved.
    assert restored.algorithm_name == "LogisticRegression"
    assert restored.problem_type == "binary_classification"
    assert restored.qc_gate_passed is True
    assert restored.training_status == "completed"
    assert restored.success_criteria_met is True

    # Mixed-type validation_metrics preserved (the regression Shard B caught).
    assert restored.validation_metrics is not None
    assert restored.validation_metrics["auc_roc"] == 0.85
    assert restored.validation_metrics["chosen_threshold_source"] == "validation"
    assert restored.validation_metrics["net_benefit_grid"] == {
        "p_t=0.05": 0.5228,
        "p_t=0.50": 0.5333,
    }

    # Same shape on test_metrics + test_metrics_at_05.
    assert restored.test_metrics is not None
    assert restored.test_metrics["chosen_threshold_source"] == "validation"
    assert restored.test_metrics_at_05 is not None
    assert isinstance(restored.test_metrics_at_05["net_benefit_grid"], dict)

    # Dict-like access via the BaseAgentSchema shim still works on the
    # restored instance — node code uses this access pattern.
    assert restored["algorithm_name"] == "LogisticRegression"
    assert restored.get("nonexistent_field", "default") == "default"


def test_model_trainer_state_minimal_checkpoint_replay() -> None:
    """ModelTrainerState lower-bound smoke test: minimal-keys checkpoint
    deserializes — every other field defaults to None per Decision 8a.
    """
    audit_id = uuid4()
    minimal = {"audit_workflow_id": str(audit_id)}
    restored = ModelTrainerState.model_validate_json(json.dumps(minimal))
    assert restored.audit_workflow_id == audit_id
    # Spot-check Decision-8a defaults.
    assert restored.algorithm_name is None
    assert restored.validation_metrics is None
    assert restored.test_metrics is None
    assert restored.training_status is None


def test_model_trainer_state_repeated_mode_fold_invocation_is_declared_field() -> None:
    """Sub-shard D4 / codex review B2: ``repeated_mode_fold_invocation`` is
    a declared pydantic field, NOT a ``model_extra`` flow-through.

    The pre-D4 code declared the field with a leading underscore
    (``_repeated_mode_fold_invocation``) which pydantic v2 treats as a
    private attribute. The field could not be a model field, so the
    sentinel flowed through ``model_extra``. LangGraph 1.0 only
    propagates declared fields through channels, not ``model_extra``,
    so the sentinel was silently dropped on every node coercion —
    breaking ``mlflow_logger.py``'s repeated-fold MLflow nesting
    (codex review B2, 2026-05-05).

    Post-fix: the field is declared without underscore. This test
    pins the declaration so a future "go back to underscore" edit
    fires a CI failure.
    """
    assert "repeated_mode_fold_invocation" in ModelTrainerState.model_fields
    assert "_repeated_mode_fold_invocation" not in ModelTrainerState.model_fields
    field_info = ModelTrainerState.model_fields["repeated_mode_fold_invocation"]
    # Pydantic 2.x stores the type via .annotation; for Optional[bool]
    # it resolves to ``Optional[bool]`` aka ``Union[bool, None]``.
    assert field_info.is_required() is False  # has default of None


def test_audit_workflow_id_propagates_when_caller_provides_uuid() -> None:
    """Backlog #1 (D1 strict-required): ``audit_workflow_id`` propagates
    correctly across all nodes of a graph when the caller provides it
    in initial state.

    Pre-D1 history (kept for context): when ``default_factory=uuid4``
    was active and the caller did NOT provide a UUID, every
    Schema(**channel_dict) reconstruction inside LangGraph re-fired
    the default_factory, minting a fresh UUID per node and breaking
    the audit chain. The ``@preserve_audit_workflow_id`` decorator was
    the surgical mitigation: it pinned the entry-node UUID into channel
    state so downstream nodes saw the same value.

    Post-D1 (backlog #1 closed 2026-05-09): the State's
    audit_workflow_id field is now required (no default_factory). The
    caller MUST provide a UUID at graph.ainvoke; LangGraph then treats
    it as a "set" channel value and propagates it through all nodes
    natively. The decorator is no longer needed and was removed.

    This test pins the propagation contract: with caller-provided UUID,
    a 3-node graph sees the SAME audit_workflow_id at every node.
    """
    import asyncio

    from langgraph.graph import END, START, StateGraph

    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    captured = []

    async def entry_node(state: ModelTrainerState) -> dict:
        captured.append(("entry", state.audit_workflow_id))
        return {"algorithm_name": "TestAlgo"}

    async def middle_node(state: ModelTrainerState) -> dict:
        captured.append(("middle", state.audit_workflow_id))
        return {}

    async def final_node(state: ModelTrainerState) -> dict:
        captured.append(("final", state.audit_workflow_id))
        return {}

    builder = StateGraph(ModelTrainerState)
    builder.add_node("entry", entry_node)
    builder.add_node("middle", middle_node)
    builder.add_node("final", final_node)
    builder.add_edge(START, "entry")
    builder.add_edge("entry", "middle")
    builder.add_edge("middle", "final")
    builder.add_edge("final", END)
    graph = builder.compile()

    # Caller MUST provide audit_workflow_id (post-D1 contract).
    expected_id = uuid4()
    asyncio.run(
        graph.ainvoke({"experiment_id": "test_d1_propagation", "audit_workflow_id": expected_id})
    )

    # All three nodes must see the SAME audit_workflow_id — the one
    # the caller provided. LangGraph's reducer pins it as the channel
    # value after the first node coercion.
    entry_id = captured[0][1]
    middle_id = captured[1][1]
    final_id = captured[2][1]
    assert entry_id == middle_id == final_id == expected_id, (
        f"audit_workflow_id should be stable across nodes and equal to the "
        f"caller-provided UUID; got entry={entry_id}, middle={middle_id}, "
        f"final={final_id}, expected={expected_id}"
    )


def test_model_trainer_state_repeated_mode_fold_invocation_propagates_through_langgraph() -> None:
    """B2 fix verification: a multi-node LangGraph using ModelTrainerState
    as schema correctly propagates ``repeated_mode_fold_invocation``
    from initial state through subsequent nodes.

    Pre-D4: model_extra-stored sentinel was dropped after the first
    node coercion, so node_b.state.get("repeated_mode_fold_invocation")
    returned False (the get-default coalescing for missing values).
    Post-D4: declared field → real channel → propagates correctly.

    This is the smoke test that catches the regression if anyone
    re-introduces underscore-prefixed sentinel handling.
    """
    import asyncio

    from langgraph.graph import END, START, StateGraph

    captured_at_node_b: dict = {}

    async def node_a(state: ModelTrainerState) -> dict:
        # Node A doesn't return repeated_mode_fold_invocation; LangGraph
        # must preserve it from the channel state established at graph entry.
        return {"algorithm_name": "LogisticRegression"}

    async def node_b(state: ModelTrainerState) -> dict:
        # Node B reads the sentinel via the dict-shim path used by
        # mlflow_logger.py:192 in production code.
        captured_at_node_b["sentinel"] = state.get("repeated_mode_fold_invocation", False)
        captured_at_node_b["evaluation_mode"] = state.get("evaluation_mode", "single")
        return {}

    builder = StateGraph(ModelTrainerState)
    builder.add_node("a", node_a)
    builder.add_node("b", node_b)
    builder.add_edge(START, "a")
    builder.add_edge("a", "b")
    builder.add_edge("b", END)
    graph = builder.compile()

    # Initial state mimics the per-fold invocation pattern from
    # _run_repeated_splits at agent.py:1096.
    initial_state = {
        "audit_workflow_id": uuid4(),
        "evaluation_mode": "repeated_k10",
        "repeated_mode_fold_invocation": True,
        "fold_idx": 3,
    }

    asyncio.run(graph.ainvoke(initial_state))

    # The sentinel MUST reach node B as True. If LangGraph drops it
    # (the pre-D4 model_extra regression), this assertion fires loud.
    assert captured_at_node_b["sentinel"] is True
    assert captured_at_node_b["evaluation_mode"] == "repeated_k10"


# --------------------------------------------------------------------------- #
# Codex review N1 follow-up: backward-compat alias for legacy underscore key  #
# --------------------------------------------------------------------------- #
#
# PR #53 (sub-shard D4) renamed _repeated_mode_fold_invocation to
# repeated_mode_fold_invocation (drop underscore prefix) so the field could
# be a declared pydantic channel. Codex's post-fix review flagged that any
# Redis/DB checkpoints persisted PRE-PR-#53 would contain the legacy
# underscore key in their JSON payloads, and would silently land in
# model_extra rather than mapping to the new field.
#
# The fix in this test file's companion PR adds
# ``validation_alias=AliasChoices("repeated_mode_fold_invocation",
# "_repeated_mode_fold_invocation")`` to the field declaration. These tests
# pin the resulting contract:
#
# 1. Construction via legacy underscore kwarg works.
# 2. ``model_validate_json`` with legacy underscore key in the JSON payload
#    works — this is the actual checkpoint-replay scenario.
# 3. Serialization uses the canonical (python field name), NOT the legacy
#    alias — newly-written checkpoints use the new format. Asymmetric on
#    purpose: read-old / write-new.
#
# Scope limit: the alias only affects pydantic schema construction +
# JSON deserialization. LangGraph's channel routing uses the python field
# name, NOT the alias. So passing the legacy underscore key to
# ``graph.ainvoke({"_repeated_mode_fold_invocation": True})`` will NOT
# propagate via channels — production callers must use the canonical name
# at the LangGraph boundary. This is why we don't have a
# "test_langgraph_legacy_key_propagates" test: the alias does not fix that
# path, and that path is not what codex N1 flagged. All production
# callers of model_trainer use the canonical name post-PR-#53.


def test_model_trainer_state_accepts_legacy_underscore_key_at_construction() -> None:
    """N1 fix: ``ModelTrainerState(_repeated_mode_fold_invocation=True)``
    constructs successfully via the validation_alias.

    This is the smallest unit of the alias contract — confirms the
    AliasChoices declaration accepts the legacy form at the kwarg
    level. Subsequent tests exercise the more realistic JSON-load path.
    """
    state = ModelTrainerState(audit_workflow_id=uuid4(), **{"_repeated_mode_fold_invocation": True})
    assert state.repeated_mode_fold_invocation is True


def test_model_trainer_state_replays_legacy_checkpoint_with_underscore_key() -> None:
    """N1 fix: pre-PR-#53 JSON checkpoints with the legacy underscore
    key deserialize cleanly via ``model_validate_json``.

    This is the actual checkpoint-replay scenario codex N1 flagged.
    Pre-fix: the underscore key would land in ``model_extra`` (via
    extra="allow") because it didn't match any declared field name —
    silently breaking the propagation contract that mlflow_logger.py
    relies on. Post-fix: the validation_alias maps the underscore key
    to the canonical field at deserialization time.
    """
    audit_id = uuid4()
    legacy_checkpoint = {
        "audit_workflow_id": str(audit_id),
        "experiment_id": "exp_test",
        "evaluation_mode": "repeated_k10",
        # Legacy underscore key — pre-PR-#53 format
        "_repeated_mode_fold_invocation": True,
        "fold_idx": 5,
    }
    restored = ModelTrainerState.model_validate_json(json.dumps(legacy_checkpoint))

    # Audit chain still round-trips (Decision 7a).
    assert restored.audit_workflow_id == audit_id

    # Legacy key mapped to canonical field via validation_alias.
    assert restored.repeated_mode_fold_invocation is True

    # Other fields preserved.
    assert restored.evaluation_mode == "repeated_k10"
    assert restored.fold_idx == 5

    # The legacy key should NOT remain in model_extra — pre-fix it would
    # have fallen there; post-fix it's promoted to the declared field.
    assert "_repeated_mode_fold_invocation" not in (restored.model_extra or {})


def test_model_trainer_state_dual_key_payload_canonical_wins() -> None:
    """N1 follow-up I1 (codex review of PR #55, 2026-05-05): pin the
    dual-key precedence behavior of the ``AliasChoices`` declaration.

    When a malformed checkpoint contains BOTH the canonical and the
    legacy underscore keys, ``AliasChoices`` resolves to the FIRST
    alias listed in the declaration — which we keep as the canonical
    name.

    D3 update (2026-05-05): under ``extra="ignore"`` (tightened from
    ``extra="allow"``) the runner-up key is silently DROPPED rather than
    landing in ``model_extra``. Both behaviors preserve the canonical-
    wins precedence; D3 makes the residue invisible (which is OK because
    no production code path actually inspects model_extra for this key
    — LangGraph drops model_extra at every channel boundary).

    Verified empirically against pydantic 2.12.5: precedence follows
    AliasChoices declaration order, NOT input dict iteration order.
    Both orderings of the input dict produce the same result below.
    """
    # Canonical wins regardless of input dict ordering.
    audit_id = uuid4()
    payload_legacy_first = {
        "audit_workflow_id": str(audit_id),
        "_repeated_mode_fold_invocation": True,  # legacy says True
        "repeated_mode_fold_invocation": False,  # canonical says False
    }
    s1 = ModelTrainerState.model_validate_json(json.dumps(payload_legacy_first))
    assert s1.repeated_mode_fold_invocation is False, (
        "Canonical alias must win when both keys present (declaration order is load-bearing)"
    )
    # D3 (2026-05-05): runner-up legacy key is silently DROPPED under
    # ``extra="ignore"``. Pre-D3 it landed in model_extra with its
    # discarded value.
    assert s1.model_extra is None or "_repeated_mode_fold_invocation" not in s1.model_extra

    payload_canonical_first = {
        "audit_workflow_id": str(audit_id),
        "repeated_mode_fold_invocation": False,
        "_repeated_mode_fold_invocation": True,
    }
    s2 = ModelTrainerState.model_validate_json(json.dumps(payload_canonical_first))
    assert s2.repeated_mode_fold_invocation is False
    assert s2.model_extra is None or "_repeated_mode_fold_invocation" not in s2.model_extra


def test_model_trainer_state_serializes_with_canonical_name_not_legacy_alias() -> None:
    """N1 fix: newly-written checkpoints use the canonical python field
    name, NOT the legacy underscore alias.

    This pins the asymmetric design: read both forms (validation_alias),
    write only the canonical form (no serialization_alias, no plain
    alias). Future loads of newly-written checkpoints will see the
    canonical name and fail-fast if some future schema change drops
    the canonical name without coordinating with checkpoint persistence.
    """
    state = ModelTrainerState(
        audit_workflow_id=uuid4(), repeated_mode_fold_invocation=True, experiment_id="exp_test"
    )
    json_str = state.model_dump_json(exclude_none=True)
    parsed = json.loads(json_str)

    # Canonical (post-PR-#53) name MUST be in the serialization output.
    assert "repeated_mode_fold_invocation" in parsed
    assert parsed["repeated_mode_fold_invocation"] is True

    # Legacy underscore form MUST NOT be in the serialization output.
    # If a future change adds ``alias=...`` (instead of validation_alias),
    # this test fires loud — the asymmetric read-old/write-new contract
    # would be broken.
    assert "_repeated_mode_fold_invocation" not in parsed
