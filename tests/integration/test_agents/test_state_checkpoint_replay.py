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
    state = FeatureAnalyzerState(shap_values=arr)
    assert state.shap_values is not None
    assert state.shap_values.shape == (2, 2)
    np.testing.assert_array_equal(state.shap_values, arr)
