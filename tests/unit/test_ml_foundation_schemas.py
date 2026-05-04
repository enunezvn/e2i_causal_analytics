"""Round-trip and contract tests for ml_foundation pydantic v2 schemas.

Pins the chore(schemas) scaffolding PR (precursor to TypedDict →
Pydantic v2 migration). Every test here exercises a contract that
Shard A / B / C will rely on:

- Round-trip serialization: schemas survive ``model_dump_json`` →
  ``model_validate_json`` without data loss for representative
  fixtures. Critical for checkpoint replay.
- UUID coercion: ``audit_workflow_id`` accepts both ``UUID`` and
  ``str`` (Decision 7a). Critical for Redis / FalkorDB / Postgres
  checkpoint compatibility.
- Optional defaults: every field declared ``Optional[T] = None`` can
  be omitted at construction (Decision 8a). Critical for partial
  state updates that LangGraph reducers emit.
- ``extra="allow"`` permissiveness: unknown keys pass through during
  the migration's transition window. Critical for staged rollout.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from pydantic import BaseModel, ValidationError

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
    coerce_uuid,
)
from src.agents.ml_foundation.data_preparer.schemas import QCReportSchema
from src.agents.ml_foundation.model_trainer.schemas import MetricsSchema
from src.agents.ml_foundation.scope_definer.schemas import (
    ScopeSpecSchema,
    SuccessCriteriaSchema,
)

# --------------------------------------------------------------------------- #
# coerce_uuid + audit_workflow_id_validator                                   #
# --------------------------------------------------------------------------- #


def test_coerce_uuid_accepts_uuid_instance() -> None:
    """A UUID instance passes through unchanged."""
    u = uuid4()
    assert coerce_uuid(u) is u


def test_coerce_uuid_accepts_string_form() -> None:
    """A canonical UUID string is coerced to a UUID instance."""
    u = uuid4()
    s = str(u)
    coerced = coerce_uuid(s)
    assert isinstance(coerced, UUID)
    assert coerced == u


def test_coerce_uuid_rejects_malformed_string() -> None:
    """A non-UUID string raises ValueError (not silent corruption)."""
    with pytest.raises(ValueError):
        coerce_uuid("not-a-uuid")


def test_coerce_uuid_rejects_unknown_type() -> None:
    """A non-str non-UUID input raises ValueError."""
    with pytest.raises(ValueError):
        coerce_uuid(12345)


class _ProbeSchema(BaseAgentSchema):
    """Test-only schema exercising the audit_workflow_id_validator factory."""

    audit_workflow_id: UUID
    _validate_audit_id = audit_workflow_id_validator()


def test_audit_workflow_id_validator_accepts_uuid() -> None:
    """The factory-built validator accepts UUID instances."""
    u = uuid4()
    schema = _ProbeSchema(audit_workflow_id=u)
    assert schema.audit_workflow_id == u


def test_audit_workflow_id_validator_accepts_string() -> None:
    """The factory-built validator coerces str → UUID."""
    u = uuid4()
    schema = _ProbeSchema(audit_workflow_id=str(u))
    assert isinstance(schema.audit_workflow_id, UUID)
    assert schema.audit_workflow_id == u


def test_audit_workflow_id_validator_rejects_malformed() -> None:
    """The factory-built validator rejects malformed strings."""
    with pytest.raises(ValidationError):
        _ProbeSchema(audit_workflow_id="not-a-uuid")


# --------------------------------------------------------------------------- #
# BaseAgentSchema config — extra=allow                                        #
# --------------------------------------------------------------------------- #


def test_base_agent_schema_allows_extra_keys() -> None:
    """``extra="allow"`` keeps unknown keys in ``model_extra``.

    Critical for the migration: a pydantic-shaped agent receiving
    state from a TypedDict-shaped upstream would otherwise reject
    keys it does not declare.
    """

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty(unknown_key="surprise", another=42)
    assert instance.model_extra == {"unknown_key": "surprise", "another": 42}


def test_base_agent_schema_arbitrary_types_allowed() -> None:
    """``arbitrary_types_allowed=True`` permits non-pydantic types.

    ``model_trainer`` and ``feature_analyzer`` need this for fields
    like ``trained_model: Any`` and ``shap_values: np.ndarray``.
    """

    class _Custom:
        pass

    class _Carrier(BaseAgentSchema):
        payload: _Custom = None  # type: ignore[assignment]

    obj = _Custom()
    instance = _Carrier(payload=obj)
    assert instance.payload is obj


# --------------------------------------------------------------------------- #
# ScopeSpecSchema                                                             #
# --------------------------------------------------------------------------- #


def test_scope_spec_schema_constructs_empty() -> None:
    """Every field is Optional → empty construction is valid."""
    schema = ScopeSpecSchema()
    assert schema.experiment_id is None
    assert schema.problem_type is None


def test_scope_spec_schema_round_trips() -> None:
    """Representative scope_spec round-trips through JSON."""
    original = ScopeSpecSchema(
        experiment_id="exp_remi_us_20260504_abc123",
        experiment_name="Remibrutinib - HCP Engagement",
        problem_type="binary_classification",
        prediction_target="hcp_will_prescribe",
        prediction_horizon_days=30,
        prediction_timestamp="2026-04-26T00:00:00",
        cost_matrix={"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0},
        target_population="HCPs treating CSU patients",
        inclusion_criteria=["hcp_is_active", "has_patient_data"],
        exclusion_criteria=["test_accounts"],
        required_features=["hcp_specialty", "patient_count"],
        excluded_features=["hcp_name", "hcp_npi"],
        feature_categories=["demographics", "engagement"],
        regulatory_constraints=["HIPAA"],
        ethical_constraints=["no_protected_attributes"],
        technical_constraints=["inference_latency_<100ms"],
        minimum_samples=500,
        brand="Remibrutinib",
        region="US",
        use_case="commercial_targeting",
        created_by="scope_definer",
        created_at="2026-05-04T00:00:00",
    )
    json_str = original.model_dump_json()
    restored = ScopeSpecSchema.model_validate_json(json_str)
    assert restored == original


def test_scope_spec_schema_rejects_invalid_problem_type() -> None:
    """``problem_type`` is a Literal — unknown values fail validation."""
    with pytest.raises(ValidationError):
        ScopeSpecSchema(problem_type="not_a_real_type")  # type: ignore[arg-type]


def test_scope_spec_schema_passes_through_unknown_keys() -> None:
    """Unknown keys are tolerated via ``extra="allow"``."""
    schema = ScopeSpecSchema(future_field="reserved")  # type: ignore[call-arg]
    assert schema.model_extra == {"future_field": "reserved"}


# --------------------------------------------------------------------------- #
# SuccessCriteriaSchema                                                       #
# --------------------------------------------------------------------------- #


def test_success_criteria_schema_constructs_empty() -> None:
    """Empty SuccessCriteriaSchema is valid."""
    schema = SuccessCriteriaSchema()
    assert schema.minimum_auc is None
    assert schema.criteria_source is None


def test_success_criteria_schema_round_trips() -> None:
    """Representative success_criteria round-trips through JSON."""
    original = SuccessCriteriaSchema(
        experiment_id="exp_remi_us_20260504_abc123",
        baseline_model="logistic_regression",
        minimum_lift_over_baseline=0.10,
        minimum_auc=0.75,
        minimum_precision=0.60,
        minimum_recall=0.50,
        minimum_f1=0.55,
        criteria_source="adaptive",
    )
    json_str = original.model_dump_json()
    restored = SuccessCriteriaSchema.model_validate_json(json_str)
    assert restored == original


def test_success_criteria_schema_rejects_invalid_source() -> None:
    """``criteria_source`` is a Literal — unknown values fail."""
    with pytest.raises(ValidationError):
        SuccessCriteriaSchema(criteria_source="not_a_real_source")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# QCReportSchema                                                              #
# --------------------------------------------------------------------------- #


def test_qc_report_schema_constructs_empty() -> None:
    """Empty QCReportSchema is valid."""
    schema = QCReportSchema()
    assert schema.status is None


def test_qc_report_schema_round_trips() -> None:
    """Representative qc_report round-trips through JSON."""
    original = QCReportSchema(
        report_id="qc_rep_001",
        experiment_id="exp_remi_us_20260504_abc123",
        status="passed",
        overall_score=0.92,
        completeness_score=0.95,
        validity_score=0.90,
        consistency_score=0.93,
        uniqueness_score=0.99,
        timeliness_score=0.85,
        expectation_results=[{"expectation": "no_nulls", "result": "passed"}],
        failed_expectations=[],
        warnings=["timeliness_below_threshold"],
        remediation_steps=[],
        blocking_issues=[],
        row_count=10000,
        column_count=42,
        validated_at="2026-05-04T00:00:00",
    )
    json_str = original.model_dump_json()
    restored = QCReportSchema.model_validate_json(json_str)
    assert restored == original


def test_qc_report_schema_rejects_invalid_status() -> None:
    """``status`` is a Literal — unknown values fail."""
    with pytest.raises(ValidationError):
        QCReportSchema(status="not_a_real_status")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# MetricsSchema                                                               #
# --------------------------------------------------------------------------- #


def test_metrics_schema_constructs_empty() -> None:
    """Empty MetricsSchema is valid (scaffolding-window permissiveness)."""
    schema = MetricsSchema()
    assert schema.problem_type is None
    assert schema.auc_roc is None


def test_metrics_schema_round_trips_classification() -> None:
    """Classification metrics round-trip cleanly."""
    original = MetricsSchema(
        problem_type="binary_classification",
        auc_roc=0.85,
        f1_score=0.72,
        precision=0.78,
        recall=0.68,
        accuracy=0.81,
        log_loss=0.42,
        business_utility=1234.56,
    )
    json_str = original.model_dump_json()
    restored = MetricsSchema.model_validate_json(json_str)
    assert restored == original


def test_metrics_schema_round_trips_regression() -> None:
    """Regression metrics round-trip cleanly."""
    original = MetricsSchema(
        problem_type="regression",
        rmse=0.05,
        mae=0.03,
        r2=0.87,
        mape=0.12,
    )
    json_str = original.model_dump_json()
    restored = MetricsSchema.model_validate_json(json_str)
    assert restored == original


def test_metrics_schema_rejects_invalid_problem_type() -> None:
    """``problem_type`` is a Literal — unknown values fail."""
    with pytest.raises(ValidationError):
        MetricsSchema(problem_type="not_a_real_type")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Discriminating-coverage guard — ensure schemas actually fire on bad input   #
# --------------------------------------------------------------------------- #


def test_schemas_reject_non_optional_int_for_optional_float() -> None:
    """Pydantic v2 default config coerces int→float; this confirms the
    coercion is in effect for ``Optional[float]`` fields. If a future
    config change disables coercion, tests fixing ``minimum_auc=0.75``
    while passing ``0`` would silently break.
    """
    schema = SuccessCriteriaSchema(minimum_auc=1)  # int coerced to float
    assert schema.minimum_auc == 1.0
    assert isinstance(schema.minimum_auc, float)


def test_base_agent_schema_is_pydantic_v2() -> None:
    """Sanity check: pydantic v2 ``BaseModel`` API is in scope.

    Catches accidental pydantic v1 imports that would fail silently
    on the new ``model_dump_json`` / ``model_validate_json`` API.
    """
    assert hasattr(BaseAgentSchema, "model_dump_json")
    assert hasattr(BaseAgentSchema, "model_validate_json")
    assert issubclass(BaseAgentSchema, BaseModel)
