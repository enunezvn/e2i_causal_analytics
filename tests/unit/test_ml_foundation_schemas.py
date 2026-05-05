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
from pydantic import BaseModel, TypeAdapter, ValidationError

from src.agents.ml_foundation._pydantic_utils import (
    BaseAgentSchema,
    audit_workflow_id_validator,
    coerce_uuid,
)
from src.agents.ml_foundation.data_preparer.schemas import QCReportSchema
from src.agents.ml_foundation.model_trainer.schemas import (
    MetricsSchema,
    OptunaCategoricalDistribution,
    OptunaDistribution,
    OptunaFloatDistribution,
    OptunaIntDistribution,
)
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
# BaseAgentSchema TypedDict-compat dict-like accessors                        #
# --------------------------------------------------------------------------- #
#
# These accessors are the Shard A enabler: they let the 270+ existing
# ``state["key"]`` / ``state.get("key")`` call sites across the
# ml_foundation node files keep working unchanged after the State
# classes migrate from TypedDict to pydantic BaseModel.


def test_dict_access_reads_declared_field() -> None:
    """``state["key"]`` returns the declared-field value."""
    schema = SuccessCriteriaSchema(minimum_auc=0.75)
    assert schema["minimum_auc"] == 0.75


def test_dict_access_reads_extra_field() -> None:
    """``state["key"]`` returns ``model_extra`` values for unknown keys."""

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty(future_key="reserved")
    assert instance["future_key"] == "reserved"


def test_dict_access_raises_keyerror_for_unknown_key() -> None:
    """``state["key"]`` raises ``KeyError`` when key is genuinely absent."""
    schema = SuccessCriteriaSchema()
    with pytest.raises(KeyError):
        _ = schema["totally_unknown_key"]


def test_dict_access_returns_none_for_unset_optional() -> None:
    """``state["key"]`` returns ``None`` for Optional fields that default to None.

    This is a documented semantic shift from TypedDict — the
    ``Optional[T] = None`` default is materialised at construction
    time. Use ``key in state`` to discriminate "missing field" from
    "field set to None" if needed.
    """
    schema = SuccessCriteriaSchema()
    assert schema["minimum_auc"] is None  # not KeyError


def test_dict_setitem_writes_declared_field() -> None:
    """``state["key"] = value`` updates the declared field via attribute set."""
    schema = SuccessCriteriaSchema()
    schema["minimum_auc"] = 0.85
    assert schema.minimum_auc == 0.85
    assert schema["minimum_auc"] == 0.85


def test_dict_setitem_writes_extra_field() -> None:
    """``state["key"] = value`` writes unknown keys to ``model_extra``."""

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty()
    instance["new_key"] = "value"
    assert instance.model_extra == {"new_key": "value"}
    assert instance["new_key"] == "value"


def test_contains_check_for_declared_field() -> None:
    """``key in state`` is True for declared fields (even when value is None)."""
    schema = SuccessCriteriaSchema()
    assert "minimum_auc" in schema  # declared, even though None
    assert "totally_unknown_key" not in schema


def test_contains_check_for_extra_field() -> None:
    """``key in state`` is True for keys in ``model_extra``."""

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty(future_key="reserved")
    assert "future_key" in instance


def test_contains_check_rejects_non_string() -> None:
    """``int in state`` is False (not TypeError); matches dict semantics for foreign types."""
    schema = SuccessCriteriaSchema()
    assert (42 in schema) is False


def test_get_returns_value_for_declared_field() -> None:
    """``state.get("key")`` returns the value when set."""
    schema = SuccessCriteriaSchema(minimum_auc=0.75)
    assert schema.get("minimum_auc") == 0.75


def test_get_returns_default_for_unset_optional() -> None:
    """``state.get("key", default)`` returns default when value is None.

    This is the Shard A migration shim: pydantic-Optional fields with
    ``=None`` default look "set to None" but the existing call sites
    expect ``default`` returned in that case.
    """
    schema = SuccessCriteriaSchema()  # minimum_auc defaults to None
    assert schema.get("minimum_auc", 0.5) == 0.5


def test_get_returns_default_for_unknown_key() -> None:
    """``state.get("key", default)`` returns default when key is genuinely absent."""
    schema = SuccessCriteriaSchema()
    assert schema.get("totally_unknown_key", "fallback") == "fallback"


def test_get_returns_none_for_unset_optional_no_default() -> None:
    """``state.get("key")`` (no default) returns None when value is None."""
    schema = SuccessCriteriaSchema()
    assert schema.get("minimum_auc") is None


def test_get_returns_value_for_extra_field() -> None:
    """``state.get("key")`` resolves keys in ``model_extra`` too."""

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty(extra_key="present")
    assert instance.get("extra_key") == "present"


def test_get_returns_default_when_extra_value_is_none() -> None:
    """``state.get("key", default)`` returns default when extra-field value is None."""

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty(extra_key=None)
    assert instance.get("extra_key", "fallback") == "fallback"


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
# OptunaDistribution discriminated union (Shard B deliverable)                #
# --------------------------------------------------------------------------- #

_OPTUNA_ADAPTER = TypeAdapter(OptunaDistribution)


def test_optuna_int_distribution_constructs() -> None:
    """Int distribution accepts low/high + optional step/log."""
    dist = OptunaIntDistribution(type="int", low=3, high=10, step=1)
    assert dist.low == 3
    assert dist.high == 10
    assert dist.log is None


def test_optuna_float_distribution_constructs() -> None:
    """Float distribution accepts low/high + log scale."""
    dist = OptunaFloatDistribution(type="float", low=1e-4, high=0.1, log=True)
    assert dist.low == 1e-4
    assert dist.high == 0.1
    assert dist.log is True


def test_optuna_categorical_distribution_requires_choices() -> None:
    """Categorical distribution rejects empty choices (min_length=1)."""
    OptunaCategoricalDistribution(type="categorical", choices=["tpe", "random"])
    with pytest.raises(ValidationError):
        OptunaCategoricalDistribution(type="categorical", choices=[])


def test_optuna_distribution_discriminator_routes_int() -> None:
    """The discriminated union routes ``{type: "int"}`` to IntDistribution."""
    dist = _OPTUNA_ADAPTER.validate_python({"type": "int", "low": 3, "high": 10})
    assert isinstance(dist, OptunaIntDistribution)


def test_optuna_distribution_discriminator_routes_float() -> None:
    """The discriminated union routes ``{type: "float"}`` to FloatDistribution."""
    dist = _OPTUNA_ADAPTER.validate_python({"type": "float", "low": 1e-4, "high": 0.1, "log": True})
    assert isinstance(dist, OptunaFloatDistribution)


def test_optuna_distribution_discriminator_routes_categorical() -> None:
    """The discriminated union routes ``{type: "categorical"}`` to CategoricalDistribution."""
    dist = _OPTUNA_ADAPTER.validate_python({"type": "categorical", "choices": ["a", "b", "c"]})
    assert isinstance(dist, OptunaCategoricalDistribution)


def test_optuna_distribution_rejects_unknown_type() -> None:
    """The discriminated union rejects unknown ``type`` values."""
    with pytest.raises(ValidationError):
        _OPTUNA_ADAPTER.validate_python({"type": "not_real", "low": 0, "high": 1})


def test_optuna_distribution_rejects_extra_keys() -> None:
    """Each variant has ``extra="forbid"`` — unknown keys fail validation.

    Strictness is intentional: a typo in distribution metadata
    (e.g., 'lwr' instead of 'low') would silently route to the wrong
    Optuna sampler if extra="allow" let it through.
    """
    with pytest.raises(ValidationError):
        OptunaIntDistribution(  # type: ignore[call-arg]
            type="int", low=0, high=1, unknown_key="oops"
        )


def test_optuna_distribution_round_trips_through_json() -> None:
    """Discriminated union round-trips: dict → adapter → JSON → adapter."""
    original = _OPTUNA_ADAPTER.validate_python(
        {"type": "float", "low": 0.001, "high": 1.0, "log": True}
    )
    json_bytes = _OPTUNA_ADAPTER.dump_json(original)
    restored = _OPTUNA_ADAPTER.validate_json(json_bytes)
    assert restored == original


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


# --------------------------------------------------------------------------- #
# Codex review fixes — pinning contracts                                      #
# --------------------------------------------------------------------------- #


def test_setitem_validates_assignment_rejects_off_spec_literal() -> None:
    """I1 fix: ``state["status"] = off_spec_value`` must raise ValidationError.

    Pre-fix: ``BaseAgentSchema.__setitem__`` called ``setattr`` which
    bypassed pydantic field validation in pydantic v2 (``setattr`` does
    NOT trigger validators unless ``validate_assignment=True`` is set
    on ``model_config``). Post-fix: model_config sets validate_assignment
    so every assignment runs the validator pipeline.

    This test pins the contract — if a future config change drops
    ``validate_assignment``, off-spec writes would silently corrupt
    state again.
    """
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    state = DataPreparerState()
    with pytest.raises(ValidationError):
        # qc_status is Optional[Literal["passed","failed","warning","skipped"]]
        # — "not_a_real_status" must be rejected loud at assignment time.
        state["qc_status"] = "not_a_real_status"


def test_setattr_also_validates_assignment() -> None:
    """I1 fix: ``state.field = off_spec_value`` (attribute form) is
    equivalent to ``state["field"] = ...`` for validation purposes.

    Pydantic v2 routes BOTH attribute and __setitem__ assignments
    through the same validator pipeline when validate_assignment=True.
    Pinning both forms catches a regression in either path.
    """
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    state = DataPreparerState()
    with pytest.raises(ValidationError):
        state.qc_status = "not_a_real_status"  # type: ignore[assignment]


def test_setitem_validates_assignment_accepts_valid_values() -> None:
    """I1 fix: valid Literal values still pass through assignment."""
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    state = DataPreparerState()
    state["qc_status"] = "passed"
    assert state.qc_status == "passed"
    state["qc_status"] = "failed"
    assert state.qc_status == "failed"


def test_contains_get_asymmetry_for_none_valued_declared_field() -> None:
    """I2 fix: pin the documented semantic asymmetry between
    ``key in state`` and ``state.get(key, default)`` for declared
    fields with value None.

    - ``"minimum_auc" in schema`` returns True (declared field exists).
    - ``schema.get("minimum_auc", 0.5)`` returns 0.5 (None coalesced).

    This is intentional — see ``BaseAgentSchema.__contains__`` and
    ``BaseAgentSchema.get`` docstrings. The test pins the contract so
    a future "change __contains__ to return False on None" edit fires
    a CI failure rather than silently breaking caller logic.
    """
    schema = SuccessCriteriaSchema()  # minimum_auc defaults to None
    assert "minimum_auc" in schema
    assert schema.get("minimum_auc", 0.5) == 0.5
    # The "discriminating" idiom for callers that need to distinguish
    # "set to None" from "default":
    assert "minimum_auc" in schema and schema.minimum_auc is None


def test_feature_analyzer_state_shap_values_json_dump_raises() -> None:
    """I4 fix: pin the documented un-serializable surface for
    ``shap_values: np.ndarray``.

    Pydantic v2 has no built-in serializer for numpy.ndarray. Calling
    ``model_dump_json()`` on a FeatureAnalyzerState with non-None
    shap_values must raise PydanticSerializationError. Sub-shard D5
    is queued to add a ``@field_serializer`` if SHAP needs JSON
    checkpointing in the future. This test fires loud if anyone
    inadvertently tries to JSON-checkpoint state with SHAP values.
    """
    import numpy as np
    from pydantic_core import PydanticSerializationError

    from src.agents.ml_foundation.feature_analyzer.state import FeatureAnalyzerState

    state = FeatureAnalyzerState(shap_values=np.array([[0.1, 0.2], [0.3, 0.4]]))
    with pytest.raises(PydanticSerializationError):
        state.model_dump_json()


def test_feature_analyzer_state_json_dump_works_when_shap_values_is_none() -> None:
    """I4 companion: state without shap_values DOES serialize cleanly.

    Pins the half of the contract where the no-SHAP path works —
    confirms the serialization failure is specifically the ndarray
    field, not a structural pydantic regression.
    """
    from src.agents.ml_foundation.feature_analyzer.state import FeatureAnalyzerState

    state = FeatureAnalyzerState(experiment_id="exp_test", problem_type="classification")
    # Should NOT raise.
    json_str = state.model_dump_json()
    assert "exp_test" in json_str


def test_metrics_schema_permits_empty_metrics_for_stated_problem_type() -> None:
    """M2 fix: pin the documented permissive behavior of
    ``MetricsSchema._check_metrics_subset_for_problem_type``.

    The validator is intentionally non-enforcing (returns ``self``
    even when ``problem_type=binary_classification`` and no metrics
    are set). Documented as a placeholder for future tightening.
    This test fires if someone changes the validator to raise without
    coordinating the trainer-side stop point.
    """
    # binary_classification with no metrics — should NOT raise.
    schema = MetricsSchema(problem_type="binary_classification")
    assert schema.problem_type == "binary_classification"
    assert all(
        getattr(schema, m) is None
        for m in ("auc_roc", "f1_score", "precision", "recall", "accuracy", "log_loss")
    )

    # regression with no metrics — should NOT raise either.
    schema_reg = MetricsSchema(problem_type="regression")
    assert schema_reg.problem_type == "regression"
    assert all(getattr(schema_reg, m) is None for m in ("rmse", "mae", "r2", "mape"))
