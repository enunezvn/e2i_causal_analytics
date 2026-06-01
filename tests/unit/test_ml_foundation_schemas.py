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
# BaseAgentSchema config — extra=ignore (D3 tightening, 2026-05-05)            #
# --------------------------------------------------------------------------- #


def test_base_agent_schema_drops_extra_keys_at_construction() -> None:
    """D3 (2026-05-05): ``extra="ignore"`` silently drops unknown keys at
    construction time. Pre-D3 they flowed through ``model_extra``; under
    the tightened config they are simply discarded.

    Note: dict-shim ``__setitem__`` continues to populate
    ``__pydantic_extra__`` directly (see ``test_dict_setitem_writes_extra_field``)
    so ``state["foo"] = v`` still works for ad-hoc keys — only construction
    is tightened.
    """

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty(unknown_key="surprise", another=42)
    # Under extra="ignore", model_extra is None (pydantic does not track
    # dropped keys).
    assert instance.model_extra is None or instance.model_extra == {}


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


def test_dict_access_reads_extra_field_after_setitem() -> None:
    """D3 (2026-05-05): under ``extra="ignore"`` constructor-time extras
    are dropped, but the dict-shim ``__setitem__`` still routes unknown
    keys to ``__pydantic_extra__``. So writing via ``instance["foo"] = v``
    and reading via ``instance["foo"]`` still round-trips.
    """

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty()
    instance["future_key"] = "reserved"
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


def test_contains_check_for_extra_field_after_setitem() -> None:
    """D3: ``key in state`` is True for keys written via ``__setitem__``
    after construction, even though constructor-time extras are dropped.
    """

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty()
    instance["future_key"] = "reserved"
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


def test_get_returns_value_for_extra_field_after_setitem() -> None:
    """D3: ``state.get("key")`` resolves keys written via ``__setitem__``
    after construction, even though constructor-time extras are dropped.
    """

    class _Empty(BaseAgentSchema):
        pass

    instance = _Empty()
    instance["extra_key"] = "present"
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


def test_scope_spec_schema_default_target_entity_codes_is_none() -> None:
    """Phase 2.9 Stage 2 PR-A: target_entity_codes is optional;
    cohort runner sets per cohort. Default None means no KG-mappable
    target representation (typical for synthetic regimes)."""
    schema = ScopeSpecSchema()
    assert schema.target_entity_codes is None
    assert schema.kg_cache_path is None


def test_scope_spec_schema_accepts_target_entity_codes() -> None:
    """Cohort runner populates target_entity_codes with the prediction
    target's KG entities (e.g., RxCUIs for bio_initiation target class)."""
    schema = ScopeSpecSchema(
        prediction_target="bio_initiation",
        target_entity_codes=[("RXNORM", "479158"), ("RXNORM", "1011295")],
        kg_cache_path="data/kg_cache/abc123__def456.json",
    )
    assert schema.target_entity_codes == [("RXNORM", "479158"), ("RXNORM", "1011295")]
    assert schema.kg_cache_path == "data/kg_cache/abc123__def456.json"


def test_scope_spec_schema_target_entity_codes_round_trips() -> None:
    """JSON round-trip preserves target_entity_codes shape (list of tuples
    becomes list of lists in JSON; Pydantic restores as tuples)."""
    original = ScopeSpecSchema(
        prediction_target="dupixent_init",
        target_entity_codes=[("RXNORM", "1011295")],
        kg_cache_path="/tmp/cache.json",
    )
    restored = ScopeSpecSchema.model_validate_json(original.model_dump_json())
    assert restored.target_entity_codes == original.target_entity_codes
    assert restored.kg_cache_path == original.kg_cache_path


def test_scope_spec_schema_drops_unknown_keys() -> None:
    """D3: under ``extra="ignore"`` unknown keys are silently dropped at
    construction. Pre-D3 they flowed through ``model_extra``; under the
    tightened config they are simply discarded.
    """
    schema = ScopeSpecSchema(future_field="reserved")  # type: ignore[call-arg]
    assert schema.model_extra is None or schema.model_extra == {}


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
        warnings=[{"expectation_type": "timeliness_below_threshold"}],
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

    state = DataPreparerState(audit_workflow_id=uuid4())
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

    state = DataPreparerState(audit_workflow_id=uuid4())
    with pytest.raises(ValidationError):
        state.qc_status = "not_a_real_status"  # type: ignore[assignment]


def test_setitem_validates_assignment_accepts_valid_values() -> None:
    """I1 fix: valid Literal values still pass through assignment."""
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    state = DataPreparerState(audit_workflow_id=uuid4())
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

    state = FeatureAnalyzerState(
        audit_workflow_id=uuid4(), shap_values=np.array([[0.1, 0.2], [0.3, 0.4]])
    )
    with pytest.raises(PydanticSerializationError):
        state.model_dump_json()


def test_feature_analyzer_state_json_dump_works_when_shap_values_is_none() -> None:
    """I4 companion: state without shap_values DOES serialize cleanly.

    Pins the half of the contract where the no-SHAP path works —
    confirms the serialization failure is specifically the ndarray
    field, not a structural pydantic regression.
    """
    from src.agents.ml_foundation.feature_analyzer.state import FeatureAnalyzerState

    state = FeatureAnalyzerState(
        audit_workflow_id=uuid4(), experiment_id="exp_test", problem_type="classification"
    )
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


# --------------------------------------------------------------------------- #
# D2.1 — hyperparameter_search_space wired into State as Dict[str, OptunaDistribution]
# --------------------------------------------------------------------------- #


def test_optuna_distribution_supports_dict_shim_get_for_low_high() -> None:
    """D2.1: extending ``BaseAgentSchema`` gives Optuna distribution
    instances the dict-shim accessors. The hyperparameter_tuner consumer
    reads ``config["low"]`` / ``config["high"]`` — we pin that the shim
    returns the declared field values rather than raising TypeError.
    """
    int_dist = OptunaIntDistribution(type="int", low=2, high=20, step=2)
    assert int_dist["low"] == 2
    assert int_dist["high"] == 20
    assert int_dist["step"] == 2
    assert int_dist.get("step") == 2
    assert int_dist.get("missing", "fallback") == "fallback"

    float_dist = OptunaFloatDistribution(type="float", low=1e-4, high=0.1, log=True)
    assert float_dist["low"] == 1e-4
    assert float_dist["high"] == 0.1
    assert float_dist["log"] is True
    assert float_dist.get("step") is None  # field declared, value None

    cat_dist = OptunaCategoricalDistribution(type="categorical", choices=["tpe", "random"])
    assert cat_dist["choices"] == ["tpe", "random"]


def test_optuna_distribution_extra_forbid_still_rejects_unknown_keys() -> None:
    """D2.1: even with BaseAgentSchema parent (which has extra="allow"),
    the OPTUNA-base override ``model_config = ConfigDict(extra="forbid")``
    is load-bearing — producer typos in algorithm_registry.py must raise
    at construction time, not silently route to model_extra.

    Pre-D2.1 this was guaranteed by extending plain BaseModel with
    extra="forbid". Post-D2.1 we extend BaseAgentSchema (which has
    extra="allow") and override extra="forbid" via ConfigDict merge.
    Pin that the merge actually preserves forbid behavior.
    """
    with pytest.raises(ValidationError, match="(?i)extra"):
        OptunaIntDistribution(  # type: ignore[call-arg]
            type="int",
            low=1,
            high=10,
            unknown_typo_field="should_raise",
        )


def test_model_trainer_state_validates_hyperparameter_search_space_dict_literal() -> None:
    """D2.1: ModelTrainerState constructed with a dict-of-dicts (today's
    producer shape from algorithm_registry.py) validates the entries
    into the OptunaDistribution discriminated union.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        hyperparameter_search_space={
            "n_estimators": {"type": "int", "low": 50, "high": 500, "step": 50},
            "learning_rate": {
                "type": "float",
                "low": 1e-4,
                "high": 0.3,
                "log": True,
            },
            "objective": {
                "type": "categorical",
                "choices": ["binary:logistic", "binary:hinge"],
            },
        },
    )

    space = state.hyperparameter_search_space
    assert space is not None
    assert isinstance(space["n_estimators"], OptunaIntDistribution)
    assert isinstance(space["learning_rate"], OptunaFloatDistribution)
    assert isinstance(space["objective"], OptunaCategoricalDistribution)

    # Consumer-side dict-shim access — what hyperparameter_tuner does today.
    assert space["n_estimators"]["low"] == 50
    assert space["learning_rate"]["log"] is True
    assert space["objective"]["choices"] == ["binary:logistic", "binary:hinge"]


def test_model_selector_state_validates_hyperparameter_search_space_dict_literal() -> None:
    """D2.1: ModelSelectorState (the producer side) accepts the same
    dict-of-dicts shape and validates into OptunaDistribution. Pins
    the cross-agent contract: model_selector emits, model_trainer reads,
    both use the same typed shape.
    """
    from src.agents.ml_foundation.model_selector.state import ModelSelectorState

    state = ModelSelectorState(
        audit_workflow_id=uuid4(),
        hyperparameter_search_space={
            "max_depth": {"type": "int", "low": 3, "high": 10},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
        },
    )
    space = state.hyperparameter_search_space
    assert space is not None
    assert isinstance(space["max_depth"], OptunaIntDistribution)
    assert isinstance(space["subsample"], OptunaFloatDistribution)


def test_model_trainer_state_rejects_invalid_hyperparameter_search_space() -> None:
    """D2.1: invalid distribution dicts raise at State construction time.
    This is the static-safety win — pre-D2.1 a producer typo like
    ``"type": "ilt"`` (typo'd "int") would silently flow as a plain dict
    until Optuna ran and probably ignored it. Now it raises immediately.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    # Invalid type discriminator
    with pytest.raises(ValidationError):
        ModelTrainerState(
            audit_workflow_id=uuid4(),
            hyperparameter_search_space={
                "x": {"type": "ilt", "low": 1, "high": 10},  # typo
            },
        )

    # Missing required field (high)
    with pytest.raises(ValidationError):
        ModelTrainerState(
            audit_workflow_id=uuid4(),
            hyperparameter_search_space={
                "x": {"type": "int", "low": 1},  # missing high
            },
        )

    # Categorical with empty choices
    with pytest.raises(ValidationError):
        ModelTrainerState(
            audit_workflow_id=uuid4(),
            hyperparameter_search_space={
                "x": {"type": "categorical", "choices": []},  # min_length=1
            },
        )


def test_model_trainer_state_hyperparameter_search_space_none_default() -> None:
    """D2.1: hyperparameter_search_space remains Optional[Dict[...]] with
    default None — the typed-schema wiring does not break partial-state
    construction.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(audit_workflow_id=uuid4())
    assert state.hyperparameter_search_space is None


def test_model_trainer_state_hyperparameter_search_space_round_trips_through_json() -> None:
    """D2.1: typed schema round-trips through model_dump → model_validate.
    Pins LangGraph checkpointer compatibility — a serialised state JSON
    round-trips back into the typed shape, not into raw dicts.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        hyperparameter_search_space={
            "lr": {"type": "float", "low": 1e-3, "high": 0.5, "log": True}
        },
    )
    dumped = state.model_dump()
    restored = ModelTrainerState.model_validate(dumped)

    space = restored.hyperparameter_search_space
    assert space is not None
    assert isinstance(space["lr"], OptunaFloatDistribution)
    assert space["lr"].low == 1e-3
    assert space["lr"].log is True


# --------------------------------------------------------------------------- #
# D2.2 — qc_report wired into ModelTrainerState/ModelSelectorState; consumer-
# contract fields qc_passed/qc_errors/qc_warnings declared on QCReportSchema
# --------------------------------------------------------------------------- #


def test_qc_report_schema_declares_consumer_contract_fields() -> None:
    """D2.2: qc_passed, qc_errors, qc_warnings are declared schema fields.
    Pre-D2.2 these were undeclared keys patched in by a runner shim at
    ``scripts/run_tier0_test.py:2295-2300, 2558+``. Hidden coupling.
    """
    schema = QCReportSchema(
        qc_passed=True,
        qc_errors=["err1", "err2"],
        qc_warnings=[{"expectation_type": "warn1"}],
    )
    assert schema.qc_passed is True
    assert schema.qc_errors == ["err1", "err2"]
    assert schema.qc_warnings == [{"expectation_type": "warn1"}]
    # Dict-shim access (consumer pattern at qc_gate_checker.py:30-46).
    assert schema["qc_passed"] is True
    assert schema.get("qc_errors", []) == ["err1", "err2"]
    assert schema.get("qc_warnings", []) == [{"expectation_type": "warn1"}]


def test_qc_report_schema_consumer_fields_default_to_none() -> None:
    """D2.2: omitted consumer fields default to None per Optional[T] = None.
    The dict-shim's ``get(key, default)`` coalesces None → default, so
    ``qc_report.get("qc_passed", False)`` returns False when the producer
    hasn't populated it. Matches pre-D2.2 missing-key behavior; this test
    pins the coalescing contract.
    """
    schema = QCReportSchema()
    assert schema.qc_passed is None
    assert schema.qc_errors is None
    assert schema.qc_warnings is None
    # Consumer pattern: get(key, default) returns default for None values.
    assert schema.get("qc_passed", False) is False
    assert schema.get("qc_errors", []) == []
    assert schema.get("qc_warnings", []) == []


def test_model_trainer_state_qc_report_validates_typed_schema() -> None:
    """D2.2: ModelTrainerState constructed with qc_report dict literal
    validates the dict into a typed QCReportSchema instance. Then consumer
    code reading ``state.qc_report.get("qc_passed", False)`` works through
    the dict-shim — same call shape as pre-D2.2 plain-dict reads.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        qc_report={
            "report_id": "qc_test_001",
            "experiment_id": "exp_xyz",
            "status": "passed",
            "overall_score": 0.92,
            "qc_passed": True,
            "qc_errors": [],
            "qc_warnings": [{"expectation_type": "minor_correlation"}],
        },
    )

    assert state.qc_report is not None
    assert isinstance(state.qc_report, QCReportSchema)
    assert state.qc_report.qc_passed is True
    assert state.qc_report.overall_score == 0.92
    # Consumer-pattern reads (qc_gate_checker.py:30-46).
    assert state.qc_report.get("qc_passed", False) is True
    assert state.qc_report.get("qc_errors", []) == []
    assert state.qc_report.get("qc_warnings", []) == [{"expectation_type": "minor_correlation"}]


def test_model_selector_state_qc_report_validates_typed_schema() -> None:
    """D2.2: same contract on ModelSelectorState (the other consumer)."""
    from src.agents.ml_foundation.model_selector.state import ModelSelectorState

    state = ModelSelectorState(
        audit_workflow_id=uuid4(),
        qc_report={
            "report_id": "qc_test_002",
            "qc_passed": False,
            "qc_errors": ["leakage_detected:f1"],
        },
    )

    assert state.qc_report is not None
    assert isinstance(state.qc_report, QCReportSchema)
    assert state.qc_report.get("qc_passed", False) is False
    assert state.qc_report.get("qc_errors", []) == ["leakage_detected:f1"]


def test_qc_report_schema_round_trips_through_json_d22() -> None:
    """D2.2: typed schema round-trips through model_dump → model_validate
    so LangGraph checkpointer can serialise/restore qc_report unchanged.
    """
    original = QCReportSchema(
        report_id="qc_rt_001",
        status="passed",
        overall_score=0.87,
        qc_passed=True,
        qc_errors=[],
        qc_warnings=[{"expectation_type": "w1"}],
    )
    dumped = original.model_dump()
    restored = QCReportSchema.model_validate(dumped)
    assert restored.qc_passed is True
    assert restored.qc_warnings == [{"expectation_type": "w1"}]


def test_qc_report_schema_consumer_contract_qc_gate_blocks_when_qc_passed_false() -> None:
    """D2.2: end-to-end consumer pattern. Replicates the QC-gate logic at
    ``model_trainer/nodes/qc_gate_checker.py:30-46`` against a typed
    qc_report. Pre-D2.2 the runner shim would default qc_passed=True
    (fail-open); post-D2.2 if the producer didn't populate qc_passed,
    the typed schema's None-default + dict-shim coalescing returns False
    (fail-closed) — a deliberate tightening.
    """
    qc_report = QCReportSchema(qc_passed=False, qc_errors=["leak:a", "leak:b"])

    qc_passed = qc_report.get("qc_passed", False)
    qc_errors = qc_report.get("qc_errors", [])

    assert qc_passed is False
    assert "leak:a" in qc_errors

    # Without qc_passed populated, fail-closed (None coalesces to default).
    qc_report_empty = QCReportSchema()
    assert qc_report_empty.get("qc_passed", False) is False


# --------------------------------------------------------------------------- #
# D2.3 — success_criteria wired into ModelTrainerState; SuccessCriteriaSchema
# adds 9 missing fields (minimum_mape, 6 v3 adaptive gates, 2 consumer keys).
# --------------------------------------------------------------------------- #


def test_success_criteria_schema_declares_v3_adaptive_gate_fields() -> None:
    """D2.3: SuccessCriteriaSchema declares the 6 v3 adaptive gate fields
    that ``adaptive_success_criteria()`` emits at criteria_validator.py
    lines 118-173. Pre-D2.3 these flowed through ``model_extra`` and the
    consumer's iteration-driven reads worked by accident.
    """
    schema = SuccessCriteriaSchema(
        minimum_net_benefit_at_p_t=0.05,
        minimum_mcc=0.30,
        maximum_calibration_slope_deviation=0.20,
        maximum_calibration_intercept_magnitude=0.10,
        maximum_calibration_error=0.05,
        maximum_train_val_delta=0.10,
    )
    assert schema.minimum_net_benefit_at_p_t == 0.05
    assert schema.minimum_mcc == 0.30
    assert schema.maximum_calibration_slope_deviation == 0.20
    assert schema.maximum_calibration_intercept_magnitude == 0.10
    assert schema.maximum_calibration_error == 0.05
    assert schema.maximum_train_val_delta == 0.10


def test_success_criteria_schema_declares_minimum_mape() -> None:
    """D2.3: minimum_mape is emitted by all 4 problem-type branches
    in criteria_validator. Pre-D2.3 it was undeclared.
    """
    schema = SuccessCriteriaSchema(minimum_mape=0.15, minimum_rmse=0.5)
    assert schema.minimum_mape == 0.15
    assert schema.minimum_rmse == 0.5


def test_success_criteria_schema_declares_consumer_injected_fields() -> None:
    """D2.3: clinical_threshold_range + dataset_disease are caller-injected
    keys read by ``model_trainer/nodes/evaluator.py``. Pre-D2.3 they
    flowed through ``model_extra``; declaring them as fields gives static
    safety + dict-shim get/contains semantics.
    """
    schema = SuccessCriteriaSchema(
        clinical_threshold_range={"low": 0.1, "high": 0.5},
        dataset_disease="diabetes",
    )
    assert schema.clinical_threshold_range == {"low": 0.1, "high": 0.5}
    assert schema.dataset_disease == "diabetes"


def test_success_criteria_schema_underscore_audit_keys_flow_via_model_extra() -> None:
    """D2.3: ``_adaptive_skipped``/``_adaptive_p_t``/``_adaptive_inputs``
    cannot be declared as pydantic v2 fields (reserved namespace). They
    continue to flow through ``model_extra`` via inherited ``extra="allow"``.
    """
    schema = SuccessCriteriaSchema.model_validate(
        {
            "minimum_auc": 0.75,
            "_adaptive_skipped": ["mcc", "net_benefit"],
            "_adaptive_p_t": {"clean": 0.5, "default": 0.6},
            "_adaptive_inputs": {"n_samples": 1000, "prevalence": 0.05},
        }
    )
    assert schema.minimum_auc == 0.75
    assert schema.model_extra is not None
    assert schema.model_extra["_adaptive_skipped"] == ["mcc", "net_benefit"]
    assert schema["_adaptive_inputs"] == {"n_samples": 1000, "prevalence": 0.05}


def test_model_trainer_state_success_criteria_validates_typed_schema() -> None:
    """D2.3: ModelTrainerState constructed with success_criteria dict
    literal validates cleanly into SuccessCriteriaSchema.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        success_criteria={
            "experiment_id": "exp_001",
            "minimum_auc": 0.75,
            "minimum_net_benefit_at_p_t": 0.04,
            "minimum_mcc": 0.30,
            "criteria_source": "adaptive",
        },
    )
    assert state.success_criteria is not None
    assert isinstance(state.success_criteria, SuccessCriteriaSchema)
    assert state.success_criteria.get("minimum_mcc", 0.0) == 0.30


def test_success_criteria_schema_v3_round_trips_through_json() -> None:
    """D2.3: full v3 SuccessCriteriaSchema round-trips through model_dump
    → model_validate. Pins LangGraph checkpointer compatibility.
    """
    original = SuccessCriteriaSchema(
        experiment_id="exp_rt",
        minimum_auc=0.80,
        minimum_mape=0.12,
        minimum_mcc=0.35,
        clinical_threshold_range={"low": 0.2, "high": 0.7},
        dataset_disease="cancer",
        criteria_source="adaptive",
    )
    dumped = original.model_dump()
    restored = SuccessCriteriaSchema.model_validate(dumped)
    assert restored.minimum_auc == 0.80
    assert restored.minimum_mape == 0.12
    assert restored.dataset_disease == "cancer"


def test_success_criteria_schema_omitted_v3_fields_default_to_none() -> None:
    """D2.3: all 9 new fields default to None when omitted (Decision 8a)."""
    schema = SuccessCriteriaSchema(minimum_auc=0.75)
    for field in (
        "minimum_mape",
        "minimum_net_benefit_at_p_t",
        "minimum_mcc",
        "maximum_calibration_slope_deviation",
        "maximum_calibration_intercept_magnitude",
        "maximum_calibration_error",
        "maximum_train_val_delta",
        "clinical_threshold_range",
        "dataset_disease",
    ):
        assert getattr(schema, field) is None


# --------------------------------------------------------------------------- #
# D2.4 — scope_spec wired into DataPreparerState/ModelSelectorState; ScopeSpec
# Schema gains 24 caller-injected consumer-side fields.
# --------------------------------------------------------------------------- #


def test_scope_spec_schema_declares_data_preparer_consumer_keys() -> None:
    """D2.4: ScopeSpecSchema declares the 24 consumer keys read across
    data_preparer nodes (date_column, required_columns, data_source,
    table_name, etc.). Pre-D2.4 these flowed through model_extra.
    """
    schema = ScopeSpecSchema(
        data_source="patients_v2",
        table_name="rwd.patients",
        date_column="event_date",
        required_columns=["patient_id", "event_date"],
        target_column="outcome",
        scaling_method="standard",
        encoding_method="onehot",
        imputation_strategy="median",
        filters={"region": "us"},
        entity_column="patient_id",
        split_date="2026-01-01",
        val_days=30,
        test_days=60,
        use_sample_data=True,
        sample_size=1000,
        event_date_column="event_date",
        target_date_column="outcome_date",
        feature_date_columns=["dx_date", "rx_date"],
        entity_key="patient",
        max_staleness_days=14.0,
        unique_columns=["patient_id"],
        expected_dtypes={"patient_id": "string"},
        exclude_columns=["raw_id"],
        extract_datetime_features=True,
    )
    assert schema.data_source == "patients_v2"
    assert schema.date_column == "event_date"
    assert schema.target_column == "outcome"
    assert schema.use_sample_data is True
    assert schema.feature_date_columns == ["dx_date", "rx_date"]


def test_data_preparer_state_scope_spec_validates_typed_schema() -> None:
    """D2.4: DataPreparerState constructed with scope_spec dict literal
    validates the dict into a typed ScopeSpecSchema instance.
    """
    from src.agents.ml_foundation.data_preparer.state import DataPreparerState

    state = DataPreparerState(
        audit_workflow_id=uuid4(),
        scope_spec={
            "experiment_id": "exp_001",
            "data_source": "rwd.patients_v2",
            "table_name": "rwd.patients_v2",
            "date_column": "event_date",
            "required_columns": ["patient_id", "event_date"],
            "split_date": "2026-01-01",
        },
    )
    assert state.scope_spec is not None
    assert isinstance(state.scope_spec, ScopeSpecSchema)
    assert state.scope_spec.get("data_source") == "rwd.patients_v2"
    assert state.scope_spec.get("date_column") == "event_date"
    assert state.scope_spec.get("split_date") == "2026-01-01"


def test_model_selector_state_scope_spec_validates_typed_schema() -> None:
    """D2.4: same contract on ModelSelectorState (lighter consumer)."""
    from src.agents.ml_foundation.model_selector.state import ModelSelectorState

    state = ModelSelectorState(
        audit_workflow_id=uuid4(),
        scope_spec={
            "experiment_id": "exp_002",
            "problem_type": "binary_classification",
            "technical_constraints": ["interpretability"],
        },
    )
    assert state.scope_spec is not None
    assert isinstance(state.scope_spec, ScopeSpecSchema)
    assert state.scope_spec.experiment_id == "exp_002"
    assert state.scope_spec.problem_type == "binary_classification"


def test_scope_spec_schema_omitted_consumer_fields_default_to_none() -> None:
    """D2.4: all 24 new consumer keys default to None when omitted."""
    schema = ScopeSpecSchema(experiment_id="e1")
    for field in (
        "data_source",
        "table_name",
        "date_column",
        "required_columns",
        "expected_dtypes",
        "unique_columns",
        "max_staleness_days",
        "target_column",
        "exclude_columns",
        "scaling_method",
        "encoding_method",
        "imputation_strategy",
        "extract_datetime_features",
        "filters",
        "entity_column",
        "split_date",
        "val_days",
        "test_days",
        "use_sample_data",
        "sample_size",
        "event_date_column",
        "target_date_column",
        "feature_date_columns",
        "entity_key",
    ):
        assert getattr(schema, field) is None, f"field {field} should be None"


def test_scope_spec_schema_d24_round_trips_through_json() -> None:
    """D2.4: scope_spec round-trips through JSON with all 24 new fields."""
    original = ScopeSpecSchema(
        experiment_id="exp_rt",
        data_source="src",
        date_column="evt",
        target_column="y",
        filters={"region": "us"},
        feature_date_columns=["dx", "rx"],
        val_days=30,
    )
    dumped = original.model_dump()
    restored = ScopeSpecSchema.model_validate(dumped)
    assert restored.data_source == "src"
    assert restored.feature_date_columns == ["dx", "rx"]
    assert restored.val_days == 30


# --------------------------------------------------------------------------- #
# D2.5 — validation_metrics wired into ModelDeployerState; MetricsSchema
# accepts roc_auc/auc_roc via AliasChoices and adds 14 runtime-emitted fields.
# --------------------------------------------------------------------------- #


def test_metrics_schema_accepts_roc_auc_alias() -> None:
    """D2.5: MetricsSchema's auc_roc field accepts both the canonical
    ``auc_roc`` python name AND the modern producer key ``roc_auc`` via
    AliasChoices.
    """
    via_canonical = MetricsSchema(auc_roc=0.85)
    via_modern = MetricsSchema.model_validate({"roc_auc": 0.85})
    assert via_canonical.auc_roc == 0.85
    assert via_modern.auc_roc == 0.85
    dumped = via_modern.model_dump(exclude_none=True)
    assert "auc_roc" in dumped
    assert dumped["auc_roc"] == 0.85


def test_metrics_schema_declares_classification_extras() -> None:
    """D2.5: per-class + extra metrics emitted by evaluator."""
    schema = MetricsSchema(
        f1_macro=0.7,
        f1_weighted=0.72,
        precision_class_0=0.65,
        precision_class_1=0.78,
        recall_class_0=0.62,
        recall_class_1=0.81,
        mcc=0.45,
        pr_auc=0.6,
        brier_score=0.18,
    )
    assert schema.f1_macro == 0.7
    assert schema.precision_class_1 == 0.78
    assert schema.mcc == 0.45


def test_metrics_schema_declares_threshold_and_calibration_fields() -> None:
    """D2.5: threshold metadata + calibration metrics."""
    schema = MetricsSchema(
        chosen_threshold=0.42,
        chosen_threshold_source="f1_optimal",
        calibration_slope=0.95,
        calibration_intercept=0.05,
        calibration_intercept_magnitude=0.05,
        calibration_slope_deviation=0.05,
        calibration_error=0.03,
        net_benefit_grid={"p_t=0.05": 0.45, "p_t=0.10": 0.40},
    )
    assert schema.chosen_threshold == 0.42
    assert schema.chosen_threshold_source == "f1_optimal"
    assert schema.calibration_slope == 0.95
    assert schema.net_benefit_grid == {"p_t=0.05": 0.45, "p_t=0.10": 0.40}


def test_metrics_schema_declares_lift_baseline_fields() -> None:
    """D2.5: lift / baseline comparison fields."""
    schema = MetricsSchema(
        baseline_test_auc=0.50,
        train_val_auc_delta=0.05,
        train_val_delta=0.05,
    )
    assert schema.baseline_test_auc == 0.50
    assert schema.train_val_auc_delta == 0.05


def test_model_deployer_state_validation_metrics_validates_typed_schema() -> None:
    """D2.5: ModelDeployerState validates dict literal into MetricsSchema."""
    from src.agents.ml_foundation.model_deployer.state import ModelDeployerState

    state = ModelDeployerState(
        audit_workflow_id=uuid4(),
        validation_metrics={
            "accuracy": 0.85,
            "precision": 0.78,
            "recall": 0.72,
            "f1_score": 0.75,
            "roc_auc": 0.88,
            "mcc": 0.40,
            "chosen_threshold": 0.5,
        },
    )
    assert state.validation_metrics is not None
    assert isinstance(state.validation_metrics, MetricsSchema)
    assert state.validation_metrics.auc_roc == 0.88
    assert state.validation_metrics.get("auc_roc", 0.0) == 0.88
    assert state.validation_metrics.get("mcc", 0.0) == 0.40


def test_metrics_schema_d25_round_trips_through_json() -> None:
    """D2.5: MetricsSchema round-trips through JSON."""
    original = MetricsSchema(
        problem_type="binary_classification",
        auc_roc=0.85,
        mcc=0.40,
        chosen_threshold=0.5,
        net_benefit_grid={"p_t=0.05": 0.45},
    )
    dumped = original.model_dump(exclude_none=True)
    restored = MetricsSchema.model_validate(dumped)
    assert restored.auc_roc == 0.85
    assert restored.mcc == 0.40
    assert restored.net_benefit_grid == {"p_t=0.05": 0.45}


def test_metrics_schema_omitted_d25_fields_default_to_none() -> None:
    """D2.5: all 14+ new fields default to None when omitted."""
    schema = MetricsSchema(auc_roc=0.85)
    for field in (
        "f1_macro",
        "f1_weighted",
        "precision_class_0",
        "precision_class_1",
        "recall_class_0",
        "recall_class_1",
        "mcc",
        "pr_auc",
        "brier_score",
        "chosen_threshold",
        "chosen_threshold_source",
        "calibration_slope",
        "calibration_intercept",
        "calibration_intercept_magnitude",
        "calibration_slope_deviation",
        "calibration_error",
        "net_benefit_grid",
        "baseline_test_auc",
        "train_val_auc_delta",
        "train_val_delta",
    ):
        assert getattr(schema, field) is None, f"{field} should default to None"


# --------------------------------------------------------------------------- #
# D2.5b — test_metrics wired into ModelTrainerState; MetricsSchema            #
# accepts both auc_roc and roc_auc producer-key forms via AliasChoices.       #
# --------------------------------------------------------------------------- #


def test_model_trainer_state_test_metrics_validates_typed_schema() -> None:
    """D2.5b: ModelTrainerState validates a producer-shape dict literal
    into MetricsSchema for the test_metrics field.

    Mirrors the D2.5 precedent
    (test_model_deployer_state_validation_metrics_validates_typed_schema)
    but for the test_metrics field, which was deferred from D2.5 to limit
    blast radius. The producer at evaluator.py:1410 emits the same shape
    for both validation_metrics and test_metrics, so this test pins that
    contract on the consumer-state side.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(
        audit_workflow_id=uuid4(),
        test_metrics={
            "roc_auc": 0.83,  # producer-modern alias resolves to auc_roc
            "f1_score": 0.70,
            "mcc": 0.38,
            "brier_score": 0.18,
            "chosen_threshold": 0.4,
            "chosen_threshold_source": "validation",
        },
    )
    assert state.test_metrics is not None
    assert isinstance(state.test_metrics, MetricsSchema)
    assert state.test_metrics.auc_roc == 0.83
    # Dict-shim access still works (load-bearing for ~30 reader sites in
    # model_trainer/nodes/* and scripts/run_tier0_test.py).
    assert state.test_metrics.get("auc_roc", 0.0) == 0.83
    assert state.test_metrics.get("mcc", 0.0) == 0.38
    assert state.test_metrics.get("nonexistent_key", 0.0) == 0.0


def test_model_trainer_state_test_metrics_round_trips_through_json() -> None:
    """D2.5b: test_metrics survives JSON checkpoint round-trip.

    Cross-references the integration test at
    tests/integration/test_agents/test_state_checkpoint_replay.py:431-476
    which exercises the same shape but in a heavier integration context.
    This unit test is the unit-level pin.
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    original = ModelTrainerState(
        audit_workflow_id=uuid4(),
        test_metrics=MetricsSchema(
            auc_roc=0.83,
            f1_score=0.70,
            problem_type="binary_classification",
        ),
    )
    # Use the actual checkpoint path (`model_validate_json` on a JSON string,
    # not the Python-object `model_dump` -> `model_validate` round-trip) per
    # codex P6 NIT — pins the alias-resolution code path that LangGraph
    # checkpointers exercise in production.
    restored = ModelTrainerState.model_validate_json(original.model_dump_json())
    assert restored.test_metrics is not None
    assert isinstance(restored.test_metrics, MetricsSchema)
    assert restored.test_metrics.auc_roc == 0.83


def test_model_trainer_state_test_metrics_omitted_defaults_to_none() -> None:
    """D2.5b: test_metrics omitted in input defaults to None per Decision 8a."""
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    state = ModelTrainerState(audit_workflow_id=uuid4())
    assert state.test_metrics is None


def test_model_trainer_state_test_metrics_accepts_realistic_evaluator_output() -> None:
    """D2.5b smoke: a realistic producer-shape dict from
    model_trainer/nodes/evaluator.py:_compute_split_classification_metrics
    validates cleanly into ModelTrainerState.test_metrics, AND the resulting
    MetricsSchema has zero model_extra (no key smuggled in unrecognized).

    Mirrors the D2.5 contract test pattern (per d2_investigation_20260505.md
    Field 4 "Add runtime smoke test: realistic evaluator output -> MetricsSchema,
    assert len(model_extra) == 0").
    """
    from src.agents.ml_foundation.model_trainer.state import ModelTrainerState

    # Shape mirrored from evaluator.py:1411 + the D2.5-added classification/
    # calibration/lift fields documented at schemas.py:84-113 + the D2.5b
    # evaluator-injected fields (minimum_lift_over_baseline at
    # evaluator.py:1226, calibrated_ece at evaluator.py:347-384) added per
    # codex P1+P5 review of PR #80.
    realistic = {
        "roc_auc": 0.83,
        "f1_score": 0.70,
        "f1_macro": 0.69,
        "f1_weighted": 0.71,
        "precision": 0.65,
        "recall": 0.75,
        "precision_class_0": 0.85,
        "precision_class_1": 0.65,
        "recall_class_0": 0.80,
        "recall_class_1": 0.75,
        "mcc": 0.38,
        "pr_auc": 0.72,
        "brier_score": 0.18,
        "chosen_threshold": 0.4,
        "chosen_threshold_source": "validation",
        "calibration_slope": 1.02,
        "calibration_intercept": 0.01,
        "calibration_intercept_magnitude": 0.01,
        "calibration_slope_deviation": 0.02,
        "calibration_error": 0.03,
        "net_benefit_grid": {"p_t=0.05": 0.51, "p_t=0.50": 0.52},
        "baseline_test_auc": 0.50,
        "train_val_auc_delta": 0.05,
        "minimum_lift_over_baseline": 0.04,  # evaluator.py:1226 (binary path)
        "calibrated_ece": 0.025,  # evaluator.py:347-384 (binary path)
    }
    state = ModelTrainerState(audit_workflow_id=uuid4(), test_metrics=realistic)
    assert state.test_metrics is not None
    assert isinstance(state.test_metrics, MetricsSchema)
    # Zero extra keys: every producer-emitted key was claimed by a declared
    # MetricsSchema field. If this assertion fires, the producer has drifted
    # and the schema needs a new field (NOT a `# type: ignore` workaround).
    assert len(state.test_metrics.model_extra or {}) == 0
