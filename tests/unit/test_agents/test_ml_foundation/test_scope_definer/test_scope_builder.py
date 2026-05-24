"""Tests for scope specification builder."""

from datetime import datetime

import pandas as pd
import pytest

from src.agents.ml_foundation.scope_definer.nodes.scope_builder import (
    _calculate_minimum_samples,
    _define_excluded_features,
    _define_inclusion_criteria,
    _define_target_population,
    _normalise_prediction_timestamp,
    _validate_cost_matrix,
    build_scope_spec,
)


@pytest.mark.asyncio
async def test_build_scope_spec_creates_complete_spec():
    """Test that build_scope_spec creates a complete ScopeSpec."""
    state = {
        "business_objective": "Increase prescriptions",
        "target_outcome": "Predict HCP prescribing",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "prediction_horizon_days": 30,
        "brand": "Remibrutinib",
        "region": "US",
        "use_case": "hcp_targeting",
    }

    result = await build_scope_spec(state)

    # Check required output fields
    assert "experiment_id" in result
    assert "experiment_name" in result
    assert "scope_spec" in result

    scope_spec = result["scope_spec"]

    # Check required ScopeSpec fields
    assert scope_spec["experiment_id"] == result["experiment_id"]
    assert scope_spec["problem_type"] == "binary_classification"
    assert scope_spec["prediction_target"] == "will_prescribe"
    assert scope_spec["prediction_horizon_days"] == 30
    assert "target_population" in scope_spec
    assert "inclusion_criteria" in scope_spec
    assert "exclusion_criteria" in scope_spec
    assert "required_features" in scope_spec
    assert "excluded_features" in scope_spec
    assert "feature_categories" in scope_spec
    assert "regulatory_constraints" in scope_spec
    assert "ethical_constraints" in scope_spec
    assert "technical_constraints" in scope_spec
    assert "minimum_samples" in scope_spec
    assert scope_spec["brand"] == "Remibrutinib"
    assert scope_spec["region"] == "US"
    assert scope_spec["created_by"] == "scope_definer"


@pytest.mark.asyncio
async def test_experiment_id_format():
    """Test that experiment_id follows expected format."""
    state = {
        "brand": "Remibrutinib",
        "region": "US",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
    }

    result = await build_scope_spec(state)

    experiment_id = result["experiment_id"]

    # Should start with "exp_"
    assert experiment_id.startswith("exp_")

    # Should contain brand code
    assert "remi" in experiment_id.lower()

    # Should contain region code
    assert "us" in experiment_id.lower()

    # Should contain timestamp (numeric) and UUID suffix
    parts = experiment_id.split("_")
    # Format: exp_{brand}_{region}_{timestamp}_{uuid}
    assert parts[-2].isdigit()  # Timestamp part
    assert len(parts[-1]) == 6  # UUID suffix (6 hex chars)


@pytest.mark.asyncio
async def test_experiment_name_includes_brand_and_outcome():
    """Test that experiment_name is human-readable."""
    state = {
        "brand": "Kisqali",
        "target_outcome": "Increase prescriptions",
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
    }

    result = await build_scope_spec(state)

    experiment_name = result["experiment_name"]

    assert "Kisqali" in experiment_name
    assert "Increase prescriptions" in experiment_name


def test_define_target_population_remibrutinib():
    """Test target population for Remibrutinib brand."""
    state = {"brand": "Remibrutinib"}

    population = _define_target_population(state)

    assert "CSU" in population or "Chronic Spontaneous Urticaria" in population


def test_define_target_population_fabhalta():
    """Test target population for Fabhalta brand."""
    state = {"brand": "Fabhalta"}

    population = _define_target_population(state)

    assert "PNH" in population or "Paroxysmal Nocturnal Hemoglobinuria" in population


def test_define_target_population_kisqali():
    """Test target population for Kisqali brand."""
    state = {"brand": "Kisqali"}

    population = _define_target_population(state)

    assert "breast cancer" in population.lower()
    assert "HR+" in population or "HER2-" in population


def test_define_target_population_generic():
    """Test target population for unknown brand."""
    state = {"brand": "UnknownBrand"}

    population = _define_target_population(state)

    # Should return generic population
    assert "HCP" in population


def test_define_inclusion_criteria_has_base_criteria():
    """Test that inclusion criteria always include base requirements."""
    state = {"brand": "test"}

    criteria = _define_inclusion_criteria(state)

    # Should always include base criteria
    assert "hcp_is_active" in criteria
    assert "has_patient_data" in criteria
    assert any("activity" in c.lower() for c in criteria)


def test_define_inclusion_criteria_brand_specific():
    """Test brand-specific inclusion criteria."""
    # Remibrutinib
    state_remi = {"brand": "Remibrutinib"}
    criteria_remi = _define_inclusion_criteria(state_remi)
    assert any("dermatology" in c.lower() or "allergy" in c.lower() for c in criteria_remi)

    # Fabhalta
    state_fab = {"brand": "Fabhalta"}
    criteria_fab = _define_inclusion_criteria(state_fab)
    assert any("hematology" in c.lower() for c in criteria_fab)

    # Kisqali
    state_kis = {"brand": "Kisqali"}
    criteria_kis = _define_inclusion_criteria(state_kis)
    assert any("oncology" in c.lower() for c in criteria_kis)


def test_define_excluded_features_prevents_pii():
    """Test that excluded features list prevents PII leakage."""
    state = {}

    excluded = _define_excluded_features(state)

    # Should exclude common PII fields
    pii_keywords = ["name", "npi", "ssn", "address", "phone", "email"]
    for keyword in pii_keywords:
        assert any(keyword in feat.lower() for feat in excluded)


def test_define_excluded_features_prevents_temporal_leakage():
    """Test that excluded features prevent temporal leakage."""
    state = {}

    excluded = _define_excluded_features(state)

    # Should exclude future data
    assert any("future" in feat.lower() for feat in excluded)


def test_calculate_minimum_samples_binary_classification():
    """Test minimum samples for binary classification."""
    min_samples = _calculate_minimum_samples("binary_classification")

    # Should require at least 500 samples for balanced classes
    assert min_samples >= 500


def test_calculate_minimum_samples_regression():
    """Test minimum samples for regression."""
    min_samples = _calculate_minimum_samples("regression")

    # Should require at least 300 samples
    assert min_samples >= 300


def test_calculate_minimum_samples_causal():
    """Test minimum samples for causal inference."""
    min_samples = _calculate_minimum_samples("causal_inference")

    # Should require more samples for treatment/control groups
    assert min_samples >= 1000


@pytest.mark.asyncio
async def test_build_scope_includes_regulatory_constraints():
    """Test that scope includes required regulatory constraints."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)

    regulatory = result["scope_spec"]["regulatory_constraints"]

    # Should include HIPAA and GDPR
    assert "HIPAA" in regulatory
    assert "GDPR" in regulatory


@pytest.mark.asyncio
async def test_build_scope_includes_ethical_constraints():
    """Test that scope includes ethical constraints."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)

    ethical = result["scope_spec"]["ethical_constraints"]

    # Should exclude protected attributes
    assert any("protected" in c.lower() or "race" in c.lower() for c in ethical)
    assert any("pii" in c.lower() for c in ethical)


@pytest.mark.asyncio
async def test_build_scope_uses_candidate_features_if_provided():
    """Test that provided candidate_features override defaults."""
    custom_features = ["custom_feature1", "custom_feature2"]

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
        "candidate_features": custom_features,
    }

    result = await build_scope_spec(state)

    required_features = result["scope_spec"]["required_features"]

    # Should use custom features
    assert required_features == custom_features


# === Block 1B: prediction_timestamp scaffolding =============================


def test_normalise_prediction_timestamp_handles_datetime():
    """``datetime`` inputs are normalised to ISO 8601 strings."""
    dt = datetime(2026, 4, 26, 12, 30, 45)
    assert _normalise_prediction_timestamp(dt) == dt.isoformat()


def test_normalise_prediction_timestamp_handles_pandas_timestamp():
    """``pd.Timestamp`` inputs are normalised to ISO 8601 strings."""
    ts = pd.Timestamp("2026-04-26T12:30:45")
    assert _normalise_prediction_timestamp(ts) == ts.isoformat()


def test_normalise_prediction_timestamp_passes_through_strings():
    """ISO-format string inputs are preserved verbatim."""
    s = "2026-04-26T12:30:45"
    assert _normalise_prediction_timestamp(s) == s


@pytest.mark.parametrize("value", [None, ""])
def test_normalise_prediction_timestamp_returns_none_for_empty(value):
    """Missing / empty inputs map to ``None`` (not "" or current time)."""
    assert _normalise_prediction_timestamp(value) is None


class _OpaqueObject:
    """Custom class with no datetime semantics — used in strict-reject tests."""


@pytest.mark.parametrize(
    "value",
    [
        1714089600,  # int (epoch-ish, but caller must convert explicitly)
        1714089600.5,  # float
        ["2026-04-26"],  # list
        {"date": "2026-04-26"},  # dict
        _OpaqueObject(),  # custom class without datetime semantics
    ],
)
def test_normalise_prediction_timestamp_rejects_unknown_types(value):
    """Block 1B-M2: unknown types fail loud rather than silently str()-coerce.

    Permissive ``str(value)`` coercion would mask upstream bugs — e.g. an
    epoch int silently becoming ``"1714089600"`` and downstream feature
    builders parsing that as the year 1714089600. The contract is now
    strict: only ``datetime``, ``pd.Timestamp``, parseable ``str``, or
    ``None``/``""``. Everything else raises ``TypeError`` at the
    scope_definer boundary.
    """
    with pytest.raises(TypeError, match="prediction_timestamp must be"):
        _normalise_prediction_timestamp(value)


def test_normalise_prediction_timestamp_rejects_unparseable_string():
    """Garbage strings that ``pd.Timestamp`` cannot parse are rejected.

    Without this check, "not a date" would round-trip verbatim through the
    helper and only blow up later inside a feature builder, with the failure
    point disconnected from the offending input. Fail loud at the boundary.
    """
    with pytest.raises(TypeError, match="not parseable by pd.Timestamp"):
        _normalise_prediction_timestamp("not a date")


@pytest.mark.asyncio
async def test_build_scope_propagates_prediction_timestamp_when_provided():
    """``prediction_timestamp`` from state lands on ``scope_spec``."""
    ts = pd.Timestamp("2026-04-26T00:00:00")
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
        "prediction_timestamp": ts,
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]

    assert "prediction_timestamp" in scope_spec
    assert scope_spec["prediction_timestamp"] == ts.isoformat()


@pytest.mark.asyncio
async def test_build_scope_prediction_timestamp_absent_when_unset():
    """Without ``prediction_timestamp`` in state, ``scope_spec`` has None."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]

    # Block 1B threading rule: the field is always present in the spec for a
    # stable schema, but its value is None when no timestamp was provided.
    assert "prediction_timestamp" in scope_spec
    assert scope_spec["prediction_timestamp"] is None


# === Block 5 (#10): cost_matrix scaffolding ================================


def test_validate_cost_matrix_accepts_full_dict():
    """All four required keys with numeric values → coerced to float dict."""
    cm = {"tp": 100, "fp": -10.5, "fn": -50, "tn": 0}
    result = _validate_cost_matrix(cm)
    assert result == {"tp": 100.0, "fp": -10.5, "fn": -50.0, "tn": 0.0}
    assert all(isinstance(v, float) for v in result.values())


@pytest.mark.parametrize("value", [None, {}])
def test_validate_cost_matrix_returns_none_for_empty(value):
    """None and {} both signal 'no cost matrix configured'."""
    assert _validate_cost_matrix(value) is None


def test_validate_cost_matrix_rejects_missing_keys():
    """A partial cost matrix is misconfiguration — fail loud at the boundary."""
    with pytest.raises(ValueError, match="missing required keys"):
        _validate_cost_matrix({"tp": 1.0, "fp": 0.0})


def test_validate_cost_matrix_rejects_non_numeric_values():
    """Strings/None/bools as values are rejected before the evaluator multiplies."""
    with pytest.raises(ValueError, match="must be int or float"):
        _validate_cost_matrix({"tp": "100", "fp": -10.0, "fn": -50.0, "tn": 0.0})


def test_validate_cost_matrix_rejects_non_dict():
    """The matrix must be a dict — lists/tuples are rejected."""
    with pytest.raises(ValueError, match="must be a dict"):
        _validate_cost_matrix([100, -10, -50, 0])


@pytest.mark.asyncio
async def test_build_scope_propagates_cost_matrix_when_provided():
    """A valid cost_matrix on state is forwarded onto scope_spec verbatim."""
    cm = {"tp": 100.0, "fp": -10.0, "fn": -50.0, "tn": 0.0}
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
        "cost_matrix": cm,
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]

    assert "cost_matrix" in scope_spec
    assert scope_spec["cost_matrix"] == cm


@pytest.mark.asyncio
async def test_build_scope_cost_matrix_absent_when_unset():
    """Without ``cost_matrix`` in state, ``scope_spec`` has None.

    The field is always present so the schema is stable; downstream code
    uses ``None`` as the "skip business_utility" signal.
    """
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "will_prescribe",
        "target_outcome": "Test",
        "brand": "Test",
    }

    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert "cost_matrix" in scope_spec
    assert scope_spec["cost_matrix"] is None


# ===========================================================================
# PR #462 hotfix F4 / D1: scope-level sufficiency.target_mde + target_mde_source
# ===========================================================================


@pytest.mark.asyncio
async def test_d1_user_override_passes_through_with_user_override_source():
    """F4 / D1: when the caller supplies sufficiency.target_mde explicitly,
    scope_builder must preserve it AND stamp target_mde_source='user_override'.
    No WARN is emitted (the user knows what they want)."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "sufficiency": {"target_mde": 0.07, "epv_floor": 10},
    }
    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert "sufficiency" in scope_spec
    assert scope_spec["sufficiency"]["target_mde"] == 0.07
    assert scope_spec["sufficiency"]["target_mde_source"] == "user_override"
    # Other user fields preserved
    assert scope_spec["sufficiency"]["epv_floor"] == 10


@pytest.mark.asyncio
async def test_d1_data_driven_binary_computes_from_baseline_rate():
    """F4 / D1: when caller pre-supplies baseline_rate at scope time
    (binary classification), scope_builder computes target_mde from data
    and stamps source='computed_from_data'. No WARN."""
    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "baseline_rate": 0.30,
    }
    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert scope_spec["sufficiency"]["target_mde_source"] == "computed_from_data"
    # max(0.05 floor, 0.20 * 0.30) = max(0.05, 0.06) = 0.06
    assert abs(scope_spec["sufficiency"]["target_mde"] - 0.06) < 1e-9


@pytest.mark.asyncio
async def test_d1_data_driven_regression_computes_from_sigma_outcome():
    """F4 / D1: regression path — pre-supplied sigma_outcome → continuous
    MDE = 0.5 * sigma, source='computed_from_data'."""
    state = {
        "inferred_problem_type": "regression",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "sigma_outcome": 4.0,
    }
    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert scope_spec["sufficiency"]["target_mde_source"] == "computed_from_data"
    assert scope_spec["sufficiency"]["target_mde"] == 2.0  # 0.5 * 4.0


@pytest.mark.asyncio
async def test_d1_literature_default_emits_loud_warning(caplog):
    """F4 / D1: when neither user override nor scope-time data signal is
    available AND the problem_type has a literature default, scope_builder
    falls back to literature AND emits a LOUD warning. The warning surfaces
    in BOTH the log AND the report's target_mde_source field (per audit
    chain need)."""
    import logging

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        # No baseline_rate, no user override → literature fallback.
    }
    with caplog.at_level(logging.WARNING):
        result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert scope_spec["sufficiency"]["target_mde_source"] == "literature_default"
    # WARN fires (audit signal in BOTH the log AND the field).
    assert any("literature default" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_d1_user_override_does_not_emit_warning(caplog):
    """F4 / D1: user-supplied target_mde → NO warning. Avoids warning
    fatigue (the prior implementation warned on every defaulted MDE
    in the DataPreparer instead of differentiating)."""
    import logging

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "sufficiency": {"target_mde": 0.10},
    }
    with caplog.at_level(logging.WARNING):
        await build_scope_spec(state)
    # No literature-default warning.
    assert not any("literature default" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_d1_data_driven_does_not_emit_warning(caplog):
    """F4 / D1: data-driven default → NO warning. The audit signal is
    in the target_mde_source field, not a noisy log."""
    import logging

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "baseline_rate": 0.30,
    }
    with caplog.at_level(logging.WARNING):
        await build_scope_spec(state)
    assert not any("literature default" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_d1_unset_for_multiclass_when_no_user_override():
    """F4 / D1: multiclass + time_series have no literature MDE convention
    at scope-time; scope_builder leaves `sufficiency` absent (or empty user
    payload). The DataPreparer's runtime resolver still has its chance
    once data is loaded."""
    state = {
        "inferred_problem_type": "multiclass_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
    }
    result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    # No sufficiency field at all (user provided nothing + no scope-time signal).
    assert "sufficiency" not in scope_spec


# ===========================================================================
# R2.4 (round-2): scope_builder must DEFER for causal_inference (no user
# override) — round-1 fell through to the literature-default branch despite
# the docstring saying "defer". Combined with the R2.3 resolver bug, every
# causal scope ended up with a fake user_override audit chain.
# ===========================================================================


@pytest.mark.asyncio
async def test_r24_causal_inference_defers_when_no_user_override(caplog):
    """R2.4: for causal_inference without a user override, scope_builder
    must NOT write a literature_default target_mde. The DataPreparer's
    resolver has access to baseline_rate from loaded data and is the
    correct place to compute the data-driven default with proper source
    attribution.

    Round-1 wrote ``target_mde=0.05, target_mde_source='literature_default'``
    AND emitted a LOUD WARN at scope-build time — contradicting the
    docstring (lines 95-97) that explicitly said "defer to the
    DataPreparer". Round-2 fix: no target_mde value, no scope-build
    warning, ``target_mde_source=None`` to signal "deferred to resolver".
    """
    import logging

    state = {
        "inferred_problem_type": "causal_inference",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        # No user override, no baseline_rate / sigma — pure defer case.
    }
    with caplog.at_level(logging.WARNING):
        result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    # No scope-time literature_default value written.
    if "sufficiency" in scope_spec:
        suff = scope_spec["sufficiency"]
        assert "target_mde" not in suff or suff.get("target_mde") is None, (
            f"causal_inference should defer target_mde to DataPreparer; got {suff}"
        )
        # target_mde_source set to None (explicit defer marker), NOT literature_default.
        assert suff.get("target_mde_source") is None, (
            f"causal_inference should set target_mde_source=None to defer; "
            f"got {suff.get('target_mde_source')!r}"
        )
    # No literature-default WARN emitted at scope-build time (the resolver
    # will warn IFF data load also fails to provide baseline_rate).
    assert not any(
        "literature default" in r.message and "causal_inference" in r.message
        for r in caplog.records
    ), (
        f"causal_inference defer path should not emit literature-default WARN; got {[r.message for r in caplog.records]}"
    )


@pytest.mark.asyncio
async def test_r24_causal_inference_user_override_still_wins(caplog):
    """R2.4: even on the deferral path, an explicit user override for
    causal_inference must still pass through with ``user_override``
    provenance. The deferral only applies when the user did NOT supply
    one — it doesn't disable the user-override entry point.
    """
    import logging

    state = {
        "inferred_problem_type": "causal_inference",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "sufficiency": {"target_mde": 0.08, "epv_floor": 10},
    }
    with caplog.at_level(logging.WARNING):
        result = await build_scope_spec(state)
    scope_spec = result["scope_spec"]
    assert scope_spec["sufficiency"]["target_mde"] == 0.08
    assert scope_spec["sufficiency"]["target_mde_source"] == "user_override"
    assert scope_spec["sufficiency"]["epv_floor"] == 10
    # And still no literature WARN.
    assert not any("literature default" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_r24_causal_inference_no_user_override_no_warn(caplog):
    """R2.4: the round-1 implementation always warned for causal at scope
    time. The R2.4 deferral path must be silent — the resolver inside
    sufficiency_check is responsible for the loud warning IFF the
    deferral can't be resolved after data load.
    """
    import logging

    state = {
        "inferred_problem_type": "causal_inference",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
    }
    with caplog.at_level(logging.WARNING):
        await build_scope_spec(state)
    assert not any("literature default" in r.message for r in caplog.records), (
        "scope-build should be silent on causal_inference defer path"
    )


# ===========================================================================
# R2.3 (round-2): scope_builder + resolver round-trip preserves the
# pre-stamped target_mde_source. Without R2.3, the resolver overwrites
# scope_builder's ``computed_from_data`` / ``literature_default`` stamps
# with ``user_override``, breaking the audit chain.
# ===========================================================================


@pytest.mark.asyncio
async def test_r23_scope_builder_resolver_roundtrip_preserves_computed_from_data():
    """R2.3: round-trip from scope_builder (which stamps
    ``computed_from_data`` for binary with baseline_rate at scope-time)
    through resolver must PRESERVE that stamp — not silently re-label as
    ``user_override``.
    """
    from src.utils.sufficiency_resolver import resolve_target_mde

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "baseline_rate": 0.30,
    }
    result = await build_scope_spec(state)
    scope_spec_suff = result["scope_spec"]["sufficiency"]
    # scope_builder stamped computed_from_data.
    assert scope_spec_suff["target_mde_source"] == "computed_from_data"
    # Round-trip through the resolver — stamp must survive.
    resolution = resolve_target_mde(
        user_config=scope_spec_suff,
        outcome_type="binary",
        baseline_rate=0.30,
    )
    assert resolution.source == "computed_from_data", (
        f"R2.3 audit-chain break: resolver re-stamped {resolution.source!r} "
        f"over scope_builder's 'computed_from_data'. The pre-stamp is "
        f"the upstream provenance signal and must be preserved."
    )
    assert resolution.value == scope_spec_suff["target_mde"]


@pytest.mark.asyncio
async def test_r23_scope_builder_resolver_roundtrip_preserves_literature_default():
    """R2.3: round-trip for the binary literature-default path (no
    user override, no baseline_rate) — scope_builder stamps
    ``literature_default``; resolver must preserve.
    """
    from src.utils.sufficiency_resolver import resolve_target_mde

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        # No baseline_rate → literature_default path.
    }
    result = await build_scope_spec(state)
    scope_spec_suff = result["scope_spec"]["sufficiency"]
    assert scope_spec_suff["target_mde_source"] == "literature_default"
    resolution = resolve_target_mde(
        user_config=scope_spec_suff,
        outcome_type="binary",
    )
    assert resolution.source == "literature_default"


@pytest.mark.asyncio
async def test_r23_user_override_stamp_is_preserved():
    """R2.3: when scope_builder stamps user_override (because the user
    DID supply target_mde), the resolver preserves it too. This case
    happens to look the same as the no-pre-stamp default; the test pins
    that the explicit ``user_override`` stamp round-trips correctly.
    """
    from src.utils.sufficiency_resolver import resolve_target_mde

    state = {
        "inferred_problem_type": "binary_classification",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "sufficiency": {"target_mde": 0.07},
    }
    result = await build_scope_spec(state)
    scope_spec_suff = result["scope_spec"]["sufficiency"]
    assert scope_spec_suff["target_mde_source"] == "user_override"
    resolution = resolve_target_mde(
        user_config=scope_spec_suff,
        outcome_type="binary",
        baseline_rate=0.30,
    )
    assert resolution.source == "user_override"
    assert resolution.value == 0.07


# ===========================================================================
# R2.2 (round-2): scope_builder + schema can produce continuous-outcome
# raw effect-size MDEs above 1.0 (sigma > 2.0). Round-1's schema bound
# (lt=1.0) silently rejected these via ScopeSpecSchema validation.
# ===========================================================================


@pytest.mark.asyncio
async def test_r22_regression_with_large_sigma_outcome_survives_schema():
    """R2.2: when scope_builder computes ``target_mde = 0.5 * sigma_outcome``
    for regression with ``sigma_outcome > 2.0``, the value is >= 1.0. The
    schema bound used to be ``lt=1.0`` → ScopeSpecSchema validation
    silently rejected the legitimate MDE. R2.2 widens to ``lt=1e6`` and
    documents the dual semantic: binary = absolute risk difference in
    (0, 1); continuous = raw effect size in outcome units, can be >= 1.0.
    """
    from src.agents.ml_foundation.scope_definer.schemas import ScopeSpecSchema

    state = {
        "inferred_problem_type": "regression",
        "inferred_target_variable": "y",
        "target_outcome": "Test",
        "brand": "Test",
        "sigma_outcome": 4.0,  # → target_mde = 2.0, was rejected pre-R2.2
    }
    result = await build_scope_spec(state)
    scope_spec_dict = result["scope_spec"]
    # scope_builder computed the value.
    assert scope_spec_dict["sufficiency"]["target_mde"] == 2.0
    assert scope_spec_dict["sufficiency"]["target_mde_source"] == "computed_from_data"
    # And the resulting scope_spec round-trips through pydantic without rejection.
    schema = ScopeSpecSchema(**scope_spec_dict)
    assert schema.sufficiency is not None
    assert schema.sufficiency.target_mde == 2.0


# ===========================================================================
# Round-3 finding: END-TO-END audit chain. The R2.3 round-trip tests above
# call resolve_target_mde directly; they never exercise the sufficiency_check
# NODE (_extract_sufficiency_config + the per-classifier mde_assumption
# construction) that actually copies the provenance into the persisted
# report.mde_assumption_used['source']. These two tests drive the full
# build_scope_spec -> run_sufficiency_check -> report chain.
# ===========================================================================


def _e2e_binary_df(n: int, prevalence: float, n_features: int):
    import numpy as np

    rng = np.random.default_rng(seed=7)
    data = {f"x{i}": rng.normal(size=n) for i in range(n_features)}
    y = np.zeros(n, dtype=int)
    y[: int(round(n * prevalence))] = 1
    rng.shuffle(y)
    data["y"] = pd.Series(y)
    return pd.DataFrame(data)


@pytest.mark.asyncio
async def test_e2e_audit_chain_binary_source_flows_to_report():
    """build_scope_spec stamps ``computed_from_data`` at the scope boundary;
    the node must carry that provenance into report.mde_assumption_used.
    """
    from uuid import uuid4

    from src.agents.ml_foundation.data_preparer.nodes.sufficiency_check import (
        run_sufficiency_check,
    )

    scope_result = await build_scope_spec(
        {
            "inferred_problem_type": "binary_classification",
            "inferred_target_variable": "y",
            "target_outcome": "Test",
            "brand": "Test",
            "baseline_rate": 0.30,
        }
    )
    sufficiency = scope_result["scope_spec"]["sufficiency"]
    assert sufficiency["target_mde_source"] == "computed_from_data"

    node_state = {
        "audit_workflow_id": uuid4(),
        "experiment_id": "test-exp",
        "scope_spec": {
            "problem_type": "binary_classification",
            "prediction_target": "y",
            "experiment_id": "test-exp",
            "sufficiency": sufficiency,
        },
        "train_df": _e2e_binary_df(4000, 0.30, 8),
        "target_rate": 0.30,
        "blocking_issues": [],
    }
    report = (await run_sufficiency_check(node_state))["sufficiency_report"]
    assert report["mde_assumption_used"]["source"] == "computed_from_data", (
        "audit-chain break: scope_builder stamped 'computed_from_data' but the "
        f"node reported {report['mde_assumption_used']['source']!r}"
    )


@pytest.mark.asyncio
async def test_e2e_audit_chain_causal_defer_no_fake_user_override():
    """R2.4 end-to-end: a causal scope with NO user override defers
    target_mde at scope time; the node then resolves from loaded data and
    must record a GENUINE provenance — never a fake ``user_override``.
    """
    from uuid import uuid4

    from src.agents.ml_foundation.data_preparer.nodes.sufficiency_check import (
        run_sufficiency_check,
    )

    scope_result = await build_scope_spec(
        {
            "inferred_problem_type": "causal_inference",
            "inferred_target_variable": "y",
            "target_outcome": "Test",
            "brand": "Test",
        }
    )
    sufficiency = scope_result["scope_spec"].get("sufficiency") or {}
    # Defer contract: no scope-time target_mde, source explicitly None.
    assert sufficiency.get("target_mde") is None
    assert sufficiency.get("target_mde_source") is None

    node_state = {
        "audit_workflow_id": uuid4(),
        "experiment_id": "test-exp",
        "scope_spec": {
            "problem_type": "causal_inference",
            "prediction_target": "y",
            "experiment_id": "test-exp",
            "sufficiency": sufficiency,
        },
        "train_df": _e2e_binary_df(3000, 0.30, 6),
        "target_rate": 0.30,
        "blocking_issues": [],
    }
    report = (await run_sufficiency_check(node_state))["sufficiency_report"]
    mde = report.get("mde_assumption_used")
    assert mde is not None, "causal node should populate mde_assumption_used"
    assert mde["source"] != "user_override", (
        "R2.4 regression: causal scope with NO user override produced a FAKE "
        "'user_override' audit label"
    )
    assert mde["source"] in ("computed_from_data", "literature_default")
