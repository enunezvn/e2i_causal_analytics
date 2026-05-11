"""Unit tests for v5 Gate B3 feature engineering.

Covers:
1. CSU + Optum manifest contracts declared for all 10 candidates with
   ``knowable_at=index_date`` and pre-anchor derivation chains.
2. Helper ``engineer_features`` correctness on small synthetic
   DataFrames (CSU + Optum dispatch).
3. Node wrapper ``engineer_features_node`` gating: OFF by default
   (no-op patch), ON applies transforms to all splits.
4. Dispatch on ``scope_spec.feature_manifest_source``: csu, optum,
   unknown (no-op).
5. Missing input columns -> feature silently skipped (best-effort).
6. Edge cases: NaN inputs, divide-by-zero clamping, categorical
   factorization stability.

Pre-spec: docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.nodes.feature_engineering import (
    CSU_ENGINEERED_FEATURES,
    OPTUM_ENGINEERED_FEATURES,
    engineer_features,
    engineer_features_node,
)
from src.data.manifests.csu_feature_manifest import (
    CSU_FEATURES,
    CSU_SAFE_FEATURES,
)
from src.data.manifests.optum_feature_manifest import (
    OPTUM_FEATURES,
    OPTUM_SAFE_FEATURES,
)

# =============================================================================
# Manifest declarations
# =============================================================================


def test_csu_engineered_features_all_in_manifest():
    """Each engineered feature has a matching FeatureContract."""
    contracts = {c.name: c for c in CSU_FEATURES}
    for name in CSU_ENGINEERED_FEATURES:
        assert name in contracts, f"{name} missing from CSU_FEATURES"


def test_optum_engineered_features_all_in_manifest():
    """Each engineered feature has a matching FeatureContract."""
    contracts = {c.name: c for c in OPTUM_FEATURES}
    for name in OPTUM_ENGINEERED_FEATURES:
        assert name in contracts, f"{name} missing from OPTUM_FEATURES"


def test_csu_engineered_features_declared_index_date():
    """All CSU engineered features are knowable_at=index_date (pre-anchor)."""
    contracts = {c.name: c for c in CSU_FEATURES}
    for name in CSU_ENGINEERED_FEATURES:
        c = contracts[name]
        assert c.knowable_at.reference == "index_date", (
            f"{name} declared knowable_at={c.knowable_at.reference!r}, expected 'index_date'"
        )


def test_optum_engineered_features_declared_index_date():
    """All Optum engineered features are knowable_at=index_date (pre-anchor)."""
    contracts = {c.name: c for c in OPTUM_FEATURES}
    for name in OPTUM_ENGINEERED_FEATURES:
        c = contracts[name]
        assert c.knowable_at.reference == "index_date", (
            f"{name} declared knowable_at={c.knowable_at.reference!r}, expected 'index_date'"
        )


def test_csu_engineered_features_in_safe_view():
    """Engineered features appear in CSU_SAFE_FEATURES (pre-or-at-index)."""
    safe = set(CSU_SAFE_FEATURES)
    for name in CSU_ENGINEERED_FEATURES:
        assert name in safe, f"{name} not in CSU_SAFE_FEATURES"


def test_optum_engineered_features_in_safe_view():
    """Engineered features appear in OPTUM_SAFE_FEATURES (pre-or-at-index)."""
    safe = set(OPTUM_SAFE_FEATURES)
    for name in OPTUM_ENGINEERED_FEATURES:
        assert name in safe, f"{name} not in OPTUM_SAFE_FEATURES"


def test_csu_engineered_derivation_chain_is_pre_anchor():
    """Every derivation_input is itself declared pre-anchor in CSU manifest."""
    contracts = {c.name: c for c in CSU_FEATURES}
    for name in CSU_ENGINEERED_FEATURES:
        c = contracts[name]
        for input_name in c.derivation_inputs:
            assert input_name in contracts, (
                f"{name} derivation input {input_name!r} is not a CSU manifest entry"
            )
            input_contract = contracts[input_name]
            assert input_contract.knowable_at.is_pre_or_at_index(), (
                f"{name} pulls from {input_name!r} which is declared "
                f"knowable_at={input_contract.knowable_at.reference!r} "
                f"— breaks pre-anchor derivation chain (leakage risk)"
            )


def test_optum_engineered_derivation_chain_is_pre_anchor():
    """Every derivation_input is itself declared pre-anchor in Optum manifest."""
    contracts = {c.name: c for c in OPTUM_FEATURES}
    for name in OPTUM_ENGINEERED_FEATURES:
        c = contracts[name]
        for input_name in c.derivation_inputs:
            assert input_name in contracts, (
                f"{name} derivation input {input_name!r} is not an Optum manifest entry"
            )
            input_contract = contracts[input_name]
            assert input_contract.knowable_at.is_pre_or_at_index(), (
                f"{name} pulls from {input_name!r} which is declared "
                f"knowable_at={input_contract.knowable_at.reference!r} "
                f"— breaks pre-anchor derivation chain (leakage risk)"
            )


# =============================================================================
# Helper correctness — CSU
# =============================================================================


def _make_csu_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age_continuous": [30.0, 45.0, 60.0, 75.0],
            "insurance_type": ["commercial", "medicare", "commercial", "medicaid"],
            "medication_claim_count": [5, 10, 3, 0],
            "procedure_claim_count": [2, 8, 1, 0],
            "eligibility_duration_days": [180.0, 365.0, 90.0, 0.0],
            "engagement_score": [10.0, 25.0, 5.0, 2.0],
            "hcp_visits": [3, 5, 1, 0],
            "prior_treatments": [1, 3, 0, 2],
            "days_on_therapy": [30.0, 120.0, 0.0, 60.0],
            "disease_severity": [1.5, 3.0, 0.5, 2.0],
        }
    )


def test_engineer_csu_features_materializes_all_five():
    df = _make_csu_df()
    out, materialized = engineer_features(df, "csu")
    assert set(materialized) == set(CSU_ENGINEERED_FEATURES)
    for name in CSU_ENGINEERED_FEATURES:
        assert name in out.columns


def test_csu_claim_intensity_ratio_correctness():
    df = _make_csu_df()
    out, _ = engineer_features(df, "csu")
    # Row 0: (5+2)/180 = 0.0389
    # Row 3: (0+0)/max(0, 1) = 0/1 = 0
    np.testing.assert_allclose(out["claim_intensity_ratio"].iloc[0], 7 / 180.0, rtol=1e-6)
    np.testing.assert_allclose(out["claim_intensity_ratio"].iloc[3], 0.0)


def test_csu_engagement_per_visit_clamps_zero_visits():
    df = _make_csu_df()
    out, _ = engineer_features(df, "csu")
    # Row 3: 2.0 / max(0, 1) = 2.0  (clamped denominator)
    np.testing.assert_allclose(out["engagement_per_visit"].iloc[3], 2.0)


def test_csu_treatment_diversity_intensity_log1p():
    df = _make_csu_df()
    out, _ = engineer_features(df, "csu")
    # Row 1: 3 * log1p(120) = 3 * log(121)
    expected = 3 * math.log1p(120.0)
    np.testing.assert_allclose(out["treatment_diversity_intensity"].iloc[1], expected, rtol=1e-6)


def test_csu_severity_engagement_product():
    df = _make_csu_df()
    out, _ = engineer_features(df, "csu")
    # Row 0: 1.5 * 10 = 15
    np.testing.assert_allclose(out["severity_engagement_product"].iloc[0], 15.0)


def test_csu_age_x_insurance_uses_categorical_codes():
    """Categorical insurance_type encoded via stable factorize."""
    df = _make_csu_df()
    out, _ = engineer_features(df, "csu")
    # Both row 0 and row 2 have insurance_type='commercial' → same code.
    code_0 = out["age_x_insurance_interaction"].iloc[0] / 30.0
    code_2 = out["age_x_insurance_interaction"].iloc[2] / 60.0
    assert code_0 == code_2, "Stable factorization broken for repeated category"


# =============================================================================
# Helper correctness — Optum
# =============================================================================


def _make_optum_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "has_atopic_dermatitis": [1, 0, 1, 0],
            "has_asthma": [0, 1, 1, 0],
            "has_allergic_rhinitis": [0, 0, 1, 0],
            "has_anxiety": [0, 1, 0, 0],
            "has_depression": [0, 1, 0, 0],
            "has_thyroid_autoimmune": [0, 0, 0, 0],
            "has_nsaid_hypersensitivity": [0, 0, 0, 0],
            "has_angioedema": [0, 0, 1, 0],
            "dx_total_csu": [5, 10, 20, 0],
            "months_since_first_dx": [6.0, 12.0, 0.0, 24.0],
            "h1_1g_ever_filled": [1, 1, 0, 0],
            "h1_2g_ever_filled": [0, 1, 1, 0],
            "h2_ever_filled": [0, 0, 0, 0],
            "ltra_ever_filled": [0, 1, 0, 0],
            "sys_steroid_ever_filled": [0, 0, 1, 0],
            "top_steroid_ever_filled": [1, 0, 0, 0],
            "immunosupp_ever_filled": [0, 0, 0, 0],
            "ige_total_tested": [1, 1, 1, 0],
            "eosinophil_tested": [1, 0, 1, 0],
            "crp_tested": [0, 1, 1, 0],
            "tpo_ab_tested": [0, 0, 0, 0],
            "free_t4_tested": [0, 0, 1, 0],
            "tsh_tested": [1, 1, 1, 0],
            "ana_tested": [0, 0, 0, 0],
            "cbc_tested": [1, 1, 1, 0],
            "office_visits_allergist": [2, 0, 5, 0],
            "office_visits_dermatology": [3, 1, 4, 0],
        }
    )


def test_engineer_optum_features_materializes_all_five():
    df = _make_optum_df()
    out, materialized = engineer_features(df, "optum")
    assert set(materialized) == set(OPTUM_ENGINEERED_FEATURES)
    for name in OPTUM_ENGINEERED_FEATURES:
        assert name in out.columns


def test_optum_comorbidity_load_total_sums_eight_flags():
    df = _make_optum_df()
    out, _ = engineer_features(df, "optum")
    # Row 0: AD only = 1
    # Row 1: asthma + anxiety + depression = 3
    # Row 2: AD + asthma + allergic_rhinitis + angioedema = 4
    # Row 3: none = 0
    np.testing.assert_array_equal(
        out["comorbidity_load_total"].values, np.array([1.0, 3.0, 4.0, 0.0])
    )


def test_optum_csu_dx_intensity_clamps_zero_months():
    df = _make_optum_df()
    out, _ = engineer_features(df, "optum")
    # Row 2: 20 / max(0, 1) = 20
    np.testing.assert_allclose(out["csu_dx_intensity"].iloc[2], 20.0)


def test_optum_polypharmacy_breadth_sums_seven_classes():
    df = _make_optum_df()
    out, _ = engineer_features(df, "optum")
    # Row 1: h1_1g + h1_2g + ltra = 3
    np.testing.assert_allclose(out["polypharmacy_breadth"].iloc[1], 3.0)


def test_optum_lab_workup_completeness_sums_eight_labs():
    df = _make_optum_df()
    out, _ = engineer_features(df, "optum")
    # Row 2: ige + eos + crp + free_t4 + tsh + cbc = 6
    np.testing.assert_allclose(out["lab_workup_completeness"].iloc[2], 6.0)


def test_optum_specialist_visit_interaction():
    df = _make_optum_df()
    out, _ = engineer_features(df, "optum")
    # Row 2: 5 * 4 = 20
    np.testing.assert_allclose(out["specialist_visit_interaction"].iloc[2], 20.0)


# =============================================================================
# Missing-input resilience
# =============================================================================


def test_csu_missing_inputs_skip_feature_silently():
    """When a required input column is absent, that feature is skipped (not crashed)."""
    df = pd.DataFrame(
        {
            "age_continuous": [30.0],
            # insurance_type ABSENT → age_x_insurance_interaction skipped
            "engagement_score": [10.0],
            "hcp_visits": [3],
        }
    )
    out, materialized = engineer_features(df, "csu")
    # engagement_per_visit can still be computed; age_x_insurance cannot.
    assert "age_x_insurance_interaction" not in materialized
    assert "engagement_per_visit" in materialized


def test_optum_missing_drug_class_columns_skips_polypharmacy():
    """Need >=2 drug-class flags to compute polypharmacy_breadth."""
    df = pd.DataFrame(
        {
            # Only one drug-class flag — below threshold
            "h1_1g_ever_filled": [1],
        }
    )
    out, materialized = engineer_features(df, "optum")
    assert "polypharmacy_breadth" not in materialized


# =============================================================================
# NaN handling
# =============================================================================


def test_csu_nan_inputs_propagate_through_ratio():
    df = pd.DataFrame(
        {
            "medication_claim_count": [5],
            "procedure_claim_count": [np.nan],
            "eligibility_duration_days": [180.0],
        }
    )
    out, _ = engineer_features(df, "csu")
    # NaN procedure_claim_count is fillna(0.0) before sum -> 5/180
    np.testing.assert_allclose(out["claim_intensity_ratio"].iloc[0], 5 / 180.0)


# =============================================================================
# Dispatch
# =============================================================================


def test_dispatch_unknown_manifest_source_noop():
    df = _make_csu_df()
    cols_before = list(df.columns)
    out, materialized = engineer_features(df, "unknown_cohort")
    assert materialized == []
    assert list(out.columns) == cols_before


def test_dispatch_none_manifest_source_noop():
    df = _make_csu_df()
    cols_before = list(df.columns)
    out, materialized = engineer_features(df, None)
    assert materialized == []
    assert list(out.columns) == cols_before


# =============================================================================
# Node wrapper
# =============================================================================


@pytest.mark.asyncio
async def test_node_default_off_returns_empty_patch():
    """When enable_feature_engineering is unset or False, node is a no-op."""
    state = {
        "train_df": _make_csu_df(),
        "validation_df": _make_csu_df(),
        "scope_spec": {"feature_manifest_source": "csu"},
        # enable_feature_engineering absent (defaults to False)
    }
    patch = await engineer_features_node(state)
    assert patch == {}


@pytest.mark.asyncio
async def test_node_enabled_csu_applies_to_all_splits():
    df_train = _make_csu_df()
    df_val = _make_csu_df()
    df_test = _make_csu_df()
    state = {
        "enable_feature_engineering": True,
        "scope_spec": {"feature_manifest_source": "csu"},
        "train_df": df_train,
        "validation_df": df_val,
        "test_df": df_test,
    }
    patch = await engineer_features_node(state)
    assert set(patch["engineered_features"]) == set(CSU_ENGINEERED_FEATURES)
    assert patch["engineered_dispatch_source"] == "csu"
    # Each split mutated in place
    for d in (df_train, df_val, df_test):
        for name in CSU_ENGINEERED_FEATURES:
            assert name in d.columns


@pytest.mark.asyncio
async def test_node_enabled_optum_dispatches_optum_family():
    state = {
        "enable_feature_engineering": True,
        "scope_spec": {"feature_manifest_source": "optum"},
        "train_df": _make_optum_df(),
    }
    patch = await engineer_features_node(state)
    assert set(patch["engineered_features"]) == set(OPTUM_ENGINEERED_FEATURES)
    assert patch["engineered_dispatch_source"] == "optum"


@pytest.mark.asyncio
async def test_node_enabled_unknown_source_noop():
    state = {
        "enable_feature_engineering": True,
        "scope_spec": {"feature_manifest_source": "synthetic"},
        "train_df": _make_csu_df(),
    }
    patch = await engineer_features_node(state)
    assert patch["engineered_features"] == []
    assert patch["engineered_dispatch_source"] == "synthetic"


@pytest.mark.asyncio
async def test_node_skips_non_dataframe_state_values():
    """If a split key holds something that isn't a DataFrame (legacy callers),
    the node logs a warning and skips that split without crashing."""
    state = {
        "enable_feature_engineering": True,
        "scope_spec": {"feature_manifest_source": "csu"},
        "train_df": _make_csu_df(),
        "validation_df": "not a dataframe",  # legacy/test path
    }
    patch = await engineer_features_node(state)
    # Train split still engineered.
    assert set(patch["engineered_features"]) == set(CSU_ENGINEERED_FEATURES)
