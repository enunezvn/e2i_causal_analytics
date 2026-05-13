"""Unit tests for v5 Gate B3 feature engineering.

Covers:
1. CSU + Optum manifest contracts declared for all engineered features
   exposed by ``feature_engineering`` (4 CSU + 5 Optum after
   ``claim_intensity_ratio`` was dropped post-audit in commit
   ``fc7c251a``; see ``docs/calibration/b3_engineered_audit_20260511.json``).
   Backlog #17 (commit ``cfa71627``, 2026-05-12) reclassified 3 of the 4
   CSU engineered features to ``knowable_at=post_index`` because their
   inputs (``engagement_score`` / ``hcp_visits`` / ``prior_treatments``
   / ``days_on_therapy`` / ``disease_severity``) became post-index;
   only ``age_x_insurance_interaction`` survives as a pre-anchor
   engineered feature. The 5 Optum engineered features remain
   pre-anchor (their inputs are pre-anchor by construction).
2. Helper ``engineer_features`` correctness on small synthetic
   DataFrames (CSU + Optum dispatch).
3. Node wrapper ``engineer_features_node`` gating: OFF by default
   (no-op patch), ON applies transforms to all splits.
4. Dispatch on ``scope_spec.feature_manifest_source``: csu, optum,
   unknown (no-op).
5. Missing input columns -> feature silently skipped (best-effort).
6. Edge cases: NaN inputs, divide-by-zero clamping, categorical
   factorization stability.

Pre-spec: docs/specs/v5_b3_feature_engineering_prespec_2026-05-11.md
(pre-dates backlog #17 and still says "all candidates declared
knowable_at=index_date"; the CSU manifest is now the source of truth and
``tests/unit/test_data/test_csu_feature_manifest.py`` pins the post-index
reclassification at the contract level).
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
    CSU_FORBIDDEN_AS_FEATURES,
    CSU_SAFE_FEATURES,
)
from src.data.manifests.optum_feature_manifest import (
    OPTUM_FEATURES,
    OPTUM_SAFE_FEATURES,
)

# =============================================================================
# Manifest declarations
# =============================================================================


# Issue #187 (2026-05-13): backlog #17 (commit cfa71627, 2026-05-12)
# reclassified 6 CSU medication-derived aggregates to
# ``knowable_at=post_index`` because the CSU prediction target
# (``treatment_initiated``) is structurally coupled to "patient appears
# in the medication panel". The chain-validity rule then forced 3 of
# the 4 B3 engineered features (``engagement_per_visit``,
# ``treatment_diversity_intensity``, ``severity_engagement_product``)
# to ``post_index`` because their inputs are post_index. Only
# ``age_x_insurance_interaction`` remains pre-anchor — its inputs
# (``age_continuous`` + ``insurance_type``) are both enrollment-knowable.
#
# This file pre-dated backlog #17 and iterated over
# ``CSU_ENGINEERED_FEATURES`` as if all four were pre-anchor. The fix
# below splits the engineered set by anchor at the manifest contract
# level. The pre-anchor sibling tests assert the pre-anchor expectations
# only over the pre-anchor bucket; a NEW test pins the post-index
# bucket as a literal frozenset so the split cannot regress silently
# (someone removing one of the names from ``CSU_ENGINEERED_FEATURES``
# would fire the new test, not silently skip the pre-anchor sibling).
def _split_engineered_by_anchor(
    engineered_names: tuple[str, ...],
    contracts: dict,
) -> tuple[list[str], list[str]]:
    """Partition engineered feature names by their declared anchor.

    Returns ``(pre_anchor, post_index)``. A pre-anchor name has its
    contract's ``knowable_at.reference`` in ``{"index_date", "enrollment"}``;
    a post-index name has ``reference == "post_index"``. Names absent from
    ``contracts`` raise a ``KeyError`` — callers must first run
    ``test_*_engineered_features_all_in_manifest``.
    """
    pre_anchor: list[str] = []
    post_index: list[str] = []
    for name in engineered_names:
        ref = contracts[name].knowable_at.reference
        if ref == "post_index":
            post_index.append(name)
        else:
            pre_anchor.append(name)
    return pre_anchor, post_index


# Literal pin on the post-index bucket. If one of these names is silently
# removed from ``CSU_ENGINEERED_FEATURES`` (e.g., a refactor drops the
# helper that materializes it) the new
# ``test_csu_engineered_features_post_index_correctly_classified`` will
# fail because the literal pin is no longer a subset of the bucket.
_BACKLOG_17_POST_INDEX_ENGINEERED: frozenset[str] = frozenset(
    {
        "engagement_per_visit",
        "treatment_diversity_intensity",
        "severity_engagement_product",
    }
)


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
    """Pre-anchor CSU engineered features are knowable_at=index_date.

    Backlog #17 split: only engineered features whose inputs are
    pre-anchor (currently ``age_x_insurance_interaction``) can themselves
    be declared at ``index_date``. The 3 medication-derived engineered
    features (``engagement_per_visit`` etc.) are post-index by chain
    validity and are pinned by
    ``test_csu_engineered_features_post_index_correctly_classified``.
    """
    contracts = {c.name: c for c in CSU_FEATURES}
    pre_anchor, _ = _split_engineered_by_anchor(CSU_ENGINEERED_FEATURES, contracts)
    assert pre_anchor, (
        "Expected at least one pre-anchor CSU engineered feature; got 0. "
        "If backlog #17 has fully retired all pre-anchor engineered "
        "features, update this test and CSU_ENGINEERED_FEATURES "
        "intentionally rather than letting the assertion silently pass."
    )
    for name in pre_anchor:
        c = contracts[name]
        assert c.knowable_at.reference == "index_date", (
            f"{name} declared knowable_at={c.knowable_at.reference!r}, expected 'index_date'"
        )


def test_csu_engineered_features_post_index_correctly_classified():
    """Post-index CSU engineered features are pinned in the forbidden view.

    Backlog #17 regression pin: the 3 medication-derived engineered
    features (``engagement_per_visit``, ``treatment_diversity_intensity``,
    ``severity_engagement_product``) MUST stay declared ``post_index``
    because their inputs are post_index. This test catches two drift
    directions:

    1. Someone removes one of these names from ``CSU_ENGINEERED_FEATURES``
       — the literal-pin subset check fires.
    2. Someone "fixes" the pre-anchor sibling test by re-classifying one
       of them back to ``index_date`` — that would re-break chain
       validity, and this test catches it via the per-name assertion
       below.
    """
    contracts = {c.name: c for c in CSU_FEATURES}
    _, post_index = _split_engineered_by_anchor(CSU_ENGINEERED_FEATURES, contracts)
    post_index_set = set(post_index)

    # Pin (1): literal frozenset must be a subset of the post-index bucket.
    missing = _BACKLOG_17_POST_INDEX_ENGINEERED - post_index_set
    assert not missing, (
        f"Backlog #17 regression: engineered features {sorted(missing)} "
        f"are no longer present in the post-index bucket of "
        f"CSU_ENGINEERED_FEATURES. Either the helper dropped them, or "
        f"they were re-classified back to pre-anchor (which would also "
        f"violate chain validity)."
    )

    # Pin (2): per-name assertions confirm manifest reference + forbidden view.
    forbidden = set(CSU_FORBIDDEN_AS_FEATURES)
    for name in post_index:
        c = contracts[name]
        assert c.knowable_at.reference == "post_index", (
            f"{name} mis-bucketed: knowable_at={c.knowable_at.reference!r}"
        )
        assert name in forbidden, (
            f"{name} should be in CSU_FORBIDDEN_AS_FEATURES (it is "
            f"post_index by manifest); manifest views have drifted."
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
    """Pre-anchor CSU engineered features appear in ``CSU_SAFE_FEATURES``.

    Backlog #17 split: only engineered features whose contract is
    pre-anchor land in the SAFE view. The 3 post-index engineered
    features appear in ``CSU_FORBIDDEN_AS_FEATURES`` instead and are
    pinned by ``test_csu_engineered_features_post_index_correctly_classified``.
    """
    contracts = {c.name: c for c in CSU_FEATURES}
    pre_anchor, _ = _split_engineered_by_anchor(CSU_ENGINEERED_FEATURES, contracts)
    safe = set(CSU_SAFE_FEATURES)
    for name in pre_anchor:
        assert name in safe, f"{name} not in CSU_SAFE_FEATURES"


def test_optum_engineered_features_in_safe_view():
    """Engineered features appear in OPTUM_SAFE_FEATURES (pre-or-at-index)."""
    safe = set(OPTUM_SAFE_FEATURES)
    for name in OPTUM_ENGINEERED_FEATURES:
        assert name in safe, f"{name} not in OPTUM_SAFE_FEATURES"


# L1 (codex): tighten the derivation-chain check from is_pre_or_at_index()
# to an explicit set membership over the canonical pre-anchor reference
# strings. is_pre_or_at_index() returns True for any reference with
# offset_days <= 0, which would silently pass an input declared with an
# unexpected reference string (e.g., "discharge_date") via the offset-based
# fallback. The explicit set keeps the test sensitive to schema drift.
_PRE_ANCHOR_REFERENCES = {"index_date", "enrollment"}


def test_csu_engineered_derivation_chain_is_pre_anchor():
    """Pre-anchor CSU engineered features pull from pre-anchor inputs only.

    Backlog #17 split: only engineered features whose own contract is
    pre-anchor are required to derive from pre-anchor inputs. The 3
    post-index engineered features (e.g., ``engagement_per_visit``)
    legitimately pull from post-index inputs (``engagement_score``,
    ``hcp_visits``, etc.) — their post-index declaration in the
    manifest is precisely WHAT chain validity enforces, and they are
    excluded from training via ``CSU_FORBIDDEN_AS_FEATURES`` (see
    ``test_csu_engineered_features_post_index_correctly_classified``).
    Full-contract chain validity is also pinned at the manifest layer
    by ``tests/unit/test_data/test_csu_feature_manifest.py``.
    """
    contracts = {c.name: c for c in CSU_FEATURES}
    pre_anchor, _ = _split_engineered_by_anchor(CSU_ENGINEERED_FEATURES, contracts)
    for name in pre_anchor:
        c = contracts[name]
        for input_name in c.derivation_inputs:
            assert input_name in contracts, (
                f"{name} derivation input {input_name!r} is not a CSU manifest entry"
            )
            input_contract = contracts[input_name]
            ref = input_contract.knowable_at.reference
            assert ref in _PRE_ANCHOR_REFERENCES, (
                f"{name} pulls from {input_name!r} declared "
                f"knowable_at={ref!r}; expected one of "
                f"{sorted(_PRE_ANCHOR_REFERENCES)} (leakage risk)"
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
            ref = input_contract.knowable_at.reference
            assert ref in _PRE_ANCHOR_REFERENCES, (
                f"{name} pulls from {input_name!r} declared "
                f"knowable_at={ref!r}; expected one of "
                f"{sorted(_PRE_ANCHOR_REFERENCES)} (leakage risk)"
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


# claim_intensity_ratio DROPPED post-audit (b3_engineered_audit_20260511).
# Test removed; module docstring + manifest comment cite the audit JSON.


def test_csu_claim_intensity_ratio_dropped_from_module():
    """Regression pin: the dropped candidate is no longer exported."""
    assert "claim_intensity_ratio" not in CSU_ENGINEERED_FEATURES


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


def test_csu_nan_engagement_score_propagates_through_ratio():
    """engagement_per_visit handles NaN engagement_score gracefully."""
    df = pd.DataFrame(
        {
            "engagement_score": [np.nan, 10.0],
            "hcp_visits": [3, 5],
        }
    )
    out, _ = engineer_features(df, "csu")
    # NaN engagement_score / clipped denominator -> NaN
    assert pd.isna(out["engagement_per_visit"].iloc[0])
    # Row 1: 10 / 5 = 2.0
    np.testing.assert_allclose(out["engagement_per_visit"].iloc[1], 2.0)


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
async def test_node_returns_mutated_dataframes_in_state_patch():
    """Codex pass-2 LOW-1: replay-safety guarantee.

    The patch MUST include the mutated DataFrames keyed by their state
    keys (train_df, validation_df, test_df, holdout_df) — not just the
    engineered_features metadata. Without this, in-place mutations are
    lost on LangGraph checkpoint replay because deserialized DataFrame
    objects are fresh.

    This test catches a silent revert of the H3 fix.
    """
    state = {
        "enable_feature_engineering": True,
        "scope_spec": {"feature_manifest_source": "csu"},
        "train_df": _make_csu_df(),
        "validation_df": _make_csu_df(),
        "test_df": _make_csu_df(),
        "holdout_df": _make_csu_df(),
    }
    patch = await engineer_features_node(state)
    # The patch MUST surface the mutated DataFrames so LangGraph's
    # reducer applies them durably.
    for split_key in ("train_df", "validation_df", "test_df", "holdout_df"):
        assert split_key in patch, (
            f"H3 fix regression: {split_key} not returned in patch — "
            "LangGraph checkpoint replay would lose engineered columns."
        )
        df_in_patch = patch[split_key]
        for name in CSU_ENGINEERED_FEATURES:
            assert name in df_in_patch.columns, (
                f"H3 fix regression: patch[{split_key!r}] missing engineered column {name!r}."
            )


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
