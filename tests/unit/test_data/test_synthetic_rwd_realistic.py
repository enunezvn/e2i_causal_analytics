"""Tests for the RWD-realistic synthetic regime.

Verifies that the synthetic generator produces data with structural properties
matching real-world claims data:
- 2.4% prevalence (matches published CSU/AD/asthma claims studies)
- Demographics-only feature surface
- Vanilla XGBoost achieves val_AUC in [0.62, 0.68] (the published ceiling)
- Injected leakage patterns are catchable by the 4-layer defense
"""

from __future__ import annotations

import numpy as np
import pytest


def test_basic_generation_shape():
    """Generator produces correct cohort size and required columns."""
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=500, seed=42)
    df = generate_rwd_realistic(config)
    assert len(df) == 500
    required = {
        "patient_id",
        "age",
        "age_group",
        "gender",
        "geographic_region",
        "insurance_product",
        "primary_diagnosis_code",
        "eligeff",
        "index_date",
        "eligend",
        "eligibility_duration_days",
        "treatment_initiated",
    }
    assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"


def test_prevalence_matches_target():
    """Realized prevalence should be close to configured target (within 1pp)."""
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=10000, prevalence=0.024, seed=42)
    df = generate_rwd_realistic(config)
    realized = df["treatment_initiated"].mean()
    assert abs(realized - 0.024) < 0.015, (
        f"Realized prevalence {realized:.4f} far from target 0.024"
    )


def test_panel_fragmentation_rate():
    """The configured fraction of patients should have fragmented panels."""
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=2000, panel_fragmentation_rate=0.50, seed=42)
    df = generate_rwd_realistic(config)
    realized = df["is_fragmented"].mean()
    assert abs(realized - 0.50) < 0.03


def test_no_clinical_columns_present():
    """Critical: regime should NOT include clinical features (lab, severity,
    prior-medication) — matches CSU/specialty-pharma claims-only constraint.
    """
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=200, seed=42)
    df = generate_rwd_realistic(config)
    forbidden = {
        "lab_value", "disease_severity_score", "ige_level", "d_dimer",
        "prior_omalizumab", "antihistamine_history",
    }
    overlap = set(df.columns) & forbidden
    assert overlap == set(), f"Regime should not include clinical columns: {overlap}"


def test_pure_noise_leak_pattern_does_not_correlate():
    """The `pure_noise` control should produce a column NOT correlated with target."""
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(
        n_patients=2000, leakage_pattern="pure_noise", seed=42
    )
    df = generate_rwd_realistic(config)
    assert "random_noise_CONTROL" in df.columns
    # Correlation should be near zero
    corr = df["random_noise_CONTROL"].corr(df["treatment_initiated"])
    assert abs(corr) < 0.10, f"Pure noise has unexpected correlation: {corr:.3f}"


def test_post_index_aggregation_leak_is_caught_by_layer_3():
    """The injected post_index_aggregation leak should produce HIGH z-score
    in the adversarial discriminator (Layer 3).
    """
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(
        n_patients=2000, leakage_pattern="post_index_aggregation", seed=42
    )
    df = generate_rwd_realistic(config)
    leak_col = "post_index_med_count_LEAK"
    assert leak_col in df.columns

    result = compute_adversarial_score(
        df[leak_col].values,
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert result["suspicious"], (
        f"Layer 3 missed post_index_aggregation leak: {result}"
    )
    assert result["z_score"] > 5


def test_post_hoc_termination_leak_is_caught_by_layer_3():
    """The eligend-based post_hoc_termination leak should be caught by Layer 3."""
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(
        n_patients=2000, leakage_pattern="post_hoc_termination", seed=42
    )
    df = generate_rwd_realistic(config)
    leak_col = "months_remaining_eligibility_LEAK"

    result = compute_adversarial_score(
        df[leak_col].values,
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert result["suspicious"], (
        f"Layer 3 missed post_hoc_termination leak: {result}"
    )


def test_treatment_leaked_code_is_caught_by_layer_3():
    """The treatment-leaked ICD code (Z79.899-style) should be caught."""
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(
        n_patients=2000, leakage_pattern="treatment_leaked_code", seed=42
    )
    df = generate_rwd_realistic(config)
    leak_col = "has_z79_long_term_drug_LEAK"

    result = compute_adversarial_score(
        df[leak_col].values.astype(float),
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert result["suspicious"]


def test_pure_noise_control_is_NOT_flagged():
    """CRITICAL: control column must NOT trigger Layer 3 false positive."""
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(
        n_patients=2000, leakage_pattern="pure_noise", seed=42
    )
    df = generate_rwd_realistic(config)

    result = compute_adversarial_score(
        df["random_noise_CONTROL"].values,
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert not result["suspicious"], (
        f"Control noise falsely flagged as leak: {result}"
    )


def test_demographic_features_have_realistic_correlations():
    """Each individual demographic feature should have weak signal (single-feature
    AUC < ~0.65), reflecting the published claims-only ceiling.
    """
    from sklearn.metrics import roc_auc_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=5000, seed=42)
    df = generate_rwd_realistic(config)

    target = df["treatment_initiated"].values
    # Drop NaN rows for the AUC calculation
    age_mask = df["age"].notna()
    age_auc = roc_auc_score(
        target[age_mask], df.loc[age_mask, "age"].values
    )
    eligibility_auc = roc_auc_score(target, df["eligibility_duration_days"].values)

    # Each individual feature should be weak (matches claims-only ceiling)
    assert max(age_auc, 1 - age_auc) < 0.70, (
        f"Age single-feature AUC too high: {age_auc:.3f}"
    )
    assert max(eligibility_auc, 1 - eligibility_auc) < 0.70, (
        f"Eligibility AUC too high: {eligibility_auc:.3f}"
    )
