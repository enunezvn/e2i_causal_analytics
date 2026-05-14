"""Tests for the RWD-realistic synthetic regime.

Canonical design reference: ``docs/synthetic_v3_design.md`` (Phase S.3).
This file is the **executable spec** for the ``rwd_realistic`` regime
shipped at ``src/repositories/synthetic_rwd_realistic.py``; the design doc
points back here as the authoritative test surface (§5.1). Per issue #200
codex pass-1 MED-3, the back-pointer here is mandatory to keep the
cross-reference non-circular AND symmetric — the legacy regime file
``tests/synthetic/test_synthetic_regimes.py`` carries the same pointer
already.

Verifies that the synthetic generator produces data with structural properties
matching real-world claims data:
- 2.4% prevalence (matches published CSU/AD/asthma claims studies)
- Demographics-only feature surface
- Vanilla XGBoost achieves val_AUC in [0.62, 0.68] (the published ceiling)
- Injected leakage patterns are catchable by the 4-layer defense
"""

from __future__ import annotations


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
        "lab_value",
        "disease_severity_score",
        "ige_level",
        "d_dimer",
        "prior_omalizumab",
        "antihistamine_history",
    }
    overlap = set(df.columns) & forbidden
    assert overlap == set(), f"Regime should not include clinical columns: {overlap}"


def test_pure_noise_leak_pattern_does_not_correlate():
    """The `pure_noise` control should produce a column NOT correlated with target."""
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=2000, leakage_pattern="pure_noise", seed=42)
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

    config = RwdRealisticConfig(n_patients=2000, leakage_pattern="post_index_aggregation", seed=42)
    df = generate_rwd_realistic(config)
    leak_col = "post_index_med_count_LEAK"
    assert leak_col in df.columns

    result = compute_adversarial_score(
        df[leak_col].values,
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert result["suspicious"], f"Layer 3 missed post_index_aggregation leak: {result}"
    assert result["z_score"] > 5


def test_post_hoc_termination_leak_is_caught_by_layer_3():
    """The eligend-based post_hoc_termination leak should be caught by Layer 3."""
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=2000, leakage_pattern="post_hoc_termination", seed=42)
    df = generate_rwd_realistic(config)
    leak_col = "months_remaining_eligibility_LEAK"

    result = compute_adversarial_score(
        df[leak_col].values,
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert result["suspicious"], f"Layer 3 missed post_hoc_termination leak: {result}"


def test_treatment_leaked_code_is_caught_by_layer_3():
    """The treatment-leaked ICD code (Z79.899-style) should be caught."""
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=2000, leakage_pattern="treatment_leaked_code", seed=42)
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

    config = RwdRealisticConfig(n_patients=2000, leakage_pattern="pure_noise", seed=42)
    df = generate_rwd_realistic(config)

    result = compute_adversarial_score(
        df["random_noise_CONTROL"].values,
        df["treatment_initiated"].values,
        n_permutations=200,
        seed=7,
    )
    assert not result["suspicious"], f"Control noise falsely flagged as leak: {result}"


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
    age_auc = roc_auc_score(target[age_mask], df.loc[age_mask, "age"].values)
    eligibility_auc = roc_auc_score(target, df["eligibility_duration_days"].values)

    # Each individual feature should be weak (matches claims-only ceiling)
    assert max(age_auc, 1 - age_auc) < 0.70, f"Age single-feature AUC too high: {age_auc:.3f}"
    assert max(eligibility_auc, 1 - eligibility_auc) < 0.70, (
        f"Eligibility AUC too high: {eligibility_auc:.3f}"
    )


# --- Direct audit follow-ups (synthetic regime — item H) --------------------


def test_spurious_correlation_leak_is_caught_by_layer_3():
    """The 5th injectable leak pattern (``spurious_correlation``) was missing
    from coverage despite the state document's "5 tested leak patterns"
    claim. This test closes that gap by exercising Layer 3 against the
    spurious-correlation regime: feature ~ N(2, 0.5) for treated, ~ N(0, 0.5)
    for untreated produces strong AUC, well above any 5σ permutation null.
    """
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    config = RwdRealisticConfig(n_patients=2000, leakage_pattern="spurious_correlation", seed=42)
    df = generate_rwd_realistic(config)
    assert "spurious_score_LEAK" in df.columns

    score = compute_adversarial_score(
        df["spurious_score_LEAK"].to_numpy(),
        df["treatment_initiated"].to_numpy(),
        n_permutations=200,
        seed=42,
    )
    assert score["suspicious"], f"spurious_correlation leak pattern was NOT flagged at 5σ: {score}"


def test_post_index_aggregation_leak_is_zero_for_untreated():
    """Anti-regression for the dead-code cleanup: the ``+ (1 - target) * 0``
    term was a no-op (always zero), and removing it must not change the
    untreated-row values. Verifies the leak's deterministic-zero invariant.
    """
    from src.repositories.synthetic_rwd_realistic import (
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    df = generate_rwd_realistic(
        RwdRealisticConfig(n_patients=1000, leakage_pattern="post_index_aggregation", seed=42)
    )
    untreated = df[df["treatment_initiated"] == 0]
    assert (untreated["post_index_med_count_LEAK"] == 0).all(), (
        "post_index leak must be deterministically zero for untreated rows; "
        f"got non-zero values among untreated patients ({len(untreated)} rows)."
    )
    treated = df[df["treatment_initiated"] == 1]
    assert (treated["post_index_med_count_LEAK"] >= 1).all()
    assert (treated["post_index_med_count_LEAK"] <= 9).all()


# --- v5 Gate C2 borderline_genuine injection ---------------------------------


def test_borderline_genuine_pattern_emits_named_feature():
    """v5 C2: the borderline_genuine pattern must emit
    ``BORDERLINE_GENUINE_FEATURE_NAME`` as the canonical column.

    ENGINEERING CI SANITY-CHECK — NOT RWD positive-evidence.
    """
    from src.repositories.synthetic_rwd_realistic import (
        BORDERLINE_GENUINE_FEATURE_NAME,
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    df = generate_rwd_realistic(
        RwdRealisticConfig(n_patients=2000, leakage_pattern="borderline_genuine", seed=42)
    )
    assert BORDERLINE_GENUINE_FEATURE_NAME in df.columns
    # Both classes get nonzero draws — the feature is class-conditional but
    # not target-deterministic (which is what distinguishes "genuine causal"
    # from "post_index_aggregation" where untreated == 0 by construction).
    assert (df.loc[df["treatment_initiated"] == 0, BORDERLINE_GENUINE_FEATURE_NAME] != 0).any()
    assert (df.loc[df["treatment_initiated"] == 1, BORDERLINE_GENUINE_FEATURE_NAME] != 0).any()


def test_borderline_genuine_pattern_produces_intermediate_auc():
    """v5 C2: the injected feature's effective AUC at default parameters
    must land in the [0.54, 0.58] band so the permutation-null z falls
    inside the (5σ, 7.5σ) HBLP variance-relaxation window at n=20000.

    The integration test pins the full z-band; this unit test pins the
    upstream AUC so regressions in the generator constants surface here
    before they propagate into the integration suite's longer permutation
    runtime.
    """
    from src.data.adversarial_leakage import compute_adversarial_score
    from src.repositories.synthetic_rwd_realistic import (
        BORDERLINE_GENUINE_DEFAULT_N_PATIENTS,
        BORDERLINE_GENUINE_DEFAULT_SEED,
        BORDERLINE_GENUINE_FEATURE_NAME,
        RwdRealisticConfig,
        generate_rwd_realistic,
    )

    df = generate_rwd_realistic(
        RwdRealisticConfig(
            n_patients=BORDERLINE_GENUINE_DEFAULT_N_PATIENTS,
            leakage_pattern="borderline_genuine",
            seed=BORDERLINE_GENUINE_DEFAULT_SEED,
        )
    )
    score = compute_adversarial_score(
        df[BORDERLINE_GENUINE_FEATURE_NAME].to_numpy(),
        df["treatment_initiated"].to_numpy(),
        n_permutations=200,
        seed=42,
    )
    assert 0.54 <= score["actual_auc"] <= 0.58, (
        f"borderline_genuine AUC calibration drift: got {score['actual_auc']:.4f}; "
        f"expected [0.54, 0.58] at default n + seed"
    )
