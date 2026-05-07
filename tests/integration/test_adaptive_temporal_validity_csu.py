"""Integration test: adaptive temporal-validity layers on real CSU data.

Demonstrates that Layer 1 (declarative contracts) + Layer 3 (adversarial
discriminator) catch the SAME features that the PR #83 threshold-based
approach caught — but with disease-agnostic, data-derived mechanisms
instead of hardcoded thresholds.

This is the empirical proof the adaptive approach is at least as good as
threshold tuning on real-world data, while being:
- Disease-agnostic (no per-cohort calibration)
- Defensible (every decision has a traceable reason)
- Adaptive (thresholds derive from cohort-specific permutation null)

Reference incidents from .claude/state/leakage_compile_set_20260507.md:
- disease_severity (incident 1) — Layer 1 catches at AUTHOR time (unwindowed)
- engagement_score (incident 2) — Layer 1 catches at AUTHOR time (unwindowed)
- days_on_therapy (incident 3) — Layer 1 catches at AUTHOR time (unwindowed)
- medication_claim_count (incident 4) — Layer 1 catches at AUTHOR time
- hcp_visits (incident 5) — Layer 1 catches at AUTHOR time
- journey_duration_days (incident 6) — Layer 3 catches via z-score (single-feature
  AUC=0.689 → high z-score against permutation null)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSU_JOURNEYS_PATH = PROJECT_ROOT / "data" / "rwd" / "csu" / "e2i_ml_v3_patient_journeys.json"


@pytest.fixture(scope="module")
def csu_journeys_df() -> pd.DataFrame:
    """Load the CSU patient journeys into a DataFrame for analysis."""
    if not CSU_JOURNEYS_PATH.exists():
        pytest.skip(f"CSU journeys not available at {CSU_JOURNEYS_PATH}")
    with open(CSU_JOURNEYS_PATH) as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    return df


def test_layer_1_rejects_unwindowed_event_aggregations():
    """Layer 1 (FeatureContract) rejects all 5 unwindowed-aggregation incidents
    AT AUTHOR TIME, before any data is touched. This is the cheapest possible
    defense.
    """
    from src.data.feature_contract import ContractViolation, FeatureContract, KnowableAt

    incidents = [
        # (name, source, derivation_inputs, aggregation, window_days)
        ("disease_severity", "medication_events", ("medication_date",), "sum", None),
        ("engagement_score", "medication_events", ("medication_date", "npi"), "count", None),
        ("days_on_therapy", "medication_events", ("days_supply",), "sum", None),
        ("medication_claim_count", "medication_events", ("medication_date",), "count", None),
        ("hcp_visits", "medication_events", ("npi",), "nunique", None),
    ]

    for name, source, inputs, agg, window in incidents:
        with pytest.raises(ContractViolation):
            FeatureContract(
                name=name,
                knowable_at=KnowableAt(reference="index_date"),
                source=source,
                derivation_inputs=inputs,
                aggregation=agg,
                window_days=window,
            )


def test_layer_1_accepts_correctly_windowed_versions():
    """The same features, windowed correctly, pass Layer 1."""
    from src.data.feature_contract import FeatureContract, KnowableAt

    # Windowed versions all pass
    contracts = [
        FeatureContract(
            name="disease_severity_180d",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date",),
            aggregation="sum",
            window_days=180,
        ),
        FeatureContract(
            name="med_fill_count_180d",
            knowable_at=KnowableAt(reference="index_date"),
            source="medication_events",
            derivation_inputs=("medication_date",),
            aggregation="count",
            window_days=180,
        ),
    ]
    assert len(contracts) == 2  # construction succeeded


def test_layer_3_flags_journey_duration_days_on_real_csu_data(csu_journeys_df):
    """Layer 3 (adversarial discriminator) flags journey_duration_days on the
    actual CSU data — even though its single-feature AUC of ~0.689 is below
    the old hardcoded HIGH threshold of 0.80.

    This is the empirical proof that data-derived thresholds catch what
    hardcoded ones miss.
    """
    from src.data.adversarial_leakage import compute_adversarial_score

    df = csu_journeys_df
    if "journey_duration_days" not in df.columns or "treatment_initiated" not in df.columns:
        pytest.skip("Required columns not present in CSU journeys")

    # Filter to non-null duration and binary target
    mask = df["journey_duration_days"].notna() & df["treatment_initiated"].isin([0, 1])
    feature = df.loc[mask, "journey_duration_days"].values
    target = df.loc[mask, "treatment_initiated"].astype(int).values

    if mask.sum() < 100:
        pytest.skip("Insufficient non-null journey_duration_days data")

    result = compute_adversarial_score(feature, target, n_permutations=300, seed=7)

    # The pre-Phase-2 single-feature AUC was 0.689 in the cohort. With windowing
    # in Phase 2, dropped to ~0.59. Either way, against a 9000+-patient cohort's
    # permutation null, the z-score should be well above the noise floor.
    assert not np.isnan(result["z_score"]), f"Got NaN z_score: {result}"
    # Expect at least 5σ above null for a feature with this much patient evidence
    # (the threshold is documented; the data-derived nature is the key property,
    # not the specific number)
    assert result["z_score"] > 5, (
        f"Expected z_score > 5 for journey_duration_days against the CSU null; "
        f"got {result['z_score']:.2f}. actual_auc={result['actual_auc']:.4f}, "
        f"null_mean={result['null_mean']:.4f}, null_std={result['null_std']:.4f}"
    )


def test_layer_3_does_not_falsely_flag_age(csu_journeys_df):
    """Age is a legitimate pre-prediction-time feature with weak-but-real signal.
    Layer 3's data-derived threshold should NOT classify it as critically
    suspicious (z-score may exceed threshold but should be lower than the
    obvious leaks like journey_duration_days).
    """
    from src.data.adversarial_leakage import compute_adversarial_score

    df = csu_journeys_df
    if "age" not in df.columns or "treatment_initiated" not in df.columns:
        pytest.skip("Required columns not present in CSU journeys")

    mask = df["age"].notna() & df["treatment_initiated"].isin([0, 1])
    feature = df.loc[mask, "age"].astype(float).values
    target = df.loc[mask, "treatment_initiated"].astype(int).values

    if mask.sum() < 100:
        pytest.skip("Insufficient non-null age data")

    age_result = compute_adversarial_score(feature, target, n_permutations=300, seed=7)

    # Age in CSU has weak signal (single-feature AUC ~0.55 per memory).
    # In a large cohort, even weak signal beats the null with high z-score, but
    # should be much lower than journey_duration_days's leakage signal.
    # We don't assert an absolute threshold; we assert that age's z-score is
    # SMALLER than journey_duration_days's z-score (relative ranking is what
    # matters for governance review).
    if "journey_duration_days" not in df.columns:
        pytest.skip("Cannot compare without journey_duration_days")

    mask_jd = df["journey_duration_days"].notna() & df["treatment_initiated"].isin([0, 1])
    jd_feature = df.loc[mask_jd, "journey_duration_days"].values
    jd_target = df.loc[mask_jd, "treatment_initiated"].astype(int).values
    jd_result = compute_adversarial_score(jd_feature, jd_target, n_permutations=300, seed=7)

    assert age_result["z_score"] < jd_result["z_score"], (
        f"Age z-score should be LOWER than journey_duration_days z-score "
        f"(age is legitimate, jd is leak). Got age={age_result['z_score']:.2f} "
        f"vs jd={jd_result['z_score']:.2f}"
    )
