"""RWD-realistic synthetic data generator (Phase S of adaptive-temporal-validity).

Faithfully reproduces the structural challenges of ConcertAI claims data
(or analogous specialty-pharma RWD) so the agentic ML pipeline can be tested
on data that LOOKS like what it'll see in production.

Why this exists (per codex synthetic-DGP fitness research, 2026-05-07):
- The existing `ml_patients()` `clean` regime produces val_AUC ~0.87, which
  is unrealistic for any specialty-pharma claims-only model.
- Published claims-only initiation models (psoriasis, AD, severe asthma)
  converge at val_AUC 0.61–0.67. Information-theoretic ceiling for the
  6-feature, 2.4%-prevalence regime is 0.62–0.68.
- A synthetic generator that produces too-clean signal trains the pipeline
  on UNREALISTIC leakage shapes. Real-world RWD has subtler leaks
  (vendor-encoded post-hoc fields, partial-panel masking) that the synthetic
  must reproduce or testing is hollow.

Structural properties this regime preserves:
1. Low prevalence (0.024 by default — matches CSU 2.4%, AD 4.1%, asthma 3.8%)
2. Demographic-only feature surface (age, gender, region, insurance, ICD subtype,
   eligibility window). NO labs, NO clinical severity, NO prior-medication
   history (matches the CSU vendor-data limitations).
3. Fragmented panels: ~50% of patients have <12 months observation; ~5% have
   demo only without clinical claims.
4. Missing data patterns tied to insurance/age (matches access-driven
   missingness in real claims data).
5. Calibrated so vanilla XGBoost achieves val_AUC 0.62–0.68 when run through
   a leakage-clean pipeline.

Plus optional injectable leakage scenarios for testing the 4-layer adaptive
defense (Phase S in `.claude/plans/adaptive_temporal_validity_redesign.md`):
- `post_index_aggregation`: feature aggregating events past prediction time
- `post_hoc_termination`: eligend reflecting actual termination (vendor-encoded)
- `treatment_leaked_code`: ICD code assigned post-treatment
- `spurious_correlation`: feature with high single-feature AUC but no causal path
- `pure_noise`: control (should NOT be flagged)
- `borderline_genuine`: pre-anchor causal feature with z in [5σ, 7.5σ] —
  the HBLP variance-relaxation band. Engineering CI sanity-check ONLY:
  validates that HBLP RETAINS when Layer 1 declared-safe while legacy 5σ
  DROPS. NOT RWD positive-evidence (per v5 plan §2 C2 + codex pass-3).

Reference: codex agent output 2026-05-07 (option (c) hybrid: keep existing
regimes for plumbing tests; add this regime for RWD-realistic testing).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np
import pandas as pd

LeakagePattern = Literal[
    "none",
    "post_index_aggregation",
    "post_hoc_termination",
    "treatment_leaked_code",
    "spurious_correlation",
    "pure_noise",
    "borderline_genuine",
]

# ============================================================================
# v5 Gate C2 — borderline_genuine injection parameters.
#
# The borderline_genuine pattern injects a pre-anchor causal feature whose
# permutation-null z lands in the HBLP variance-relaxation band [5σ, 7.5σ].
# At n_patients=20000 with prevalence=0.024 (≈480 positives, far above the
# HBLP variance-inflation reference N=50) the effective HBLP threshold for
# a manifest-declared-safe feature is exactly 5σ × 1.5 = 7.5σ.
#
# The injection adds a single normally-distributed feature with class-
# conditional means tuned so the resulting feature AUC ≈ 0.55 (~6σ of
# permutation-null evidence) at n=20000. The default constants below were
# calibrated against compute_adversarial_score with n_permutations=200 and
# seed=42; the integration test pins the empirical z-value AND the
# legacy-drops vs HBLP-retains contrast.
#
# This is a v5 Gate C2 ENGINEERING CI SANITY-CHECK, not RWD positive
# evidence (v5 plan §2 C2 + codex pass-3 MEDIUM-7). The synthetic generator
# can produce any AUC by construction; what this test pins is that the
# pipeline routing (legacy vs HBLP) decides correctly at the boundary.
# ============================================================================
BORDERLINE_GENUINE_FEATURE_NAME = "borderline_genuine_feature"
# Class-conditional Gaussian offset. Calibrated empirically against
# ``adaptive_validity_check`` defaults (n_permutations=200, seed=7) at
# n_patients=20000, prevalence=0.024, generator seed=42:
#   AUC ≈ 0.553, z ≈ 6.10σ via the pipeline path — comfortably in
#   (5.0, 7.5) band.
# Different injection seeds (passed to RwdRealisticConfig.seed) shift the
# observed z by ±1.5σ via the small-positive-class sample variance; the
# integration test pins a single seed for reproducibility.
BORDERLINE_GENUINE_TREATED_MEAN = 0.06
BORDERLINE_GENUINE_UNTREATED_MEAN = 0.0
BORDERLINE_GENUINE_SHARED_STD = 1.0
BORDERLINE_GENUINE_DEFAULT_N_PATIENTS = 20000
BORDERLINE_GENUINE_DEFAULT_SEED = 42


@dataclass(frozen=True)
class RwdRealisticConfig:
    """Configuration for the RWD-realistic synthetic regime."""

    n_patients: int = 7000
    prevalence: float = 0.024
    panel_fragmentation_rate: float = 0.50  # Fraction with <12mo observation
    missing_demo_rate: float = 0.05  # Fraction with demo missing fields
    leakage_pattern: LeakagePattern = "none"
    leakage_strength: float = 1.0  # Multiplier on injected leak signal (0–1)
    # Backlog #135: multiplier on the 4 demographic coefficients in
    # _generate_target. Default 1.0 reproduces the [0.62, 0.68] AUC band
    # calibrated for the published claims-only ceiling. signal_scale=0
    # produces a pure-noise cohort (single-feature AUC ≈ 0.50);
    # signal_scale > 1 produces higher AUCs. The T2.2 calibration sweep
    # at scripts/calibration/run_t22_synth_sweep.py varies this knob to
    # span target AUCs [0.55, 0.85].
    signal_scale: float = 1.0
    seed: int = 42
    start_date: str = "2022-01-01"
    end_date: str = "2024-12-31"


def _generate_demographics(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Demographics matching CSU-shape distributions (codex research)."""
    # Age: bimodal (CSU has both pediatric-onset and adult cohorts)
    age = (
        np.where(
            rng.random(n) < 0.30,
            rng.normal(35, 12, n),  # Adult-onset
            rng.normal(55, 15, n),  # Mature
        )
        .clip(18, 90)
        .astype(int)
    )

    age_group = pd.cut(age, bins=[0, 50, 65, 200], labels=["<50", "50-65", ">65"]).astype(str)

    gender = rng.choice(["F", "M"], n, p=[0.65, 0.35])  # CSU is female-skewed
    geographic_region = rng.choice(
        ["northeast", "south", "midwest", "west"], n, p=[0.20, 0.35, 0.25, 0.20]
    )

    # Insurance products (matches ConcertAI bus categories)
    insurance_product = rng.choice(
        [
            "commercial_PPO",
            "commercial_HMO",
            "medicare_advantage",
            "medicaid_managed",
            "self_insured",
            "exchange",
            "other",
        ],
        n,
        p=[0.30, 0.20, 0.15, 0.15, 0.10, 0.05, 0.05],
    )

    # Primary diagnosis subtype (CSU has L50.0, L50.1, L50.8, L50.9)
    primary_diagnosis_code = rng.choice(
        ["L50.0", "L50.1", "L50.8", "L50.9"],
        n,
        p=[0.55, 0.25, 0.15, 0.05],
    )

    # Generate UUID-style patient IDs from rng — assemble from two int64s
    # since np.random.Generator.integers caps at int64 bounds
    def _make_patient_id() -> str:
        hi = int(rng.integers(0, 2**63))
        lo = int(rng.integers(0, 2**63))
        return str(uuid.UUID(int=(hi << 64) | lo))

    return pd.DataFrame(
        {
            "patient_id": [_make_patient_id() for _ in range(n)],
            "age": age,
            "age_group": age_group,
            "gender": gender,
            "geographic_region": geographic_region,
            "insurance_product": insurance_product,
            "primary_diagnosis_code": primary_diagnosis_code,
        }
    )


def _generate_eligibility(
    rng: np.random.Generator, demographics: pd.DataFrame, config: RwdRealisticConfig
) -> pd.DataFrame:
    """Generate enrollment and eligibility windows with realistic fragmentation.

    Returns same DataFrame with added columns: eligeff (start), index_date,
    eligend, eligibility_duration_days, observation_months.
    """
    n = len(demographics)
    start = datetime.fromisoformat(config.start_date)
    end = datetime.fromisoformat(config.end_date)
    panel_days = (end - start).days

    # Enrollment start uniform across panel
    eligeff_offset = rng.integers(0, max(panel_days - 60, 1), n)
    eligeff = [start + timedelta(days=int(d)) for d in eligeff_offset]

    # Index date: 6-12 months post enrollment for most; sooner for fragmented
    is_fragmented = rng.random(n) < config.panel_fragmentation_rate
    months_to_index = np.where(
        is_fragmented,
        rng.integers(1, 6, n),  # Fragmented: 1-6 months
        rng.integers(6, 18, n),  # Standard: 6-18 months
    )
    index_date = [
        e + timedelta(days=int(m * 30)) for e, m in zip(eligeff, months_to_index, strict=True)
    ]

    # Eligibility duration as known at INDEX time (pre-prediction-time)
    eligibility_duration_days = [
        max((idx - e).days, 0) for e, idx in zip(eligeff, index_date, strict=True)
    ]

    # Total observation months (pre + post index combined)
    observation_months = np.where(
        is_fragmented,
        rng.integers(6, 18, n),  # Total <18mo
        rng.integers(18, 36, n),  # Standard 18-36mo
    )

    # eligend = "actual" end date in the dataset; note this can be post-index
    # (a vendor-encoding pattern that creates the post_hoc_termination leak).
    eligend = [
        idx + timedelta(days=int((obs - m) * 30))
        for idx, obs, m in zip(index_date, observation_months, months_to_index, strict=True)
    ]

    out = demographics.copy()
    out["eligeff"] = [d.isoformat() for d in eligeff]
    out["index_date"] = [d.isoformat() for d in index_date]
    out["eligend"] = [d.isoformat() for d in eligend]
    out["eligibility_duration_days"] = eligibility_duration_days
    out["is_fragmented"] = is_fragmented
    out["observation_months"] = observation_months
    return out


def _generate_target(
    rng: np.random.Generator, df: pd.DataFrame, config: RwdRealisticConfig
) -> np.ndarray:
    """Generate treatment_initiated target with weak demographic signal.

    Calibrated so that single-feature AUC for any demographic stays in
    [0.50, 0.65] and joint AUC achievable by a vanilla classifier is
    [0.62, 0.68] — matching the published claims-only ceiling.
    T2.3 derives this range dynamically per-cohort (see evaluator._emit_cohort_derived_honest_band).
    """
    n = len(df)
    # Coefficients tuned so signal is REAL but WEAK
    # (matches the 0.61-0.67 published-AUC range)
    age_norm = (df["age"].values - 50) / 20  # Normalized age centered at 50
    icd_severe = (df["primary_diagnosis_code"].isin(["L50.1", "L50.8"])).astype(int).values
    insurance_premium = (
        (df["insurance_product"].isin(["commercial_PPO", "self_insured"])).astype(int).values
    )
    long_eligibility = (df["eligibility_duration_days"].values > 365).astype(int)

    # Logit linear combination — coefficients tuned for AUC 0.62-0.68 at
    # signal_scale=1.0. Backlog #135: signal_scale multiplier on the 4
    # demographic coefficients enables T2.2 calibration sweep across
    # target AUCs [0.55, 0.85]. The base-rate and noise terms are NOT
    # scaled (otherwise scale=0 would still inherit the prevalence offset
    # and noise floor and not produce a pure-noise cohort).
    scale = config.signal_scale
    logit = (
        np.log(config.prevalence / (1 - config.prevalence))  # Base rate
        + scale * 0.25 * age_norm
        + scale * 0.45 * icd_severe
        + scale * 0.20 * insurance_premium
        + scale * 0.15 * long_eligibility
        + rng.normal(0, 0.5, n)  # Noise dominates over modest signal
    )
    prob = 1 / (1 + np.exp(-logit))
    return (rng.random(n) < prob).astype(int)


def _inject_leakage(
    df: pd.DataFrame,
    target: np.ndarray,
    rng: np.random.Generator,
    config: RwdRealisticConfig,
) -> pd.DataFrame:
    """Optionally inject one leakage pattern for testing the 4-layer defense.

    Each pattern adds exactly ONE column whose name encodes the pattern. The
    pipeline's defense layers should detect each (Layer 1 catches some at
    author time; Layer 3 catches statistical leaks; etc.).
    """
    out = df.copy()
    if config.leakage_pattern == "none":
        return out

    n = len(df)
    strength = config.leakage_strength

    if config.leakage_pattern == "post_index_aggregation":
        # Feature counts events post-index; deterministic-zero for untreated.
        # Multiplying by target gives 1-9 for treated, 0 for untreated; the
        # ``(1 - target) * 0`` term that used to live here was always 0 by
        # construction and was removed during the Layer-H audit (item H).
        out["post_index_med_count_LEAK"] = target * rng.integers(1, 10, n)

    elif config.leakage_pattern == "post_hoc_termination":
        # Feature derives from eligend - index, where eligend reflects actual
        # post-hoc termination (untreated patients have systematically different
        # eligibility-end patterns)
        out["months_remaining_eligibility_LEAK"] = (
            (1 - target) * (12 + rng.normal(6, 3, n))  # Untreated: long
            + target * (3 + rng.normal(2, 1, n))  # Treated: short
        ).clip(0, 24)

    elif config.leakage_pattern == "treatment_leaked_code":
        # ICD code assigned post-treatment (treatment_leaked_code='Z79.899' = encounter for long-term drug therapy)
        out["has_z79_long_term_drug_LEAK"] = target * (rng.random(n) < 0.85 * strength).astype(
            int
        ) + (1 - target) * (rng.random(n) < 0.05).astype(int)

    elif config.leakage_pattern == "spurious_correlation":
        # High single-feature AUC, no causal path (deliberately chosen wrong feature)
        out["spurious_score_LEAK"] = target * rng.normal(2, 0.5, n) + (1 - target) * rng.normal(
            0, 0.5, n
        )

    elif config.leakage_pattern == "pure_noise":
        # Control: should NOT be flagged by any layer
        out["random_noise_CONTROL"] = rng.normal(0, 1, n)

    elif config.leakage_pattern == "borderline_genuine":
        # v5 Gate C2 ENGINEERING CI SANITY-CHECK — NOT RWD positive evidence.
        #
        # A class-conditional Gaussian whose effect size (treated-mean offset
        # scaled by ``leakage_strength``) produces z in [5σ, 7.5σ] at
        # n_patients=20000. The injected feature is declared knowable_at=
        # index_date in the synthetic manifest (manifest source "synthetic"),
        # so the pipeline sees it as Layer 1 declared-safe.
        #
        # Contract under v5 §2 C2: legacy 5σ threshold → DROP (z > 5σ).
        # HBLP threshold for declared-safe = 5σ × 1.5 → RETAIN (z < 7.5σ).
        # The integration test pins this contrast.
        treated_mean = BORDERLINE_GENUINE_TREATED_MEAN * strength
        out[BORDERLINE_GENUINE_FEATURE_NAME] = target * rng.normal(
            treated_mean, BORDERLINE_GENUINE_SHARED_STD, n
        ) + (1 - target) * rng.normal(
            BORDERLINE_GENUINE_UNTREATED_MEAN, BORDERLINE_GENUINE_SHARED_STD, n
        )

    return out


def _apply_missing_data(
    df: pd.DataFrame, rng: np.random.Generator, config: RwdRealisticConfig
) -> pd.DataFrame:
    """Apply realistic missing-data patterns tied to insurance/age."""
    out = df.copy()
    n = len(out)

    # Insurance-tied missingness: medicaid + exchange more likely to have
    # missing demographic fields (matches access-driven RWD patterns)
    high_miss_insurance = out["insurance_product"].isin(["medicaid_managed", "exchange", "other"])
    miss_mask = (rng.random(n) < config.missing_demo_rate) & high_miss_insurance.values
    # Apply missingness to age (selectively)
    out.loc[miss_mask, "age"] = np.nan
    out.loc[miss_mask, "age_group"] = None

    return out


def generate_rwd_realistic(config: RwdRealisticConfig) -> pd.DataFrame:
    """Generate one complete RWD-realistic synthetic cohort.

    Returns:
        DataFrame with columns:
        - patient_id, age, age_group, gender, geographic_region,
          insurance_product, primary_diagnosis_code (demographics)
        - eligeff, index_date, eligend, eligibility_duration_days,
          is_fragmented, observation_months (eligibility/timing)
        - treatment_initiated (target)
        - Optional leakage column (named *_LEAK or *_CONTROL) per config
    """
    rng = np.random.default_rng(config.seed)

    df = _generate_demographics(rng, config.n_patients)
    df = _generate_eligibility(rng, df, config)
    target = _generate_target(rng, df, config)
    df["treatment_initiated"] = target
    df = _inject_leakage(df, target, rng, config)
    df = _apply_missing_data(df, rng, config)

    return df
