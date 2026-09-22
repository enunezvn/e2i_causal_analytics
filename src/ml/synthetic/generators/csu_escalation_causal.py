"""Planted-truth synthetic CSU escalation cohort in the ``csu_escalation_causal``
contract (Lane C, spec 2026-09-22 §3C.2-3).

Remibrutinib (Rhapsido) is absent from every real claims drop, so the registry
entry ``csu_escalation_causal`` (treatment ``treatment_remibrutinib`` vs the
competitor biologics, the four Lane A outcomes, the 64 ``MART_SAFE_FEATURES``
covariates) is backed by THIS cohort until the post-launch refresh. Every row is
``is_synthetic = True``; the DGP plants a known, recoverable effect so the whole
registry -> loader -> causal_impact agent path can be certified end to end
(``tests/unit/test_api/test_causal_csu_escalation_planted_truth.py``) with a
fixed seed and no data files.

The DGP reuses the platform's causal primitives (``src.ml.synthetic.dgp``):

* treatment ``T ~ Bernoulli(e(X))`` through :func:`assign_arm_from_spec`,
  CONFOUNDED on three measured contract columns -- ``age_at_index`` (older
  patients less likely to start the new oral), ``charlson_score`` (more
  comorbid, less likely) and ``payer_category`` (commercial more likely; a
  TEXT column, so the planted backdoor only closes if the loader's one-hot
  path delivers it to the estimator) -- with overlap clipped to [0.01, 0.99];
* each binary outcome through :func:`binary_outcome_rd`: latent = baseline(X)
  + T * tau_latent(segment) + noise, thresholded at a fixed prevalence, with
  the per-unit RISK-DIFFERENCE ``tau_i`` computed analytically so
  ``mean(tau_i)`` is the RD-scale TRUE_ATE the estimators target. The segment
  (three age bands) is a function of a MEASURED covariate, so the heterogeneity
  is identified.

The planted confounding is deliberately large: the naive diff-in-means sits
well outside the tolerance, so a run that silently drops the adjustment set
(or the one-hot payer dummies) fails the recovery check instead of passing by
luck. ``PlantedTruth`` records both numbers.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.manifests import MART_SAFE_FEATURES
from src.ml.synthetic.dgp.treatment_arm import (
    SEGMENT_HIGH,
    SEGMENT_LOW,
    SEGMENT_MEDIUM,
    ArmSpec,
    assign_arm_from_spec,
    binary_outcome_rd,
)

DATASET = "csu_escalation_causal"
TREATMENT = "treatment_remibrutinib"
PRIMARY_OUTCOME = "persistent_at_180d_g28"
OUTCOMES: Tuple[str, ...] = (
    PRIMARY_OUTCOME,
    "discontinued_180d",
    "biologic_switch_180d_flag",
    "persistent_at_180d",
)
TREATED_ARM = "RHAPSIDO"
COMPETITOR_ARMS: Tuple[str, ...] = ("XOLAIR", "DUPIXENT")
# The measured contract columns the arm is confounded on. ``payer_category``
# enters the propensity through its ``commercial`` level.
PLANTED_CONFOUNDERS: Tuple[str, ...] = ("age_at_index", "charlson_score", "payer_category")
DEFAULT_N = 3000
DEFAULT_SEED = 20260922
# The tolerance the ground-truth sidecar of the synthetic CSU set uses
# (data/rwd/synthetic_CSU/ground_truth_*.json: 0.10 at n = 8,420 per brand).
TOLERANCE = 0.10
# The planted-truth E2E's DISCRIMINATING check (verifier MED-A, 2026-09-22):
# at this DGP the 0.10 sidecar window admits losing the whole payer backdoor,
# so recovery is asserted as a capability -- the error must be inside
# RECOVERY_TOLERANCE AND below NAIVE_ERROR_FRACTION of the naive contrast's
# error. Measured through the real graph on the default frame (evidence
# README D2b): full adjustment 0.406 (error 0.042), payer_category dropped
# 0.436 (error 0.072), naive 0.498 (error 0.134); run-to-run spread ~0.001.
# 0.06 is the midpoint of the two measured outcomes (0.057, rounded up), so
# each side keeps a >= 0.012 margin.
RECOVERY_TOLERANCE = 0.06
NAIVE_ERROR_FRACTION = 0.5

IDENTITY_COLUMNS: Tuple[str, ...] = (
    "patient_id",
    "patient_journey_id",
    "patient_hash",
    "index_date",
    "journey_start_date",
    "journey_status",
    "discontinuation_flag",
    "data_quality_score",
    "data_split",
    "index_biologic_brand",
    TREATMENT,
    "treatment_start_date",
)
# The column order of the table (migration 149) minus the server-defaulted
# created_at / updated_at.
CONTRACT_COLUMNS: Tuple[str, ...] = (
    *IDENTITY_COLUMNS,
    *OUTCOMES,
    *MART_SAFE_FEATURES,
    "is_synthetic",
)

_CATEGORICAL_LEVELS: Dict[str, Tuple[Tuple[str, ...], Tuple[float, ...]]] = {
    "gdr_cd": (("F", "M"), (0.67, 0.33)),
    "payer_category": (("commercial", "medicare", "medicaid", "other"), (0.55, 0.25, 0.15, 0.05)),
    "payer_product": (("HMO", "PPO", "POS", "EPO"), (0.35, 0.40, 0.15, 0.10)),
    "payer_bus": (("COM", "MCR", "MCD"), (0.60, 0.25, 0.15)),
    "elixhauser_risk_band": (("low", "medium", "high"), (0.60, 0.30, 0.10)),
    "geographic_region": (("midwest", "south", "northeast", "west"), (0.22, 0.38, 0.18, 0.22)),
}
# Share of rows whose geographic_region is NULL (the real cohort has 8.1%): the
# loader must give those rows their own ``geographic_region=__missing__`` dummy.
_REGION_NULL_SHARE = 0.06
_COUNT_COLUMNS: Tuple[str, ...] = (
    "elixhauser_van_walraven_score",
    "comorbidity_diag_distinct_count",
    "comorbidity_diag_claim_count",
)


@dataclass(frozen=True)
class PlantedTruth:
    """The recoverable ground truth for ONE outcome of the planted cohort."""

    outcome_variable: str
    true_ate: float
    tolerance: float
    confounders: List[str]
    treatment_variable: str
    naive_diff: float
    cate_by_segment: Dict[str, float]
    n_samples: int
    arm_split: Dict[str, int]
    seed: int
    dgp: str = field(default="csu_escalation_causal.confounded")

    def is_estimate_valid(self, estimated_ate: float) -> bool:
        return abs(float(estimated_ate) - self.true_ate) <= self.tolerance

    def get_error(self, estimated_ate: float) -> float:
        return abs(float(estimated_ate) - self.true_ate)

    def naive_error(self) -> float:
        return abs(self.naive_diff - self.true_ate)

    def is_recovery_convincing(self, estimated_ate: float) -> bool:
        """The capability check: inside ``RECOVERY_TOLERANCE`` AND below
        ``NAIVE_ERROR_FRACTION`` of the naive contrast's error, so an estimate
        that lost a planted confounder (or never adjusted) fails."""
        err = self.get_error(estimated_ate)
        return err < RECOVERY_TOLERANCE and err < NAIVE_ERROR_FRACTION * self.naive_error()

    def recovery_report(self, estimated_ate: float) -> Dict[str, float]:
        """The numbers behind :meth:`is_recovery_convincing`, for an assertion message."""
        return {
            "ate": float(estimated_ate),
            "true_ate": self.true_ate,
            "error": self.get_error(estimated_ate),
            "recovery_tolerance": RECOVERY_TOLERANCE,
            "naive_diff": self.naive_diff,
            "naive_error": self.naive_error(),
            "naive_error_ceiling": NAIVE_ERROR_FRACTION * self.naive_error(),
            "sidecar_tolerance": self.tolerance,
        }

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _rd_by_segment(segment: np.ndarray, tau_i: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for s in (SEGMENT_HIGH, SEGMENT_MEDIUM, SEGMENT_LOW):
        m = segment == s
        if m.any():
            out[s] = round(float(np.mean(tau_i[m])), 4)
    return out


def _age_segment(age: np.ndarray) -> np.ndarray:
    return np.where(age > 60, SEGMENT_HIGH, np.where(age > 40, SEGMENT_MEDIUM, SEGMENT_LOW))


def _draw_baseline_covariates(n: int, rng: np.random.Generator) -> Dict[str, np.ndarray]:
    cov: Dict[str, np.ndarray] = {}
    cov["age_at_index"] = np.clip(rng.normal(48.0, 15.0, n), 18, 89).round()
    for col, (levels, probs) in _CATEGORICAL_LEVELS.items():
        cov[col] = rng.choice(levels, size=n, p=probs).astype(object)
    region_null = rng.random(n) < _REGION_NULL_SHARE
    # ``np.full(..., None, dtype=object)`` rather than a bare ``None``: numpy's
    # ``where`` stub only accepts ArrayLike branches; the produced object
    # column (None for the null share) is identical.
    cov["geographic_region"] = np.where(
        region_null, np.full(n, None, dtype=object), cov["geographic_region"]
    )
    cov["health_exchange_flag"] = rng.binomial(1, 0.06, n)
    cov["lis_dual_flag"] = rng.binomial(1, 0.09, n)
    cov["enrollment_duration_days"] = rng.integers(365, 3650, n)
    cov["charlson_score"] = rng.poisson(0.8, n)
    cov["charlson_risk_band"] = np.where(
        cov["charlson_score"] >= 3, "high", np.where(cov["charlson_score"] >= 1, "medium", "low")
    ).astype(object)
    for col in _COUNT_COLUMNS:
        cov[col] = rng.poisson(2.0, n)
    cov["high_comorbidity_burden_flag"] = (cov["charlson_score"] >= 3).astype(int)
    for col in MART_SAFE_FEATURES:
        if col not in cov:  # the cci_* / elx_* 0-1 flags
            cov[col] = rng.binomial(1, 0.08, n)
    return cov


def generate_csu_escalation_cohort(
    n: int = DEFAULT_N,
    seed: int = DEFAULT_SEED,
    *,
    tolerance: float = TOLERANCE,
) -> Tuple[pd.DataFrame, Dict[str, PlantedTruth]]:
    """Return ``(frame, truths)``: the planted cohort in the contract's column
    order and the recoverable truth per outcome (keyed by outcome column)."""
    if n < 200:
        raise ValueError(f"n={n}: the planted cohort needs at least 200 rows")
    rng = np.random.default_rng(seed)
    cov = _draw_baseline_covariates(n, rng)
    age = cov["age_at_index"].astype(float)
    charlson = cov["charlson_score"].astype(float)
    is_commercial = (cov["payer_category"] == "commercial").astype(float)
    depression = cov["elx_depression"].astype(float)

    arm_spec = ArmSpec(
        name=TREATMENT,
        confounders={"age_at_index": -0.03, "charlson_score": -0.35, "is_commercial": 0.9},
        intercept=-0.6,
        cate_by_segment={},
        target_outcomes=OUTCOMES,
        center={"age_at_index": 48.0},
        propensity_col="",
    )
    arm, _propensity = assign_arm_from_spec(
        arm_spec,
        {"age_at_index": age, "charlson_score": charlson, "is_commercial": is_commercial},
        rng,
    )
    segment = _age_segment(age)

    # Prognostic baseline shared by the persistence outcomes: the SAME measured
    # columns that drive the arm (so adjustment is required) plus depression.
    persist_baseline = (
        -0.02 * (age - 48.0) - 0.25 * charlson + 0.40 * is_commercial - 0.30 * depression
    )
    truths: Dict[str, PlantedTruth] = {}
    outcome_cols: Dict[str, np.ndarray] = {}

    def _plant(outcome: str, baseline: np.ndarray, cate_map: Dict[str, float], prev: float):
        y, tau_i = binary_outcome_rd(
            arm, baseline, segment, cate_map, rng, target_prevalence=prev, noise_std=0.6
        )
        outcome_cols[outcome] = y.astype(int)
        naive = float(y[arm == 1].mean() - y[arm == 0].mean())
        truths[outcome] = PlantedTruth(
            outcome_variable=outcome,
            true_ate=round(float(np.mean(tau_i)), 4),
            tolerance=tolerance,
            confounders=list(PLANTED_CONFOUNDERS),
            treatment_variable=TREATMENT,
            naive_diff=round(naive, 4),
            cate_by_segment=_rd_by_segment(segment, tau_i),
            n_samples=n,
            arm_split={"0": int((arm == 0).sum()), "1": int((arm == 1).sum())},
            seed=seed,
        )

    _plant(
        PRIMARY_OUTCOME,
        persist_baseline,
        {SEGMENT_HIGH: 0.9, SEGMENT_MEDIUM: 0.7, SEGMENT_LOW: 0.5},
        0.45,
    )
    _plant(
        "discontinued_180d",
        -persist_baseline,
        {SEGMENT_HIGH: -0.7, SEGMENT_MEDIUM: -0.55, SEGMENT_LOW: -0.4},
        0.35,
    )
    # The SHIPPED persistence definition: same latent structure, attenuated
    # effect (days-supply sensitive per brand in the real cohort).
    _plant(
        "persistent_at_180d",
        persist_baseline,
        {SEGMENT_HIGH: 0.45, SEGMENT_MEDIUM: 0.35, SEGMENT_LOW: 0.25},
        0.40,
    )
    # Switching is rare and, by design, NOT affected by the arm: a planted null.
    switch = rng.binomial(1, 0.03, n)
    outcome_cols["biologic_switch_180d_flag"] = switch
    truths["biologic_switch_180d_flag"] = PlantedTruth(
        outcome_variable="biologic_switch_180d_flag",
        true_ate=0.0,
        tolerance=tolerance,
        confounders=list(PLANTED_CONFOUNDERS),
        treatment_variable=TREATMENT,
        naive_diff=round(float(switch[arm == 1].mean() - switch[arm == 0].mean()), 4),
        cate_by_segment={SEGMENT_HIGH: 0.0, SEGMENT_MEDIUM: 0.0, SEGMENT_LOW: 0.0},
        n_samples=n,
        arm_split={"0": int((arm == 0).sum()), "1": int((arm == 1).sum())},
        seed=seed,
    )

    # Identity + journey metadata in the converter's record shape.
    ids = np.arange(1, n + 1)
    patient_id = np.array([f"PAT_SYN_{i:09d}" for i in ids])
    index_date = pd.Timestamp("2026-01-01") + pd.to_timedelta(rng.integers(0, 180, n), unit="D")
    treatment_start = index_date + pd.to_timedelta(rng.integers(0, 61, n), unit="D")
    brand = np.where(
        arm == 1,
        TREATED_ARM,
        rng.choice(COMPETITOR_ARMS, size=n, p=(0.72, 0.28)),
    ).astype(object)
    split = rng.choice(
        ["train", "validation", "test", "holdout"], size=n, p=(0.60, 0.20, 0.15, 0.05)
    )
    frame = pd.DataFrame(
        {
            "patient_id": patient_id,
            "patient_journey_id": np.array([f"PJ_SYN_{i:09d}" for i in ids]),
            "patient_hash": np.array(
                [
                    # a synthetic-cohort join key, not a security hash (Bandit B324)
                    hashlib.sha1(p.encode(), usedforsecurity=False).hexdigest()[:16]
                    for p in patient_id
                ]
            ),
            "index_date": index_date.date,
            "journey_start_date": index_date.date,
            "journey_status": "active",
            "discontinuation_flag": outcome_cols["discontinued_180d"],
            "data_quality_score": rng.uniform(0.70, 1.0, n).round(3),
            "data_split": split,
            "index_biologic_brand": brand,
            TREATMENT: arm.astype(int),
            "treatment_start_date": treatment_start.date,
        }
    )
    for outcome in OUTCOMES:
        frame[outcome] = outcome_cols[outcome]
    for col in MART_SAFE_FEATURES:
        frame[col] = cov[col]
    frame["is_synthetic"] = True
    frame = frame[list(CONTRACT_COLUMNS)]
    return frame, truths
