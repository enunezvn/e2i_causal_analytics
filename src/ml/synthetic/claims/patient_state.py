"""Latent state + demographics + inpatient emit (Shard 10 P1.1).

DGP items 1–3:
  1. Correlated latent block (severity / response_propensity /
     adherence_propensity) drawn from a shared standard-normal axis. These NEVER
     reach the parquet output — they are dropped before write (P1.3) — they only
     drive the downstream claim event DATES, which is what the converter reads.
  2. Enrollment gate: ``eligeff``/``eligend`` bracket the claim-dated index so
     non-fragmented patients clear ``_check_enrollment_window`` (production
     regime 360/180). The fragmentation knob deliberately fails ~half the panel
     to mirror real-world panel churn (converter drops them at :2017).
  3. Inpatient L50.x admits (≥2 distinct dates → cohort-A index = the 2nd) plus
     pre-index comorbidity admits whose RATE scales with ``severity`` so the
     converter's has_<comorbidity>/charlson/elixhauser features carry real,
     non-degenerate pre-index signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import COMORBIDITY_DX, CSU_DX_CODES, ClaimsDGPConfig


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-x)))


def generate_patients(rng: np.random.Generator, cfg: ClaimsDGPConfig) -> pd.DataFrame:
    """Generate the patient-level latent state + demographics frame.

    The returned frame is the seed for every other emitter. The latent columns
    (``severity``/``response_propensity``/``adherence_propensity``/
    ``claim_index``) are internal — the CLI drops them before writing
    demographics.parquet.
    """
    n = cfg.n_patients
    scale = cfg.signal_scale

    # DGP item 1 — TWO independent latent axes (both RAW standard normals):
    #   * ``severity``  — disease burden; drives comorbidity / hospitalisation /
    #     office-visit counts (the converter's has_<comorbidity>/charlson/
    #     elixhauser/office_visits features).
    #   * ``tx_burden`` — prior-therapy escalation; drives non-biologic
    #     antihistamine/LTRA fill counts (the converter's NON_TARGET_DRUG_CLASSES
    #     *_fill_count / *_days_supply_total features).
    # Keeping them INDEPENDENT is what gives the longitudinal prior-therapy
    # features signal BEYOND the comorbidity-only baseline — the exact margin the
    # cheapest-disproof checks (> 0.03). The target depends on BOTH axes, so the
    # comorbidity-only subset CANNOT recover the full effect.
    severity = rng.standard_normal(n)
    tx_burden = rng.standard_normal(n)

    # The post-index TARGET propensities are logistic in BOTH latents plus a
    # calibrated noise term that caps the recoverable AUC at the honest band.
    # The latents and the noise are the ONLY drivers, so the pre-index features
    # (which track the latents) statistically predict the post-index target
    # without the target ever being a feature (Fact #3).
    init_logit = (
        scale * cfg.init_severity_coef * severity
        + scale * cfg.init_tx_coef * tx_burden
        + cfg.init_noise_sd * rng.standard_normal(n)
    )
    response = _sigmoid(init_logit)
    # Higher severity AND higher prior-therapy burden -> LOWER adherence (sicker,
    # more treatment-experienced patients drop off / gap), so the disc/
    # persistence labels are also recoverable from BOTH pre-index axes.
    adh_logit = (
        -scale * cfg.adherence_severity_coef * severity
        - scale * cfg.adherence_tx_coef * tx_burden
        + cfg.adherence_noise_sd * rng.standard_normal(n)
    )
    adherence = _sigmoid(adh_logit)

    # Rolling claim index spread over the past ~7–30 months. A WIDE temporal
    # span is required so the tier-0 ENTITY+TEMPORAL split can form balanced
    # time buckets (a narrow span collapses the temporal split and discards most
    # rows). Still far enough back that the synthetic eligend (index + post +
    # slack) is plausible and the converter's 360/180 windows are satisfiable.
    base = np.datetime64("today")
    claim_index = base - rng.integers(210, 930, n).astype("timedelta64[D]")

    # DGP item 2 — enrollment window. Non-fragmented patients clear the strict
    # gate (eligeff <= index - pre AND eligend >= index + post).
    eligeff = (
        claim_index
        - np.timedelta64(cfg.pre_days, "D")
        - rng.integers(0, 120, n).astype("timedelta64[D]")
    )
    eligend = (
        claim_index
        + np.timedelta64(cfg.post_days, "D")
        + rng.integers(0, 120, n).astype("timedelta64[D]")
    )
    frag = rng.random(n) < cfg.panel_fragmentation_rate
    # Fragmented patients violate eligeff <= index - pre -> dropped at :2017.
    eligeff = np.where(
        frag,
        claim_index - np.timedelta64(cfg.pre_days // 2, "D"),
        eligeff,
    )

    return pd.DataFrame(
        {
            "patid": np.arange(1, n + 1, dtype=np.int64),
            "eligeff": eligeff,
            "eligend": eligend,
            "claim_index": claim_index,
            "diagcode": CSU_DX_CODES[0],  # demographics CSU gate :1663
            "age": rng.integers(18, 80, n),
            "gdr_cd": rng.choice(["M", "F"], n),
            "zipcode_5": rng.integers(10000, 99999, n).astype(str),
            "bus": rng.choice(["CDH", "HMO", "PPO"], n),  # -> insurance_product :2276
            "product": "COM",
            "health_exch": 0,
            "lis_dual": 0,
            "continuous_enrollment": (~frag).astype(int),  # gate :1658
            "severity": severity,
            "tx_burden": tx_burden,
            "response_propensity": response,
            "adherence_propensity": adherence,
        }
    )


def emit_inpatient(
    rng: np.random.Generator, pats: pd.DataFrame, cfg: ClaimsDGPConfig
) -> pd.DataFrame:
    """Emit inpatient claims: ≥2 L50.x admits (index anchor) + comorbidity admits.

    Only the columns the converter reads are emitted: ``patid``, ``admit_date``,
    ``disch_date``, ``diag1..5``, ``tos_cd``.
    """
    rows: list[dict[str, object]] = []
    diag_cols = ["diag1", "diag2", "diag3", "diag4", "diag5"]
    for _, p in pats.iterrows():
        pid = int(p["patid"])
        idx = p["claim_index"]
        # Fact #1 — ≥2 distinct L50.x admit dates so _derive_index_date returns
        # the 2nd. The earlier admit is ~20d before the chosen index date.
        for d in (idx - np.timedelta64(20, "D"), idx):
            row = {
                "patid": pid,
                "admit_date": d,
                "disch_date": d + np.timedelta64(1, "D"),
                "diag1": CSU_DX_CODES[int(rng.integers(0, len(CSU_DX_CODES)))],
                "tos_cd": "IP",
            }
            for c in diag_cols[1:]:
                row[c] = ""
            rows.append(row)

        # DGP item 3 — pre-index comorbidity admits, log-linear rate in severity.
        # A log-linear rate keeps the count non-negative for the raw-normal
        # severity AND makes the observed count encode severity with good SNR.
        #
        # The admits are spread over the FULL 180d pre-index window. For cohort
        # B/C the converter re-anchors index to the first biologic fill (up to
        # 120d AFTER the cohort-A index), so its 180d lookback only partially
        # overlaps this window. A higher base rate ensures enough comorbidity
        # admits land in EITHER lookback window for the severity signal to be
        # recoverable in all three cohorts.
        rate_com = float(np.exp(np.log(4.0) + cfg.feature_log_rate_coef * float(p["severity"])))
        n_com = int(rng.poisson(rate_com))
        for _ in range(n_com):
            offset = int(rng.integers(1, 175))
            d = idx - np.timedelta64(offset, "D")
            row = {
                "patid": pid,
                "admit_date": d,
                "disch_date": d,
                "diag1": COMORBIDITY_DX[int(rng.integers(0, len(COMORBIDITY_DX)))],
                "tos_cd": "IP",
            }
            for c in diag_cols[1:]:
                row[c] = ""
            rows.append(row)

    cols = ["patid", "admit_date", "disch_date", *diag_cols, "tos_cd"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]
