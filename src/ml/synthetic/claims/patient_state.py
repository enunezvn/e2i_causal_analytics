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
    return 1.0 / (1.0 + np.exp(-x))


def generate_patients(rng: np.random.Generator, cfg: ClaimsDGPConfig) -> pd.DataFrame:
    """Generate the patient-level latent state + demographics frame.

    The returned frame is the seed for every other emitter. The latent columns
    (``severity``/``response_propensity``/``adherence_propensity``/
    ``claim_index``) are internal — the CLI drops them before writing
    demographics.parquet.
    """
    n = cfg.n_patients
    scale = cfg.signal_scale

    # DGP item 1 — correlated latent block on a shared axis ``z``.
    z = rng.standard_normal(n)
    severity = _sigmoid(scale * (0.9 * z) + 0.4 * rng.standard_normal(n))
    response = _sigmoid(scale * (0.7 * z) + 0.6 * rng.standard_normal(n))
    adherence = _sigmoid(scale * (-0.6 * z) + 0.6 * rng.standard_normal(n))

    # Rolling claim index in the past ~7–18 months. Far enough back that the
    # synthetic eligend (index + post + slack) is plausible and the converter's
    # 360/180 windows are satisfiable on real, parseable dates.
    base = np.datetime64("today")
    claim_index = base - rng.integers(210, 540, n).astype("timedelta64[D]")

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

        # DGP item 3 — pre-index comorbidity admits, rate ~ severity, all inside
        # the (index - 180, index - 1] lookback window so they reach the
        # has_<comorbidity>/charlson/elixhauser features.
        n_com = int(rng.poisson(0.5 + 3.0 * float(p["severity"]) * cfg.signal_scale))
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
