"""Medication / procedure / lab / provider claim emit (Shard 10 P1.2).

DGP items 3–5:
  * Pre-index prior therapy (non-biologic antihistamines/LTRA) -> the
    converter's NON_TARGET_DRUG_CLASSES *_fill_count / *_days_supply_total
    pre-index features.
  * Cohort-A initiation: the FIRST CSU-biologic fill lands in (index,
    index+180], gated by ``response_propensity`` so ``treatment_initiated``
    (== ``initiated_biologic_180d``) is recoverable from pre-index signal.
  * disc/persistence: the POST-index biologic fill SEQUENCE has inter-fill gaps
    driven by ``adherence_propensity``. The converter reads those gaps to BUILD
    the disc/persistence *label* (Fact #3); because adherence shares the latent
    ``z`` axis with severity, the pre-index features predict the label.

Provenance honesty: labs carry only claims-plausible LOINC results, never a
serum-IgE patient-reported-outcome score (which claims cannot observe).

All emitters emit ONLY the columns the converter reads — no phantom columns.

The optional ``npi_for`` callable lets P1b inject network-aware NPI assignment
(shared-patient cliques). When absent, a deterministic per-patient HCP pool is
used so the med.npi ∪ proc.npi graph is still well-formed.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from .config import (
    BIOLOGIC_BRAND,
    BIOLOGIC_DAYS_SUP,
    BIOLOGIC_DISCONT_GAP_DAYS,
    BIOLOGIC_GENERIC,
    BIOLOGIC_NDC,
    PRIOR_THERAPY_GENERICS,
    ClaimsDGPConfig,
)

# Provider taxonomy codes — allergy/immunology + dermatology + PCP so the
# converter's saw_allergist/saw_dermatologist features populate. These are real
# NUCC codes recognised by rwd_common's NUCC_* groupings.
_TAXONOMIES = (
    "207K00000X",  # Allergy & Immunology
    "207N00000X",  # Dermatology
    "207Q00000X",  # Family Medicine (PCP)
    "208D00000X",  # General Practice (PCP)
)

# A non-biologic prior-therapy NDC prefix that does NOT collide with any
# CSU_BIOLOGIC_NDC_PREFIXES (50242/00024/0024).
_PRIOR_NDC = "00078"


def _default_npi_pool(rng: np.random.Generator, n_patients: int, n_hcps: int) -> np.ndarray:
    """Assign each patient a treating HCP NPI from a shared pool.

    Sharing HCPs across patients is what makes the converter's shared-patient
    co-treatment graph non-degenerate.
    """
    pool = np.array([f"{1000000000 + i}" for i in range(n_hcps)], dtype=object)
    return rng.choice(pool, size=n_patients)


def emit_medication(
    rng: np.random.Generator,
    pats: pd.DataFrame,
    cfg: ClaimsDGPConfig,
    npi_for: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Emit medication claims (prior therapy + biologic initiation/sequence)."""
    n = len(pats)
    n_hcps = cfg.n_hcps if cfg.n_hcps else max(8, n // 10)
    default_pool = None
    if npi_for is None:
        default_pool = _default_npi_pool(rng, n, n_hcps)

    rows: list[dict[str, object]] = []
    for pos, (_, p) in enumerate(pats.iterrows()):
        pid = int(p["patid"])
        idx = p["claim_index"]
        npi = npi_for(pid) if npi_for is not None else str(default_pool[pos])

        # Pre-index prior therapy (non-biologic) -> *_fill_count features.
        n_prior = int(rng.poisson(2 + 2.0 * float(p["severity"])))
        for _ in range(n_prior):
            offset = int(rng.integers(1, cfg.pre_days))
            d = idx - np.timedelta64(offset, "D")
            gen = PRIOR_THERAPY_GENERICS[int(rng.integers(0, len(PRIOR_THERAPY_GENERICS)))]
            rows.append(
                _med_row(
                    pid,
                    d,
                    npi,
                    code=_PRIOR_NDC,
                    brand=gen.upper(),
                    generic=gen,
                    days_sup=30,
                )
            )

        # Cohort-A initiation: first biologic fill in (index, index+180] iff the
        # responder draw fires. response_propensity drives the rate.
        p_init = float(
            np.clip((0.20 + 0.55 * float(p["response_propensity"])) * cfg.signal_scale, 0, 1)
        )
        if rng.random() < p_init:
            # First fill 0–120d post-index (inside the 180d initiation window).
            start = idx + np.timedelta64(int(rng.integers(0, 120)), "D")
            # Item 5 — post-index refill sequence. Adherent patients refill on
            # schedule (~28d); non-adherent patients open a gap > the disc
            # threshold, which the converter reads to set discontinued_180d.
            adherence = float(p["adherence_propensity"])
            base_gap = BIOLOGIC_DAYS_SUP
            extra_gap = (1.0 - adherence) * (BIOLOGIC_DISCONT_GAP_DAYS + 40)
            gap = int(max(base_gap + rng.normal(extra_gap, 8), 7))
            n_fills = int(rng.integers(2, 7))
            for k in range(n_fills):
                d = start + np.timedelta64(k * gap, "D")
                rows.append(
                    _med_row(
                        pid,
                        d,
                        npi,
                        code=BIOLOGIC_NDC,
                        brand=BIOLOGIC_BRAND,
                        generic=BIOLOGIC_GENERIC,
                        days_sup=BIOLOGIC_DAYS_SUP,
                    )
                )

    cols = [
        "patid",
        "medication_date",
        "npi",
        "code",
        "days_sup",
        "strength",
        "Brand_Name",
        "Generic_Name",
    ]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def _med_row(
    pid: int,
    d: np.datetime64,
    npi: str,
    *,
    code: str,
    brand: str,
    generic: str,
    days_sup: int,
) -> dict[str, object]:
    return {
        "patid": pid,
        "medication_date": d,
        "npi": npi,
        "code": code,
        "days_sup": days_sup,
        "strength": "150 MG",
        "Brand_Name": brand,
        "Generic_Name": generic,
    }


def emit_procedure(
    rng: np.random.Generator,
    pats: pd.DataFrame,
    cfg: ClaimsDGPConfig,
    npi_for: Callable[[int], str] | None = None,
) -> pd.DataFrame:
    """Emit procedure (office-visit) claims feeding office_visits + HCP graph."""
    n = len(pats)
    n_hcps = cfg.n_hcps if cfg.n_hcps else max(8, n // 10)
    default_pool = None
    if npi_for is None:
        default_pool = _default_npi_pool(rng, n, n_hcps)

    rows: list[dict[str, object]] = []
    for pos, (_, p) in enumerate(pats.iterrows()):
        pid = int(p["patid"])
        idx = p["claim_index"]
        npi = npi_for(pid) if npi_for is not None else str(default_pool[pos])
        # Pre-index office visits (E&M 99213/99214), rate ~ severity. These feed
        # office_visits_* + unique_providers + the shared-patient graph.
        n_visits = int(rng.poisson(1 + 2.0 * float(p["severity"])))
        for _ in range(n_visits):
            offset = int(rng.integers(1, cfg.pre_days))
            d = idx - np.timedelta64(offset, "D")
            code = "99213" if rng.random() < 0.6 else "99214"
            rows.append({"patid": pid, "proc_date": d, "proc_code": code, "npi": npi})

    cols = ["patid", "proc_date", "proc_code", "npi"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def emit_lab(
    rng: np.random.Generator,
    pats: pd.DataFrame,
    cfg: ClaimsDGPConfig,
) -> pd.DataFrame:
    """Emit lab claims with claims-plausible LOINCs ONLY (provenance honesty).

    The converter's CSU_LABS_LOINC keys (ige_total, eosinophil, crp, tsh, ...)
    are observable in claims via LOINC. A serum-IgE *PRO score* is NOT — so the
    DGP never emits an "IGE_TOTAL" pseudo-LOINC.
    """
    # Verified CSU_LABS_LOINC codes (convert_optum_rwd.py:297-314).
    lab_menu = [
        ("19113-0", "IMMUNOGLOBULIN E, TOTAL"),
        ("711-2", "EOSINOPHIL COUNT"),
        ("1988-5", "C-REACTIVE PROTEIN"),
        ("3016-3", "TSH"),
        ("58410-2", "CBC PANEL"),
    ]
    rows: list[dict[str, object]] = []
    for _, p in pats.iterrows():
        pid = int(p["patid"])
        idx = p["claim_index"]
        n_labs = int(rng.poisson(1 + float(p["severity"])))
        for _ in range(n_labs):
            offset = int(rng.integers(1, cfg.pre_days))
            d = idx - np.timedelta64(offset, "D")
            loinc, desc = lab_menu[int(rng.integers(0, len(lab_menu)))]
            result = float(rng.gamma(2.0, 10.0))
            abnl = "H" if result > 30 else ""
            rows.append(
                {
                    "patid": pid,
                    "fst_dt": d,
                    "loinc_cd": loinc,
                    "rslt_nbr": result,
                    "abnl_cd": abnl,
                    "tst_desc": desc,
                }
            )

    cols = ["patid", "fst_dt", "loinc_cd", "rslt_nbr", "abnl_cd", "tst_desc"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]


def emit_provider(
    rng: np.random.Generator,
    med: pd.DataFrame,
    proc: pd.DataFrame,
    cfg: ClaimsDGPConfig,
) -> pd.DataFrame:
    """Emit the npi -> taxonomy1 lookup (the sole provider read, :1550-1553)."""
    npis: set[str] = set()
    if "npi" in med.columns:
        npis |= set(med["npi"].dropna().astype(str))
    if "npi" in proc.columns:
        npis |= set(proc["npi"].dropna().astype(str))
    npis.discard("")
    npis.discard("nan")
    ordered = sorted(npis)
    rows = [
        {"npi": npi, "taxonomy1": _TAXONOMIES[int(rng.integers(0, len(_TAXONOMIES)))]}
        for npi in ordered
    ]
    cols = ["npi", "taxonomy1"]
    if not rows:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(rows)[cols]
