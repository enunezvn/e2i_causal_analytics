"""End-to-end integration test for `feat/phase4b-csu-converter-masking`.

Builds a synthetic CSU dataset where treated and untreated patients have
intentionally distinct event distributions on the *post-index* side of the
timeline (replicating the real CSU vendor pattern documented in
`docs/lineage/csu_field_audit.md` §3 rows 36-39, 43). Then asserts that the
masked converter (`--lookback-days=180`) produces single-feature AUC < 0.85
on every column flagged by the leakage detector, while the unmasked
converter produces AUC ≥ 0.85 on most of them (i.e. demonstrating that the
masking is what eliminates the leakage).

This is the §8.3 acceptance signal in test form, runnable in CI without the
gitignored `data/rwd/csu/csu_data.xlsx` workbook.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from convert_csu_rwd import CSUDataConverter  # noqa: E402,I001


LEAKY_FEATURES = [
    "disease_severity",
    "engagement_score",
    "days_on_therapy",
    "hcp_visits",
    "medication_claim_count",
]


def _build_synthetic_cohort(
    n_treated: int = 60, n_untreated: int = 40, seed: int = 42
) -> dict[str, pd.DataFrame]:
    """Synthesise demo / med / proc / lab sheets with intentional CSU-pattern leakage.

    - All patients share the same `index_date = 2024-01-01`.
    - Treated patients have many medication fills concentrated AFTER index_date
      (the CSU vendor pattern).
    - Untreated patients have no medication rows.
    - Procedures and labs are randomly distributed across the [-365, +365] window.

    With masking off, single-feature AUC should be near 1.0 for med-derived
    features. With masking to a 180d pre-index window, the AUC should collapse
    because the discriminating signal (post-index events) is masked out.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    index_date = pd.Timestamp("2024-01-01")

    # ---- demo sheet -----------------------------------------------------
    n = n_treated + n_untreated
    demo = pd.DataFrame(
        {
            "patid": list(range(1, n + 1)),
            "indexdt": [index_date] * n,
            "eligeff": [index_date - timedelta(days=730)] * n,
            "eligend": [index_date + timedelta(days=365)] * n,
            "age": rng.integers(25, 75, size=n).astype(float),
            "gdr_cd": rng.choice(["F", "M"], size=n).tolist(),
            "bus": ["COM"] * n,
            "diagcode": ["L508"] * n,
            "continuous_enrollment": [1] * n,
            "zipcode_5": ["12345"] * n,
        }
    )

    # ---- medication sheet (only treated patients) -----------------------
    med_rows: list[dict[str, Any]] = []
    for patid in range(1, n_treated + 1):
        # 5-12 fills, mostly post-index (mimics the CSU vendor pattern)
        n_fills = int(rng.integers(5, 13))
        for _ in range(n_fills):
            # 80% post-index, 20% pre-index but PRE-lookback (so they should
            # also be masked out — guarantees AUC drops to near-0.5)
            if rng.random() < 0.8:
                offset = int(rng.integers(1, 365))
            else:
                offset = -int(rng.integers(200, 720))
            med_rows.append(
                {
                    "patid": patid,
                    "medication_date": index_date + timedelta(days=offset),
                    "days_sup": int(rng.integers(15, 90)),
                    "npi": f"NPI{rng.integers(1, 50):03d}",
                    "brand_normalised": "drugA",
                    "indexdt": index_date,
                }
            )
    medication = pd.DataFrame(med_rows)

    # ---- procedure sheet (random across the panel) ----------------------
    proc_rows: list[dict[str, Any]] = []
    for patid in range(1, n + 1):
        n_procs = int(rng.integers(0, 5))
        for _ in range(n_procs):
            offset = int(rng.integers(-365, 365))
            proc_rows.append(
                {
                    "patid": patid,
                    "proc_date": index_date + timedelta(days=offset),
                    "proc_code": "J2357" if rng.random() < 0.5 else "OTHER",
                    "indexdt": index_date,
                }
            )
    proc = (
        pd.DataFrame(proc_rows)
        if proc_rows
        else pd.DataFrame(columns=["patid", "proc_date", "proc_code", "indexdt"])
    )

    # ---- lab sheet (random across the panel) ----------------------------
    lab_rows: list[dict[str, Any]] = []
    for patid in range(1, n + 1):
        n_labs = int(rng.integers(0, 4))
        for _ in range(n_labs):
            offset = int(rng.integers(-365, 365))
            lab_rows.append(
                {
                    "patid": patid,
                    "fst_dt": index_date + timedelta(days=offset),
                    "abnl_cd": "A" if rng.random() < 0.3 else None,
                    "tst_desc": "lab1",
                    "rslt_nbr": float(rng.normal(50, 10)),
                    "loinc_cd": "12345-6",
                    "indexdt": index_date,
                }
            )
    lab = (
        pd.DataFrame(lab_rows)
        if lab_rows
        else pd.DataFrame(
            columns=[
                "patid",
                "fst_dt",
                "abnl_cd",
                "tst_desc",
                "rslt_nbr",
                "loinc_cd",
                "indexdt",
            ]
        )
    )

    return {"demo": demo, "medication": medication, "proc": proc, "lab": lab}


def _run_converter(
    sheets: dict[str, pd.DataFrame], lookback_days: int | None, tmp_path: Path
) -> list[dict[str, Any]]:
    """Inject sheets directly and run the journey-builder pipeline."""
    converter = CSUDataConverter(
        excel_path=tmp_path / "fake.xlsx",
        output_dir=tmp_path / f"out_{lookback_days or 'off'}",
        lookback_days=lookback_days,
    )
    converter.sheets = {k: v.copy() for k, v in sheets.items()}
    converter._index_clinical_data()
    converter._build_patient_id_map()
    return converter._build_patient_journeys()


def _single_feature_auc(journeys: list[dict[str, Any]], feature: str) -> float:
    """Single-feature AUC against treatment_initiated, with None→0 imputation."""
    y = [int(r["treatment_initiated"]) for r in journeys]
    scores = [r.get(feature) for r in journeys]
    scores_imp = [0 if s is None else float(s) for s in scores]
    return float(roc_auc_score(y, scores_imp))


@pytest.fixture(scope="module")
def synthetic_sheets() -> dict[str, pd.DataFrame]:
    return _build_synthetic_cohort()


def test_unmasked_converter_produces_high_auc_leakage(
    synthetic_sheets: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """Sanity check: without masking, the leaky features show their leakage."""
    journeys = _run_converter(synthetic_sheets, lookback_days=None, tmp_path=tmp_path)
    # At least 4 of the 5 leaky features should show AUC >= 0.85 in the
    # unmasked baseline; this confirms the synthetic cohort actually has
    # the leakage pattern we're trying to mask.
    high_auc_count = sum(
        _single_feature_auc(journeys, feature) >= 0.85 for feature in LEAKY_FEATURES
    )
    assert high_auc_count >= 4, (
        "synthetic cohort failed to reproduce the CSU leakage pattern; "
        "expected ≥4 features with AUC≥0.85 in unmasked baseline"
    )


def test_masked_converter_drives_all_leaky_features_below_threshold(
    synthetic_sheets: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    """§8.3 acceptance: every leaky feature has single-feature AUC < 0.85 under masking."""
    journeys = _run_converter(synthetic_sheets, lookback_days=180, tmp_path=tmp_path)
    failing: list[tuple[str, float]] = []
    for feature in LEAKY_FEATURES:
        auc = _single_feature_auc(journeys, feature)
        if auc >= 0.85:
            failing.append((feature, auc))
    assert not failing, (
        f"masked converter still produced AUC ≥ 0.85 for: {failing}; masking is incomplete"
    )


def test_masked_journeys_carry_lookback_masked_status(
    synthetic_sheets: dict[str, pd.DataFrame], tmp_path: Path
) -> None:
    journeys = _run_converter(synthetic_sheets, lookback_days=180, tmp_path=tmp_path)
    assert journeys, "expected non-empty journeys list"
    statuses = {j["journey_status"] for j in journeys}
    assert statuses == {"lookback_masked"}, (
        f"masked converter should set journey_status to 'lookback_masked' "
        f"for all journeys, got: {statuses}"
    )
