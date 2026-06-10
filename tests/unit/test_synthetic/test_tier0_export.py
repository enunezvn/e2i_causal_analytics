"""Red-first tests for scripts/export_synthetic_tier0.py.

The exporter turns the synthetic parquet snapshot dir into per-cohort
tier0-contract inputs (``tier0/<cohort>/e2i_ml_v3_patient_journeys.parquet`` +
``e2i_ml_v3_split_registry.json``) that ``scripts/run_tier0_test.py``'s
``load_rwd_data`` consumes. Tier0 output (trained artifacts + val_AUC gate)
feeds the tier1-5 agent stages.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.export_synthetic_tier0 import COHORT_TARGETS, export_tier0

N_PJ = 400
N_HCP = 200


@pytest.fixture()
def snapshot_dir(tmp_path: Path) -> Path:
    """Minimal synthetic snapshot dir with the real column surface."""
    rng = np.random.default_rng(0)
    brands = rng.choice(["Remibrutinib", "Kisqali", "Fabhalta"], size=N_PJ)
    initiated = rng.integers(0, 2, size=N_PJ)
    disc = np.where(initiated == 1, rng.integers(0, 2, size=N_PJ), np.nan)
    pj = pd.DataFrame(
        {
            "patient_journey_id": [f"pj_{i}" for i in range(N_PJ)],
            "patient_id": [f"p_{i}" for i in range(N_PJ)],
            "hcp_id": [f"h_{i % N_HCP}" for i in range(N_PJ)],
            "brand": brands,
            "journey_start_date": pd.Timestamp("2026-01-01"),
            "data_split": "holdout",  # scrambled by anchor remap: exporter must reassign
            "disease_severity": rng.normal(5, 2, N_PJ).clip(0, 10),
            "academic_hcp": rng.integers(0, 2, size=N_PJ),
            "engagement_score": rng.random(N_PJ),
            "treatment_initiated": initiated,
            "days_to_treatment": np.where(initiated == 1, 30, np.nan),
            "geographic_region": "northeast",
            "insurance_type": "commercial",
            "age_at_diagnosis": rng.integers(20, 80, size=N_PJ),
            # the generator populates indication covariates for EVERY row
            "urticaria_severity_uas7": rng.normal(25, 8, N_PJ),
            "hr_status": rng.choice(["positive", "negative"], size=N_PJ),
            "ldh_ratio": rng.normal(1.5, 0.4, N_PJ),
            "treatment_arm": rng.integers(0, 2, size=N_PJ),
            "propensity_score": rng.random(N_PJ),
            "segment_assignment": "medium_severity",
            "treatment_effect_estimate": 0.17,
            "discontinued_180d": disc,
            "persistent_180d": 1 - disc,
            "is_synthetic": True,
        }
    )
    hcp = pd.DataFrame(
        {
            "hcp_id": [f"h_{i}" for i in range(N_HCP)],
            "specialty": "dermatology",
            "practice_type": "academic",
            "geographic_region": "northeast",
            "years_experience": rng.integers(1, 40, size=N_HCP),
            "academic_hcp": rng.integers(0, 2, size=N_HCP),
            "total_patient_volume": rng.integers(50, 500, size=N_HCP),
            "brand": rng.choice(["Remibrutinib", "Kisqali", "Fabhalta"], size=N_HCP),
            "peer_influence_score": rng.random(N_HCP),
            "influence_network_size": rng.integers(1, 100, size=N_HCP),
            "adoption_category": rng.choice(["ADOPTER", "NON_ADOPTER"], size=N_HCP),
            "is_synthetic": True,
        }
    )
    pj.to_parquet(tmp_path / "patient_journeys.parquet", index=False)
    hcp.to_parquet(tmp_path / "hcp_profiles.parquet", index=False)
    return tmp_path


LEAK_COLS = {"propensity_score", "treatment_effect_estimate", "days_to_treatment"}


def test_exports_all_four_cohort_contract_dirs(snapshot_dir):
    written = export_tier0(snapshot_dir, brand="Remibrutinib", seed=42)
    assert set(written) == {"initiation", "discontinuation", "persistence", "hcp_adoption"}
    for cohort in written:
        d = snapshot_dir / "tier0" / cohort
        frame_path = d / "e2i_ml_v3_patient_journeys.parquet"
        registry_path = d / "e2i_ml_v3_split_registry.json"
        assert frame_path.exists(), f"{cohort}: missing tier0 frame"
        assert registry_path.exists(), f"{cohort}: missing split registry"
        df = pd.read_parquet(frame_path)
        target = COHORT_TARGETS[cohort]
        # tier0 pre-flight: target present with >=2 classes
        assert target in df.columns, f"{cohort}: target {target} missing"
        assert df[target].nunique(dropna=True) >= 2, f"{cohort}: single-class target"
        # contract: usable splits with a real train share, provenance kept
        assert "data_split" in df.columns
        share = df["data_split"].value_counts(normalize=True)
        assert share.get("train", 0) > 0.5, f"{cohort}: scrambled splits propagated"
        assert df["is_synthetic"].all()
        registry = json.loads(registry_path.read_text())
        assert registry[0]["split_strategy"] == "stratified_random"


def test_patient_cohorts_exclude_leak_and_cross_outcome_columns(snapshot_dir):
    export_tier0(snapshot_dir, brand="Remibrutinib", seed=42)
    for cohort, other_outcomes in [
        ("initiation", {"discontinued_180d", "persistent_180d"}),
        ("discontinuation", {"persistent_180d", "treatment_initiated"}),
        ("persistence", {"discontinued_180d", "treatment_initiated"}),
    ]:
        df = pd.read_parquet(snapshot_dir / "tier0" / cohort / "e2i_ml_v3_patient_journeys.parquet")
        leaked = (LEAK_COLS | other_outcomes) & set(df.columns)
        assert not leaked, f"{cohort}: leak columns present: {leaked}"
        # off-indication covariate panels pruned, own panel kept
        for off in ("hr_status", "ldh_ratio"):
            assert off not in df.columns, f"{cohort}: off-indication column {off} kept"
        assert "urticaria_severity_uas7" in df.columns
        assert (df["brand"] == "Remibrutinib").all()


def test_disc_persistence_are_initiators_only(snapshot_dir):
    export_tier0(snapshot_dir, brand="Remibrutinib", seed=42)
    pj = pd.read_parquet(snapshot_dir / "patient_journeys.parquet")
    n_initiators = int(((pj["brand"] == "Remibrutinib") & (pj["treatment_initiated"] == 1)).sum())
    for cohort in ("discontinuation", "persistence"):
        df = pd.read_parquet(snapshot_dir / "tier0" / cohort / "e2i_ml_v3_patient_journeys.parquet")
        assert len(df) == n_initiators, f"{cohort}: not initiators-only"
        assert df[COHORT_TARGETS[cohort]].notna().all()


def test_hcp_adoption_is_hcp_grain_with_binary_target(snapshot_dir):
    export_tier0(snapshot_dir, brand="Remibrutinib", seed=42)
    df = pd.read_parquet(
        snapshot_dir / "tier0" / "hcp_adoption" / "e2i_ml_v3_patient_journeys.parquet"
    )
    assert df["hcp_id"].is_unique
    assert set(df["adopted_target_brand"].unique()) <= {0, 1}
    assert (df["brand"] == "Remibrutinib").all()
