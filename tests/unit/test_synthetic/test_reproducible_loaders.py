"""Tests for the reproducible gold-standard loaders (chore/goldstd-reproducible-loaders).

Two ad-hoc-loaded artifacts are made reproducible by committed, idempotent scripts:
  * scripts/load_hcp_brand_adoption.py      -> hcp_brand_adoption (15k rows)
  * scripts/regenerate_cohort_outcomes.py   -> patient_journeys persist/disc labels

These tests exercise the PURE, deterministic generation + serialization logic (no DB,
no network). Properties asserted (red-first contract):
  - determinism: same seed -> identical frame
  - idempotency: the upsert/update is keyed on a natural key (re-run is a no-op given
    ON CONFLICT semantics) -- asserted via the natural-key uniqueness of the payload
    + record-builder determinism
  - is_synthetic == True on every generated row
  - complement property: persistent_180d == 1 - discontinued_180d
  - split ratios 60/20/10/10 (train/validation/test/holdout; #44 holdout enlargement)

Run:
    cd <worktree> && PYTHONPATH=$PWD python -m pytest \
        tests/unit/test_synthetic/test_reproducible_loaders.py -n0 -q
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from scripts.load_hcp_brand_adoption import (
    BRANDS,
    END_DATE,
    N_MONTHS,
    ON_CONFLICT,
    _records,
    build_adoption_frame,
    build_hcp_frame,
)
from scripts.regenerate_cohort_outcomes import (
    KEY,
    regenerate,
)

# ===========================================================================
# hcp_brand_adoption loader
# ===========================================================================


@pytest.fixture(scope="module")
def hcp_frame() -> pd.DataFrame:
    # n=300 keeps the test fast but exercises the full split machinery.
    return build_hcp_frame(seed=42, n_hcps=300)


@pytest.fixture(scope="module")
def adoption_seed() -> int:
    return 427


def _build(n_hcps: int, seed: int) -> pd.DataFrame:
    """Helper that builds an adoption frame from an n_hcps HCP cohort."""
    hcp = build_hcp_frame(seed=42, n_hcps=n_hcps)
    from src.ml.synthetic.generators.hcp_brand_adoption_generator import (
        generate_hcp_brand_adoption_frame,
    )

    frame = generate_hcp_brand_adoption_frame(
        hcp, seed=seed, end_date=END_DATE, brands=BRANDS, n_months=N_MONTHS
    )
    frame["is_synthetic"] = True
    return frame


def test_hcp_determinism():
    """Same seeds + same HCP cohort -> byte-identical frame."""
    a = _build(n_hcps=300, seed=427)
    b = _build(n_hcps=300, seed=427)
    assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


def test_hcp_build_adoption_frame_columns_and_shape():
    """build_adoption_frame yields exactly the DB load columns, 3 rows/HCP."""
    # Use the real 5000 cohort: build_adoption_frame has no n_hcps param (it is the
    # canonical 5000-HCP reconstruction), so assert the contract on the full frame.
    df = build_adoption_frame(adoption_seed=427)
    assert list(df.columns) == [
        "hcp_id",
        "brand",
        "consideration_date",
        "adopted",
        "adoption_category",
        "data_split",
        "is_synthetic",
    ]
    assert len(df) == df["hcp_id"].nunique() * len(BRANDS)


def test_hcp_is_synthetic_all_true(hcp_frame, adoption_seed):
    df = _build(n_hcps=300, seed=adoption_seed)
    assert df["is_synthetic"].all(), "every hcp_brand_adoption row must be is_synthetic=True"


def test_hcp_natural_key_unique_enables_idempotent_upsert():
    """(hcp_id, brand) is unique in the payload -> ON CONFLICT (hcp_id, brand) upsert
    is a no-op on re-run (no duplicate-key fan-out)."""
    df = _build(n_hcps=300, seed=427)
    conflict_cols = ON_CONFLICT.split(",")
    assert conflict_cols == ["hcp_id", "brand"]
    dups = df.duplicated(subset=conflict_cols)
    assert not dups.any(), f"natural key {conflict_cols} not unique: {int(dups.sum())} dups"


def test_hcp_records_builder_deterministic_and_json_safe():
    """_records() is deterministic and emits ISO date strings + python ints (idempotent
    payload -> identical upsert body on every run)."""
    df = _build(n_hcps=50, seed=427)
    r1 = _records(df)
    r2 = _records(df)
    assert r1 == r2, "record serialization must be deterministic"
    sample = r1[0]
    # consideration_date serialized to an ISO 'YYYY-MM-DD' string
    assert isinstance(sample["consideration_date"], str)
    date.fromisoformat(sample["consideration_date"])  # parses
    # adopted is a plain python int (not numpy) so the JSON body is stable
    assert isinstance(sample["adopted"], int)
    assert sample["is_synthetic"] is True


def test_hcp_split_ratios_60_20_10_10():
    """Stratified split lands on the designed 60/20/10/10 proportions (within rounding).

    #44 holdout enlargement (2026-07-21): test 15%→10%, holdout 5%→10%.
    """
    df = _build(n_hcps=1000, seed=427)
    # Per brand the split is computed independently; check one brand's distribution.
    sub = df[df["brand"] == "Remibrutinib"]
    n = len(sub)
    frac = sub["data_split"].value_counts(normalize=True)
    assert abs(frac.get("train", 0) - 0.60) <= 0.02, f"train {frac.get('train')}"
    assert abs(frac.get("validation", 0) - 0.20) <= 0.02, f"validation {frac.get('validation')}"
    assert abs(frac.get("test", 0) - 0.10) <= 0.02, f"test {frac.get('test')}"
    assert abs(frac.get("holdout", 0) - 0.10) <= 0.02, f"holdout {frac.get('holdout')}"
    # exact counts for n=1000: 600/200/100/100
    assert n == 1000


def test_hcp_consideration_date_span():
    """consideration_date spans N_MONTHS buckets ending at END_DATE (37 -> 2023-06..2026-06)."""
    df = _build(n_hcps=500, seed=427)
    months = pd.to_datetime(df["consideration_date"]).dt.to_period("M")
    assert months.nunique() <= N_MONTHS
    assert str(months.max()) == "2026-06"


def test_hcp_canonical_cohort_kisqali_oncology_first():
    """#1551 acceptance mirror on the CANONICAL 5000-HCP cohort (seed 42 HCPs +
    seed 427 adoption — exactly what the regen load ships): Kisqali adoption must
    be oncology-first, with rheumatology/dermatology clearly below (the served
    propensity ordering the champion LR fits follows these label means)."""
    hcp = build_hcp_frame(seed=42)
    frame = build_adoption_frame(adoption_seed=427)
    joined = frame.merge(hcp[["hcp_id", "specialty"]], on="hcp_id", how="left")
    means = joined[joined["brand"] == "Kisqali"].groupby("specialty")["adopted"].mean()
    assert means.idxmax() == "oncology", (
        f"Kisqali top specialty must be oncology; got\n{means.sort_values(ascending=False)}"
    )
    assert means["oncology"] > means["rheumatology"] + 0.10
    assert means["oncology"] > means["dermatology"] + 0.10


# ===========================================================================
# cohort outcomes regeneration
# ===========================================================================


def _patient_covariates(n: int = 3000, seed: int = 13) -> pd.DataFrame:
    """In-memory patient covariate fixture mirroring the live patient_journeys shape
    (the inputs regenerate() reads from the DB)."""
    rng = np.random.default_rng(seed)
    severity = np.clip(rng.normal(5.0, 2.0, n), 0, 10)
    segment = np.where(
        severity > 7, "high_severity", np.where(severity > 4, "medium_severity", "low_severity")
    )
    return pd.DataFrame(
        {
            KEY: [f"scvpt_{i:06d}" for i in range(n)],
            "brand": rng.choice(list(BRANDS), n),
            "treatment_initiated": (rng.random(n) < 0.35).astype(int),
            "treatment_arm": rng.integers(0, 2, n),
            "disease_severity": severity,
            "academic_hcp": (rng.random(n) < 0.3).astype(int),
            "geographic_region": rng.choice(["midwest", "northeast", "south", "west"], n),
            "segment_assignment": segment,
            # T9 drivers present on live rows (insurance + age). comorbidity_burden /
            # prior_therapy_lines are omitted here to exercise regenerate()'s backfill draw.
            "insurance_type": rng.choice(
                ["commercial", "medicare", "medicaid"], n, p=[0.6, 0.3, 0.1]
            ),
            "age_at_diagnosis": rng.integers(18, 85, n),
            # current "live" labels (unused by regenerate, present for shape parity)
            "persistent_180d": rng.integers(0, 2, n),
            "discontinued_180d": rng.integers(0, 2, n),
        }
    )


def test_cohort_determinism():
    """Same seed + same covariates -> identical regenerated labels."""
    cov = _patient_covariates()
    a = regenerate(cov, seed=74)
    b = regenerate(cov, seed=74)
    assert_frame_equal(
        a.sort_values(KEY).reset_index(drop=True),
        b.sort_values(KEY).reset_index(drop=True),
    )


def test_cohort_outcomes_invariant_to_driver_backfill_path():
    """codex FINDING-2 regression: re-deriving with comorbidity_burden/prior_therapy_lines
    ALREADY POPULATED (the rerun path, where _read_or_draw READS not DRAWS) must give
    byte-identical treatment_initiated + persist/disc as the NULL path that draws them.
    The reseed spawns independent per-component rng streams so the driver backfill draws
    never advance the disc/initiation outcome streams. Without that, the populated rerun
    would silently re-realize the outcomes off a shifted offset (non-idempotent)."""
    cov = _patient_covariates()  # comorbidity/prior absent → DRAWN this pass
    drawn = regenerate(cov, seed=74)
    # Second pass: the driver columns are now persisted on the live rows → READ.
    cov_pop = cov.merge(
        drawn[[KEY, "comorbidity_burden", "prior_therapy_lines"]], on=KEY, how="left"
    )
    read = regenerate(cov_pop, seed=74)
    a = drawn.sort_values(KEY).reset_index(drop=True)
    b = read.sort_values(KEY).reset_index(drop=True)
    for col in (
        "treatment_initiated",
        "persistent_180d",
        "discontinued_180d",
        "comorbidity_burden",
        "prior_therapy_lines",
    ):
        assert (a[col].to_numpy() == b[col].to_numpy()).all(), (
            f"{col} differs across the draw-vs-read driver backfill path (rng bifurcation)"
        )


def test_cohort_regenerates_treatment_initiated_in_band():
    """T11: treatment_initiated is re-derived from the enriched eqn (prevalence-banded
    ~0.35), and days_to_treatment stays consistent with the label (value iff initiated)."""
    cov = _patient_covariates()
    out = regenerate(cov, seed=74)
    assert "treatment_initiated" in out.columns
    prev = float(out["treatment_initiated"].mean())
    assert 0.25 <= prev <= 0.45, f"init prevalence {prev} out of band"
    assert set(out["treatment_initiated"].unique()) <= {0, 1}
    init = out["treatment_initiated"].to_numpy() == 1
    days = out["days_to_treatment"].to_numpy()
    assert np.all(~np.isnan(days[init])), "initiators must have a days_to_treatment value"
    assert np.all(np.isnan(days[~init])), "non-initiators must have NULL days_to_treatment"


def test_cohort_complement_property():
    """persistent_180d == 1 - discontinued_180d for every row (no violations)."""
    cov = _patient_covariates()
    out = regenerate(cov, seed=74)
    viol = int((out["persistent_180d"] + out["discontinued_180d"] != 1).sum())
    assert viol == 0, f"{viol} complement violations (persist != 1 - disc)"
    assert np.array_equal(
        out["persistent_180d"].to_numpy(), 1 - out["discontinued_180d"].to_numpy()
    )


def test_cohort_labels_are_binary():
    cov = _patient_covariates()
    out = regenerate(cov, seed=74)
    assert set(out["persistent_180d"].unique()) <= {0, 1}
    assert set(out["discontinued_180d"].unique()) <= {0, 1}


def test_cohort_idempotent_key_is_patient_id():
    """The UPDATE is keyed on patient_id; the regenerated frame has a unique patient_id
    so re-applying the UPDATE is a no-op (one row per key)."""
    cov = _patient_covariates()
    out = regenerate(cov, seed=74)
    assert KEY == "patient_id"
    assert not out[KEY].duplicated().any(), "patient_id must be unique for an idempotent UPDATE"


def test_cohort_prevalence_in_designed_band():
    """Regenerated discontinuation prevalence lands in the DGP's designed [0.05, 0.60]
    band (anti-degeneracy), matching the committed generate_discontinuation_outcomes."""
    cov = _patient_covariates(n=5000)
    out = regenerate(cov, seed=74)
    disc = out["discontinued_180d"].mean()
    assert 0.05 <= disc <= 0.60, f"disc prevalence {disc} outside designed band"


def test_cohort_covariates_not_redrawn():
    """regenerate must preserve the causal INPUTS (patient_id set + brand) exactly --
    disease_severity/academic_hcp/geographic_region/treatment_arm/segment are READ, never
    re-drawn. T11: treatment_initiated is no longer a preserved input -- it is RE-DERIVED
    as the enriched initiation OUTCOME (covered by test_cohort_regenerates_treatment_
    initiated_in_band), so it is NOT asserted equal to the input here."""
    cov = _patient_covariates()
    out = regenerate(cov, seed=74)
    # same patients, no fabricated/dropped ids
    assert set(out[KEY]) == set(cov[KEY])
    merged = cov[[KEY, "brand"]].merge(out, on=KEY, suffixes=("_in", "_out"))
    assert (merged["brand_in"] == merged["brand_out"]).all()
