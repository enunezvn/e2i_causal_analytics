"""patient_generator emits disc/persist columns with band-valid prevalence,
consuming the Shard-03 canonical treatment_arm + segment (no second arm source)."""

from __future__ import annotations

import numpy as np

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator


def _gen(n=6000, seed=3, brand=Brand.REMIBRUTINIB):
    cfg = GeneratorConfig(seed=seed, n_records=n, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
    return PatientGenerator(cfg).generate()


def test_new_driver_columns_present_and_varied():
    df = _gen()
    for col in ("comorbidity_burden", "prior_therapy_lines"):
        assert col in df.columns, f"{col} missing from generated frame"
        assert df[col].notna().all(), f"{col} has nulls"
        assert df[col].nunique() > 1, f"{col} has no per-patient variance"


def test_drivers_independent_of_treatment_arm():
    df = _gen()
    # Prognostic-only contract: |corr(driver, treatment_arm)| must be ~0.
    for col in ("comorbidity_burden", "prior_therapy_lines", "age_at_diagnosis"):
        corr = np.corrcoef(df[col].to_numpy(float), df["treatment_arm"].to_numpy(float))[0, 1]
        assert abs(corr) < 0.05, f"{col} must be independent of treatment_arm; corr={corr}"


def test_persistence_carries_driver_signal():
    df = _gen(n=12000)
    # Commercial insurance should persist more than medicaid (signal wired through).
    p = df["persistent_180d"]
    ins = df["insurance_type"]
    assert p[ins == "commercial"].mean() > p[ins == "medicaid"].mean()


def test_generator_emits_cohort_outcome_columns():
    cfg = GeneratorConfig(
        seed=42, n_records=3000, brand=Brand.KISQALI, dgp_type=DGPType.HETEROGENEOUS
    )
    df = PatientGenerator(cfg).generate()
    for col in ("discontinued_180d", "persistent_180d"):
        assert col in df.columns
        assert df[col].isin([0, 1]).all()
        assert 0.05 <= df[col].mean() <= 0.60
    # complement holds row-for-row
    assert (df["discontinued_180d"] + df["persistent_180d"] == 1).all()


def test_disc_persist_present_across_brands():
    for brand in (Brand.REMIBRUTINIB, Brand.FABHALTA):
        cfg = GeneratorConfig(seed=9, n_records=2000, brand=brand, dgp_type=DGPType.HETEROGENEOUS)
        df = PatientGenerator(cfg).generate()
        assert 0.05 <= df["discontinued_180d"].mean() <= 0.60


def test_priorc5_arm_marginalizes_over_fabhalta_segments_on_sparse_mixed_frame():
    """#1321 regression: on a SMALL MIXED-brand frame the Fabhalta cohort spans only a
    subset of segments, so the prior-C5-rebuilt copay/psp persistence RD maps carry
    only those keys. Marginalizing the arm ATE over the whole-frame segment array (the
    first cut) KeyErrors on a non-Fabhalta segment absent from the subset map — this is
    exactly the n=5 frame that broke test_id_namespacing. The ATE must be weighted over
    the FABHALTA rows' own segments, whose keys match the RD map exactly.

    seed=1/n=5: the Fabhalta cohort spans a STRICT subset of segments while the full
    frame carries others from Kisqali/Remibrutinib rows — so the old full-frame
    marginalization raises and the subset-marginalization passes."""
    cfg = GeneratorConfig(seed=1, n_records=5, dgp_type=DGPType.HETEROGENEOUS)
    df = PatientGenerator(cfg).generate()  # must not raise KeyError

    fab = df[df["brand"] == Brand.FABHALTA.value]
    assert len(fab) > 0 and fab["complement_inhibitor_status"].notna().all(), "no prior-C5 cohort"
    fab_segments = set(fab["segment_assignment"].astype(str))
    # The regression only bites when Fabhalta is SPARSE vs the whole frame.
    assert fab_segments < set(df["segment_assignment"].astype(str)), (
        "frame not sparse enough to guard"
    )

    arms = df.attrs["true_ate_by_arm"]
    assert "complement_inhibitor_status" in arms, "prior-C5 main-effect arm missing"
    for arm in ("copay_support", "psp_enrolled"):
        cate = arms[arm]["persistent_180d"]["cate_by_segment"]
        # Keys are the Fabhalta subset's segments — NEVER a whole-frame segment the
        # subset lacks (that is the KeyError this test guards).
        assert set(cate) <= fab_segments, f"{arm} RD map leaked a non-Fabhalta segment: {set(cate)}"
        ate = arms[arm]["persistent_180d"]["ate"]
        assert np.isfinite(ate), f"{arm} persistent ATE is not finite: {ate}"
