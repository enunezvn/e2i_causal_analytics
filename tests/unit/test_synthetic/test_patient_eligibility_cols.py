"""Shard 04 Task 5 — patient_journeys must carry the brand-specific eligibility
columns cohort_constructor reads (configs.py required_fields), populated to pass
each brand's inclusion gate so the cohort is non-degenerate."""

from src.ml.synthetic.config import Brand, DGPType
from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.patient_generator import PatientGenerator

ELIG = [
    "urticaria_severity_uas7",
    "prior_antihistamine_therapy",
    "hr_status",
    "her2_status",
    "disease_stage",
    "ecog_performance_status",
    "ldh_ratio",
    "complement_inhibitor_status",
    "proteinuria_g_day",
    "egfr",
]


def test_kisqali_patients_get_breast_eligibility_fields():
    cfg = GeneratorConfig(seed=5, n_records=200, brand=Brand.KISQALI)
    df = PatientGenerator(cfg).generate()
    for c in ELIG:
        assert c in df.columns, f"{c} missing"
    assert (df["hr_status"] == "positive").all()
    assert (df["her2_status"] == "negative").all()
    assert (
        df["disease_stage"].isin(["advanced", "metastatic", "locally_advanced", "stage_iv"]).all()
    )
    assert df["ecog_performance_status"].isin([0, 1]).all()
    assert df["primary_diagnosis_code"].str.startswith("C50").all()


def test_remi_meets_uas7_and_antihistamine_gate():
    cfg = GeneratorConfig(seed=5, n_records=200, brand=Brand.REMIBRUTINIB)
    df = PatientGenerator(cfg).generate()
    assert (df["urticaria_severity_uas7"] >= 16).mean() >= 0.8  # UAS7>=16 inclusion
    assert df["prior_antihistamine_therapy"].all()
    assert df["primary_diagnosis_code"].str.startswith("L50").all()


def test_fabhalta_meets_pnh_gate():
    cfg = GeneratorConfig(seed=5, n_records=200, brand=Brand.FABHALTA)
    df = PatientGenerator(cfg).generate()
    assert (df["ldh_ratio"] >= 1.5).mean() >= 0.8
    assert df["complement_inhibitor_status"].isin(["current", "prior"]).all()
    assert (df["primary_diagnosis_code"] == "D59.5").all()


def test_loader_registers_eligibility_columns():
    from src.ml.synthetic.loaders.batch_loader import TABLE_COLUMNS

    cols = TABLE_COLUMNS["patient_journeys"]
    for c in ELIG + ["primary_diagnosis_code", "biologic_experienced", "ige_level"]:
        assert c in cols, f"{c} not registered -> loader will drop it"


# --- Phase 2 brand-gating (2026-07-13): indication-specific columns are populated
# ONLY for their own brand; off-brand values are NULL rather than fabricated. -------

# Every gated eligibility column mapped to the ONE brand it is clinically valid for.
_OWNER = {
    "urticaria_severity_uas7": "Remibrutinib",
    "prior_antihistamine_therapy": "Remibrutinib",
    "biologic_experienced": "Remibrutinib",
    "ige_level": "Remibrutinib",
    "hr_status": "Kisqali",
    "her2_status": "Kisqali",
    "disease_stage": "Kisqali",
    "ecog_performance_status": "Kisqali",
    "ldh_ratio": "Fabhalta",
    "complement_inhibitor_status": "Fabhalta",
    "proteinuria_g_day": "Fabhalta",
    "egfr": "Fabhalta",
}


def test_kisqali_config_nulls_offbrand_eligibility():
    df = PatientGenerator(GeneratorConfig(seed=5, n_records=200, brand=Brand.KISQALI)).generate()
    for c, owner in _OWNER.items():
        if owner == "Kisqali":
            assert df[c].notna().all(), f"{c} must be populated for its own brand"
        else:
            assert df[c].isna().all(), f"{c} must be NULL for off-brand Kisqali rows"


def test_remibrutinib_gets_real_biologic_ige_axis():
    df = PatientGenerator(
        GeneratorConfig(seed=5, n_records=200, brand=Brand.REMIBRUTINIB)
    ).generate()
    # The axis the chatbot used to fabricate is now real — for CSU/Remibrutinib.
    assert df["biologic_experienced"].dropna().isin([0, 1]).all()
    assert df["biologic_experienced"].notna().all()
    assert (df["ige_level"].dropna() > 0).all()
    assert df["ige_level"].notna().all()
    # Off-brand (oncology/PNH) eligibility fields are NULL for CSU rows.
    for c in ("hr_status", "ecog_performance_status", "egfr", "ldh_ratio"):
        assert df[c].isna().all(), f"{c} must be NULL for Remibrutinib rows"


def test_mixed_brand_gating_is_brand_exclusive():
    """brand=None → each gated column is non-null IFF the row's brand owns it."""
    df = PatientGenerator(
        GeneratorConfig(seed=7, n_records=900, brand=None, dgp_type=DGPType.HETEROGENEOUS)
    ).generate()
    for c, owner in _OWNER.items():
        own = df["brand"] == owner
        assert df.loc[own, c].notna().all(), f"{c} must be populated for {owner} rows"
        assert df.loc[~own, c].isna().all(), f"{c} must be NULL for non-{owner} rows"
