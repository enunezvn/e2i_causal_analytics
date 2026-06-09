"""Shard 04 Task 5 — patient_journeys must carry the brand-specific eligibility
columns cohort_constructor reads (configs.py required_fields), populated to pass
each brand's inclusion gate so the cohort is non-degenerate."""
from src.ml.synthetic.config import Brand
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
    assert df["disease_stage"].isin(
        ["advanced", "metastatic", "locally_advanced", "stage_iv"]
    ).all()
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
    for c in ELIG + ["primary_diagnosis_code"]:
        assert c in cols, f"{c} not registered -> loader will drop it"
