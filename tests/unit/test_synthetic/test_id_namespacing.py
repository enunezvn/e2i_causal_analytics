"""Synthetic entity ids must be namespaceable so a validation dataset never
collides with (and UPSERT-clobbers) the existing dev baseline. Ids must also fit
the varchar(20) id columns."""

from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator, PatientGenerator


def test_generator_id_prefix_namespaces_and_fits_varchar20():
    hcp = HCPGenerator(GeneratorConfig(seed=1, n_records=5, id_prefix="scv")).generate()
    assert hcp["hcp_id"].astype(str).str.startswith("scv").all()
    assert (hcp["hcp_id"].astype(str).str.len() <= 20).all()
    # FK linkage must remain consistent: patient.hcp_id references the prefixed hcp id.
    pat = PatientGenerator(
        GeneratorConfig(seed=1, n_records=5, id_prefix="scv"), hcp_df=hcp
    ).generate()
    for col in ("patient_journey_id", "patient_id", "hcp_id"):
        vals = pat[col].astype(str)
        assert vals.str.startswith("scv").all(), f"{col} not namespaced"
        assert (vals.str.len() <= 20).all(), f"{col} exceeds varchar(20)"
    assert set(pat["hcp_id"]).issubset(set(hcp["hcp_id"])), "FK linkage broken by prefix"


def test_empty_prefix_reproduces_legacy_ids():
    hcp = HCPGenerator(GeneratorConfig(seed=1, n_records=3, id_prefix="")).generate()
    assert hcp["hcp_id"].iloc[0].startswith("hcp_")
    assert not hcp["hcp_id"].iloc[0].startswith("scv")
