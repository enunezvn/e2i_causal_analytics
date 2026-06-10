"""P1.3 — CLI generate_synthetic_claims.py + 6-file writer (Shard 10).

Asserts the converter's exact 6 filenames are written and that the latent
helper columns + phantom columns the converter never reads are dropped.
"""

import pandas as pd

from scripts.generate_synthetic_claims import generate_to

SIX_FILES = ("demographics", "medication", "procedure", "lab", "inpatientdata", "provider")


def test_cli_writes_exactly_six_named_parquets(tmp_path):
    generate_to(out_dir=tmp_path, n_patients=100, seed=3)
    for name in SIX_FILES:
        assert (tmp_path / f"{name}.parquet").exists(), name
    # No extra parquet files (the converter raises on a missing one but the
    # contract is "exactly these six").
    written = {p.stem for p in tmp_path.glob("*.parquet")}
    assert written == set(SIX_FILES)


def test_demographics_drops_latent_and_phantom_columns(tmp_path):
    generate_to(out_dir=tmp_path, n_patients=80, seed=5)
    demo = pd.read_parquet(tmp_path / "demographics.parquet")
    # Latent helper cols never reach output.
    latent = {"severity", "response_propensity", "adherence_propensity", "claim_index"}
    assert latent & set(demo.columns) == set()
    # Phantom cols the converter never reads.
    phantom = {"indexdt", "yrdob", "family_id", "ahfsclss"}
    assert phantom & set(demo.columns) == set()
    # Required demographics read sites present.
    required = {
        "patid",
        "eligeff",
        "eligend",
        "diagcode",
        "age",
        "gdr_cd",
        "zipcode_5",
        "bus",
        "continuous_enrollment",
    }
    assert required <= set(demo.columns)


def test_medication_carries_no_phantom_columns(tmp_path):
    generate_to(out_dir=tmp_path, n_patients=60, seed=7)
    med = pd.read_parquet(tmp_path / "medication.parquet")
    phantom = {"quantity", "clmid", "ahfsclss", "indexdt"}
    assert phantom & set(med.columns) == set()
    assert len(med) > 0


def test_provider_is_npi_taxonomy_only(tmp_path):
    generate_to(out_dir=tmp_path, n_patients=50, seed=2)
    prov = pd.read_parquet(tmp_path / "provider.parquet")
    assert set(prov.columns) == {"npi", "taxonomy1"}


def test_generation_is_deterministic_for_seed(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    generate_to(out_dir=a, n_patients=40, seed=99)
    generate_to(out_dir=b, n_patients=40, seed=99)
    for name in SIX_FILES:
        da = pd.read_parquet(a / f"{name}.parquet")
        db = pd.read_parquet(b / f"{name}.parquet")
        pd.testing.assert_frame_equal(da, db)
