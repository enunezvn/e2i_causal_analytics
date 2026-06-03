"""Contract tests for the leakage-safe HCP-feature cohort enrichment.

The load-bearing invariant is the LEAKAGE filter: a prescriber the patient only
saw AFTER their index_date must NOT contribute any HCP feature, or the
enrichment would inject post-index information into a Tier0 cohort whose whole
point is leakage-safe shaping.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

_MOD_PATH = Path(__file__).resolve().parents[3] / "scripts" / "enrich_cohort_with_hcp_features.py"
_spec = importlib.util.spec_from_file_location("enrich_hcp", _MOD_PATH)
enrich = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(enrich)


def _pj() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patient_id": ["PAT_100", "PAT_200", "PAT_300", "PAT_400"],
            "index_date": ["2020-06-01"] * 4,
            "target": [1, 0, 1, 0],
        }
    )


def _medication() -> pd.DataFrame:
    # P100: pre-index npiA. P200: ONLY post-index npiB. P300: pre-index npiA+npiC.
    # P400: no meds.
    return pd.DataFrame(
        {
            "patid": [100, 200, 300, 300],
            "npi": ["npiA", "npiB", "npiA", "npiC"],
            "medication_date": ["2020-01-01", "2020-12-01", "2020-02-01", "2020-03-01"],
        }
    )


def _targeting() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "npi": ["npiA", "npiB", "npiC"],
            "decile": [5, 10, 9],
            "priority_tier": [2, 1, 1],
            "is_specialist_hcp": [1, 1, 0],
            "priority_label": ["P2", "P1", "P1"],
        }
    )


def _kol() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "npi": ["npiA", "npiB", "npiC"],
            "kol_score": [7.0, 10.0, 9.0],
            "kol_score_100pt": [70.0, 100.0, 90.0],
            "influence_network_size": [50, 999, 80],
            "kol_category": ["Moderate KOL Proxy", "High KOL Proxy", "High KOL Proxy"],
        }
    )


def _run() -> pd.DataFrame:
    out = enrich.build_patient_hcp_features(_pj(), _medication(), _targeting(), _kol())
    return out.set_index("patient_id")


def test_rows_and_order_preserved() -> None:
    out = enrich.build_patient_hcp_features(_pj(), _medication(), _targeting(), _kol())
    assert list(out["patient_id"]) == ["PAT_100", "PAT_200", "PAT_300", "PAT_400"]
    assert len(out) == 4
    for c in enrich.HCP_FEATURE_COLUMNS:
        assert c in out.columns


def test_post_index_prescriber_does_not_leak() -> None:
    """P200's only prescriber (npiB) is POST-index → it must contribute NOTHING."""
    r = _run().loc["PAT_200"]
    assert r["treating_hcp_match_count"] == 0
    assert pd.isna(r["treating_hcp_kol_score_max"])
    assert pd.isna(r["treating_hcp_targeting_decile_max"])
    # npiB's signature values (decile 10 / kol 10 / network 999) must appear NOWHERE.
    out = enrich.build_patient_hcp_features(_pj(), _medication(), _targeting(), _kol())
    assert (out["treating_hcp_kol_score_max"] == 10.0).sum() == 0
    assert (out["treating_hcp_influence_network_size_max"] == 999).sum() == 0


def test_single_preindex_provider() -> None:
    r = _run().loc["PAT_100"]
    assert r["treating_hcp_match_count"] == 1
    assert r["treating_hcp_targeting_decile_max"] == 5
    assert r["treating_hcp_kol_score_max"] == 7.0
    assert r["treating_hcp_is_specialist_any"] == 1


def test_multi_provider_aggregation_uses_strongest() -> None:
    """P300 saw npiA(decile5,kol7) + npiC(decile9,kol9) pre-index → take the max."""
    r = _run().loc["PAT_300"]
    assert r["treating_hcp_match_count"] == 2
    assert r["treating_hcp_targeting_decile_max"] == 9
    assert r["treating_hcp_priority_tier_best"] == 1  # min(2,1)
    assert r["treating_hcp_kol_score_max"] == 9.0
    # kol_category of the MAX-kol provider (npiC).
    assert r["treating_hcp_kol_category_top"] == "High KOL Proxy"


def test_no_medication_patient_is_zero_not_null_count() -> None:
    r = _run().loc["PAT_400"]
    assert r["treating_hcp_match_count"] == 0
    assert pd.isna(r["treating_hcp_kol_score_max"])
