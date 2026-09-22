"""Lane C (remibrutinib pre-wiring, spec 2026-09-22 §3C.1): the CSU biologic
matcher in ``scripts/convert_optum_rwd.py`` recognises remibrutinib (brand
RHAPSIDO, generic remibrutinib, the synthetic-placeholder product NDC
00078-1100 from ``src/ml/synthetic/clinical_codes.py``), and the real-drop
converter maps the journey / treatment-event ``brand`` to the platform's
``brand_type`` label instead of collapsing every CSU row to ``competitor``.

The vocabulary lives in ``src/data/csu_biologics.py`` (one source for both
converters); this file pins the converter's behaviour on top of it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.convert_optum_rwd import (
    CSU_BIOLOGIC_BRANDS,
    CSU_BIOLOGIC_GENERICS,
    CSU_BIOLOGIC_HCPCS,
    CSU_BIOLOGIC_NDC_PREFIXES,
    OptumDataConverter,
)
from src.data.csu_biologics import (
    CSU_BIOLOGICS,
    REMIBRUTINIB_ARM_LABEL,
    canonical_csu_arm,
    classify_csu_biologic,
    csu_biologic_by_key,
    journey_brand_for,
)

pytestmark = pytest.mark.unit


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s)


def _converter(cohorts=("discontinuation",)) -> OptumDataConverter:
    return OptumDataConverter(parquet_dir=Path("."), output_dir=Path("."), cohorts=cohorts)


def _row(brand=None, generic=None, code=None) -> pd.Series:
    return pd.Series({"Brand_Name": brand, "Generic_Name": generic, "code": code})


# ---------------------------------------------------------------------------
# vocabulary: one source, both converters read it
# ---------------------------------------------------------------------------


class TestVocabulary:
    def test_three_csu_biologics_are_declared(self) -> None:
        assert [b.key for b in CSU_BIOLOGICS] == ["xolair", "dupixent", "remibrutinib"]

    def test_remibrutinib_entry_matches_the_spec(self) -> None:
        remi = csu_biologic_by_key("remibrutinib")
        assert remi.arm_label == REMIBRUTINIB_ARM_LABEL == "RHAPSIDO"
        assert "RHAPSIDO" in remi.brand_names
        assert "remibrutinib" in remi.generic_names
        # The synthetic-placeholder product code (clinical_codes.BRAND_NDC), both
        # the 11-digit dashless form the drop uses and the dashed 5-4 form.
        assert remi.ndc_prefixes == ("000781100", "00078-1100")
        assert remi.hcpcs == frozenset()  # oral BTK inhibitor: no J-code
        assert remi.journey_brand == "Remibrutinib"

    def test_the_novartis_labeler_alone_is_never_a_prefix(self) -> None:
        """Kisqali (00078-0903) and Fabhalta (00078-1175) share the labeler:
        a bare labeler prefix would pull every Novartis fill into the CSU mask."""
        assert "00078" not in CSU_BIOLOGIC_NDC_PREFIXES
        assert "0078" not in CSU_BIOLOGIC_NDC_PREFIXES

    def test_converter_constants_are_derived_from_the_vocabulary(self) -> None:
        assert set(CSU_BIOLOGIC_BRANDS) == {"XOLAIR", "DUPIXENT", "RHAPSIDO"}
        assert set(CSU_BIOLOGIC_GENERICS) == {"omalizumab", "dupilumab", "remibrutinib"}
        assert CSU_BIOLOGIC_HCPCS == {"J2357", "J0517"}
        assert set(CSU_BIOLOGIC_NDC_PREFIXES) == {
            "50242",
            "00024",
            "0024",
            "000781100",
            "00078-1100",
        }

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("XOLAIR", "XOLAIR"),
            ("xolair", "XOLAIR"),
            ("OMALIZUMAB", "XOLAIR"),
            ("DUPIXENT", "DUPIXENT"),
            ("DUPILUMAB", "DUPIXENT"),
            ("RHAPSIDO", "RHAPSIDO"),
            ("Rhapsido", "RHAPSIDO"),
            ("REMIBRUTINIB", "RHAPSIDO"),
            ("remibrutinib", "RHAPSIDO"),
            ("no_treatment", None),
            ("", None),
            (None, None),
            (float("nan"), None),
            ("KISQALI", None),
        ],
    )
    def test_canonical_arm_label_from_brand_or_molecule(self, label, expected) -> None:
        assert canonical_csu_arm(label) == expected

    def test_journey_brand_maps_remibrutinib_and_collapses_competitors(self) -> None:
        assert journey_brand_for("remibrutinib") == "Remibrutinib"
        assert journey_brand_for("xolair") == "competitor"
        assert journey_brand_for("dupixent") == "competitor"
        assert journey_brand_for(None) == "competitor"


# ---------------------------------------------------------------------------
# classify_csu_biologic / _classify_biologic_brand
# ---------------------------------------------------------------------------


class TestClassifyRemibrutinib:
    def test_brand_name(self) -> None:
        assert classify_csu_biologic(brand_name="Rhapsido") == "remibrutinib"
        assert OptumDataConverter._classify_biologic_brand(_row(brand="RHAPSIDO")) == (
            "remibrutinib"
        )

    def test_generic_name(self) -> None:
        assert classify_csu_biologic(generic_name="remibrutinib") == "remibrutinib"
        assert OptumDataConverter._classify_biologic_brand(_row(generic="Remibrutinib")) == (
            "remibrutinib"
        )

    def test_placeholder_ndc_dashless_and_dashed(self) -> None:
        assert OptumDataConverter._classify_biologic_brand(_row(code="00078110030")) == (
            "remibrutinib"
        )
        assert OptumDataConverter._classify_biologic_brand(_row(code="00078-1100-30")) == (
            "remibrutinib"
        )

    def test_other_novartis_products_on_the_same_labeler_are_not_matched(self) -> None:
        assert OptumDataConverter._classify_biologic_brand(_row(code="00078090351")) is None
        assert OptumDataConverter._classify_biologic_brand(_row(code="00078117566")) is None

    def test_brand_name_precedence_is_unchanged(self) -> None:
        # Brand_Name wins over Generic_Name wins over code (the pre-existing order).
        assert (
            OptumDataConverter._classify_biologic_brand(
                _row(brand="XOLAIR", generic="remibrutinib", code="00078110030")
            )
            == "xolair"
        )
        assert (
            OptumDataConverter._classify_biologic_brand(
                _row(brand="aspirin", generic="dupilumab", code="00078110030")
            )
            == "dupixent"
        )


# ---------------------------------------------------------------------------
# _csu_biologic_mask: a remibrutinib fill IS a CSU biologic fill
# ---------------------------------------------------------------------------


class TestMaskRemibrutinib:
    def _med(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "medication_date": [_ts("2026-01-05"), _ts("2026-02-05"), _ts("2026-03-05")],
                "code": ["00078110030", "12345678901", "50242021501"],
                "Brand_Name": ["RHAPSIDO", "ZYRTEC", "XOLAIR"],
                "Generic_Name": ["remibrutinib", "cetirizine", "omalizumab"],
            }
        )

    def test_mask_fires_on_each_remibrutinib_signal_alone(self) -> None:
        conv = _converter()
        by_brand = pd.DataFrame(
            {"code": ["99999999999"], "Brand_Name": ["Rhapsido"], "Generic_Name": ["?"]}
        )
        by_generic = pd.DataFrame(
            {"code": ["99999999999"], "Brand_Name": ["?"], "Generic_Name": ["REMIBRUTINIB"]}
        )
        by_ndc = pd.DataFrame({"code": ["00078110030"], "Brand_Name": ["?"], "Generic_Name": ["?"]})
        assert conv._csu_biologic_mask(by_brand).tolist() == [True]
        assert conv._csu_biologic_mask(by_generic).tolist() == [True]
        assert conv._csu_biologic_mask(by_ndc).tolist() == [True]

    def test_mask_keeps_xolair_and_rejects_antihistamines(self) -> None:
        conv = _converter()
        assert conv._csu_biologic_mask(self._med()).tolist() == [True, False, True]

    def test_first_biologic_fill_sees_the_remibrutinib_row(self) -> None:
        conv = _converter()
        conv._med_by_pat = {7: self._med()}
        assert conv._first_biologic_fill(7) == _ts("2026-01-05")
        assert conv._index_biologic_key(7) == "remibrutinib"

    def test_index_biologic_key_is_none_without_a_biologic_fill(self) -> None:
        conv = _converter()
        conv._med_by_pat = {
            7: pd.DataFrame(
                {
                    "medication_date": [_ts("2026-01-05")],
                    "code": ["12345678901"],
                    "Brand_Name": ["ZYRTEC"],
                    "Generic_Name": ["cetirizine"],
                }
            )
        }
        assert conv._index_biologic_key(7) is None
        assert conv._index_biologic_key(8) is None  # no medication rows at all


# ---------------------------------------------------------------------------
# journey + treatment-event brand mapping (through the real cohort build)
# ---------------------------------------------------------------------------


def _demo_frame(patid: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "patid": patid,
                "age": 35,
                "continuous_enrollment": 1,
                "eligeff": _ts("2019-01-01"),
                "eligend": _ts("2027-12-31"),
                "diagcode_raw": "L509",
                "diagcode": "L509",
                "gdr_cd": "F",
                "zipcode_5": "10001",
            }
        ]
    )


def _med_for(brand: str, generic: str, code: str, dates: list[str], patid: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "patid": [patid] * len(dates),
            "medication_date": [_ts(d) for d in dates],
            "code": [code] * len(dates),
            "Brand_Name": [brand] * len(dates),
            "Generic_Name": [generic] * len(dates),
            "days_sup": [30] * len(dates),
            "strength": ["25 MG"] * len(dates),
            "npi": ["1234567893"] * len(dates),
        }
    )


_SIX_MONTHLY = ["2022-07-01", "2022-08-01", "2022-09-01", "2022-10-01", "2022-11-01", "2022-12-01"]


def _cohort_converter(patid: int, med: pd.DataFrame, cohort: str) -> OptumDataConverter:
    conv = _converter((cohort,))
    conv.now_iso = "2026-01-01T00:00:00"
    conv.source_timestamp_iso = "2026-01-01T00:00:00"
    conv.ingestion_timestamp_iso = "2026-01-01T00:00:00"
    conv.data_lag_hours = 0
    conv.demo = _demo_frame(patid)
    conv._med_by_pat = {patid: med}
    conv._proc_by_pat = {}
    conv._lab_by_pat = {}
    conv._inpatient_by_pat = {
        patid: pd.DataFrame(
            {
                "diag1": ["L509"],
                "diag2": [None],
                "diag3": [None],
                "diag4": [None],
                "diag5": [None],
                "admit_date": [_ts("2022-05-15")],
            }
        )
    }
    return conv


class TestBrandMapping:
    def test_remibrutinib_initiator_journey_carries_the_platform_brand(self) -> None:
        conv = _cohort_converter(
            1, _med_for("RHAPSIDO", "remibrutinib", "00078110030", _SIX_MONTHLY, 1), "persistence"
        )
        journeys, _events, _hcps, _split = conv._build_cohort("persistence")
        assert [j["patient_id"] for j in journeys] == ["PAT_000000000001"]
        assert journeys[0]["brand"] == "Remibrutinib"
        assert journeys[0]["journey_start_date"] == "2022-07-01"  # index = first Rhapsido fill

    def test_xolair_initiator_journey_stays_competitor(self) -> None:
        conv = _cohort_converter(
            2, _med_for("XOLAIR", "omalizumab", "50242021501", _SIX_MONTHLY, 2), "persistence"
        )
        journeys, _events, _hcps, _split = conv._build_cohort("persistence")
        assert [j["patient_id"] for j in journeys] == ["PAT_000000000002"]
        assert journeys[0]["brand"] == "competitor"

    def test_journey_brand_helper_without_a_biologic_fill_is_competitor(self) -> None:
        conv = _converter(("initiation",))
        conv._med_by_pat = {}
        assert conv._journey_brand(3) == "competitor"

    def test_treatment_events_carry_each_fill_rows_own_brand(self) -> None:
        """Discontinuation cohort: the labeled first fill AND the trailing fills
        map per row — a Xolair→Rhapsido switch labels the Rhapsido row
        Remibrutinib and the Xolair rows competitor."""
        med = pd.concat(
            [
                _med_for("XOLAIR", "omalizumab", "50242021501", ["2022-07-01", "2022-08-01"], 4),
                _med_for("RHAPSIDO", "remibrutinib", "00078110030", ["2022-09-15"], 4),
            ],
            ignore_index=True,
        )
        conv = _cohort_converter(4, med, "discontinuation")
        journey = {
            "_patid": 4,
            "patient_id": "PAT_000000000004",
            "patient_journey_id": "PJ_000000000004",
            "index_date": _ts("2022-07-01"),
            "lookback_start_date": _ts("2022-01-02"),
        }
        events = conv._build_treatment_events(
            {4}, [journey], cohort="discontinuation", init_date_by_patid={4: _ts("2022-07-01")}
        )
        post = [
            e
            for e in events
            if e["event_type"] == "prescription"
            and e["event_date"] is not None
            and _ts(e["event_date"]) >= _ts("2022-07-01")
        ]
        by_date = {e["event_date"]: e["brand"] for e in post}
        assert by_date == {
            "2022-07-01": "competitor",
            "2022-08-01": "competitor",
            "2022-09-15": "Remibrutinib",
        }
