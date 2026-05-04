"""Unit tests for the lookback-window masking added by `feat/phase4b-csu-converter-masking`.

These tests verify the §8.3 spec from `docs/lineage/csu_field_audit.md`:

- A `--lookback-days` CLI flag (default `None` = off, backwards-compat).
- When set, aggregate features behind `disease_severity`, `engagement_score`,
  `days_on_therapy`, `hcp_visits`, `medication_claim_count`,
  `procedure_claim_count`, `lab_claim_count`, and `eligibility_duration_days`
  must be computed only from events in `[index_date - lookback_days, index_date)`.
- Patient-journey records get `journey_status: "lookback_masked"` so downstream
  code can detect the mode.

Tests use synthetic in-memory fixtures (no Excel workbook required).
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

from convert_csu_rwd import CSUDataConverter  # noqa: E402,I001


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #


@pytest.fixture
def synthetic_index_date() -> pd.Timestamp:
    return pd.Timestamp("2024-01-01")


@pytest.fixture
def med_df_three_fills(synthetic_index_date: pd.Timestamp) -> pd.DataFrame:
    """A 3-fill panel: pre-lookback, in-lookback, post-index."""
    return pd.DataFrame(
        {
            "patid": [1, 1, 1],
            "medication_date": [
                synthetic_index_date - timedelta(days=300),
                synthetic_index_date - timedelta(days=90),
                synthetic_index_date + timedelta(days=10),
            ],
            "days_sup": [30, 45, 60],
            "npi": ["NPI001", "NPI002", "NPI002"],
            "brand_normalised": ["drugA", "drugB", "drugC"],
        }
    )


@pytest.fixture
def proc_df_three(synthetic_index_date: pd.Timestamp) -> pd.DataFrame:
    """3 procedure rows; 1 in lookback, 1 pre-lookback, 1 post-index."""
    return pd.DataFrame(
        {
            "patid": [1, 1, 1],
            "proc_date": [
                synthetic_index_date - timedelta(days=400),
                synthetic_index_date - timedelta(days=30),
                synthetic_index_date + timedelta(days=5),
            ],
            "proc_code": ["J2357", "J2357", "J2357"],
        }
    )


@pytest.fixture
def lab_df_three(synthetic_index_date: pd.Timestamp) -> pd.DataFrame:
    """3 lab rows; 1 in lookback, 1 pre-lookback, 1 post-index."""
    return pd.DataFrame(
        {
            "patid": [1, 1, 1],
            "fst_dt": [
                synthetic_index_date - timedelta(days=500),
                synthetic_index_date - timedelta(days=10),
                synthetic_index_date + timedelta(days=20),
            ],
            "abnl_cd": ["A", "A", "A"],
        }
    )


@pytest.fixture
def converter_off(tmp_path: Path) -> CSUDataConverter:
    """Constructed without lookback masking (backwards-compat default)."""
    return CSUDataConverter(
        excel_path=tmp_path / "fake.xlsx",
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def converter_on(tmp_path: Path) -> CSUDataConverter:
    """Constructed with lookback_days=180 (masking on)."""
    return CSUDataConverter(
        excel_path=tmp_path / "fake.xlsx",
        output_dir=tmp_path / "out",
        lookback_days=180,
    )


# --------------------------------------------------------------------------- #
# Constructor and config                                                      #
# --------------------------------------------------------------------------- #


def test_lookback_days_defaults_to_none_for_backwards_compat(
    converter_off: CSUDataConverter,
) -> None:
    assert converter_off.lookback_days is None


def test_lookback_days_is_stored_on_constructor(
    converter_on: CSUDataConverter,
) -> None:
    assert converter_on.lookback_days == 180


# --------------------------------------------------------------------------- #
# _apply_lookback_window — pure helper                                        #
# --------------------------------------------------------------------------- #


def test_apply_lookback_window_off_returns_df_unchanged(
    converter_off: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    out = converter_off._apply_lookback_window(
        med_df_three_fills, "medication_date", synthetic_index_date
    )
    pd.testing.assert_frame_equal(out, med_df_three_fills)


def test_apply_lookback_window_on_filters_to_window(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    out = converter_on._apply_lookback_window(
        med_df_three_fills, "medication_date", synthetic_index_date
    )
    assert len(out) == 1
    assert out.iloc[0]["brand_normalised"] == "drugB"


def test_apply_lookback_window_includes_lookback_start_boundary(
    converter_on: CSUDataConverter,
    synthetic_index_date: pd.Timestamp,
) -> None:
    df = pd.DataFrame(
        {
            "medication_date": [
                synthetic_index_date - timedelta(days=180),
                synthetic_index_date - timedelta(days=181),
            ],
        }
    )
    out = converter_on._apply_lookback_window(df, "medication_date", synthetic_index_date)
    assert len(out) == 1
    assert out.iloc[0]["medication_date"] == synthetic_index_date - timedelta(days=180)


def test_apply_lookback_window_excludes_index_date_boundary(
    converter_on: CSUDataConverter,
    synthetic_index_date: pd.Timestamp,
) -> None:
    df = pd.DataFrame(
        {
            "medication_date": [
                synthetic_index_date - timedelta(days=1),
                synthetic_index_date,
                synthetic_index_date + timedelta(days=1),
            ],
        }
    )
    out = converter_on._apply_lookback_window(df, "medication_date", synthetic_index_date)
    assert len(out) == 1
    assert out.iloc[0]["medication_date"] == synthetic_index_date - timedelta(days=1)


def test_apply_lookback_window_with_none_index_returns_empty_when_masking_on(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
) -> None:
    out = converter_on._apply_lookback_window(med_df_three_fills, "medication_date", None)
    assert len(out) == 0


def test_apply_lookback_window_with_missing_date_col_returns_empty_when_masking_on(
    converter_on: CSUDataConverter,
    synthetic_index_date: pd.Timestamp,
) -> None:
    df = pd.DataFrame({"some_other_col": [1, 2, 3]})
    out = converter_on._apply_lookback_window(df, "medication_date", synthetic_index_date)
    assert len(out) == 0


# --------------------------------------------------------------------------- #
# _derive_disease_severity — windowed counts                                  #
# --------------------------------------------------------------------------- #


def test_disease_severity_uses_lookback_for_med_fill_count(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    proc_df_three: pd.DataFrame,
    lab_df_three: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
    tmp_path: Path,
) -> None:
    converter_on._med_by_pat[1] = med_df_three_fills
    converter_on._proc_by_pat[1] = proc_df_three
    converter_on._lab_by_pat[1] = lab_df_three
    converter_on.sheets["demo"] = pd.DataFrame({"patid": [1], "diagcode": ["L508"]})

    score_masked = converter_on._derive_disease_severity(1, synthetic_index_date)

    converter_off = CSUDataConverter(excel_path=tmp_path / "fake.xlsx", output_dir=tmp_path / "out")
    converter_off._med_by_pat[1] = med_df_three_fills
    converter_off._proc_by_pat[1] = proc_df_three
    converter_off._lab_by_pat[1] = lab_df_three
    converter_off.sheets["demo"] = pd.DataFrame({"patid": [1], "diagcode": ["L508"]})
    score_unmasked = converter_off._derive_disease_severity(1, synthetic_index_date)

    # Masked score should be strictly less because pre-lookback and post-index
    # events are excluded from all three contributors.
    assert score_masked < score_unmasked


def test_disease_severity_unchanged_when_masking_off(
    converter_off: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_off._med_by_pat[1] = med_df_three_fills
    converter_off.sheets["demo"] = pd.DataFrame({"patid": [1], "diagcode": ["L508"]})
    # Three fills * 0.5 capped at 3.0 → +1.5 above base 2.0 = 3.5
    score = converter_off._derive_disease_severity(1, synthetic_index_date)
    expected_unmasked = 2.0 + min(3 * 0.5, 3.0)
    assert score == pytest.approx(expected_unmasked, abs=0.01)


# --------------------------------------------------------------------------- #
# _derive_engagement_score — windowed counts                                  #
# --------------------------------------------------------------------------- #


def test_engagement_score_uses_lookback_for_hcp_set(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    lab_df_three: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
    tmp_path: Path,
) -> None:
    converter_on._med_by_pat[1] = med_df_three_fills
    converter_on._lab_by_pat[1] = lab_df_three
    score_masked = converter_on._derive_engagement_score(
        1, continuous_enrollment=0, index_date=synthetic_index_date
    )

    converter_off = CSUDataConverter(excel_path=tmp_path / "fake.xlsx", output_dir=tmp_path / "out")
    converter_off._med_by_pat[1] = med_df_three_fills
    converter_off._lab_by_pat[1] = lab_df_three
    score_unmasked = converter_off._derive_engagement_score(
        1, continuous_enrollment=0, index_date=synthetic_index_date
    )

    assert score_masked < score_unmasked


# --------------------------------------------------------------------------- #
# Per-aggregate counts inside _build_patient_journeys                         #
# --------------------------------------------------------------------------- #


def _build_minimal_demo_sheet(synthetic_index_date: pd.Timestamp) -> pd.DataFrame:
    """A one-patient demo row covering all index_clinical_data needs."""
    return pd.DataFrame(
        {
            "patid": [1],
            "indexdt": [synthetic_index_date],
            "eligeff": [synthetic_index_date - timedelta(days=730)],
            "eligend": [synthetic_index_date + timedelta(days=365)],
            "age": [55.0],
            "gdr_cd": ["F"],
            "bus": ["COM"],
            "diagcode": ["L508"],
            "continuous_enrollment": [1],
            "zipcode_5": ["12345"],
        }
    )


def test_journey_status_lookback_masked_when_masking_on(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    proc_df_three: pd.DataFrame,
    lab_df_three: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on._med_by_pat[1] = med_df_three_fills
    converter_on._proc_by_pat[1] = proc_df_three
    converter_on._lab_by_pat[1] = lab_df_three
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()

    journeys = converter_on._build_patient_journeys()
    assert len(journeys) == 1
    assert journeys[0]["journey_status"] == "lookback_masked"


def test_journey_status_keeps_semantic_enum_when_masking_off(
    converter_off: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    proc_df_three: pd.DataFrame,
    lab_df_three: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_off._med_by_pat[1] = med_df_three_fills
    converter_off._proc_by_pat[1] = proc_df_three
    converter_off._lab_by_pat[1] = lab_df_three
    converter_off.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_off._build_patient_id_map()

    journeys = converter_off._build_patient_journeys()
    assert len(journeys) == 1
    assert journeys[0]["journey_status"] in {"completed", "active", "monitoring"}


def test_medication_claim_count_uses_window_when_masking_on(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on._med_by_pat[1] = med_df_three_fills
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()

    journeys = converter_on._build_patient_journeys()
    # Only the day-90 fill is in [day-180, day) → count == 1
    assert journeys[0]["medication_claim_count"] == 1


def test_medication_claim_count_full_panel_when_masking_off(
    converter_off: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_off._med_by_pat[1] = med_df_three_fills
    converter_off.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_off._build_patient_id_map()
    journeys = converter_off._build_patient_journeys()
    assert journeys[0]["medication_claim_count"] == 3


def test_procedure_claim_count_uses_window_when_masking_on(
    converter_on: CSUDataConverter,
    proc_df_three: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on._proc_by_pat[1] = proc_df_three
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()
    journeys = converter_on._build_patient_journeys()
    assert journeys[0]["procedure_claim_count"] == 1


def test_lab_claim_count_uses_window_when_masking_on(
    converter_on: CSUDataConverter,
    lab_df_three: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on._lab_by_pat[1] = lab_df_three
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()
    journeys = converter_on._build_patient_journeys()
    assert journeys[0]["lab_claim_count"] == 1


def test_days_on_therapy_uses_window_when_masking_on(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on._med_by_pat[1] = med_df_three_fills
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()
    journeys = converter_on._build_patient_journeys()
    # Only day-90 fill (45 days_sup) is in window
    assert journeys[0]["days_on_therapy"] == 45


def test_hcp_visits_uses_window_when_masking_on(
    converter_on: CSUDataConverter,
    med_df_three_fills: pd.DataFrame,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on._med_by_pat[1] = med_df_three_fills
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()
    journeys = converter_on._build_patient_journeys()
    # Only the day-90 visit pair (NPI002, day-90) is in window
    assert journeys[0]["hcp_visits"] == 1


# --------------------------------------------------------------------------- #
# Eligibility duration                                                        #
# --------------------------------------------------------------------------- #


def test_eligibility_duration_clipped_to_window_when_masking_on(
    converter_on: CSUDataConverter,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_on.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_on._build_patient_id_map()
    journeys = converter_on._build_patient_journeys()
    # eligeff = -730d, eligend = +365d, lookback = 180
    # clipped_start = max(-730, -180) = -180 (lookback_start)
    # clipped_end = min(+365, 0) = 0 (index_date)
    # duration = 180 days
    assert journeys[0]["eligibility_duration_days"] == 180


def test_eligibility_duration_full_when_masking_off(
    converter_off: CSUDataConverter,
    synthetic_index_date: pd.Timestamp,
) -> None:
    converter_off.sheets["demo"] = _build_minimal_demo_sheet(synthetic_index_date)
    converter_off._build_patient_id_map()
    journeys = converter_off._build_patient_journeys()
    # 730 + 365 = 1095 days
    assert journeys[0]["eligibility_duration_days"] == 1095


# --------------------------------------------------------------------------- #
# CLI surface                                                                 #
# --------------------------------------------------------------------------- #


def test_cli_accepts_lookback_days_argument() -> None:
    """Contract test: CLI exposes --lookback-days and threads it through."""
    import importlib

    convert_csu_rwd = importlib.import_module("convert_csu_rwd")
    src = Path(convert_csu_rwd.__file__).read_text()
    assert "--lookback-days" in src
    assert "lookback_days=args.lookback_days" in src
