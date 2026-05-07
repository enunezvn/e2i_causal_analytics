"""Unit tests for CSU converter journey_duration_days windowing.

Phase 2 of ml-leakage-holistic-fix. Pre-fix, `journey_duration_days` derives from
`end_date = max(eligend, last_med_date + days_supply, last_proc_date, last_lab_date)`
without applying lookback_days masking. Treated patients accumulate post-index
events → larger end_date → longer duration → target-correlated.

Post-fix, with `lookback_days=180`, end_date is capped at index_date when
deriving from clinical events, so the maximum journey_duration_days is bounded
by `(eligend - index_date)` for patients with valid demo eligend, OR 0 for
clinical-only patients (their last event is at-or-before index_date).
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

# Allow direct import of the script
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from convert_csu_rwd import CSUDataConverter  # noqa: E402


def _build_converter(lookback_days: int | None) -> CSUDataConverter:
    """Construct a converter with stub paths (no Excel read)."""
    converter = CSUDataConverter(
        excel_path="/tmp/_nonexistent_for_test.xlsx",
        output_dir="/tmp/_unused",
        max_patients=None,
        lookback_days=lookback_days,
    )
    return converter


def _populate_minimal_state(converter: CSUDataConverter) -> None:
    """Populate the minimum internal state required by `_build_patient_journeys`.

    Two patients:
    - patient 1 ("treated"): index = 2024-01-01; eligend = 2024-04-30 (+120d);
      medication fills at index-30 (in-window), index+30 (post-index leak),
      index+200 (post-index leak with +30 days_sup).
    - patient 2 ("untreated"): same dates as patient 1; no clinical events.

    Pre-fix: patient 1's end_date = max(eligend=+120, index+200+30=+230) = +230.
             duration = 230. Patient 2's duration = 120.
    Post-fix (lookback=180): patient 1's only in-window med fill (index-30,
             days_sup=30, capped at index) → no extension past eligend.
             end_date = eligend = +120 → duration = 120. Patient 2 = 120.
             INVARIANT: treated == untreated (no clinical-driven leakage).
    """
    index_date = pd.Timestamp("2024-01-01")
    eligend = pd.Timestamp("2024-04-30")  # 120 days post-index

    # demo sheet: 2 patients with eligibility and index date
    converter.sheets = {
        "demo": pd.DataFrame(
            [
                {
                    "patid": 1,
                    "indexdt": index_date,
                    "eligend": eligend,
                    "eligeff": pd.Timestamp("2023-01-01"),
                    "gdr_cd": "F",
                    "age": 45,
                    "zipcode_5": "10001",
                    "bus": "COMM",
                    "diagcode": "L50.1",
                    "continuous_enrollment": 1,
                },
                {
                    "patid": 2,
                    "indexdt": index_date,
                    "eligend": eligend,
                    "eligeff": pd.Timestamp("2023-01-01"),
                    "gdr_cd": "M",
                    "age": 55,
                    "zipcode_5": "10001",
                    "bus": "COMM",
                    "diagcode": "L50.1",
                    "continuous_enrollment": 1,
                },
            ]
        ),
        "medication": pd.DataFrame(),
        "proc": pd.DataFrame(),
        "lab": pd.DataFrame(),
    }

    # Patient 1 has med fills at index-30 (in-window), index+30, index+200.
    # Pre-fix: max-date candidate is index+200 + days_sup(30) = index+230.
    # Post-fix windowed (180d): only index-30 remains → capped at index_date.
    converter._med_by_pat = {
        1: pd.DataFrame(
            [
                {
                    "medication_date": index_date - timedelta(days=30),
                    "days_sup": 30,
                    "npi": "NPI001",
                },
                {
                    "medication_date": index_date + timedelta(days=30),
                    "days_sup": 30,
                    "npi": "NPI001",
                },
                {
                    "medication_date": index_date + timedelta(days=200),
                    "days_sup": 30,
                    "npi": "NPI002",
                },
            ]
        )
    }
    converter._proc_by_pat = {}
    converter._lab_by_pat = {}

    # ID maps
    converter.patient_id_map = {1: "PAT_000001", 2: "PAT_000002"}
    converter.journey_id_map = {1: "PJ_000001", 2: "PJ_000002"}


def test_journey_duration_days_unwindowed_off_mode_includes_post_index():
    """OFF mode (lookback_days=None): treated patient duration extends post-index.

    This is the documented baseline behavior. With no lookback, the converter
    uses last_med_date + days_supply as a candidate end_date — the +200 fill +
    30 days_sup = +230 post-index. journey_duration_days = 230.
    """
    converter = _build_converter(lookback_days=None)
    _populate_minimal_state(converter)
    journeys = converter._build_patient_journeys()

    treated = next(j for j in journeys if j["patient_id"] == "PAT_000001")
    untreated = next(j for j in journeys if j["patient_id"] == "PAT_000002")

    # Pre-fix: clinical events past index extend treated duration well past eligend.
    # last_med_date (+200) + days_sup (30) = +230 days. Untreated stays at +120 (eligend).
    assert treated["journey_duration_days"] >= 200, (
        f"Expected ≥200 duration in OFF mode (post-index leakage); "
        f"got {treated['journey_duration_days']}"
    )
    assert treated["journey_duration_days"] > untreated["journey_duration_days"], (
        f"OFF-mode invariant: clinical events should extend treated duration. "
        f"treated={treated['journey_duration_days']} vs untreated={untreated['journey_duration_days']}"
    )


def test_journey_duration_days_bounded_by_lookback_window():
    """Phase 2 fix: under lookback_days=180, journey_duration_days must not
    extend past index_date due to post-index events.

    The converter should:
    - Filter med/proc/lab event dates to the [index - 180d, index_date] window
      before computing the max date candidate.
    - Cap (last_in_window_med_date + days_sup) at index_date so that supply
      tails do not leak post-index.

    For the treated fixture (index = 2024-01-01, eligend = 2024-12-31), the
    clinical-derived end_date is bounded at index_date. Since eligend is post-
    index (+365), the demo-derived end_date is +365 — and journey_duration_days
    is then 365. The KEY invariant is: there must NOT be a difference in
    duration between the treated and untreated patients driven by clinical
    events (both should now resolve to eligend - index_date).
    """
    converter = _build_converter(lookback_days=180)
    _populate_minimal_state(converter)
    journeys = converter._build_patient_journeys()

    treated = next(j for j in journeys if j["patient_id"] == "PAT_000001")
    untreated = next(j for j in journeys if j["patient_id"] == "PAT_000002")

    # The smoking-gun invariant: treated and untreated patients with the same
    # demo eligend must have IDENTICAL journey_duration_days. Pre-fix the
    # treated patient's clinical events extend duration; post-fix they don't.
    assert treated["journey_duration_days"] == untreated["journey_duration_days"], (
        f"Post-fix invariant violated: treated={treated['journey_duration_days']}, "
        f"untreated={untreated['journey_duration_days']}. Clinical events should "
        f"NOT extend journey_duration past index_date when lookback is set."
    )


def test_journey_duration_days_clinical_only_patient_under_lookback():
    """A patient with NO demo eligend AND clinical events in the window must
    have journey_duration_days = 0 (events collapse to ≤ index_date).
    """
    converter = _build_converter(lookback_days=180)
    _populate_minimal_state(converter)
    # Strip patient 1's demo eligend so end_date can only come from clinical events
    converter.sheets["demo"].loc[converter.sheets["demo"]["patid"] == 1, "eligend"] = pd.NaT

    journeys = converter._build_patient_journeys()
    treated = next(j for j in journeys if j["patient_id"] == "PAT_000001")

    # All clinical events (capped at index_date) → duration = 0
    assert treated["journey_duration_days"] == 0, (
        f"Clinical-only patient under lookback should have duration=0; "
        f"got {treated['journey_duration_days']}"
    )
