"""Backlog #45 PR-A — synthetic claims ARRIVAL plane (stamp_claim_arrival).

The stamp is a post-generation pass on treatment_events (same discipline as
stamp_data_lag_hours / stamp_sequence_number): per-row lags KEYED on
(seed, treatment_event_id) — hashed uniform through the inverse gamma CDF, no
rng stream consumed — with vocabulary-driven parameters
(data_constraints.adjudication_lag_dgp), and two NEW columns only —
claim_available_date (= event_date + drawn adjudication lag) and
adjudication_lag_days. NO existing column may change
(additive-only, the migration-113 guard), no KPI reads the new columns, and
the #43 persistence/initiation calibration pins are structurally immune
(they train on PatientGenerator output, never treatment_events).

Contracts under test:
* pharmacy_claims (prescription) lags land inside the config clip band and the
  empirical MEDIAN lands inside the 1-3-month adjudication band (30..90 days);
* medical_claims (lab_test/procedure/consultation) adjudicate SLOWER than
  pharmacy claims (median ordering);
* CRM / configured-zero-lag event types get claim_available_date NULL and
  adjudication_lag_days 0 (they are not on the claims arrival plane);
* rows with no event_date get NULL for both columns (fail-empty);
* deterministic given seed; ORDER-INDEPENDENT per treatment_event_id (codex
  diff-review 2026-07-21 — reordering reassigns nothing); input never mutated;
* the REAL append entrypoint (build_frontier_datasets) emits stamped,
  frontier-filtered claims rows (arrival dates may exceed the frontier);
* ADDITIVE INVARIANCE on a REAL generated treatment_events frame: every
  pre-existing column byte-identical, exactly the two new columns added;
* loader carries both columns (TABLE_COLUMNS) and tolerates their absence
  (OPTIONAL_COLUMNS — frozen base + pre-#45 append cohorts predate them);
* load-script + frontier-append wiring: stamp runs at seed+10 (load path) and
  seed+10 within the weekly cohort stamp block (append path).
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.generators.data_lag import stamp_claim_arrival
from src.ml.synthetic.loaders import batch_loader
from src.ml.synthetic.loaders.batch_loader import (
    OPTIONAL_COLUMNS,
    TABLE_COLUMNS,
    BatchLoader,
    LoaderConfig,
)
from src.ontology.vocabulary_registry import VocabularyRegistry

_REPO_ROOT = Path(__file__).resolve().parents[3]

# The 1-3-month adjudication band (data_constraints.claims_lag_band) in days.
_BAND_DAYS = (30, 90)


def _dgp_config() -> dict:
    cfg = VocabularyRegistry.load().get_data_constraints().get("adjudication_lag_dgp")
    assert cfg, "data_constraints.adjudication_lag_dgp must be authored (backlog #45)"
    return cfg


def _frame(event_types: list[str], n_per_type: int, start: date = date(2026, 1, 1)):
    rows = []
    i = 0
    for et in event_types:
        for k in range(n_per_type):
            rows.append(
                {
                    "treatment_event_id": f"te{i}",
                    "patient_id": f"p{i % 500}",
                    "brand": "Kisqali",
                    "event_type": et,
                    "event_date": (pd.Timestamp(start) + pd.Timedelta(days=k % 365)).strftime(
                        "%Y-%m-%d"
                    ),
                    "is_synthetic": True,
                }
            )
            i += 1
    return pd.DataFrame(rows)


# =============================================================================
# distribution (seeded draws against the authored vocab parameters)
# =============================================================================


@pytest.mark.unit
def test_pharmacy_lags_in_clip_band_and_median_in_1_3_month_band():
    cfg = _dgp_config()
    clip_lo, clip_hi = cfg["source_classes"]["pharmacy_claims"]["clip_days"]
    df = _frame(["prescription"], 20_000)
    out = stamp_claim_arrival(df, seed=52)
    lags = out["adjudication_lag_days"].astype(int)
    assert lags.between(clip_lo, clip_hi).all(), "pharmacy lag escaped the config clip band"
    med = float(np.median(lags))
    assert _BAND_DAYS[0] <= med <= _BAND_DAYS[1], (
        f"pharmacy empirical median {med}d outside the 1-3-month adjudication band"
    )
    # claim_available_date == event_date + lag, exactly.
    expect = pd.to_datetime(out["event_date"]) + pd.to_timedelta(lags, unit="D")
    assert (pd.to_datetime(out["claim_available_date"]) == expect).all()


@pytest.mark.unit
def test_medical_claims_adjudicate_slower_than_pharmacy():
    df = _frame(["prescription", "lab_test", "procedure", "consultation"], 5_000)
    out = stamp_claim_arrival(df, seed=52)
    rx_med = float(np.median(out.loc[out["event_type"] == "prescription", "adjudication_lag_days"]))
    for et in ("lab_test", "procedure", "consultation"):
        et_med = float(np.median(out.loc[out["event_type"] == et, "adjudication_lag_days"]))
        assert et_med > rx_med, f"medical claims ({et}) must adjudicate slower than pharmacy"


@pytest.mark.unit
def test_crm_zero_lag_event_types_get_null_date_and_zero_lag():
    df = _frame(["crm_engagement", "veeva_call"], 50)
    out = stamp_claim_arrival(df, seed=52)
    assert out["claim_available_date"].isna().all(), "CRM events are not on the claims plane"
    assert (out["adjudication_lag_days"] == 0).all(), "configured-zero-lag class must stamp 0"


@pytest.mark.unit
def test_missing_event_date_yields_null_for_both_columns():
    df = _frame(["prescription"], 5)
    df.loc[[1, 3], "event_date"] = None
    out = stamp_claim_arrival(df, seed=52)
    assert out.loc[[1, 3], "claim_available_date"].isna().all()
    assert out.loc[[1, 3], "adjudication_lag_days"].isna().all()
    assert out.loc[[0, 2, 4], "claim_available_date"].notna().all()


@pytest.mark.unit
def test_deterministic_given_seed_and_input_not_mutated():
    df = _frame(["prescription", "lab_test"], 200)
    before = df.copy(deep=True)
    a = stamp_claim_arrival(df, seed=52)
    b = stamp_claim_arrival(df, seed=52)
    c = stamp_claim_arrival(df, seed=53)
    pd.testing.assert_frame_equal(a, b)
    assert not a["adjudication_lag_days"].equals(c["adjudication_lag_days"])
    pd.testing.assert_frame_equal(df, before)  # operates on a copy
    assert "claim_available_date" not in df.columns


# =============================================================================
# ADDITIVE INVARIANCE (migration-113 guard) on a real generated frame
# =============================================================================


@pytest.mark.unit
def test_additive_invariance_on_generated_treatment_events():
    """Stamp a REAL TreatmentGenerator frame (post-rename, as the load path
    does): every pre-existing column must be byte-identical and exactly the
    two arrival columns added."""
    from src.ml.synthetic.generators import (
        GeneratorConfig,
        PatientGenerator,
        TreatmentGenerator,
    )

    patients = PatientGenerator(GeneratorConfig(n_records=60, seed=7)).generate()
    treatments = TreatmentGenerator(
        GeneratorConfig(n_records=150, seed=7), patient_df=patients
    ).generate()
    treatments = treatments.rename(
        columns={
            "treatment_date": "event_date",
            "treatment_type": "event_type",
            "days_supply": "duration_days",
        }
    )
    snapshot = treatments.copy(deep=True)
    out = stamp_claim_arrival(treatments, seed=52)
    # pre-existing columns byte-identical
    pd.testing.assert_frame_equal(out[snapshot.columns.tolist()], snapshot)
    # exactly the two new columns added
    assert set(out.columns) - set(snapshot.columns) == {
        "claim_available_date",
        "adjudication_lag_days",
    }


# =============================================================================
# loader registration (silent-drop trap) + optionality (pre-#45 frames)
# =============================================================================


@pytest.mark.unit
def test_loader_whitelist_carries_both_arrival_columns():
    registered = set(TABLE_COLUMNS["treatment_events"])
    for col in ("claim_available_date", "adjudication_lag_days"):
        assert col in registered, f"{col} unregistered -> loader silently drops it"
        assert col in OPTIONAL_COLUMNS, (
            f"{col} not optional -> pre-#45 frames would trip critical_missing"
        )


@pytest.mark.unit
def test_frames_without_arrival_columns_still_validate():
    """Frozen base data + pre-#45 weekly cohorts predate the arrival columns:
    their absence must NOT be a critical validation error."""
    required = [
        c
        for c in TABLE_COLUMNS["treatment_events"]
        if not c.endswith("_split") and c not in OPTIONAL_COLUMNS
    ]
    assert "claim_available_date" not in required and "adjudication_lag_days" not in required
    df = pd.DataFrame({c: ["x"] for c in required})
    loader = BatchLoader(LoaderConfig(dry_run=True))
    is_valid, errors = loader.validate_datasets({"treatment_events": df})
    assert is_valid, f"pre-#45 frame must validate without arrival columns: {errors}"


@pytest.mark.unit
def test_legacy_source_timestamp_columns_stay_unwhitelisted():
    """Codex-folded (design C2 review): treatment_events.source_timestamp and
    .data_source are LEGACY-generator columns (data_generator.py, not in the
    reseed path) measured live-NULL on all synthetic rows — deliberately NOT
    whitelisted; the arrival plane uses its own explicit columns instead."""
    registered = set(batch_loader.TABLE_COLUMNS["treatment_events"])
    assert "source_timestamp" not in registered
    assert "data_source" not in registered


# =============================================================================
# wiring: load script (seed+10) + frontier weekly append cohorts
# =============================================================================


@pytest.mark.unit
def test_load_script_wires_stamp_at_seed_plus_10():
    """Tripwire on the load-path wiring (scripts/ is not importable here):
    the stamp must run on treatment_events at seed+10 — the next free stamp
    offset (+6 data lag, +8 model metrics, +9 change tracking)."""
    src = (_REPO_ROOT / "scripts" / "load_synthetic_data.py").read_text()
    assert "stamp_claim_arrival" in src, "load_synthetic_data.py never stamps the arrival plane"
    assert "seed=seed + 10" in src, "arrival stamp must use the seed+10 offset"


@pytest.mark.unit
def test_frontier_week_cohort_carries_arrival_columns():
    """Weekly append cohorts must be stamped identically to the base load —
    an unstamped cohort loads NULL arrival columns forever (the T9-driver
    silent-drop regression class)."""
    from src.ml.synthetic.frontier_append import EPOCH, generate_week_cohort
    from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator

    hcp_df = HCPGenerator(GeneratorConfig(id_prefix="scv", seed=42, n_records=60)).generate()
    cohort = generate_week_cohort(EPOCH, hcp_df)
    te = cohort["treatment_events"]
    assert "claim_available_date" in te.columns and "adjudication_lag_days" in te.columns
    rx = te[te["event_type"] == "prescription"]
    assert not rx.empty
    assert rx["claim_available_date"].notna().all()
    assert rx["adjudication_lag_days"].astype(int).ge(1).all()


@pytest.mark.unit
def test_lags_are_order_independent_per_event_id():
    """Codex diff-review finding 1 (2026-07-21): per-row lags must be a pure
    function of (seed, treatment_event_id), NOT of the frame's row order —
    the substrate's PKs are deterministic (positional f-strings in
    base._generate_ids / 'pnh_<pid>' / 'trxc_<i>'), so a keyed draw makes the
    frontier byte-equal re-run upsert claim structural instead of incidental
    to generator ordering. Reordering the same frame must reassign NOTHING."""
    df = _frame(["prescription", "lab_test", "consultation", "crm_engagement"], 300)
    df.loc[[5, 17], "event_date"] = None
    shuffled = df.sample(frac=1, random_state=99).reset_index(drop=True)
    a = stamp_claim_arrival(df, seed=52).set_index("treatment_event_id")
    b = stamp_claim_arrival(shuffled, seed=52).set_index("treatment_event_id")
    b = b.loc[a.index]  # align on the stable PK
    pd.testing.assert_series_equal(
        a["adjudication_lag_days"], b["adjudication_lag_days"], check_names=False
    )
    pd.testing.assert_series_equal(
        a["claim_available_date"], b["claim_available_date"], check_names=False
    )


@pytest.mark.unit
def test_build_frontier_datasets_stamps_arrival_and_survives_filter():
    """Codex diff-review finding 2 (2026-07-21): cover the REAL append
    entrypoint, not just generate_week_cohort — build_frontier_datasets must
    emit treatment_events whose claims rows carry the arrival columns AFTER
    the frontier filter (stamps run pre-filter; surviving rows keep their
    drawn lags). claim_available_date legitimately EXCEEDS the frontier for
    not-yet-arrived claims — the intended nowcast mechanism."""
    from datetime import timedelta

    from src.ml.synthetic.frontier_append import EPOCH, build_frontier_datasets
    from src.ml.synthetic.generators import GeneratorConfig, HCPGenerator

    frontier = EPOCH + timedelta(days=13)
    datasets = build_frontier_datasets(
        frontier=frontier,
        include_coverage=False,
        hcp_frame_factory=lambda: HCPGenerator(
            GeneratorConfig(id_prefix="scv", seed=42, n_records=60)
        ).generate(),
    )
    te = datasets["treatment_events"]
    assert "claim_available_date" in te.columns and "adjudication_lag_days" in te.columns
    # every surviving row respects the frontier filter (keyed on event_date)
    assert (pd.to_datetime(te["event_date"]) <= pd.Timestamp(frontier)).all()
    rx = te[te["event_type"] == "prescription"]
    assert not rx.empty
    assert rx["claim_available_date"].notna().all()
    assert rx["adjudication_lag_days"].astype(int).ge(1).all()
    # the arrival plane extends BEYOND the frontier (not-yet-arrived claims)
    assert (pd.to_datetime(rx["claim_available_date"]) > pd.Timestamp(frontier)).any()
