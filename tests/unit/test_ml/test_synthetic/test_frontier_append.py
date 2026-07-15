"""Frontier-append cohort invariants (gold-standard supplement workstream).

The append design's safety rests on three properties verified here (first
established by the 2026-07-04 cheapest-disproof experiments):

1. DETERMINISM  — a cohort is a pure function of its calendar key: two
   generations emit identical frames, so re-running a week upserts as no-ops.
2. DISJOINTNESS — cohort PKs never collide with the frozen ``scv`` base
   substrate or with other cohorts, so upserts append and never clobber.
3. FRONTIER FILTER — rows are held back until their occurrence date crosses
   the frontier, and re-emitting them later reproduces the SAME rows (the
   "stateless dribble" that replaces date-capping).

If any test here fails after touching frontier_append.py constants (sizes,
seeds, prefixes, epoch), STOP: changing cohort identity regenerates different
values under already-loaded PKs and silently rewrites history.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from src.ml.synthetic.frontier_append import (
    BM_EPOCH,
    EPOCH,
    TRAIL_WEEKS,
    WEEKLY_SIZES,
    base_hcp_frame,
    build_frontier_datasets,
    filter_to_frontier,
    generate_month_cohort,
    generate_week_cohort,
    iter_month_starts,
    iter_week_starts,
    month_prefix,
    month_seed,
    week_prefix,
    week_seed,
    week_start_of,
)

WEEK_A = date(2026, 7, 6)  # 2026-W28 (== EPOCH)
WEEK_B = date(2026, 7, 13)  # 2026-W29

# Wall-clock stamp columns excluded from frame-equality checks.
VOLATILE = {"created_at", "updated_at", "generated_at", "inserted_at"}

# FeatureStoreSeeder mints DETERMINISTIC uuid5 ids (pure function of the natural
# key), so group/feature ids are stable across seeder instances and cohorts —
# build_frontier_datasets keeps only the FIRST cohort's features frame, and every
# cohort's feature_values must resolve against it (the loader's #852 reconcile can
# only remap ids it finds in that frame). feature_values row PKs are still random
# per generation (FeatureValueGenerator uuid4); the loader upserts them on the
# natural key (feature_id, entity_values, event_timestamp), so only that PK column
# stays excluded from determinism checks.
VOLATILE_BY_TABLE = {
    "feature_values": {"id", "feature_value_id"},
}

# The one PK column the loader upserts on, per fact table.
PK = {
    "patient_journeys": "patient_journey_id",
    "treatment_events": "treatment_event_id",
    "ml_predictions": "prediction_id",
    "triggers": "trigger_id",
    "business_metrics": "metric_id",
}


@pytest.fixture(scope="module")
def hcp_df() -> pd.DataFrame:
    return base_hcp_frame()


@pytest.fixture(scope="module")
def cohort_a(hcp_df) -> dict:
    return generate_week_cohort(WEEK_A, hcp_df)


@pytest.fixture(scope="module")
def cohort_a_again(hcp_df) -> dict:
    return generate_week_cohort(WEEK_A, hcp_df)


@pytest.fixture(scope="module")
def cohort_b(hcp_df) -> dict:
    return generate_week_cohort(WEEK_B, hcp_df)


def _stable(df: pd.DataFrame, table: str = "") -> pd.DataFrame:
    drop = VOLATILE | VOLATILE_BY_TABLE.get(table, set())
    return df[[c for c in df.columns if c not in drop]].reset_index(drop=True)


# =============================================================================
# calendar keying
# =============================================================================


def test_week_start_of_returns_monday():
    assert week_start_of(date(2026, 7, 9)) == WEEK_A
    assert week_start_of(WEEK_A) == WEEK_A


def test_prefixes_and_seeds_are_unique_per_cohort():
    assert week_prefix(WEEK_A) == "w2628"
    assert week_prefix(WEEK_A) != week_prefix(WEEK_B)
    assert week_seed(WEEK_A) != week_seed(WEEK_B)
    # month keys never collide with week keys (week 1..53 < month offset 500)
    assert month_seed(date(2026, 8, 1)) != week_seed(WEEK_A)
    assert month_prefix(date(2026, 8, 1)) == "m2608"


def test_iter_week_starts_clamps_at_epoch_and_trails():
    assert iter_week_starts(EPOCH - timedelta(days=1)) == []
    assert iter_week_starts(EPOCH) == [EPOCH]
    # mid-week frontier still includes the current (partial) week
    assert iter_week_starts(EPOCH + timedelta(days=3)) == [EPOCH]
    # far future: exactly TRAIL_WEEKS cohorts, ending at the frontier's week
    far = EPOCH + timedelta(weeks=100)
    weeks = iter_week_starts(far)
    assert len(weeks) == TRAIL_WEEKS
    assert weeks[-1] == week_start_of(far)


def test_iter_month_starts_from_bm_epoch():
    assert iter_month_starts(BM_EPOCH - timedelta(days=1)) == []
    assert iter_month_starts(date(2026, 9, 15)) == [date(2026, 8, 1), date(2026, 9, 1)]


# =============================================================================
# determinism (idempotent re-runs)
# =============================================================================


def test_week_cohort_is_deterministic(cohort_a, cohort_a_again):
    assert set(cohort_a) == set(cohort_a_again)
    for table in cohort_a:
        pd.testing.assert_frame_equal(
            _stable(cohort_a[table], table), _stable(cohort_a_again[table], table)
        )


def test_month_cohort_is_deterministic():
    m1 = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
    m2 = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
    pd.testing.assert_frame_equal(_stable(m1), _stable(m2))


# =============================================================================
# disjointness (append, never clobber)
# =============================================================================


def test_cohort_pks_are_prefixed_and_disjoint_from_base(cohort_a):
    """Every PK must EMBED the week prefix (most ids lead with it; PNH lab
    events are `pnh_<patient_id>` natural keys, so the prefix sits inside) —
    embedding is what guarantees disjointness from the base `scv` namespace."""
    prefix = week_prefix(WEEK_A)
    for table, pk in PK.items():
        if table not in cohort_a:
            continue
        ids = cohort_a[table][pk].astype(str)
        assert ids.str.contains(prefix, regex=False).all(), f"{table}.{pk} not namespaced"
        assert not ids.str.contains("scv", regex=False).any(), f"{table}.{pk} collides with base"


def test_cohort_pks_disjoint_across_weeks(cohort_a, cohort_b):
    for table, pk in PK.items():
        if table not in cohort_a or table not in cohort_b:
            continue
        overlap = set(cohort_a[table][pk]) & set(cohort_b[table][pk])
        assert not overlap, f"{table}.{pk}: {len(overlap)} ids shared across weeks"


def test_cohorts_are_not_value_clones(cohort_a, cohort_b):
    """Per-week seeds must vary the draws — otherwise every week appends the
    same 160 patients under different ids and drift/HTE structure flatlines."""
    a = cohort_a["patient_journeys"]["propensity_score"].reset_index(drop=True)
    b = cohort_b["patient_journeys"]["propensity_score"].reset_index(drop=True)
    assert not a.equals(b)


def test_feature_values_resolve_against_any_cohorts_features_frame(cohort_a, cohort_b):
    """THE 23503 regression (2026-07-15 append run: feature_values 0/340).

    build_frontier_datasets keeps only the FIRST cohort's feature_groups/features
    frames but concatenates feature_values from EVERY cohort. The loader's #852
    reconcile remaps feature_id only for ids present in the kept features frame,
    so any cohort minting different feature ids sends orphaned feature_ids to the
    DB -> FK 23503 -> the whole batch fails atomically. Feature ids must therefore
    be identical across cohorts (deterministic from the natural key), making every
    cohort's feature_values resolvable against whichever frame is kept."""
    kept_ids = set(cohort_a["features"]["id"])
    for cohort in (cohort_a, cohort_b):
        referenced = set(cohort["feature_values"]["feature_id"])
        orphans = referenced - kept_ids
        assert not orphans, f"{len(orphans)} feature_ids unresolvable against the kept frame"


def test_ids_fit_varchar20(cohort_a):
    for table, pk in PK.items():
        if table not in cohort_a:
            continue
        width = cohort_a[table][pk].astype(str).str.len().max()
        assert width <= 20, f"{table}.{pk} max width {width} exceeds varchar(20)"


def test_patients_reference_frozen_hcp_universe(cohort_a, hcp_df):
    assert cohort_a["patient_journeys"]["hcp_id"].isin(hcp_df["hcp_id"]).all()


# =============================================================================
# frontier filter (stateless dribble)
# =============================================================================


def test_filter_holds_back_future_occurrence_rows(cohort_a):
    frontier = WEEK_A + timedelta(days=6)  # week end: derived events overshoot it
    filtered = filter_to_frontier(cohort_a, frontier)
    cutoff = frontier.isoformat()
    checked = 0
    for table, col in (
        ("patient_journeys", "journey_start_date"),
        ("treatment_events", "event_date"),
        ("ml_predictions", "prediction_timestamp"),
        ("triggers", "trigger_timestamp"),
    ):
        vals = filtered[table][col].astype(str).str[:10]
        assert (vals <= cutoff).all(), f"{table}.{col} leaked past the frontier"
        checked += 1
    assert checked == 4
    # derived events DO overshoot the week (the reason the filter exists):
    # something must actually get held back at week-end frontier.
    assert len(filtered["treatment_events"]) < len(cohort_a["treatment_events"])


def test_expiration_dates_may_exceed_frontier(cohort_a):
    """expiration_date is a deadline, not an occurrence — never filtered on.
    (Frontier +34d so some triggers have occurred; trigger timestamps only
    start ~2 days after week end.)"""
    frontier = WEEK_A + timedelta(days=34)
    filtered = filter_to_frontier(cohort_a, frontier)
    assert len(filtered["triggers"]) > 0
    exp = pd.to_datetime(filtered["triggers"]["expiration_date"].astype(str).str[:10])
    assert (exp > pd.Timestamp(frontier)).any()


def test_rows_stable_as_frontier_advances(cohort_a):
    """THE append invariant: a row emitted at frontier F1 is byte-identical
    when the cohort is regenerated and filtered at F2 > F1 — so later runs
    upsert already-loaded rows as no-ops and only add newly-crossed rows."""
    f1 = WEEK_A + timedelta(days=6)
    f2 = WEEK_A + timedelta(days=34)
    at_f1 = filter_to_frontier(cohort_a, f1)
    at_f2 = filter_to_frontier(cohort_a, f2)
    for table, pk in PK.items():
        if table not in at_f1:
            continue
        early = _stable(at_f1[table])
        late = _stable(at_f2[table])
        assert len(late) >= len(early)
        merged = late[late[pk].isin(set(early[pk]))].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            early.sort_values(pk).reset_index(drop=True),
            merged.sort_values(pk).reset_index(drop=True),
        )


def test_sequence_numbers_survive_the_filter(cohort_a):
    """stamp_sequence_number runs on the FULL cohort before filtering: an rx
    visible at an early frontier keeps its sequence number as later rx of the
    same (patient, brand) cross the frontier in later runs (NRx honesty)."""
    # +34d: prescriptions only occur after the journey week (diagnosis->rx
    # lag), so a week-end frontier keeps zero rx rows.
    frontier = WEEK_A + timedelta(days=34)
    full = cohort_a["treatment_events"]
    kept = filter_to_frontier(cohort_a, frontier)["treatment_events"]
    by_id = full.set_index("treatment_event_id")["sequence_number"]
    rx = kept[kept["sequence_number"].notna()]
    assert len(rx) > 0
    assert len(rx) < full["sequence_number"].notna().sum()  # some rx still held back
    for _, row in rx.iterrows():
        assert row["sequence_number"] == by_id[row["treatment_event_id"]]


# =============================================================================
# monthly business_metrics cohorts
# =============================================================================


def test_month_cohort_rows_land_on_month_start():
    """Pins the monthly-grain assumption: the generator floors metric dates to
    month starts, so ONE cohort per month (weekly cohorts would multiply)."""
    bm = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
    dates = bm["metric_date"].astype(str).str[:10].unique().tolist()
    assert dates == ["2026-08-01"]
    assert bm["metric_id"].astype(str).str.startswith("m2608").all()


# =============================================================================
# build_frontier_datasets (whole-run assembly)
# =============================================================================


def test_build_pre_epoch_returns_empty(hcp_df):
    assert build_frontier_datasets(EPOCH - timedelta(days=1)) == {}


def test_build_scopes_tables_and_stamps_synthetic(hcp_df):
    frontier = EPOCH + timedelta(days=2)  # 1 weekly cohort, 0 bm months — cheap
    datasets = build_frontier_datasets(frontier, hcp_frame_factory=lambda: hcp_df)
    # fixed-universe Shard-09 tables must NOT grow with the calendar
    for absent in (
        "ml_experiments",
        "ml_model_registry",
        "ab_experiment_assignments",
        "ml_observability_spans",
        "causal_paths",
        "hcp_profiles",
    ):
        assert absent not in datasets, f"{absent} must not be in append runs"
    for present in (
        "patient_journeys",
        "treatment_events",
        "feature_values",
        "user_sessions",
    ):
        assert present in datasets, f"{present} missing from append run"
    # Filter-emptied tables are OMITTED (loader validation rejects empty
    # frames): no trigger occurs within 2 days of a journey start.
    assert "triggers" not in datasets
    assert "business_metrics" not in datasets  # frontier precedes BM_EPOCH
    for table, df in datasets.items():
        assert df["is_synthetic"].all(), f"{table} rows missing is_synthetic"
    # weekly cohort scale sanity (one week's patients, frontier-filtered)
    assert 0 < len(datasets["patient_journeys"]) <= WEEKLY_SIZES["patient"]
