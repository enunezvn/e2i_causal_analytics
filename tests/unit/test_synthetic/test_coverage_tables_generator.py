"""Shard 09 Task 5 + #1115: the 5 view-backed KPI tables already hold rows but all
pre-date now()-30d (max 2025-11-28), so MAU/WAU/intent-delta/data-lag/label-quality
read 0. CoverageTablesGenerator INSERTs fresh is_synthetic=true rows anchored to the
rolling window (we do NOT delete the stale real rows). Enum-exact user_region;
columns are the REAL ones (match_rate_vs_iqvia, not vs_claims).

#1115: user_sessions carries a realistic HETEROGENEOUS user population (daily /
weekly / occasional cohorts, ~n_records//4 users) so MAU (trailing 30d) clears its
production-scale target (2000) and WAU (trailing 7d) its 1200 target at the frontier,
with WAU < MAU and week-to-week variation — instead of the old `i % 30` cap that
pinned both KPIs to 30 forever. All 5 tables use DETERMINISTIC uuid5 PKs (reseed
upserts UPDATE in place instead of accumulating; cf. the #1105/#1106 incident).
"""

from datetime import date, datetime, timedelta

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.coverage_tables_generator import CoverageTablesGenerator

# Mirrors the production loader call (scripts/load_synthetic_data.py):
# n_records=max(50, FULL_SIZES["business_metrics"])=10000, seed=42+5, id_prefix="scv".
_PROD_CONFIG = {"seed": 47, "n_records": 10_000, "id_prefix": "scv"}
_RUN = date(2026, 7, 3)


def _session_dates(us):
    return us["session_start"].map(lambda s: datetime.fromisoformat(s).date())


def _distinct_users(us, run, days_back):
    """COUNT(DISTINCT user_id) over the trailing window ending at `run`, mirroring
    the registry SQL `session_start >= NOW() - INTERVAL 'N days'` (conservative:
    strictly-inside calendar days 0..days_back-1)."""
    dates = _session_dates(us)
    return us.loc[dates >= run - timedelta(days=days_back - 1), "user_id"].nunique()


def test_five_view_tables_fresh_and_tagged():
    run = date(2026, 6, 9)
    out = CoverageTablesGenerator(GeneratorConfig(seed=5, n_records=50), run_date=run).generate()
    assert set(out) == {
        "user_sessions",
        "hcp_intent_surveys",
        "data_source_tracking",
        "etl_pipeline_metrics",
        "ml_annotations",
    }
    us = out["user_sessions"]
    # sessions land inside the trailing 7d/30d KPI windows and never in the future
    starts = _session_dates(us)
    assert (starts <= run).all()
    assert (starts >= run - timedelta(days=89)).all()
    assert (starts >= run - timedelta(days=6)).any()  # WAU window non-empty
    assert set(us["user_region"]).issubset({"northeast", "south", "midwest", "west"})
    # data_source_tracking columns must be the REAL ones (vs_iqvia, not vs_claims)
    dst = out["data_source_tracking"]
    assert {"match_rate_vs_iqvia", "stacking_lift_percentage"}.issubset(dst.columns)
    # intent delta non-null for the BR-002 KPI
    assert out["hcp_intent_surveys"]["intent_to_prescribe_change"].notna().all()
    assert set(out["hcp_intent_surveys"]["brand"]).issubset({"Remibrutinib", "Kisqali", "Fabhalta"})
    # etl_pipeline_metrics carries a non-null TTR (WS1-DQ-009)...
    assert out["etl_pipeline_metrics"]["time_to_release_hours"].notna().all()
    # ...on status='success' rows — the WS1-DQ-009 registry query (migration
    # 095) averages TTR over status='success' only, so a revert to the old
    # 'completed' literal would silently zero the KPI's row set again.
    assert (out["etl_pipeline_metrics"]["status"] == "success").all()
    # ml_annotations carries an iaa_group_id (WS1-DQ-008) + categorical label
    # strings the label-quality KPI counts (positive/negative/uncertain, not 0/1).
    ann = out["ml_annotations"]
    assert ann["iaa_group_id"].notna().all()
    labels = {v["label"] for v in ann["annotation_value"]}
    assert labels.issubset({"positive", "negative", "uncertain"})
    for f in out.values():
        assert f["is_synthetic"].all()


def test_recent_rows_land_within_30d_of_run_date():
    run = date(2026, 6, 9)
    out = CoverageTablesGenerator(GeneratorConfig(seed=6, n_records=100), run_date=run).generate()
    # at least one row in each view-backed table inside the now()-30d window
    et = out["etl_pipeline_metrics"]["run_end"].apply(lambda s: datetime.fromisoformat(s).date())
    assert (et >= date(2026, 5, 10)).any()


def test_surveys_reference_provided_hcp_id_pool():
    """hcp_intent_surveys.hcp_id has a NO-ACTION FK to hcp_profiles.hcp_id. The
    generator must sample from the run's REAL generated hcp ids (namespaced, e.g.
    scvhcp_*) instead of minting hardcoded hcp_{i%50:05d} ids that only resolve
    against legacy stub rows (the 2026-06-11 cleanup had to keep 50 untagged
    legacy profiles solely because of that hardcoding)."""
    run = date(2026, 6, 9)
    pool = [f"scvhcp_{i:05d}" for i in range(200)]
    out = CoverageTablesGenerator(
        GeneratorConfig(seed=7, n_records=100), run_date=run, hcp_ids=pool
    ).generate()
    assert set(out["hcp_intent_surveys"]["hcp_id"]).issubset(set(pool))
    # more than one distinct HCP gets surveyed (sampling, not a constant)
    assert out["hcp_intent_surveys"]["hcp_id"].nunique() > 1


# --------------------------------------------------------------------------
# #1115: realistic user population — MAU/WAU carry signal vs production targets
# --------------------------------------------------------------------------


def test_user_population_scales_with_n_records_not_capped_at_30():
    """The old `i % 30` cap saturated distinct users at 30 regardless of
    n_records (WS3-BI-001/002 permanently CRITICAL). At the production loader
    config the population must land at ~n_records//4 = 2500 users."""
    out = CoverageTablesGenerator(GeneratorConfig(**_PROD_CONFIG), run_date=_RUN).generate()
    us = out["user_sessions"]
    assert 2400 <= us["user_id"].nunique() <= 2800


def test_mau_wau_clear_production_targets_at_frontier():
    """MAU (trailing 30d) >= 2000 (WS3-BI-001 target) and WAU (trailing 7d)
    >= 1200 (WS3-BI-002 target) at the data frontier, WITHOUT everyone being
    active every week: WAU < MAU and WAU/MAU in a realistic 0.5-0.75 band."""
    out = CoverageTablesGenerator(GeneratorConfig(**_PROD_CONFIG), run_date=_RUN).generate()
    us = out["user_sessions"]
    mau = _distinct_users(us, _RUN, 30)
    wau = _distinct_users(us, _RUN, 7)
    assert mau >= 2000, f"MAU {mau} below WS3-BI-001 target 2000"
    assert wau >= 1200, f"WAU {wau} below WS3-BI-002 target 1200"
    assert wau < mau, "WAU==MAU means the population is degenerate (everyone weekly)"
    assert 0.5 <= wau / mau <= 0.75, f"WAU/MAU {wau / mau:.2f} outside realistic band"


def test_weekly_wau_varies_and_activity_is_heterogeneous():
    """Week-to-week WAU must vary (no constant-30 plateau) and per-user session
    counts must be heterogeneous (daily/weekly/occasional cohorts, not a
    uniform activity rate)."""
    out = CoverageTablesGenerator(GeneratorConfig(**_PROD_CONFIG), run_date=_RUN).generate()
    us = out["user_sessions"]
    dates = _session_dates(us)
    weekly_wau = []
    for w in range(8):
        hi = _RUN - timedelta(days=7 * w)
        lo = hi - timedelta(days=6)
        weekly_wau.append(us.loc[(dates >= lo) & (dates <= hi), "user_id"].nunique())
    assert len(set(weekly_wau)) > 1, f"WAU constant across weeks: {weekly_wau}"
    # heterogeneity: the busiest decile is far more active than the quietest
    per_user = us.groupby("user_id").size().sort_values()
    n = len(per_user)
    q10 = per_user.iloc[: max(1, n // 10)].mean()
    q90 = per_user.iloc[-max(1, n // 10) :].mean()
    assert q90 / q10 >= 4, f"activity not heterogeneous (p90 {q90:.1f} / p10 {q10:.1f})"
    # total volume sanity: ~8-25 sessions/user keeps the table in the same
    # order of magnitude as the previous substrate (~40k rows)
    assert 8 <= len(us) / n <= 25


def test_user_attributes_stable_per_user():
    """A user has ONE role and ONE region across all their sessions (the old
    per-session redraw gave the same user different roles/regions)."""
    out = CoverageTablesGenerator(GeneratorConfig(seed=9, n_records=400), run_date=_RUN).generate()
    us = out["user_sessions"]
    per_user = us.groupby("user_id")[["user_role", "user_region"]].nunique()
    assert (per_user["user_role"] == 1).all()
    assert (per_user["user_region"] == 1).all()


# --------------------------------------------------------------------------
# #1115 (+#1105/#1106 pattern): deterministic PKs -> reseed-idempotent upserts
# --------------------------------------------------------------------------

_PK = {
    "user_sessions": "session_id",
    "hcp_intent_surveys": "survey_id",
    "data_source_tracking": "tracking_id",
    "etl_pipeline_metrics": "pipeline_run_id",
    "ml_annotations": "annotation_id",
}


def test_pks_deterministic_across_identical_runs():
    """uuid4 PKs made every reseed INSERT fresh rows the upsert-on-PK could
    never match -> user_sessions accumulated to 40k rows (4 reseeds x 10k).
    Same config + run_date must reproduce the SAME PKs so the loader's upsert
    UPDATEs in place (idempotent), mirroring experiment/mlops generators."""
    cfg = {"seed": 11, "n_records": 60, "id_prefix": "scv"}
    a = CoverageTablesGenerator(GeneratorConfig(**cfg), run_date=_RUN).generate()
    b = CoverageTablesGenerator(GeneratorConfig(**cfg), run_date=_RUN).generate()
    for table, pk in _PK.items():
        assert sorted(a[table][pk]) == sorted(b[table][pk]), f"{table}.{pk} not deterministic"
        assert a[table][pk].is_unique, f"{table}.{pk} has duplicates"
    # iaa_group_id groups must also be stable across runs
    assert sorted(a["ml_annotations"]["iaa_group_id"]) == sorted(
        b["ml_annotations"]["iaa_group_id"]
    )


def test_pks_disjoint_across_id_prefixes():
    """id_prefix namespaces a run's ids so a namespaced validation run cannot
    clobber the dev baseline (base.py contract)."""
    cfg = {"seed": 11, "n_records": 60}
    a = CoverageTablesGenerator(GeneratorConfig(id_prefix="scv", **cfg), run_date=_RUN).generate()
    b = CoverageTablesGenerator(GeneratorConfig(id_prefix="alt", **cfg), run_date=_RUN).generate()
    for table, pk in _PK.items():
        assert set(a[table][pk]).isdisjoint(set(b[table][pk])), f"{table}.{pk} collides"


def test_session_rows_stable_across_shifted_run_dates():
    """Session activity is keyed to the ABSOLUTE calendar date (not the offset
    from run_date): a later reseed regenerates IDENTICAL rows for overlapping
    dates (upsert no-op) instead of re-rolling them into near-duplicates."""
    cfg = {"seed": 13, "n_records": 200, "id_prefix": "scv"}
    a = CoverageTablesGenerator(GeneratorConfig(**cfg), run_date=_RUN).generate()
    b = CoverageTablesGenerator(
        GeneratorConfig(**cfg), run_date=_RUN + timedelta(days=14)
    ).generate()
    us_a, us_b = a["user_sessions"], b["user_sessions"]
    overlap_a = us_a[_session_dates(us_a) >= _RUN + timedelta(days=14) - timedelta(days=89)]
    overlap_b = us_b[_session_dates(us_b) <= _RUN]
    cols = ["session_id", "user_id", "session_start", "session_end"]
    fa = overlap_a[cols].sort_values("session_id").reset_index(drop=True)
    fb = overlap_b[cols].sort_values("session_id").reset_index(drop=True)
    assert fa.equals(fb)
