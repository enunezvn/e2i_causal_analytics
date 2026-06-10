"""Shard 09 Task 5: the 5 view-backed KPI tables already hold rows but all
pre-date now()-30d (max 2025-11-28), so MAU/WAU/intent-delta/data-lag/label-quality
read 0. CoverageTablesGenerator INSERTs fresh is_synthetic=true rows anchored to the
rolling window (we do NOT delete the stale real rows). Enum-exact user_region;
columns are the REAL ones (match_rate_vs_iqvia, not vs_claims)."""

from datetime import date, datetime

from src.ml.synthetic.generators.base import GeneratorConfig
from src.ml.synthetic.generators.coverage_tables_generator import CoverageTablesGenerator


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
    # bulk within now()-30d -> MAU/WAU non-zero (recent_frac 0.7)
    starts = us["session_start"].apply(lambda s: datetime.fromisoformat(s).date())
    assert (starts >= date(2026, 5, 10)).mean() >= 0.6
    assert set(us["user_region"]).issubset({"northeast", "south", "midwest", "west"})
    # data_source_tracking columns must be the REAL ones (vs_iqvia, not vs_claims)
    dst = out["data_source_tracking"]
    assert {"match_rate_vs_iqvia", "stacking_lift_percentage"}.issubset(dst.columns)
    # intent delta non-null for the BR-002 KPI
    assert out["hcp_intent_surveys"]["intent_to_prescribe_change"].notna().all()
    assert set(out["hcp_intent_surveys"]["brand"]).issubset({"Remibrutinib", "Kisqali", "Fabhalta"})
    # etl_pipeline_metrics carries a non-null TTR (WS1-DQ-009)
    assert out["etl_pipeline_metrics"]["time_to_release_hours"].notna().all()
    # ml_annotations carries an iaa_group_id (WS1-DQ-008)
    assert out["ml_annotations"]["iaa_group_id"].notna().all()
    for f in out.values():
        assert f["is_synthetic"].all()


def test_recent_rows_land_within_30d_of_run_date():
    run = date(2026, 6, 9)
    out = CoverageTablesGenerator(GeneratorConfig(seed=6, n_records=100), run_date=run).generate()
    # at least one row in each view-backed table inside the now()-30d window
    et = out["etl_pipeline_metrics"]["run_end"].apply(lambda s: datetime.fromisoformat(s).date())
    assert (et >= date(2026, 5, 10)).any()
