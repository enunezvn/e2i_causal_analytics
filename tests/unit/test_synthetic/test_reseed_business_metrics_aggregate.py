"""#1833 reseed script — offline (pure) parts.

The DB-facing parts (reading the live aggregate rows, the upsert) are
exercised by running ``--dry-run`` against the real local DB; the frame
assembly and the diff summary are pure and pinned here.
"""

from datetime import date

import pandas as pd

import src.ml.synthetic.frontier_append as fa
from scripts.reseed_business_metrics_aggregate import (
    AGGREGATE_METRIC_NAMES,
    build_reseed_frame,
    diff_summary,
)


class TestBuildReseedFrame:
    def test_covers_base_and_cohort_months_through_frontier(self):
        frame = build_reseed_frame(frontier=date(2026, 8, 30))
        base = frame[frame["metric_id"].str.startswith("metric_")]
        cohort = frame[~frame["metric_id"].str.startswith("metric_")]
        assert len(base) == 9780
        assert base["metric_date"].max() == "2026-07-01"
        # BM_EPOCH (2026-08) through the frontier month: exactly one cohort
        assert len(cohort) == 60
        assert cohort["metric_id"].str.startswith("m2608_").all()
        assert frame["metric_id"].is_unique
        assert frame["is_synthetic"].all()
        assert frame["value"].notna().all()
        assert set(frame["metric_name"]) == set(AGGREGATE_METRIC_NAMES)

    def test_cohort_rows_are_the_cron_cohort(self):
        frame = build_reseed_frame(frontier=date(2026, 9, 15))
        cohort = fa.generate_month_cohort(date(2026, 9, 1))["business_metrics"]
        got = frame[frame["metric_id"].str.startswith("m2609_")].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            got[["metric_id", "metric_date", "value", "target"]],
            cohort[["metric_id", "metric_date", "value", "target"]],
        )

    def test_frontier_before_epoch_is_base_only(self):
        frame = build_reseed_frame(frontier=date(2026, 7, 31))
        assert len(frame) == 9780


def _rows(*specs):
    cols = ["metric_id", "metric_date", "brand", "region", "metric_name", "value", "target"]
    return pd.DataFrame([dict(zip(cols, s, strict=True)) for s in specs])


class TestDiffSummary:
    def test_counts_and_scale(self):
        db = _rows(
            ("a", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0),
            ("b", "2026-07-01", "Kisqali", "west", "trx", 100.0, 110.0),
            ("c", "2026-07-01", "Kisqali", "west", "market_share", 0.5, 0.6),
            ("stale", "2026-07-01", "Fabhalta", "west", "trx", 5.0, 6.0),
        )
        regen = _rows(
            ("a", "2026-07-01", "Kisqali", "midwest", "trx", 88.0, 110.0),
            ("b", "2026-07-01", "Kisqali", "west", "trx", 100.0, 110.0),
            ("c", "2026-07-01", "Kisqali", "west", "market_share", 0.5, 0.6),
            ("new", "2026-08-01", "Kisqali", "west", "trx", 120.0, 130.0),
        )
        s = diff_summary(db, regen, scale_month="2026-07-01")
        assert s["rows_to_upsert"] == 4
        assert s["ids_only_in_regen"] == ["new"]
        assert s["ids_only_in_db"] == ["stale"]
        assert s["value_changed"] == 1
        assert s["value_unchanged"] == 2
        assert s["target_changed"] == 0
        # per-brand national trx scale for the month, before -> after
        assert s["national_trx"]["Kisqali"] == {"before": 200.0, "after": 188.0, "ratio": 0.94}
        # a brand only present on one side is reported, not dropped
        assert s["national_trx"]["Fabhalta"]["before"] == 5.0
        assert s["national_trx"]["Fabhalta"]["after"] == 0.0

    def test_identical_frames_are_a_no_op(self):
        db = _rows(("a", "2026-07-01", "Kisqali", "midwest", "trx", 100.0, 110.0))
        s = diff_summary(db, db.copy(), scale_month="2026-07-01")
        assert s["value_changed"] == 0 and s["target_changed"] == 0
        assert s["ids_only_in_db"] == [] and s["ids_only_in_regen"] == []
