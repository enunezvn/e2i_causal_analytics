"""Frontier-append business_metrics cohort tests (#1566 D1).

The frozen base (loaded 2026-07-03) spans 163 months 2013-01..2026-07, so its
July-2026 trx rows carry trend_factor 1 + 0.02*162. Monthly frontier cohorts
(``generate_month_cohort``) are single-date runs, so the positional
``month_idx = dates.index(metric_date)`` reset to 0 and every appended month
from BM_EPOCH (2026-08-01) forward collapsed to the 2013 baseline (~24% of
July's level) — deterministically and permanently. D1 anchors the cohort's
trend to the absolute calendar origin ``BM_TREND_ORIGIN = date(2013, 1, 1)``
instead; ``trend_origin=None`` preserves the positional behavior for every
other caller byte-for-byte.
"""

from datetime import date

import pandas as pd

import src.ml.synthetic.frontier_append as fa
from src.ml.synthetic.frontier_append import generate_month_cohort
from src.ml.synthetic.generators import BusinessMetricsGenerator, GeneratorConfig

# Frozen-base trx model pinned literally (NOT read from METRIC_CONFIGS): the DB
# base was generated from these values on 2026-07-03 and does not retune with
# code edits — if someone retunes the generator config, appended cohorts WOULD
# diverge from the frozen base, and this test should break.
TRX_BASE = {"Remibrutinib": 15000.0, "Fabhalta": 8000.0, "Kisqali": 50000.0}
REGION_FACTORS = {"northeast": 1.15, "south": 0.95, "midwest": 0.90, "west": 1.00}
TRX_TREND = 0.02  # 2%/month
# (2026-08 - 2013-01) in calendar months: (2026-2013)*12 + (8-1) = 163.
AUG_2026_IDX = 163


def _cohort_config(**overrides) -> GeneratorConfig:
    """Mirror generate_month_cohort's GeneratorConfig for August 2026."""
    month_start = date(2026, 8, 1)
    params = {
        "id_prefix": fa.month_prefix(month_start),
        "seed": fa.month_seed(month_start),
        "n_records": fa.MONTHLY_SIZES["business_metrics"],
        "start_date": month_start,
        "end_date": date(2026, 8, 31),
    }
    params.update(overrides)
    return GeneratorConfig(**params)


class TestMonthCohortContinuity:
    """The #1566 red test: appended months must sit at frozen-base scale."""

    def test_august_2026_trx_sits_at_base_scale(self):
        bm = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
        trx = bm[bm["metric_type"] == "trx"]
        # 3 brands x 4 regions on the single cohort date.
        assert len(trx) == 12

        expected = trx.apply(
            lambda r: (
                TRX_BASE[r["brand"]]
                * REGION_FACTORS[r["region"]]
                * (1 + TRX_TREND * AUG_2026_IDX)
            ),
            axis=1,
        )
        ratio = float((trx["value"] / expected).mean())
        # Per-row noise is N(0, 0.15); the 12-row mean ratio must sit near 1.
        # Under the positional month_idx=0 defect it sits at ~0.235.
        assert 0.85 <= ratio <= 1.15, (
            f"appended-month trx collapsed off frozen-base scale: mean ratio {ratio:.3f}"
        )

    def test_bm_trend_origin_frozen_constant(self):
        # FROZEN: 2013-01-01 is the first month of the base actually loaded in
        # the DB (2026-07-03 load) — July 2026 = index 162, first appended
        # month 2026-08 = index 163. NOT re-derivable by re-running the base
        # generator (its default date range re-anchors to the run date).
        assert fa.BM_TREND_ORIGIN == date(2013, 1, 1)


class TestMonthCohortDeterminism:
    def test_same_month_twice_is_identical(self):
        a = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
        b = generate_month_cohort(date(2026, 8, 1))["business_metrics"]
        pd.testing.assert_frame_equal(a, b)


class TestTrendOriginChangedColumnInvariance:
    """trend_origin feeds arithmetic only — it must consume no RNG variates.

    Pin the EXACT set of columns that differ between trend_origin=None and
    trend_origin=2013-01-01 on the same cohort config (measured, then frozen):

    - value:  base * region * trend_factor * (1 + noise) — trend_factor moves
      from 1.00 (positional idx 0) to 4.26 (calendar idx 163).
    - target: same trend_factor term times the target multiplier draw.
    - confidence_interval_lower/upper: derived from value (std_error is
      value * volatility / sqrt(sample_size)).
    - achievement_rate: value/target is trend_factor-INVARIANT before rounding
      ((1+noise)/multiplier), but (a) value and target are rounded to 2
      decimals before the ratio, so sub-1.0 metrics (market_share,
      conversion_rate) re-quantize differently at different scales (measured
      max |diff| 0.077 / 0.042 on the Aug-2026 cohort), and (b) the
      hcp_engagement_score value cap at 10.0 engages at calendar scale
      (10 of 12 rows; measured max |diff| 0.289) while its target is uncapped
      (the inherited cap defect noted in #1566 — out of D1 scope). trx/nrx
      agree byte-for-byte (measured max |diff| 0.0).

    Byte-identical (pinned below): metric_id, metric_date, metric_type,
    metric_name, brand, region, year_over_year_change, month_over_month_change,
    roi, statistical_significance, sample_size, data_split — the yoy/mom/roi
    stamps are model-parameter draws (D2), the id/statistical draws precede no
    month_idx-dependent RNG call, and data_split depends on dates only.
    """

    EXPECTED_CHANGED = {
        "value",
        "target",
        "achievement_rate",
        "confidence_interval_lower",
        "confidence_interval_upper",
    }

    def _frames(self):
        df_none = BusinessMetricsGenerator(_cohort_config()).generate()
        df_origin = BusinessMetricsGenerator(
            _cohort_config(trend_origin=date(2013, 1, 1))
        ).generate()
        return df_none, df_origin

    def test_changed_column_set_is_exactly_the_value_leg(self):
        df_none, df_origin = self._frames()
        assert list(df_none.columns) == list(df_origin.columns)
        changed = {
            col for col in df_none.columns if not df_none[col].equals(df_origin[col])
        }
        assert changed == self.EXPECTED_CHANGED, (
            f"unexpected trend_origin-sensitive columns: "
            f"extra={changed - self.EXPECTED_CHANGED}, "
            f"missing={self.EXPECTED_CHANGED - changed}"
        )

    def test_rng_stream_columns_byte_identical(self):
        df_none, df_origin = self._frames()
        for col in (
            "metric_id",
            "year_over_year_change",
            "month_over_month_change",
            "roi",
            "statistical_significance",
            "sample_size",
            "data_split",
        ):
            pd.testing.assert_series_equal(df_none[col], df_origin[col]), col

    def test_achievement_rate_agrees_where_value_cap_never_engages(self):
        # value/target is scale-invariant modulo the 2-decimal rounding of
        # value and target; for the large-value prescription metrics that
        # quantization is negligible (~1e-7 relative), so agreement holds
        # within the spec's ~2e-3 rounding tolerance. Sub-1.0 metrics and the
        # capped hcp_engagement_score are excluded (see class docstring).
        df_none, df_origin = self._frames()
        mask = df_none["metric_type"].isin(["trx", "nrx"])
        diff = (
            (df_none.loc[mask, "achievement_rate"] - df_origin.loc[mask, "achievement_rate"])
            .abs()
            .max()
        )
        assert diff <= 2e-3, f"trx/nrx achievement_rate moved beyond rounding: {diff}"


class TestPKStability:
    def test_metric_id_set_identical_with_and_without_trend_origin(self):
        df_none = BusinessMetricsGenerator(_cohort_config()).generate()
        df_origin = BusinessMetricsGenerator(
            _cohort_config(trend_origin=date(2013, 1, 1))
        ).generate()
        assert list(df_none["metric_id"]) == list(df_origin["metric_id"])
