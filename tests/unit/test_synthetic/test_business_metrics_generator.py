"""Tests for Business Metrics Generator.

Tests metric generation per brand/region, achievement rate calculations,
time-series continuity, and data split distribution.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.generators import BusinessMetricsGenerator, GeneratorConfig


class TestBusinessMetricsGeneratorBasic:
    """Test basic BusinessMetricsGenerator functionality."""

    def test_generate_returns_dataframe(self):
        """Test generate returns a DataFrame."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_generate_respects_n_records(self):
        """Test generate produces approximately requested record count."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Should be close to requested, within tolerance for rounding
        assert len(df) >= 400
        assert len(df) <= 600

    def test_required_columns_present(self):
        """Test all required columns are present."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        required_columns = [
            "metric_id",
            "metric_date",
            "metric_type",
            "metric_name",
            "brand",
            "region",
            "value",
            "target",
            "achievement_rate",
            "year_over_year_change",
            "month_over_month_change",
            "roi",
            "statistical_significance",
            "confidence_interval_lower",
            "confidence_interval_upper",
            "sample_size",
            "data_split",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_entity_type_property(self):
        """Test entity_type property."""
        gen = BusinessMetricsGenerator(GeneratorConfig(n_records=10))
        assert gen.entity_type == "business_metrics"


class TestBrandAndRegionCoverage:
    """Test brand and region coverage."""

    def test_all_brands_covered(self):
        """Test all three brands are represented."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()
        brands = df["brand"].unique()

        expected_brands = ["Remibrutinib", "Fabhalta", "Kisqali"]
        for brand in expected_brands:
            assert brand in brands, f"Missing brand: {brand}"

    def test_all_regions_covered(self):
        """Test all regions are represented."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()
        regions = df["region"].unique()

        expected_regions = ["northeast", "south", "midwest", "west"]
        for region in expected_regions:
            assert region in regions, f"Missing region: {region}"

    def test_brand_region_combinations(self):
        """Test brand/region combinations exist."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Check at least some combinations exist
        combinations = df.groupby(["brand", "region"]).size()
        assert len(combinations) >= 8  # At least 8 of 12 possible combinations


class TestMetricTypes:
    """Test metric type generation."""

    def test_metric_types_covered(self):
        """Test all metric types are generated."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()
        metric_types = df["metric_type"].unique()

        # metric_type field contains the metric key (trx, nrx, etc.)
        expected_metrics = ["trx", "nrx", "market_share", "conversion_rate", "hcp_engagement_score"]
        for metric in expected_metrics:
            assert metric in metric_types, f"Missing metric: {metric}"

    def test_metric_type_field_valid(self):
        """Test metric_type field has valid values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # metric_type contains the metric keys, not categories
        valid_types = ["trx", "nrx", "market_share", "conversion_rate", "hcp_engagement_score"]
        for metric_type in df["metric_type"].unique():
            assert metric_type in valid_types, f"Invalid metric_type: {metric_type}"

    def test_metric_name_field_contains_connector_keys(self):
        """Canonical v1.1: metric_name now holds the lowercase gap-connector key.

        gap_analyzer queries business_metrics on metric_name == connector key
        (business_metric.py:79 .eq("metric_name", kpi_name)); the human-readable
        description was a 0-match value and is no longer stored in metric_name.
        """
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()
        metric_names = set(df["metric_name"].unique())

        expected_keys = {
            "trx",
            "nrx",
            "market_share",
            "conversion_rate",
            "hcp_engagement_score",
        }
        assert expected_keys <= metric_names, f"missing keys: {expected_keys - metric_names}"
        # No title-case alias / description rows leaked in.
        assert metric_names <= expected_keys, f"unexpected rows: {metric_names - expected_keys}"


class TestAchievementRateCalculations:
    """Test achievement rate calculations."""

    def test_achievement_rate_range(self):
        """Test achievement rates are in valid range."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Achievement rates should generally be between 0 and 2 (0% to 200%)
        assert df["achievement_rate"].min() >= 0.0
        assert df["achievement_rate"].max() <= 2.5  # Allow some outliers

    def test_achievement_rate_calculation_consistency(self):
        """Test achievement rate is consistent with value/target."""
        config = GeneratorConfig(n_records=200, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # For rows where target is not zero, achievement should be value/target
        valid_rows = df[df["target"] > 0]
        if len(valid_rows) > 0:
            expected = valid_rows["value"] / valid_rows["target"]
            actual = valid_rows["achievement_rate"]
            # Allow for rounding differences (generator rounds to 3 decimals)
            np.testing.assert_array_almost_equal(actual, expected, decimal=2)

    def test_target_values_reasonable(self):
        """Test target values are positive and reasonable."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        assert df["target"].min() > 0
        assert df["value"].min() >= 0


class TestTimeSeriesContinuity:
    """Test time-series continuity."""

    def test_dates_within_range(self):
        """Test metric dates are within configured range."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        config = GeneratorConfig(
            n_records=500,
            seed=42,
            start_date=start,
            end_date=end,
        )
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Convert dates if needed
        dates = pd.to_datetime(df["metric_date"]).dt.date
        assert dates.min() >= start
        assert dates.max() <= end

    def test_date_distribution(self):
        """Test dates are distributed across the range."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        config = GeneratorConfig(
            n_records=1000,
            seed=42,
            start_date=start,
            end_date=end,
        )
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Convert to datetime for month extraction
        df["month"] = pd.to_datetime(df["metric_date"]).dt.month

        # Should have records in multiple months
        unique_months = df["month"].nunique()
        assert unique_months >= 6, "Dates should be distributed across months"

    def test_time_series_per_brand_metric(self):
        """Test time series exists per brand/metric combination."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Group by brand and metric_name
        groups = df.groupby(["brand", "metric_name"]).size()

        # Each major combination should have multiple entries (time series)
        major_combos = groups[groups > 5]
        assert len(major_combos) > 5, (
            "Should have time-series data for multiple brand/metric combos"
        )


class TestDataSplitDistribution:
    """Test data split distribution."""

    def test_data_split_values_valid(self):
        """Test data_split has valid values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # Base generator uses 4 splits: train, validation, test, holdout
        valid_splits = ["train", "validation", "test", "holdout"]
        for split in df["data_split"].unique():
            assert split in valid_splits, f"Invalid split: {split}"

    def test_data_split_approximate_ratios(self):
        """Test data_split follows approximate 60/20/10/10 ratios (#44 policy)."""
        config = GeneratorConfig(n_records=2000, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        split_counts = df["data_split"].value_counts(normalize=True)

        # Base generator uses 60/20/10/10 ratios with tolerance. _assign_splits is a
        # row-share quota (not date-banded), so at n=2000 every split — including the
        # 10% holdout — is reliably present.
        assert abs(split_counts.get("train", 0) - 0.60) < 0.15
        assert abs(split_counts.get("validation", 0) - 0.20) < 0.10
        assert abs(split_counts.get("test", 0) - 0.10) < 0.10
        assert abs(split_counts.get("holdout", 0) - 0.10) < 0.10


class TestStatisticalFields:
    """Test statistical fields."""

    def test_statistical_significance_values(self):
        """Test statistical_significance is a p-value in valid range."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # statistical_significance is a p-value (float between 0.001 and 0.10)
        assert df["statistical_significance"].min() >= 0.0
        assert df["statistical_significance"].max() <= 0.15  # Allow small buffer

    def test_confidence_intervals_ordered(self):
        """Test confidence interval lower <= upper."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        assert (df["confidence_interval_lower"] <= df["confidence_interval_upper"]).all()

    def test_sample_size_positive(self):
        """Test sample_size is positive."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        assert df["sample_size"].min() > 0

    def test_roi_values_reasonable(self):
        """Test ROI values are in reasonable range."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # ROI typically between -1 (100% loss) and 10 (1000% return)
        assert df["roi"].min() >= -1.0
        assert df["roi"].max() <= 15.0


class TestYoYMoMChanges:
    """Test year-over-year and month-over-month changes."""

    def test_yoy_change_range(self):
        """Test YoY change is in reasonable range."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # YoY change typically between -100% and +100%
        assert df["year_over_year_change"].min() >= -1.0
        assert df["year_over_year_change"].max() <= 1.5

    def test_mom_change_range(self):
        """Test MoM change is in reasonable range."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        # MoM change typically smaller than YoY
        assert df["month_over_month_change"].min() >= -0.5
        assert df["month_over_month_change"].max() <= 0.5


class TestReproducibility:
    """Test reproducibility with seed."""

    def test_same_seed_same_results(self):
        """Test same seed produces same results."""
        config1 = GeneratorConfig(n_records=100, seed=42)
        config2 = GeneratorConfig(n_records=100, seed=42)

        gen1 = BusinessMetricsGenerator(config1)
        gen2 = BusinessMetricsGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_results(self):
        """Test different seeds produce different results."""
        config1 = GeneratorConfig(n_records=100, seed=42)
        config2 = GeneratorConfig(n_records=100, seed=123)

        gen1 = BusinessMetricsGenerator(config1)
        gen2 = BusinessMetricsGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Values should be different
        assert not df1["value"].equals(df2["value"])


class TestTrendOrigin:
    """#1566 D1: absolute calendar anchoring of the trend index.

    trend_origin=None (the default) preserves the positional month_idx
    (= dates.index(metric_date)) for all existing callers; when set,
    month_idx = (d.year - origin.year)*12 + (d.month - origin.month).
    """

    # trx model constants (METRIC_CONFIGS): base values, 2%/month trend.
    _TRX_BASE = {"Remibrutinib": 15000.0, "Fabhalta": 8000.0, "Kisqali": 50000.0}
    _REGION = {"northeast": 1.15, "south": 0.95, "midwest": 0.90, "west": 1.00}
    _TRX_TREND = 0.02

    def _first_date_trx_ratio(self, df: pd.DataFrame, trend_factor: float) -> float:
        first = df["metric_date"].min()  # ISO strings sort chronologically
        trx = df[(df["metric_date"] == first) & (df["metric_type"] == "trx")]
        assert len(trx) == 12  # 3 brands x 4 regions
        expected = trx.apply(
            lambda r: self._TRX_BASE[r["brand"]] * self._REGION[r["region"]] * trend_factor,
            axis=1,
        )
        return float((trx["value"] / expected).mean())

    def test_default_none_keeps_positional_first_date_at_factor_one(self):
        # Explicit non-default start_date so _generate_date_range honors it
        # as-is (the 2022 default triggers the recency re-anchor instead).
        config = GeneratorConfig(
            n_records=500,
            seed=42,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
        )
        df = BusinessMetricsGenerator(config).generate()

        # Positional behavior: the run's first date has index 0 -> factor 1.
        ratio = self._first_date_trx_ratio(df, trend_factor=1.0)
        # Per-row noise N(0, 0.15); the 12-row mean must sit near 1.
        assert 0.85 <= ratio <= 1.15, f"positional first-date ratio drifted: {ratio:.3f}"

    def test_trend_origin_24_months_before_start_scales_first_date(self):
        config = GeneratorConfig(
            n_records=500,
            seed=42,
            start_date=date(2024, 1, 1),
            end_date=date(2024, 12, 31),
            trend_origin=date(2022, 1, 1),  # 24 calendar months before start
        )
        df = BusinessMetricsGenerator(config).generate()

        ratio = self._first_date_trx_ratio(df, trend_factor=1.0 + self._TRX_TREND * 24)
        assert 0.85 <= ratio <= 1.15, f"calendar-anchored first-date ratio off: {ratio:.3f}"


class TestMetricIDUniqueness:
    """Test metric ID uniqueness."""

    def test_metric_ids_unique(self):
        """Test metric_id values are unique."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        assert df["metric_id"].nunique() == len(df)

    def test_metric_ids_format(self):
        """Test metric_id values have expected format (metric_{hex})."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        for metric_id in df["metric_id"].head(10):
            # Format: metric_{12 hex characters}
            assert metric_id.startswith("metric_"), (
                f"metric_id should start with 'metric_': {metric_id}"
            )
            hex_part = metric_id[7:]  # Remove "metric_" prefix
            assert len(hex_part) == 12, f"hex part should be 12 chars: {hex_part}"
            # Verify it's valid hex
            try:
                int(hex_part, 16)
            except ValueError:
                pytest.fail(f"Invalid hex in metric_id: {hex_part}")


GAP_KEYS = {"trx", "nrx", "market_share", "conversion_rate", "hcp_engagement_score"}


class TestMetricNameContract:
    def test_emits_gap_connector_keys(self):
        df = BusinessMetricsGenerator(GeneratorConfig(n_records=2000, seed=42)).generate()
        present = set(df["metric_name"].unique())
        assert GAP_KEYS <= present, f"missing gap keys: {GAP_KEYS - present}"

    def test_emits_only_lowercase_connector_keys(self):
        # Canonical v1.1: NO title-case alias rows -- title-case tokens are
        # get_kpi_summary response field keys (kpi_query RPC over treatment_events),
        # not business_metrics.metric_name filters.
        df = BusinessMetricsGenerator(GeneratorConfig(n_records=2000, seed=42)).generate()
        present = set(df["metric_name"].unique())
        assert present <= GAP_KEYS, (
            f"unexpected non-connector metric_name rows: {present - GAP_KEYS}"
        )

    def test_value_and_target_nonnull_and_differ(self):
        df = BusinessMetricsGenerator(GeneratorConfig(n_records=2000, seed=42)).generate()
        sub = df[df["metric_name"] == "trx"]
        assert sub["value"].notna().all() and sub["target"].notna().all()
        assert (sub["value"] != sub["target"]).all(), "target must differ from value"

    def test_covers_three_brands_four_regions(self):
        df = BusinessMetricsGenerator(GeneratorConfig(n_records=2000, seed=42)).generate()
        assert set(df["brand"].unique()) == {"Remibrutinib", "Kisqali", "Fabhalta"}
        assert set(df["region"].unique()) == {"northeast", "south", "midwest", "west"}

    def test_metric_date_is_recent_not_2022(self):
        # Column must be populated with a recent date (rolling mechanism = Shard 04);
        # here just assert it is not the stale 2022 default start.
        df = BusinessMetricsGenerator(GeneratorConfig(n_records=2000, seed=42)).generate()
        years = {d[:4] for d in df["metric_date"].astype(str)}
        assert "2022" not in years, "metric_date still anchored to the 2022 staleness root"


# ---------------------------------------------------------------------------
# #1833: brand x region structure planted on VALUE only (targets stay on the
# market-size trend line); both terms are deterministic lookups that consume
# no RNG, so metric_ids / dates / every other column reproduce byte-for-byte.
# ---------------------------------------------------------------------------

# The frozen-base identity (scripts/load_synthetic_data.py: seed 42, n=10000;
# start pinned to the 2026-07-03 load's first month). #1833 Step 0 measured this
# config byte-identical to all 9,780 DB base rows (16 columns, 0 mismatches).
BASE_CONFIG = {
    "id_prefix": "scv",
    "seed": 42,
    "n_records": 10000,
    "start_date": date(2013, 1, 1),
}
BRANDS = ("Remibrutinib", "Fabhalta", "Kisqali")
REGIONS = ("northeast", "south", "midwest", "west")

# Literal pre-change fingerprints read from the live DB (2026-08-30). metric_id
# is a seeded RNG draw and target is a seeded draw on the market-size line:
# neither may move when brand x region terms are planted on value.
DB_FINGERPRINTS = [
    # (metric_date, brand, region, metric_name, metric_id, target)
    ("2013-01-01", "Remibrutinib", "northeast", "trx", "metric_f9b7c98320d6", 18869.57),
    ("2020-06-01", "Fabhalta", "south", "market_share", "metric_b40527712eab", 0.13),
    ("2026-06-01", "Remibrutinib", "west", "trx", "metric_fccbf69412a6", 72161.66),
    ("2026-07-01", "Kisqali", "midwest", "trx", "metric_2ca0d492f13b", 219010.90),
]


def _flat_frame(monkeypatch) -> pd.DataFrame:
    """The pre-#1833 formula: every brand x region term at identity, no events.

    Patches only the planted lookup TABLES (class constants); the generator code
    path — and therefore its RNG consumption — is the real one.
    """
    monkeypatch.setattr(
        BusinessMetricsGenerator,
        "BRAND_REGION_PERFORMANCE",
        {b: dict.fromkeys(REGIONS, 1.0) for b in BRANDS},
    )
    monkeypatch.setattr(BusinessMetricsGenerator, "BRAND_REGION_EVENTS", ())
    return BusinessMetricsGenerator(GeneratorConfig(**BASE_CONFIG)).generate()


def _later_events_product(brand, region, metric, after) -> float:
    """Product of the factors of events on (brand, region, metric) starting
    strictly after ``after`` — compounding steps are allowed."""
    prod = 1.0
    for ev in BusinessMetricsGenerator.BRAND_REGION_EVENTS:
        if (
            ev.brand == brand
            and ev.region == region
            and metric in ev.metric_types
            and ev.start > after
        ):
            prod *= ev.factor
    return prod


class TestBrandRegionStructure1833:
    def test_value_carries_brand_region_factor_and_target_does_not(self, monkeypatch):
        planted = BusinessMetricsGenerator(GeneratorConfig(**BASE_CONFIG)).generate()
        # Resolve the planted factors BEFORE _flat_frame patches the tables away.
        mask = planted["metric_type"].isin(["trx", "nrx"])
        expected = planted[mask].apply(
            lambda r: BusinessMetricsGenerator.brand_region_factor(
                r["brand"], r["region"], r["metric_type"], date.fromisoformat(r["metric_date"])
            ),
            axis=1,
        )
        flat = _flat_frame(monkeypatch)
        assert list(planted["metric_id"]) == list(flat["metric_id"])
        # target: byte-identical (market-size line only)
        pd.testing.assert_series_equal(planted["target"], flat["target"])
        # value: planted / flat == the deterministic brand x region factor, on
        # the uncapped prescription metrics (market_share / conversion_rate /
        # engagement carry caps that re-quantize the ratio).
        mask &= flat["value"] > 0
        expected = expected[mask.loc[expected.index]]
        ratio = planted.loc[mask, "value"] / flat.loc[mask, "value"]
        assert np.allclose(ratio, expected, rtol=1e-3), "value does not carry the planted factor"
        # The plant is real: at least one brand x region pair differs from 1.
        assert (expected != 1.0).any()

    def test_rng_stream_untouched_literal_db_fingerprints(self):
        df = BusinessMetricsGenerator(GeneratorConfig(**BASE_CONFIG)).generate()
        assert len(df) == 9780
        for metric_date, brand, region, metric_name, metric_id, target in DB_FINGERPRINTS:
            row = df[
                (df["metric_date"] == metric_date)
                & (df["brand"] == brand)
                & (df["region"] == region)
                & (df["metric_name"] == metric_name)
            ]
            assert len(row) == 1, (metric_date, brand, region, metric_name)
            assert row["metric_id"].iloc[0] == metric_id, "metric_id moved: RNG stream consumed"
            assert row["target"].iloc[0] == pytest.approx(target, abs=0.005), "target moved"

    def test_rng_only_columns_identical_to_flat_formula(self, monkeypatch):
        planted = BusinessMetricsGenerator(GeneratorConfig(**BASE_CONFIG)).generate()
        flat = _flat_frame(monkeypatch)
        for col in (
            "metric_id",
            "metric_date",
            "target",
            "year_over_year_change",
            "month_over_month_change",
            "roi",
            "statistical_significance",
            "sample_size",
            "data_split",
        ):
            pd.testing.assert_series_equal(planted[col], flat[col]), col

    def test_per_brand_national_scale_within_3pct(self, monkeypatch):
        planted = BusinessMetricsGenerator(GeneratorConfig(**BASE_CONFIG)).generate()
        flat = _flat_frame(monkeypatch)
        last = planted["metric_date"].max()
        for brand in BRANDS:
            for metric in ("trx", "nrx"):
                p = planted[(planted["brand"] == brand) & (planted["metric_type"] == metric)]
                f = flat[(flat["brand"] == brand) & (flat["metric_type"] == metric)]
                ratio_all = p["value"].sum() / f["value"].sum()
                assert 0.97 <= ratio_all <= 1.03, (brand, metric, "all", ratio_all)
                # The frontier month carries each brand's FIRST planted step
                # (-12% in a region that is <=25% of national under
                # REGION_FACTORS -> <=-3.0% national by design); the #1640
                # substrate note re-measures this magnitude. On a single
                # month the planted region's share of national swings with
                # its own row noise (N(0, 0.15) trx / N(0, 0.20) nrx), so the
                # realised step effect reads -2..-4.5% (measured: Fabhalta
                # nrx 2026-07 at 0.957).
                ratio_last = (
                    p[p["metric_date"] == last]["value"].sum()
                    / f[f["metric_date"] == last]["value"].sum()
                )
                assert 0.95 <= ratio_last <= 1.03, (brand, metric, last, ratio_last)

    def test_performance_matrix_is_market_size_weighted_mean_one(self):
        # National scale stays ~unchanged BY CONSTRUCTION: each brand's execution
        # factors average to 1 under the REGION_FACTORS market-size weights.
        w = BusinessMetricsGenerator.REGION_FACTORS
        assert set(BusinessMetricsGenerator.BRAND_REGION_PERFORMANCE) == set(BRANDS)
        for brand, row in BusinessMetricsGenerator.BRAND_REGION_PERFORMANCE.items():
            assert set(row) == set(w), brand
            weighted = sum(row[r] * w[r] for r in w) / sum(w.values())
            assert abs(weighted - 1.0) <= 0.01, (brand, weighted)

    def test_each_brand_has_a_distinct_planted_region(self):
        # The point of #1833: the weakest execution region differs per brand.
        weakest = {
            brand: min(row, key=row.get)
            for brand, row in BusinessMetricsGenerator.BRAND_REGION_PERFORMANCE.items()
        }
        assert len(set(weakest.values())) == 3, weakest
        # and every event lands in its brand's weakest region
        for ev in BusinessMetricsGenerator.BRAND_REGION_EVENTS:
            assert ev.region == weakest[ev.brand], ev

    def test_event_is_a_step_that_never_reverts(self):
        events = BusinessMetricsGenerator.BRAND_REGION_EVENTS
        assert events, "no planted events"
        for ev in events:
            assert 0 < ev.factor < 1, ev  # a shortfall the gap analyzer can rank
            assert ev.start.day == 1, ev  # rows sit on month starts
            day_before = date.fromordinal(ev.start.toordinal() - 1)
            for metric in ev.metric_types:
                f_before = BusinessMetricsGenerator.brand_region_factor(
                    ev.brand, ev.region, metric, day_before
                )
                f_at = BusinessMetricsGenerator.brand_region_factor(
                    ev.brand, ev.region, metric, ev.start
                )
                f_far = BusinessMetricsGenerator.brand_region_factor(
                    ev.brand, ev.region, metric, date(2099, 1, 1)
                )
                assert f_at == pytest.approx(f_before * ev.factor)
                assert f_far == pytest.approx(
                    f_at * _later_events_product(ev.brand, ev.region, metric, ev.start)
                )

    def test_events_do_not_touch_other_metrics(self):
        for ev in BusinessMetricsGenerator.BRAND_REGION_EVENTS:
            for metric in BusinessMetricsGenerator.METRIC_CONFIGS:
                if metric in ev.metric_types:
                    continue
                perf = BusinessMetricsGenerator.BRAND_REGION_PERFORMANCE[ev.brand][ev.region]
                got = BusinessMetricsGenerator.brand_region_factor(
                    ev.brand, ev.region, metric, date(2099, 1, 1)
                )
                assert got == pytest.approx(
                    perf * _later_events_product(ev.brand, ev.region, metric, date(1900, 1, 1))
                )

    def test_unknown_brand_or_region_is_identity(self):
        assert (
            BusinessMetricsGenerator.brand_region_factor(
                "competitor", "west", "trx", date(2099, 1, 1)
            )
            == 1.0
        )
        assert (
            BusinessMetricsGenerator.brand_region_factor("Kisqali", "mars", "trx", date(2099, 1, 1))
            == 1.0
        )
