"""Tests for Business Metrics Generator.

Tests metric generation per brand/region, achievement rate calculations,
time-series continuity, and data split distribution.
"""

from datetime import date, timedelta

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

    def test_metric_name_field_contains_descriptions(self):
        """Test metric_name field contains human-readable descriptions."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()
        metric_names = df["metric_name"].unique()

        # metric_name contains the description
        expected_descriptions = [
            "Total Prescriptions",
            "New Prescriptions",
            "Market Share Percentage",
            "HCP Conversion Rate",
            "HCP Engagement Score (0-10)",
        ]
        for desc in expected_descriptions:
            assert desc in metric_names, f"Missing description: {desc}"


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
        assert len(major_combos) > 5, "Should have time-series data for multiple brand/metric combos"


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
        """Test data_split follows approximate 60/20/15/5 ratios."""
        config = GeneratorConfig(n_records=2000, seed=42)
        gen = BusinessMetricsGenerator(config)

        df = gen.generate()

        split_counts = df["data_split"].value_counts(normalize=True)

        # Base generator uses 60/20/15/5 ratios with tolerance
        assert abs(split_counts.get("train", 0) - 0.60) < 0.15
        assert abs(split_counts.get("validation", 0) - 0.20) < 0.10
        assert abs(split_counts.get("test", 0) - 0.15) < 0.10
        # holdout is small (5%), may or may not be present depending on date distribution


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
            assert metric_id.startswith("metric_"), f"metric_id should start with 'metric_': {metric_id}"
            hex_part = metric_id[7:]  # Remove "metric_" prefix
            assert len(hex_part) == 12, f"hex part should be 12 chars: {hex_part}"
            # Verify it's valid hex
            try:
                int(hex_part, 16)
            except ValueError:
                pytest.fail(f"Invalid hex in metric_id: {hex_part}")
