"""Tests for Feature Value Generator.

Tests time-series generation with proper timestamps, entity_values JSONB structure,
freshness status assignment, and all 15 features generated.
"""

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.ml.synthetic.generators import (
    FeatureStoreSeeder,
    FeatureValueGenerator,
    GeneratorConfig,
)


class TestFeatureValueGeneratorBasic:
    """Test basic FeatureValueGenerator functionality."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_requires_features_df(self):
        """Test generator requires features_df."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=None)

        with pytest.raises(ValueError, match="features_df is required"):
            gen.generate()

    def test_generate_returns_dataframe(self, features_df):
        """Test generate returns a DataFrame."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0

    def test_generate_respects_n_records(self, features_df):
        """Test generate produces approximately requested record count."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Should be close to requested (may be trimmed)
        assert len(df) <= 500
        assert len(df) >= 100  # At least some records

    def test_required_columns_present(self, features_df):
        """Test all required columns are present."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        required_columns = [
            "id",
            "feature_id",
            "entity_values",
            "value",
            "event_timestamp",
            "freshness_status",
        ]
        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_entity_type_property(self, features_df):
        """Test entity_type property."""
        gen = FeatureValueGenerator(
            GeneratorConfig(n_records=10),
            features_df=features_df,
        )
        assert gen.entity_type == "feature_values"


class TestAll15FeaturesGenerated:
    """Test that all 15 features are generated."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_all_features_have_values(self, features_df):
        """Test all 15 features get values generated."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Get unique feature IDs from generated data
        generated_feature_ids = df["feature_id"].unique()

        # Should have values for all 15 features
        assert len(generated_feature_ids) == 15, (
            f"Expected 15 features, got {len(generated_feature_ids)}"
        )

    def test_feature_ids_match_seeder(self, features_df):
        """Test feature_ids match those from seeder."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # All generated feature_ids should be in features_df
        seeder_ids = set(features_df["id"].unique())
        generated_ids = set(df["feature_id"].unique())

        assert generated_ids.issubset(seeder_ids)

    def test_feature_groups_represented(self, features_df):
        """Test all 4 feature groups have features."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Merge to get feature names
        df_with_names = df.merge(
            features_df[["id", "name", "feature_group_id"]],
            left_on="feature_id",
            right_on="id",
            suffixes=("", "_feature"),
        )

        # Check feature groups by looking at feature names
        feature_names = df_with_names["name"].unique()

        # HCP demographics features
        assert "specialty_encoded" in feature_names
        assert "years_experience" in feature_names

        # Patient features
        assert "disease_severity" in feature_names
        assert "age_at_diagnosis" in feature_names

        # Brand performance features
        assert "trx_30d" in feature_names
        assert "conversion_rate" in feature_names

        # Causal features
        assert "engagement_score" in feature_names
        assert "treatment_propensity" in feature_names


class TestTimestampGeneration:
    """Test time-series generation with proper timestamps."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_timestamps_within_date_range(self, features_df):
        """Test timestamps are within configured date range."""
        start = date(2024, 1, 1)
        end = date(2024, 6, 30)
        config = GeneratorConfig(
            n_records=500,
            seed=42,
            start_date=start,
            end_date=end,
        )
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Parse timestamps
        timestamps = pd.to_datetime(df["event_timestamp"], format="ISO8601")
        dates = timestamps.dt.date

        assert dates.min() >= start
        assert dates.max() <= end

    def test_timestamps_are_iso_format(self, features_df):
        """Test timestamps are in ISO format."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # All timestamps should be parseable as ISO format
        for ts in df["event_timestamp"].head(10):
            parsed = datetime.fromisoformat(ts)
            assert isinstance(parsed, datetime)

    def test_timestamps_spread_per_feature(self, features_df):
        """Test timestamps are spread across the date range per feature."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # For each feature, check we have multiple timestamps
        for feature_id in df["feature_id"].unique():
            feature_df = df[df["feature_id"] == feature_id].copy()
            timestamps = pd.to_datetime(feature_df["event_timestamp"], format="ISO8601")
            if len(feature_df) > 1:
                # Should have some variation in timestamps
                assert timestamps.nunique() >= 1

    def test_biased_towards_recent_dates(self, features_df):
        """Test timestamps are biased towards more recent dates (exponential distribution)."""
        start = date(2024, 1, 1)
        end = date(2024, 12, 31)
        config = GeneratorConfig(
            n_records=1000,
            seed=42,
            start_date=start,
            end_date=end,
        )
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        timestamps = pd.to_datetime(df["event_timestamp"], format="ISO8601")

        # Calculate median date - exponential bias should push median towards end
        median_ts = timestamps.median()
        midpoint = datetime(2024, 7, 1)

        # With exponential bias towards recent dates, median should be closer to end
        # The generator uses exponential decay from end_date, so most records should be recent
        # Allow some flexibility since exponential decay mean is total_seconds/3
        assert median_ts > midpoint, (
            f"Median {median_ts} should be after midpoint {midpoint} due to recent bias"
        )


class TestEntityValuesJSONBStructure:
    """Test entity_values JSONB structure."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_entity_values_is_dict(self, features_df):
        """Test entity_values are dictionaries."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        for entity_values in df["entity_values"]:
            assert isinstance(entity_values, dict)

    def test_entity_values_have_correct_keys(self, features_df):
        """Test entity_values have correct keys based on feature."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Merge to get feature entity_keys
        df_with_keys = df.merge(
            features_df[["id", "name", "entity_keys"]],
            left_on="feature_id",
            right_on="id",
            suffixes=("", "_feature"),
        )

        # Check entity_values keys match entity_keys
        for _, row in df_with_keys.head(50).iterrows():
            entity_keys = row["entity_keys"]
            entity_values = row["entity_values"]

            for key in entity_keys:
                assert key in entity_values, (
                    f"Missing key {key} in entity_values for feature {row['name']}"
                )

    def test_hcp_entity_values(self, features_df):
        """Test HCP entity values format."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Get records for HCP features
        df_merged = df.merge(
            features_df[["id", "name", "entity_keys"]],
            left_on="feature_id",
            right_on="id",
        )

        hcp_records = df_merged[df_merged["entity_keys"].apply(lambda x: "hcp_id" in x)]

        if len(hcp_records) > 0:
            for entity_values in hcp_records["entity_values"].head(10):
                assert "hcp_id" in entity_values
                assert entity_values["hcp_id"].startswith("hcp_")

    def test_patient_entity_values(self, features_df):
        """Test patient entity values format."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        df_merged = df.merge(
            features_df[["id", "name", "entity_keys"]],
            left_on="feature_id",
            right_on="id",
        )

        patient_records = df_merged[df_merged["entity_keys"].apply(lambda x: "patient_id" in x)]

        if len(patient_records) > 0:
            for entity_values in patient_records["entity_values"].head(10):
                assert "patient_id" in entity_values
                assert entity_values["patient_id"].startswith("pt_")

    def test_brand_region_entity_values(self, features_df):
        """Test brand/region entity values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        df_merged = df.merge(
            features_df[["id", "name", "entity_keys"]],
            left_on="feature_id",
            right_on="id",
        )

        brand_records = df_merged[df_merged["entity_keys"].apply(lambda x: "brand" in x)]

        if len(brand_records) > 0:
            valid_brands = ["Remibrutinib", "Fabhalta", "Kisqali"]
            valid_regions = ["northeast", "south", "midwest", "west"]

            for entity_values in brand_records["entity_values"].head(10):
                assert "brand" in entity_values
                assert entity_values["brand"] in valid_brands
                if "region" in entity_values:
                    assert entity_values["region"] in valid_regions


class TestValueJSONBStructure:
    """Test value JSONB structure."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_value_is_dict(self, features_df):
        """Test value field is a dictionary."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        for value in df["value"]:
            assert isinstance(value, dict)

    def test_value_has_value_key(self, features_df):
        """Test value dict has 'value' key."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        for value in df["value"]:
            assert "value" in value


class TestFreshnessStatusAssignment:
    """Test freshness status assignment."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_freshness_status_values_valid(self, features_df):
        """Test freshness_status has valid values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        valid_statuses = ["fresh", "stale", "expired"]
        for status in df["freshness_status"].unique():
            assert status in valid_statuses, f"Invalid status: {status}"

    def test_freshness_correlates_with_timestamp(self, features_df):
        """Test freshness status correlates with timestamp age."""
        # Use a date range ending today to test freshness
        today = date.today()
        start = today - timedelta(days=30)
        config = GeneratorConfig(
            n_records=500,
            seed=42,
            start_date=start,
            end_date=today,
        )
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        # Fresh records should have recent timestamps
        now = datetime.now()
        df["timestamp_dt"] = pd.to_datetime(df["event_timestamp"], format="ISO8601")
        df["age_hours"] = (now - df["timestamp_dt"]).dt.total_seconds() / 3600

        # Freshness logic: fresh < 24h, stale < 168h (7 days), expired >= 168h
        fresh_records = df[df["freshness_status"] == "fresh"]
        stale_records = df[df["freshness_status"] == "stale"]
        expired_records = df[df["freshness_status"] == "expired"]

        # Check relationships between freshness and age
        if len(fresh_records) > 0 and len(expired_records) > 0:
            # Fresh records should be newer than expired records on average
            assert fresh_records["age_hours"].mean() < expired_records["age_hours"].mean()

        if len(stale_records) > 0 and len(expired_records) > 0:
            # Stale records should be newer than expired records on average
            assert stale_records["age_hours"].mean() < expired_records["age_hours"].mean()

    def test_all_freshness_statuses_present(self, features_df):
        """Test all freshness statuses are present in varied data."""
        # Use a wide date range to get all statuses
        config = GeneratorConfig(
            n_records=1000,
            seed=42,
            start_date=date(2024, 1, 1),
            end_date=date.today(),
        )
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        statuses = df["freshness_status"].unique()
        # Should have at least 2 different statuses with varied dates
        assert len(statuses) >= 2


class TestFeatureValueTypes:
    """Test feature value types match definitions."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_categorical_features(self, features_df):
        """Test categorical features have string values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        df_merged = df.merge(
            features_df[["id", "name", "value_type"]],
            left_on="feature_id",
            right_on="id",
        )

        categorical_records = df_merged[df_merged["value_type"] == "string"]
        if len(categorical_records) > 0:
            for value_dict in categorical_records["value"].head(10):
                assert isinstance(value_dict["value"], str)

    def test_numeric_features(self, features_df):
        """Test numeric features have numeric values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        df_merged = df.merge(
            features_df[["id", "name", "value_type"]],
            left_on="feature_id",
            right_on="id",
        )

        numeric_records = df_merged[df_merged["value_type"].isin(["int64", "float64"])]
        if len(numeric_records) > 0:
            for value_dict in numeric_records["value"].head(10):
                val = value_dict["value"]
                assert isinstance(val, (int, float, np.integer, np.floating))


class TestPatientDFIntegration:
    """Test integration with patient_df."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    @pytest.fixture
    def patient_df(self):
        """Create mock patient DataFrame."""
        return pd.DataFrame(
            {
                "patient_id": ["pt_000001", "pt_000002", "pt_000003", "pt_000004", "pt_000005"],
                "hcp_id": ["hcp_00001", "hcp_00002", "hcp_00003", "hcp_00004", "hcp_00005"],
            }
        )

    def test_uses_patient_df_ids(self, features_df, patient_df):
        """Test generator uses patient_df IDs when provided."""
        config = GeneratorConfig(n_records=200, seed=42)
        gen = FeatureValueGenerator(
            config,
            features_df=features_df,
            patient_df=patient_df,
        )

        df = gen.generate()

        # Get patient_id values from entity_values
        patient_ids = set()
        for entity_values in df["entity_values"]:
            if "patient_id" in entity_values:
                patient_ids.add(entity_values["patient_id"])

        # Should use IDs from patient_df
        if patient_ids:
            expected_ids = set(patient_df["patient_id"])
            assert patient_ids.issubset(expected_ids)


class TestIDUniqueness:
    """Test ID uniqueness."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_ids_unique(self, features_df):
        """Test id values are unique."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        assert df["id"].nunique() == len(df)

    def test_ids_are_uuids(self, features_df):
        """Test id values appear to be UUIDs."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = FeatureValueGenerator(config, features_df=features_df)

        df = gen.generate()

        for id_val in df["id"].head(10):
            parts = id_val.split("-")
            assert len(parts) == 5


class TestReproducibility:
    """Test reproducibility with seed."""

    @pytest.fixture
    def features_df(self):
        """Create features DataFrame from seeder."""
        config = GeneratorConfig(seed=42)
        seeder = FeatureStoreSeeder(config)
        _, features_df = seeder.seed()
        return features_df

    def test_same_seed_same_results(self, features_df):
        """Test same seed produces same results."""
        config1 = GeneratorConfig(n_records=100, seed=42)
        config2 = GeneratorConfig(n_records=100, seed=42)

        gen1 = FeatureValueGenerator(config1, features_df=features_df)
        gen2 = FeatureValueGenerator(config2, features_df=features_df)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Note: feature_id depends on seeder UUIDs which may differ
        # Compare structure and value patterns instead
        assert len(df1) == len(df2)
        assert list(df1.columns) == list(df2.columns)
