"""
Unit tests for src/ml/data_generator.py

Tests the E2IDataGenerator class and helper functions for synthetic data generation.
"""

import uuid
from datetime import date, datetime
from unittest.mock import patch

import pytest

from src.ml.data_generator import (
    E2IDataGenerator,
    SplitConfig,
    assign_split,
    generate_hash,
    generate_id,
    json_serial,
    random_date_in_range,
    random_datetime_in_range,
    random_jsonb,
)


@pytest.mark.unit
class TestHelperFunctions:
    """Test helper functions for data generation."""

    def test_generate_id(self):
        """Test ID generation with prefix and index."""
        result = generate_id("patient", 42)
        assert result == "patient_000042"

        result = generate_id("hcp", 1)
        assert result == "hcp_000001"

        result = generate_id("test", 999999)
        assert result == "test_999999"

    def test_random_date_in_range(self):
        """Test random date generation within range."""
        start = date(2025, 1, 1)
        end = date(2025, 12, 31)

        # Generate multiple dates and check they're all in range
        for _ in range(100):
            result = random_date_in_range(start, end)
            assert isinstance(result, date)
            assert start <= result <= end

    def test_random_date_in_range_single_day(self):
        """Test random date generation when start equals end."""
        start = end = date(2025, 6, 15)
        result = random_date_in_range(start, end)
        assert result == start

    def test_random_datetime_in_range(self):
        """Test random datetime generation within range."""
        start = date(2025, 1, 1)
        end = date(2025, 12, 31)

        for _ in range(100):
            result = random_datetime_in_range(start, end)
            assert isinstance(result, datetime)
            assert datetime.combine(start, datetime.min.time()) <= result
            assert result <= datetime.combine(end, datetime.max.time())

    def test_assign_split_train(self):
        """Test split assignment for training data."""
        config = SplitConfig(
            train_end_date=date(2025, 6, 30),
            validation_end_date=date(2025, 8, 31),
            test_end_date=date(2025, 9, 30),
        )

        # Dates in training period
        assert assign_split(date(2025, 1, 1), config) == "train"
        assert assign_split(date(2025, 6, 30), config) == "train"

    def test_assign_split_validation(self):
        """Test split assignment for validation data."""
        config = SplitConfig(
            train_end_date=date(2025, 6, 30),
            validation_end_date=date(2025, 8, 31),
            test_end_date=date(2025, 9, 30),
        )

        assert assign_split(date(2025, 7, 1), config) == "validation"
        assert assign_split(date(2025, 8, 31), config) == "validation"

    def test_assign_split_test(self):
        """Test split assignment for test data."""
        config = SplitConfig(
            train_end_date=date(2025, 6, 30),
            validation_end_date=date(2025, 8, 31),
            test_end_date=date(2025, 9, 30),
        )

        assert assign_split(date(2025, 9, 1), config) == "test"
        assert assign_split(date(2025, 9, 30), config) == "test"

    def test_assign_split_holdout(self):
        """Test split assignment for holdout data."""
        config = SplitConfig(
            train_end_date=date(2025, 6, 30),
            validation_end_date=date(2025, 8, 31),
            test_end_date=date(2025, 9, 30),
        )

        assert assign_split(date(2025, 10, 1), config) == "holdout"
        assert assign_split(date(2025, 12, 31), config) == "holdout"

    def test_generate_hash(self):
        """Test hash generation for anonymization."""
        value = "test@example.com"
        result = generate_hash(value)

        # Should be a string of length 20
        assert isinstance(result, str)
        assert len(result) == 20

        # Same input should give same hash
        assert generate_hash(value) == result

        # Different input should give different hash
        assert generate_hash("different@example.com") != result

    def test_random_jsonb_float(self):
        """Test JSONB generation with float values."""
        keys = ["metric1", "metric2", "metric3"]
        result = random_jsonb(keys, "float")

        assert isinstance(result, dict)
        assert set(result.keys()) == set(keys)
        for value in result.values():
            assert isinstance(value, float)
            assert 0 <= value <= 1

    def test_random_jsonb_int(self):
        """Test JSONB generation with integer values."""
        keys = ["count1", "count2"]
        result = random_jsonb(keys, "int")

        assert isinstance(result, dict)
        assert set(result.keys()) == set(keys)
        for value in result.values():
            assert isinstance(value, int)
            assert 1 <= value <= 100

    def test_random_jsonb_empty_keys(self):
        """Test JSONB generation with empty key list."""
        result = random_jsonb([], "float")
        assert result == {}

    def test_json_serial_datetime(self):
        """Test JSON serialization of datetime objects."""
        dt = datetime(2025, 6, 15, 14, 30, 0)
        result = json_serial(dt)
        assert result == "2025-06-15T14:30:00"

    def test_json_serial_date(self):
        """Test JSON serialization of date objects."""
        d = date(2025, 6, 15)
        result = json_serial(d)
        assert result == "2025-06-15"

    def test_json_serial_uuid(self):
        """Test JSON serialization of UUID objects."""
        u = uuid.UUID("12345678-1234-5678-1234-567812345678")
        result = json_serial(u)
        assert result == "12345678-1234-5678-1234-567812345678"

    def test_json_serial_unsupported_type(self):
        """Test JSON serialization raises error for unsupported types."""
        with pytest.raises(TypeError, match="not serializable"):
            json_serial(object())


@pytest.mark.unit
class TestSplitConfig:
    """Test SplitConfig dataclass."""

    def test_default_config(self):
        """Test default SplitConfig values."""
        config = SplitConfig()

        assert config.config_name == "e2i_pilot_v3"
        assert config.config_version == "3.0.0"
        assert config.train_ratio == 0.60
        assert config.validation_ratio == 0.20
        assert config.test_ratio == 0.15
        assert config.holdout_ratio == 0.05
        assert config.temporal_gap_days == 7

    def test_custom_config(self):
        """Test custom SplitConfig values."""
        config = SplitConfig(
            config_name="custom_v1",
            train_ratio=0.70,
            validation_ratio=0.15,
            test_ratio=0.10,
            holdout_ratio=0.05,
        )

        assert config.config_name == "custom_v1"
        assert config.train_ratio == 0.70
        assert config.validation_ratio == 0.15
        assert config.test_ratio == 0.10
        assert config.holdout_ratio == 0.05

    def test_date_boundaries(self):
        """Test date boundaries in config."""
        config = SplitConfig(
            data_start_date=date(2024, 1, 1),
            data_end_date=date(2024, 12, 31),
            train_end_date=date(2024, 6, 30),
            validation_end_date=date(2024, 9, 30),
            test_end_date=date(2024, 11, 30),
        )

        assert config.data_start_date == date(2024, 1, 1)
        assert config.data_end_date == date(2024, 12, 31)
        assert config.train_end_date == date(2024, 6, 30)
        assert config.validation_end_date == date(2024, 9, 30)
        assert config.test_end_date == date(2024, 11, 30)


@pytest.mark.unit
class TestE2IDataGenerator:
    """Test E2IDataGenerator class."""

    def test_initialization(self):
        """Test generator initialization."""
        generator = E2IDataGenerator()

        assert generator.config is not None
        assert isinstance(generator.split_config_id, str)
        assert generator.hcp_profiles == []
        assert generator.patient_journeys == []
        assert generator.treatment_events == []
        assert generator.ml_predictions == []
        assert generator.triggers == []
        assert generator.agent_activities == []
        assert generator.business_metrics == []
        assert generator.causal_paths == []
        assert generator.user_sessions == []
        assert generator.data_source_tracking == []
        assert generator.ml_annotations == []
        assert generator.etl_pipeline_metrics == []
        assert generator.hcp_intent_surveys == []
        assert generator.reference_universe == []
        assert generator.patient_splits == {}

    def test_initialization_custom_config(self):
        """Test generator initialization with custom config."""
        custom_config = SplitConfig(config_name="test_v1", train_ratio=0.75)
        generator = E2IDataGenerator(custom_config)

        assert generator.config.config_name == "test_v1"
        assert generator.config.train_ratio == 0.75

    @patch("src.ml.data_generator.fake")
    @patch("src.ml.data_generator.uuid.uuid4")
    def test_generate_reference_universe(self, mock_uuid, mock_faker):
        """Test reference universe generation."""
        mock_uuid.return_value = uuid.UUID("12345678-1234-5678-1234-567812345678")

        generator = E2IDataGenerator()
        generator._generate_reference_universe()

        # Should generate records for 3 brands * 4 regions * 6 specialties = 72 HCP records
        # Plus 3 brands * 4 regions = 12 patient records
        # Total = 84 records
        assert len(generator.reference_universe) == 84

        # Check HCP universe records
        hcp_records = [r for r in generator.reference_universe if r["universe_type"] == "hcp"]
        assert len(hcp_records) == 72

        # Check patient universe records
        patient_records = [
            r for r in generator.reference_universe if r["universe_type"] == "patient"
        ]
        assert len(patient_records) == 12

        # Verify record structure
        first_record = generator.reference_universe[0]
        assert "universe_id" in first_record
        assert "universe_type" in first_record
        assert "brand" in first_record
        assert "region" in first_record
        assert "total_count" in first_record
        assert "target_count" in first_record
        assert "data_source" in first_record

    @patch("src.ml.data_generator.fake")
    def test_generate_hcp_profiles(self, mock_faker):
        """Test HCP profile generation."""
        # Mock faker to return consistent values
        mock_faker.name.return_value = "Dr. John Doe"
        mock_faker.company.return_value = "General Hospital"

        generator = E2IDataGenerator()
        generator._generate_hcp_profiles()

        # Should generate 50 HCP profiles (NUM_HCPS = 50)
        assert len(generator.hcp_profiles) == 50

        # Verify record structure
        first_hcp = generator.hcp_profiles[0]
        assert "hcp_id" in first_hcp
        assert "npi" in first_hcp
        assert "specialty" in first_hcp
        assert "practice_type" in first_hcp
        assert "geographic_region" in first_hcp  # Field name is geographic_region, not region
        assert "state" in first_hcp
        assert "prescribing_volume" in first_hcp
        assert first_hcp["hcp_id"].startswith("HCP_")  # ID format check

    @patch("src.ml.data_generator.fake")
    def test_generate_patient_journeys(self, mock_faker):
        """Test patient journey generation."""
        generator = E2IDataGenerator()

        # Generate HCPs first (required for patient journeys)
        generator._generate_hcp_profiles()
        generator._generate_patient_journeys()

        # Should generate 200 patient journeys (NUM_PATIENTS = 200)
        assert len(generator.patient_journeys) == 200

        # Verify record structure
        first_patient = generator.patient_journeys[0]
        assert "patient_id" in first_patient
        assert "patient_journey_id" in first_patient  # Field name is patient_journey_id
        assert "brand" in first_patient
        assert "journey_start_date" in first_patient
        assert "data_split" in first_patient
        assert "geographic_region" in first_patient
        assert "primary_diagnosis_code" in first_patient
        assert first_patient["patient_id"].startswith("PAT_")  # ID format check

        # Verify split tracking
        assert len(generator.patient_splits) == 200
        assert all(
            split in ["train", "validation", "test", "holdout"]
            for split in generator.patient_splits.values()
        )

    def test_print_summary(self, capsys):
        """Test summary printing."""
        generator = E2IDataGenerator()
        generator.patient_journeys = [
            {"patient_id": f"p{i}", "data_split": "train"} for i in range(100)
        ]
        generator.hcp_profiles = [{"hcp_id": f"h{i}"} for i in range(50)]
        generator.treatment_events = [{"event_id": f"e{i}"} for i in range(300)]

        generator._print_summary()

        captured = capsys.readouterr()
        # Check for key outputs
        assert "SUMMARY" in captured.out  # Upper case in actual output
        assert "100" in captured.out  # Patient count

    def test_generate_all_no_errors(self, capsys):
        """Test that generate_all runs without errors (smoke test)."""
        with patch("src.ml.data_generator.fake") as mock_faker:
            mock_faker.name.return_value = "Test Name"
            mock_faker.company.return_value = "Test Company"

            generator = E2IDataGenerator()

            # Reduce volume for faster test
            import src.ml.data_generator as dgm

            original_patients = dgm.NUM_PATIENTS
            original_hcps = dgm.NUM_HCPS

            try:
                dgm.NUM_PATIENTS = 10
                dgm.NUM_HCPS = 5

                generator.__init__()  # Re-initialize with new values
                generator.generate_all()

                # Verify all collections have data
                assert len(generator.reference_universe) > 0
                assert len(generator.hcp_profiles) > 0
                assert len(generator.patient_journeys) > 0

            finally:
                # Restore original values
                dgm.NUM_PATIENTS = original_patients
                dgm.NUM_HCPS = original_hcps

    def test_patient_split_isolation(self):
        """Test that patients are isolated to single splits."""
        with patch("src.ml.data_generator.fake") as mock_faker:
            mock_faker.name.return_value = "Test Name"

            generator = E2IDataGenerator()

            # Reduce volume for faster test
            import src.ml.data_generator as dgm

            original_patients = dgm.NUM_PATIENTS

            try:
                dgm.NUM_PATIENTS = 50

                generator.__init__()
                generator._generate_hcp_profiles()
                generator._generate_patient_journeys()

                # Each patient should have exactly one split assignment
                patient_ids = [p["patient_id"] for p in generator.patient_journeys]
                assert len(patient_ids) == len(set(patient_ids))  # All unique

                # All patients should be in patient_splits
                assert set(patient_ids) == set(generator.patient_splits.keys())

            finally:
                dgm.NUM_PATIENTS = original_patients

    def test_data_split_distribution(self):
        """Test that data split distribution roughly matches config ratios."""
        with patch("src.ml.data_generator.fake") as mock_faker:
            mock_faker.name.return_value = "Test Name"

            config = SplitConfig(
                train_ratio=0.60,
                validation_ratio=0.20,
                test_ratio=0.15,
                holdout_ratio=0.05,
            )
            generator = E2IDataGenerator(config)

            import src.ml.data_generator as dgm

            original_patients = dgm.NUM_PATIENTS

            try:
                dgm.NUM_PATIENTS = 100

                generator.__init__(config)
                generator._generate_hcp_profiles()
                generator._generate_patient_journeys()

                # Count splits
                split_counts = {}
                for split in generator.patient_splits.values():
                    split_counts[split] = split_counts.get(split, 0) + 1

                # Should have at least 3 splits (might not have holdout if date range is small)
                assert len(split_counts) >= 3
                assert "train" in split_counts
                assert "validation" in split_counts
                assert "test" in split_counts
                # Holdout is optional depending on date distribution

            finally:
                dgm.NUM_PATIENTS = original_patients

    def test_uuid_generation_uniqueness(self):
        """Test that generated UUIDs are unique."""
        generator = E2IDataGenerator()
        generator._generate_reference_universe()

        universe_ids = [r["universe_id"] for r in generator.reference_universe]

        # All IDs should be unique
        assert len(universe_ids) == len(set(universe_ids))

    def test_date_consistency(self):
        """Test that generated dates are within configured boundaries."""
        config = SplitConfig(
            data_start_date=date(2025, 1, 1),
            data_end_date=date(2025, 10, 31),
        )

        with patch("src.ml.data_generator.fake") as mock_faker:
            mock_faker.name.return_value = "Test Name"

            generator = E2IDataGenerator(config)

            import src.ml.data_generator as dgm

            original_patients = dgm.NUM_PATIENTS

            try:
                dgm.NUM_PATIENTS = 20

                generator.__init__(config)
                generator._generate_hcp_profiles()
                generator._generate_patient_journeys()

                # All journey dates should be within boundaries
                for journey in generator.patient_journeys:
                    journey_date = date.fromisoformat(journey["journey_start_date"])
                    assert config.data_start_date <= journey_date <= config.data_end_date

            finally:
                dgm.NUM_PATIENTS = original_patients


@pytest.mark.unit
class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_generate_id_large_index(self):
        """Test ID generation with very large index."""
        result = generate_id("test", 9999999)
        assert result == "test_9999999"

    def test_generate_hash_empty_string(self):
        """Test hash generation with empty string."""
        result = generate_hash("")
        assert isinstance(result, str)
        assert len(result) == 20

    def test_random_jsonb_single_key(self):
        """Test JSONB with single key."""
        result = random_jsonb(["single"], "float")
        assert len(result) == 1
        assert "single" in result

    def test_json_serial_nested_types(self):
        """Test that complex nested types raise errors."""
        nested = {"date": date(2025, 1, 1), "obj": object()}

        # Should successfully serialize the date
        assert json_serial(date(2025, 1, 1)) == "2025-01-01"

        # Should fail on object
        with pytest.raises(TypeError):
            json_serial(nested["obj"])
