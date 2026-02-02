"""
Tests for Batch Loader.

Tests batch loading functionality with dry run mode.
"""

import pandas as pd
import pytest

from src.ml.synthetic.config import DGPType
from src.ml.synthetic.generators import (
    GeneratorConfig,
    HCPGenerator,
    PatientGenerator,
    PredictionGenerator,
    TreatmentGenerator,
    TriggerGenerator,
)
from src.ml.synthetic.loaders import (
    LOADING_ORDER,
    TABLE_COLUMNS,
    AsyncBatchLoader,
    BatchLoader,
    LoaderConfig,
    LoadResult,
)


class TestLoadResult:
    """Test suite for LoadResult."""

    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = LoadResult(
            table_name="test",
            records_loaded=90,
            records_failed=10,
            total_batches=10,
        )
        assert result.success_rate == 0.9

    def test_success_rate_empty(self):
        """Test success rate with no records."""
        result = LoadResult(
            table_name="test",
            records_loaded=0,
            records_failed=0,
            total_batches=0,
        )
        assert result.success_rate == 0.0

    def test_is_success_true(self):
        """Test is_success with high success rate."""
        result = LoadResult(
            table_name="test",
            records_loaded=96,
            records_failed=4,
            total_batches=10,
        )
        assert result.is_success is True

    def test_is_success_false(self):
        """Test is_success with low success rate."""
        result = LoadResult(
            table_name="test",
            records_loaded=90,
            records_failed=10,
            total_batches=10,
        )
        assert result.is_success is False


class TestLoaderConfig:
    """Test suite for LoaderConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = LoaderConfig()
        assert config.batch_size == 1000
        assert config.max_retries == 3
        assert config.validate_before_load is True
        assert config.dry_run is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = LoaderConfig(
            batch_size=500,
            max_retries=5,
            dry_run=True,
        )
        assert config.batch_size == 500
        assert config.max_retries == 5
        assert config.dry_run is True


class TestLoadingOrder:
    """Test suite for loading order constants."""

    def test_loading_order_has_all_tables(self):
        """Test that loading order includes expected tables.

        Note: engagement_events and business_outcomes were removed
        as they don't exist in the current Supabase schema.
        """
        expected_tables = [
            "hcp_profiles",
            "patient_journeys",
            "treatment_events",
            "ml_predictions",
            "triggers",
        ]
        for table in expected_tables:
            assert table in LOADING_ORDER

    def test_hcp_profiles_first(self):
        """Test that HCP profiles load first (no dependencies)."""
        assert LOADING_ORDER.index("hcp_profiles") == 0

    def test_patient_journeys_after_hcps(self):
        """Test that patient journeys load after HCPs."""
        assert LOADING_ORDER.index("patient_journeys") > LOADING_ORDER.index("hcp_profiles")

    def test_events_after_patients(self):
        """Test that events load after patients."""
        patient_idx = LOADING_ORDER.index("patient_journeys")
        assert LOADING_ORDER.index("treatment_events") > patient_idx
        assert LOADING_ORDER.index("ml_predictions") > patient_idx
        assert LOADING_ORDER.index("triggers") > patient_idx


class TestTableColumns:
    """Test suite for table column mappings."""

    def test_all_loading_order_tables_have_columns(self):
        """Test that all tables in loading order have column mappings."""
        for table in LOADING_ORDER:
            assert table in TABLE_COLUMNS
            assert len(TABLE_COLUMNS[table]) > 0

    def test_hcp_profiles_has_id(self):
        """Test that HCP profiles has ID column."""
        assert "hcp_id" in TABLE_COLUMNS["hcp_profiles"]

    def test_hcp_profiles_has_geographic_region(self):
        """Test that HCP profiles has geographic_region column."""
        assert "geographic_region" in TABLE_COLUMNS["hcp_profiles"]

    def test_patient_journeys_has_required_columns(self):
        """Test that patient journeys has required columns.

        Note: hcp_id was removed from TABLE_COLUMNS for batch loading
        as it's not a required column in the Supabase patient_journeys table.
        """
        required = ["patient_journey_id", "patient_id", "brand", "data_split", "geographic_region"]
        for col in required:
            assert col in TABLE_COLUMNS["patient_journeys"]

    def test_triggers_has_required_columns(self):
        """Test that triggers table has required columns."""
        required = ["trigger_id", "patient_id", "trigger_type", "priority"]
        for col in required:
            assert col in TABLE_COLUMNS["triggers"]


class TestBatchLoader:
    """Test suite for BatchLoader."""

    @pytest.fixture
    def loader(self):
        """Create a BatchLoader in dry run mode."""
        config = LoaderConfig(
            batch_size=100,
            dry_run=True,
            verbose=False,
        )
        return BatchLoader(config)

    @pytest.fixture
    def sample_datasets(self):
        """Generate sample datasets for testing.

        Note: Only includes tables that exist in current Supabase schema
        (engagement_events and business_outcomes were removed).
        """
        # Generate HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=20)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Generate patients
        patient_config = GeneratorConfig(seed=42, n_records=50, dgp_type=DGPType.CONFOUNDED)
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        # Generate treatment events
        treatment_config = GeneratorConfig(seed=42, n_records=100)
        treatment_df = TreatmentGenerator(treatment_config, patient_df=patient_df).generate()
        # Rename columns to match database schema
        if "treatment_date" in treatment_df.columns:
            treatment_df = treatment_df.rename(columns={"treatment_date": "event_date"})
        if "treatment_type" in treatment_df.columns:
            treatment_df = treatment_df.rename(columns={"treatment_type": "event_type"})
        if "days_supply" in treatment_df.columns:
            treatment_df = treatment_df.rename(columns={"days_supply": "duration_days"})

        # Generate ML predictions
        prediction_config = GeneratorConfig(seed=42, n_records=60)
        prediction_df = PredictionGenerator(prediction_config, patient_df=patient_df).generate()
        # Rename columns to match database schema
        if "prediction_date" in prediction_df.columns:
            prediction_df = prediction_df.rename(
                columns={"prediction_date": "prediction_timestamp"}
            )

        # Generate triggers
        trigger_config = GeneratorConfig(seed=42, n_records=40)
        trigger_df = TriggerGenerator(
            trigger_config, patient_df=patient_df, hcp_df=hcp_df
        ).generate()

        return {
            "hcp_profiles": hcp_df,
            "patient_journeys": patient_df,
            "treatment_events": treatment_df,
            "ml_predictions": prediction_df,
            "triggers": trigger_df,
        }

    def test_loader_initialization(self, loader):
        """Test loader initialization."""
        assert loader.config.batch_size == 100
        assert loader.config.dry_run is True

    def test_validate_datasets_valid(self, loader, sample_datasets):
        """Test dataset validation with valid data."""
        is_valid, errors = loader.validate_datasets(sample_datasets)
        assert is_valid is True
        assert len(errors) == 0

    def test_validate_datasets_unknown_table(self, loader):
        """Test dataset validation with unknown table."""
        datasets = {"unknown_table": pd.DataFrame({"col": [1, 2, 3]})}
        is_valid, errors = loader.validate_datasets(datasets)
        assert is_valid is False
        assert any("Unknown table" in e for e in errors)

    def test_validate_datasets_empty(self, loader):
        """Test dataset validation with empty DataFrame."""
        datasets = {"hcp_profiles": pd.DataFrame()}
        is_valid, errors = loader.validate_datasets(datasets)
        assert is_valid is False
        assert any("Empty" in e for e in errors)

    def test_load_table_dry_run(self, loader, sample_datasets):
        """Test loading a single table in dry run mode."""
        result = loader.load_table("hcp_profiles", sample_datasets["hcp_profiles"])

        assert result.table_name == "hcp_profiles"
        assert result.records_loaded == len(sample_datasets["hcp_profiles"])
        assert result.records_failed == 0
        assert result.is_success is True

    def test_load_all_dry_run(self, loader, sample_datasets):
        """Test loading all tables in dry run mode."""
        results = loader.load_all(sample_datasets)

        # All tables should be loaded
        for table_name in sample_datasets:
            assert table_name in results
            assert results[table_name].is_success is True

    def test_loading_order_respected(self, loader, sample_datasets):
        """Test that tables are loaded in dependency order."""
        loaded_order = []

        def track_progress(table, current, total):
            loaded_order.append(table)

        loader.load_all(sample_datasets, progress_callback=track_progress)

        # Verify order matches LOADING_ORDER
        expected_order = [t for t in LOADING_ORDER if t in sample_datasets]
        assert loaded_order == expected_order

    def test_get_loading_summary(self, loader, sample_datasets):
        """Test loading summary generation."""
        results = loader.load_all(sample_datasets)
        summary = loader.get_loading_summary(results)

        assert "SYNTHETIC DATA LOADING SUMMARY" in summary
        assert "hcp_profiles" in summary
        assert "TOTAL:" in summary

    def test_batch_calculation(self, loader, sample_datasets):
        """Test that batches are calculated correctly."""
        df = sample_datasets["hcp_profiles"]
        result = loader.load_table("hcp_profiles", df)

        expected_batches = (len(df) + loader.config.batch_size - 1) // loader.config.batch_size
        assert result.total_batches == expected_batches


class TestAsyncBatchLoader:
    """Test suite for AsyncBatchLoader."""

    @pytest.fixture
    def async_loader(self):
        """Create an AsyncBatchLoader in dry run mode."""
        config = LoaderConfig(
            batch_size=100,
            dry_run=True,
            verbose=False,
        )
        return AsyncBatchLoader(config)

    @pytest.fixture
    def sample_datasets(self):
        """Generate sample datasets for testing."""
        hcp_config = GeneratorConfig(seed=42, n_records=20)
        hcp_df = HCPGenerator(hcp_config).generate()

        patient_config = GeneratorConfig(seed=42, n_records=50, dgp_type=DGPType.CONFOUNDED)
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        return {
            "hcp_profiles": hcp_df,
            "patient_journeys": patient_df,
        }

    @pytest.mark.asyncio
    async def test_load_all_async_dry_run(self, async_loader, sample_datasets):
        """Test async loading in dry run mode."""
        results = await async_loader.load_all_async(sample_datasets)

        for table_name in sample_datasets:
            assert table_name in results
            assert results[table_name].is_success is True

    @pytest.mark.asyncio
    async def test_load_table_async_dry_run(self, async_loader, sample_datasets):
        """Test async table loading in dry run mode."""
        result = await async_loader.load_table_async(
            "hcp_profiles", sample_datasets["hcp_profiles"]
        )

        assert result.table_name == "hcp_profiles"
        assert result.is_success is True


class TestBatchLoaderIntegration:
    """Integration tests for batch loader with full pipeline."""

    @pytest.fixture
    def full_pipeline_datasets(self):
        """Generate full pipeline datasets.

        Note: Only includes tables that exist in current Supabase schema.
        """
        # HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=30)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Patients
        patient_config = GeneratorConfig(seed=42, n_records=100, dgp_type=DGPType.CONFOUNDED)
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        # Treatment events
        treatment_df = TreatmentGenerator(
            GeneratorConfig(seed=42, n_records=200), patient_df=patient_df
        ).generate()
        # Rename columns to match database schema
        if "treatment_date" in treatment_df.columns:
            treatment_df = treatment_df.rename(columns={"treatment_date": "event_date"})
        if "treatment_type" in treatment_df.columns:
            treatment_df = treatment_df.rename(columns={"treatment_type": "event_type"})
        if "days_supply" in treatment_df.columns:
            treatment_df = treatment_df.rename(columns={"days_supply": "duration_days"})

        # ML Predictions
        prediction_df = PredictionGenerator(
            GeneratorConfig(seed=42, n_records=100), patient_df=patient_df
        ).generate()
        # Rename columns to match database schema
        if "prediction_date" in prediction_df.columns:
            prediction_df = prediction_df.rename(
                columns={"prediction_date": "prediction_timestamp"}
            )

        # Triggers
        trigger_df = TriggerGenerator(
            GeneratorConfig(seed=42, n_records=80), patient_df=patient_df, hcp_df=hcp_df
        ).generate()

        return {
            "hcp_profiles": hcp_df,
            "patient_journeys": patient_df,
            "treatment_events": treatment_df,
            "ml_predictions": prediction_df,
            "triggers": trigger_df,
        }

    def test_full_pipeline_dry_run(self, full_pipeline_datasets):
        """Test full pipeline loading in dry run mode."""
        loader = BatchLoader(LoaderConfig(batch_size=50, dry_run=True))

        # Validate
        is_valid, errors = loader.validate_datasets(full_pipeline_datasets)
        assert is_valid is True, f"Validation errors: {errors}"

        # Load
        results = loader.load_all(full_pipeline_datasets)

        # Check all succeeded
        for table_name, result in results.items():
            assert result.is_success is True, f"{table_name} failed: {result.errors}"

        # Get summary
        summary = loader.get_loading_summary(results)
        assert "100.0%" in summary or "100%" in summary

    def test_total_records_loaded(self, full_pipeline_datasets):
        """Test that total records loaded matches input."""
        loader = BatchLoader(LoaderConfig(batch_size=50, dry_run=True))
        results = loader.load_all(full_pipeline_datasets)

        total_loaded = sum(r.records_loaded for r in results.values())
        total_input = sum(len(df) for df in full_pipeline_datasets.values())

        assert total_loaded == total_input
