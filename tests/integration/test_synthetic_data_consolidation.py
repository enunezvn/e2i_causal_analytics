"""
Integration Tests for Synthetic Data Consolidation.

These tests validate the consolidation of the main synthetic data system
(src/ml/synthetic/) with the external system (E2i synthetic data/).

Key validations:
1. Brand ENUM compatibility with Supabase (capitalized values)
2. All 4 data splits present (train, validation, test, holdout)
3. All 5 DGP types generate valid data
4. Ground truth store persistence
5. Batch loader FK compliance

Phase 2.1 of Synthetic Data Audit Plan.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta

from src.ml.synthetic.config import (
    Brand,
    DGPType,
    DataSplit,
    DGP_CONFIGS,
    SyntheticDataConfig,
    BRANDS,
    DGP_TYPES,
)
from src.ml.synthetic.validators import SchemaValidator
from src.ml.synthetic.ground_truth.causal_effects import (
    GroundTruthStore,
    GroundTruthEffect,
    create_ground_truth_from_dgp_config,
    get_global_store,
)
from src.ml.synthetic.loaders.batch_loader import (
    BatchLoader,
    LoaderConfig,
    LOADING_ORDER,
)


class TestSyntheticDataConsolidation:
    """
    Consolidation tests for synthetic data systems.

    These tests ensure the main system (src/ml/synthetic/) is the
    canonical source and catches issues found in the external system.
    """

    @pytest.fixture
    def schema_validator(self):
        """Create a schema validator."""
        return SchemaValidator()

    @pytest.fixture
    def ground_truth_store(self):
        """Create a fresh ground truth store."""
        store = GroundTruthStore()
        yield store
        store.clear()

    @pytest.fixture
    def sample_hcp_df(self):
        """Create sample HCP data with correct ENUMs."""
        np.random.seed(42)
        n_hcps = 50

        return pd.DataFrame({
            "hcp_id": [f"hcp_{i:05d}" for i in range(n_hcps)],
            "npi": [f"{1000000000 + i}" for i in range(n_hcps)],
            "specialty": np.random.choice(
                ["dermatology", "hematology", "oncology", "neurology"],
                n_hcps,
            ),
            "practice_type": np.random.choice(
                ["academic", "community", "private"],
                n_hcps,
            ),
            "geographic_region": np.random.choice(
                ["northeast", "south", "midwest", "west"],
                n_hcps,
            ),
            "years_experience": np.random.randint(2, 35, n_hcps),
            "academic_hcp": np.random.binomial(1, 0.3, n_hcps),
            "total_patient_volume": np.random.randint(50, 500, n_hcps),
            "brand": np.random.choice(
                ["Remibrutinib", "Fabhalta", "Kisqali"],  # Capitalized - correct
                n_hcps,
            ),
        })

    @pytest.fixture
    def sample_patient_df(self, sample_hcp_df):
        """Create sample patient data with correct ENUMs and all splits."""
        np.random.seed(42)
        n_patients = 200

        # Get HCP IDs for FK
        hcp_ids = np.random.choice(sample_hcp_df["hcp_id"].values, n_patients)

        # Generate dates spanning all splits
        dates = []
        for i in range(n_patients):
            # Distribute across date range to get all splits
            if i < 120:  # 60% train
                d = date(2022, 1, 1) + timedelta(days=np.random.randint(0, 540))
            elif i < 160:  # 20% validation
                d = date(2023, 7, 1) + timedelta(days=np.random.randint(0, 270))
            elif i < 190:  # 15% test
                d = date(2024, 4, 1) + timedelta(days=np.random.randint(0, 180))
            else:  # 5% holdout
                d = date(2024, 10, 1) + timedelta(days=np.random.randint(0, 90))
            dates.append(d)

        dates = sorted(dates)

        # Assign splits based on date boundaries
        train_cutoff = date(2023, 6, 30)
        val_cutoff = date(2024, 3, 31)
        test_cutoff = date(2024, 9, 30)

        splits = []
        for d in dates:
            if d <= train_cutoff:
                splits.append("train")
            elif d <= val_cutoff:
                splits.append("validation")
            elif d <= test_cutoff:
                splits.append("test")
            else:
                splits.append("holdout")

        return pd.DataFrame({
            "patient_journey_id": [f"journey_{i:06d}" for i in range(n_patients)],
            "patient_id": [f"pt_{i:06d}" for i in range(n_patients)],
            "hcp_id": hcp_ids,
            "brand": np.random.choice(
                ["Remibrutinib", "Fabhalta", "Kisqali"],  # Capitalized - correct
                n_patients,
            ),
            "journey_start_date": [d.strftime("%Y-%m-%d") for d in dates],
            "data_split": splits,
            "disease_severity": np.clip(np.random.normal(5, 2, n_patients), 0, 10),
            "academic_hcp": np.random.binomial(1, 0.3, n_patients),
            "engagement_score": np.random.uniform(0, 10, n_patients),
            "treatment_initiated": np.random.binomial(1, 0.5, n_patients),
            "geographic_region": np.random.choice(
                ["northeast", "south", "midwest", "west"],
                n_patients,
            ),
            "insurance_type": np.random.choice(
                ["commercial", "medicare", "medicaid"],
                n_patients,
            ),
            "age_at_diagnosis": np.random.randint(18, 80, n_patients),
        })

    # =========================================================================
    # Test 1: Brand ENUM Supabase Compatibility
    # =========================================================================

    def test_brand_enum_supabase_compatibility(self, schema_validator, sample_hcp_df, sample_patient_df):
        """
        Test that Brand ENUM values match Supabase schema exactly.

        External system bug: Uses lowercase ('remibrutinib')
        Supabase schema: Uses capitalized ('Remibrutinib')
        Main system: Uses capitalized (correct)

        This test ensures the main system's Brand enum is Supabase-compatible.
        """
        # Verify Brand enum values are capitalized (matching Supabase)
        assert Brand.REMIBRUTINIB.value == "Remibrutinib", "Brand should be capitalized"
        assert Brand.FABHALTA.value == "Fabhalta", "Brand should be capitalized"
        assert Brand.KISQALI.value == "Kisqali", "Brand should be capitalized"

        # Verify BRANDS list has capitalized values
        assert "Remibrutinib" in BRANDS
        assert "remibrutinib" not in BRANDS  # Lowercase should NOT be present

        # Test schema validation passes with capitalized brands
        hcp_result = schema_validator.validate(sample_hcp_df, "hcp_profiles")
        assert hcp_result.is_valid, f"HCP schema failed: {hcp_result.errors}"

        patient_result = schema_validator.validate(
            sample_patient_df,
            "patient_journeys",
            reference_dfs={"hcp_profiles": sample_hcp_df},
        )
        assert patient_result.is_valid, f"Patient schema failed: {patient_result.errors}"

        # Test schema validation FAILS with lowercase brands (external system bug)
        bad_hcp_df = sample_hcp_df.copy()
        bad_hcp_df.loc[0, "brand"] = "remibrutinib"  # External system bug

        bad_result = schema_validator.validate(bad_hcp_df, "hcp_profiles")
        assert bad_result.is_valid is False, "Lowercase brand should fail validation"
        assert any(
            e.error_type == "invalid_enum" and "remibrutinib" in str(e.sample_values)
            for e in bad_result.errors
        ), "Should identify lowercase brand as invalid enum"

    # =========================================================================
    # Test 2: Holdout Split Presence
    # =========================================================================

    def test_holdout_split_presence(self, sample_patient_df):
        """
        Test that all 4 data splits are present, including holdout.

        External system bug: Only 3 splits (train, validation, test) - NO holdout
        Main system: All 4 splits (train, validation, test, holdout)
        Target split ratios: 60/20/15/5
        """
        # Verify DataSplit enum has all 4 values
        split_values = {s.value for s in DataSplit}
        assert split_values == {"train", "validation", "test", "holdout"}, (
            f"DataSplit should have 4 values, got {split_values}"
        )

        # Verify sample data has all 4 splits
        unique_splits = set(sample_patient_df["data_split"].unique())
        assert "train" in unique_splits, "Missing train split"
        assert "validation" in unique_splits, "Missing validation split"
        assert "test" in unique_splits, "Missing test split"
        assert "holdout" in unique_splits, (
            "Missing holdout split - this was a bug in external system"
        )

        # Verify split ratios are approximately correct
        split_counts = sample_patient_df["data_split"].value_counts()
        total = len(sample_patient_df)

        train_ratio = split_counts.get("train", 0) / total
        val_ratio = split_counts.get("validation", 0) / total
        test_ratio = split_counts.get("test", 0) / total
        holdout_ratio = split_counts.get("holdout", 0) / total

        # Allow 10% tolerance due to date-based assignment
        assert 0.50 <= train_ratio <= 0.70, f"Train ratio {train_ratio:.2f} outside 50-70%"
        assert 0.10 <= val_ratio <= 0.30, f"Validation ratio {val_ratio:.2f} outside 10-30%"
        assert 0.05 <= test_ratio <= 0.25, f"Test ratio {test_ratio:.2f} outside 5-25%"
        assert holdout_ratio > 0, "Holdout split must have records"

    # =========================================================================
    # Test 3: All DGP Types Generate Valid Data
    # =========================================================================

    def test_all_dgp_types_generate_valid_data(self):
        """
        Test that all 5 DGP types are defined and have valid configurations.

        External system: Only CONFOUNDED DGP implemented
        Main system: All 5 DGPs (SIMPLE_LINEAR, CONFOUNDED, HETEROGENEOUS,
                     TIME_SERIES, SELECTION_BIAS)
        """
        # Verify all 5 DGP types are defined
        expected_dgps = {
            "simple_linear",
            "confounded",
            "heterogeneous",
            "time_series",
            "selection_bias",
        }
        actual_dgps = {d.value for d in DGPType}
        assert actual_dgps == expected_dgps, f"Missing DGPs: {expected_dgps - actual_dgps}"

        # Verify DGP_TYPES convenience export
        assert len(DGP_TYPES) == 5, f"Expected 5 DGP types, got {len(DGP_TYPES)}"

        # Verify each DGP has a valid configuration
        for dgp_type in DGPType:
            assert dgp_type in DGP_CONFIGS, f"Missing config for {dgp_type.value}"

            config = DGP_CONFIGS[dgp_type]

            # Validate required fields
            assert config.dgp_type == dgp_type
            assert config.true_ate is not None and config.true_ate > 0, (
                f"{dgp_type.value}: true_ate must be positive"
            )
            assert config.tolerance > 0, f"{dgp_type.value}: tolerance must be positive"
            assert config.treatment_variable is not None
            assert config.outcome_variable is not None

        # Verify specific TRUE_ATE values match ground truth reference
        assert DGP_CONFIGS[DGPType.SIMPLE_LINEAR].true_ate == 0.40
        assert DGP_CONFIGS[DGPType.CONFOUNDED].true_ate == 0.25
        assert DGP_CONFIGS[DGPType.HETEROGENEOUS].true_ate == 0.30
        assert DGP_CONFIGS[DGPType.TIME_SERIES].true_ate == 0.30
        assert DGP_CONFIGS[DGPType.SELECTION_BIAS].true_ate == 0.35

        # Verify confounders are specified where needed
        assert len(DGP_CONFIGS[DGPType.SIMPLE_LINEAR].confounders) == 0, (
            "SIMPLE_LINEAR should have no confounders"
        )
        assert "disease_severity" in DGP_CONFIGS[DGPType.CONFOUNDED].confounders
        assert "academic_hcp" in DGP_CONFIGS[DGPType.CONFOUNDED].confounders

        # Verify heterogeneous DGP has CATE by segment
        hetero_config = DGP_CONFIGS[DGPType.HETEROGENEOUS]
        assert hetero_config.cate_by_segment is not None, (
            "HETEROGENEOUS DGP should have CATE by segment"
        )
        assert "high_severity" in hetero_config.cate_by_segment
        assert hetero_config.cate_by_segment["high_severity"] == 0.50

    # =========================================================================
    # Test 4: Ground Truth Store Persistence
    # =========================================================================

    def test_ground_truth_store_persistence(self, ground_truth_store):
        """
        Test that ground truth effects can be stored and retrieved.

        Main system: Uses GroundTruthStore class with validate_estimate() method
        External system: Uses hardcoded dict in synthetic_metadata table
        """
        # Store ground truth for each DGP type and brand combination
        for brand in Brand:
            for dgp_type in DGPType:
                ground_truth = create_ground_truth_from_dgp_config(
                    brand=brand,
                    dgp_type=dgp_type,
                    n_samples=500,
                    data_split_counts={"train": 300, "validation": 100, "test": 75, "holdout": 25},
                )
                ground_truth_store.store(ground_truth)

        # Verify all combinations can be retrieved
        for brand in Brand:
            for dgp_type in DGPType:
                retrieved = ground_truth_store.get(brand, dgp_type)
                assert retrieved is not None, (
                    f"Failed to retrieve ground truth for {brand.value}/{dgp_type.value}"
                )

                # Verify retrieved data matches expected
                expected_config = DGP_CONFIGS[dgp_type]
                assert retrieved.true_ate == expected_config.true_ate
                assert retrieved.tolerance == expected_config.tolerance
                assert retrieved.confounders == expected_config.confounders
                assert retrieved.n_samples == 500

        # Test validate_estimate functionality
        validation = ground_truth_store.validate_estimate(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.CONFOUNDED,
            estimated_ate=0.24,  # Within tolerance of 0.25
        )
        assert validation["is_valid"] is True
        assert validation["true_ate"] == 0.25
        assert validation["error"] < 0.05

        # Test validation failure
        validation_fail = ground_truth_store.validate_estimate(
            brand=Brand.REMIBRUTINIB,
            dgp_type=DGPType.CONFOUNDED,
            estimated_ate=0.50,  # Outside tolerance of 0.25 ± 0.05
        )
        assert validation_fail["is_valid"] is False
        assert validation_fail["error"] > 0.05

    # =========================================================================
    # Test 5: Batch Loader FK Compliance
    # =========================================================================

    def test_batch_loader_fk_compliance(self, sample_hcp_df, sample_patient_df):
        """
        Test that batch loader respects foreign key dependencies.

        Loading order must be:
        1. hcp_profiles (no FK dependencies)
        2. patient_journeys (FK: hcp_id -> hcp_profiles)
        3. treatment_events (FK: patient_journey_id -> patient_journeys)
        4. ml_predictions (FK: patient_id, hcp_id)
        5. triggers (FK: patient_id, hcp_id)
        """
        # Verify loading order respects FK dependencies
        assert LOADING_ORDER.index("hcp_profiles") < LOADING_ORDER.index("patient_journeys"), (
            "hcp_profiles must load before patient_journeys"
        )
        assert LOADING_ORDER.index("patient_journeys") < LOADING_ORDER.index("treatment_events"), (
            "patient_journeys must load before treatment_events"
        )

        # Create batch loader in dry-run mode
        config = LoaderConfig(
            batch_size=50,
            dry_run=True,
            validate_before_load=True,
        )
        loader = BatchLoader(config)

        # Test FK validation with valid data
        datasets = {
            "hcp_profiles": sample_hcp_df,
            "patient_journeys": sample_patient_df,
        }

        # In dry-run mode, load_all should succeed with valid FK relationships
        results = loader.load_all(datasets)

        # Verify results structure
        assert "hcp_profiles" in results
        assert "patient_journeys" in results

        # Dry run should report success (no actual loading)
        for table_name, result in results.items():
            assert result.records_failed == 0, (
                f"{table_name} had failures: {result.errors}"
            )

        # Test with orphan FK (patient references non-existent HCP)
        bad_patient_df = sample_patient_df.copy()
        bad_patient_df.loc[0, "hcp_id"] = "hcp_nonexistent_99999"

        # Schema validator should catch orphan FK
        schema_validator = SchemaValidator()
        result = schema_validator.validate(
            bad_patient_df,
            "patient_journeys",
            reference_dfs={"hcp_profiles": sample_hcp_df},
        )

        assert result.is_valid is False, "Orphan FK should fail validation"
        assert any(
            e.error_type == "orphan_fk"
            for e in result.errors
        ), "Should identify orphan FK error"


class TestConsolidationSummary:
    """
    Summary tests that verify the main system addresses all external system issues.
    """

    def test_external_system_issues_addressed(self):
        """
        Verify all issues found in external system are fixed in main system.

        External System Issues:
        1. Brand ENUM lowercase ('remibrutinib') - FIXED: capitalized
        2. Missing holdout split - FIXED: 4 splits
        3. Only 1 DGP (CONFOUNDED) - FIXED: 5 DGPs
        4. Hardcoded ground truth - FIXED: GroundTruthStore
        """
        issues_addressed = []

        # Issue 1: Brand ENUM case
        if Brand.REMIBRUTINIB.value == "Remibrutinib":
            issues_addressed.append("Brand ENUM capitalization")

        # Issue 2: Missing holdout split
        if DataSplit.HOLDOUT.value == "holdout":
            issues_addressed.append("Holdout split present")

        # Issue 3: Limited DGP coverage
        if len(DGP_CONFIGS) == 5:
            issues_addressed.append("All 5 DGP types implemented")

        # Issue 4: Ground truth storage
        store = GroundTruthStore()
        if hasattr(store, "validate_estimate"):
            issues_addressed.append("GroundTruthStore with validation")

        assert len(issues_addressed) == 4, (
            f"Not all external system issues addressed. Fixed: {issues_addressed}"
        )

    def test_config_version_documented(self):
        """Verify configuration is versioned for tracking."""
        config = SyntheticDataConfig()

        assert hasattr(config, "config_name")
        assert hasattr(config, "config_version")
        assert config.config_name is not None
        assert config.config_version is not None
