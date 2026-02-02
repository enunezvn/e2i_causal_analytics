"""
Tests for Event Generators (Treatment, Engagement, Outcome).

Tests event generation with referential integrity and causal relationships.
"""

import pytest

from src.ml.synthetic.config import DGPType, EngagementTypeEnum
from src.ml.synthetic.generators import (
    EngagementGenerator,
    GeneratorConfig,
    HCPGenerator,
    OutcomeGenerator,
    PatientGenerator,
    TreatmentGenerator,
)


class TestTreatmentGenerator:
    """Test suite for TreatmentGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a TreatmentGenerator instance."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
        )
        return TreatmentGenerator(config)

    @pytest.fixture
    def linked_generator(self):
        """Create generator with linked patient data."""
        # First generate HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=20)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Then generate patients with some initiated
        patient_config = GeneratorConfig(
            seed=42,
            n_records=50,
            dgp_type=DGPType.CONFOUNDED,
        )
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        # Create treatment generator
        config = GeneratorConfig(seed=42, n_records=100)
        return TreatmentGenerator(config, patient_df=patient_df), patient_df

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_records == 100
        assert generator.entity_type == "treatment_events"

    def test_generate_correct_columns(self, generator):
        """Test that generated DataFrame has correct columns."""
        df = generator.generate()

        required_columns = [
            "treatment_event_id",
            "patient_journey_id",
            "patient_id",
            "brand",
            "treatment_date",
            "treatment_type",
            "days_supply",
            "refill_number",
            "adherence_score",
            "efficacy_score",
            "data_split",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_treatment_ids_unique(self, generator):
        """Test that treatment IDs are unique."""
        df = generator.generate()
        assert df["treatment_event_id"].nunique() == len(df)

    def test_adherence_score_range(self, generator):
        """Test that adherence scores are within 0-1."""
        df = generator.generate()
        assert df["adherence_score"].min() >= 0
        assert df["adherence_score"].max() <= 1

    def test_efficacy_score_range(self, generator):
        """Test that efficacy scores are within 0-1."""
        df = generator.generate()
        assert df["efficacy_score"].min() >= 0
        assert df["efficacy_score"].max() <= 1

    def test_days_supply_range(self, generator):
        """Test that days supply is within valid range."""
        df = generator.generate()
        assert df["days_supply"].min() >= 7
        assert df["days_supply"].max() <= 90

    def test_linked_generation_referential_integrity(self, linked_generator):
        """Test that linked generation maintains referential integrity."""
        treatment_gen, patient_df = linked_generator
        treatment_df = treatment_gen.generate()

        if len(treatment_df) > 0:
            # All treatment patient IDs should be from initiated patients
            initiated_patients = set(
                patient_df[patient_df["treatment_initiated"] == 1]["patient_id"]
            )
            for patient_id in treatment_df["patient_id"].unique():
                assert patient_id in initiated_patients


class TestEngagementGenerator:
    """Test suite for EngagementGenerator."""

    @pytest.fixture
    def generator(self):
        """Create an EngagementGenerator instance."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
        )
        return EngagementGenerator(config)

    @pytest.fixture
    def linked_generator(self):
        """Create generator with linked HCP data."""
        # Generate HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=30)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Create engagement generator
        config = GeneratorConfig(seed=42, n_records=200)
        return EngagementGenerator(config, hcp_df=hcp_df), hcp_df

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_records == 100
        assert generator.entity_type == "engagement_events"

    def test_generate_correct_columns(self, generator):
        """Test that generated DataFrame has correct columns."""
        df = generator.generate()

        required_columns = [
            "engagement_event_id",
            "hcp_id",
            "rep_id",
            "brand",
            "engagement_date",
            "engagement_type",
            "quality_score",
            "duration_minutes",
            "data_split",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_engagement_ids_unique(self, generator):
        """Test that engagement IDs are unique."""
        df = generator.generate()
        assert df["engagement_event_id"].nunique() == len(df)

    def test_engagement_types_valid(self, generator):
        """Test that engagement types are valid enums."""
        df = generator.generate()
        valid_types = {e.value for e in EngagementTypeEnum}

        for eng_type in df["engagement_type"].unique():
            assert eng_type in valid_types, f"Invalid type: {eng_type}"

    def test_quality_score_range(self, generator):
        """Test that quality scores are within 0-10."""
        df = generator.generate()
        assert df["quality_score"].min() >= 0
        assert df["quality_score"].max() <= 10

    def test_duration_positive(self, generator):
        """Test that durations are positive."""
        df = generator.generate()
        assert df["duration_minutes"].min() > 0

    def test_linked_generation_referential_integrity(self, linked_generator):
        """Test that linked generation maintains referential integrity."""
        engagement_gen, hcp_df = linked_generator
        engagement_df = engagement_gen.generate()

        valid_hcp_ids = set(hcp_df["hcp_id"])

        for hcp_id in engagement_df["hcp_id"].unique():
            assert hcp_id in valid_hcp_ids


class TestOutcomeGenerator:
    """Test suite for OutcomeGenerator."""

    @pytest.fixture
    def generator(self):
        """Create an OutcomeGenerator instance."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
        )
        return OutcomeGenerator(config)

    @pytest.fixture
    def linked_generator(self):
        """Create generator with linked patient data."""
        # Generate patients
        patient_config = GeneratorConfig(
            seed=42,
            n_records=100,
            dgp_type=DGPType.CONFOUNDED,
        )
        patient_df = PatientGenerator(patient_config).generate()

        # Create outcome generator
        config = GeneratorConfig(seed=42, n_records=50)
        return OutcomeGenerator(config, patient_df=patient_df), patient_df

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_records == 100
        assert generator.entity_type == "business_outcomes"

    def test_generate_correct_columns(self, generator):
        """Test that generated DataFrame has correct columns."""
        df = generator.generate()

        required_columns = [
            "outcome_id",
            "patient_journey_id",
            "patient_id",
            "brand",
            "outcome_type",
            "outcome_date",
            "outcome_value",
            "data_split",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_outcome_ids_unique(self, generator):
        """Test that outcome IDs are unique."""
        df = generator.generate()
        assert df["outcome_id"].nunique() == len(df)

    def test_outcome_types_valid(self, generator):
        """Test that outcome types are from valid set."""
        df = generator.generate()
        valid_types = {
            "prescription_written",
            "prescription_filled",
            "treatment_adherent",
            "treatment_switch",
            "treatment_discontinue",
        }

        for outcome_type in df["outcome_type"].unique():
            assert outcome_type in valid_types

    def test_outcome_value_non_negative(self, generator):
        """Test that outcome values are non-negative."""
        df = generator.generate()
        # Some outcomes like discontinue have 0 value
        assert df["outcome_value"].min() >= 0

    def test_linked_generation_referential_integrity(self, linked_generator):
        """Test that linked generation maintains referential integrity."""
        outcome_gen, patient_df = linked_generator
        outcome_df = outcome_gen.generate()

        if len(outcome_df) > 0:
            # All outcome patient IDs should be from initiated patients
            initiated_patients = set(
                patient_df[patient_df["treatment_initiated"] == 1]["patient_id"]
            )
            for patient_id in outcome_df["patient_id"].unique():
                assert patient_id in initiated_patients


class TestEventGeneratorIntegration:
    """Test integration between event generators."""

    @pytest.fixture
    def full_pipeline_data(self):
        """Generate complete pipeline data."""
        # HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=50)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Patients
        patient_config = GeneratorConfig(
            seed=42,
            n_records=200,
            dgp_type=DGPType.CONFOUNDED,
        )
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        # Events
        treatment_config = GeneratorConfig(seed=42, n_records=500)
        treatment_df = TreatmentGenerator(treatment_config, patient_df=patient_df).generate()

        engagement_config = GeneratorConfig(seed=42, n_records=300)
        engagement_df = EngagementGenerator(engagement_config, hcp_df=hcp_df).generate()

        outcome_config = GeneratorConfig(seed=42, n_records=100)
        outcome_df = OutcomeGenerator(outcome_config, patient_df=patient_df).generate()

        return {
            "hcps": hcp_df,
            "patients": patient_df,
            "treatments": treatment_df,
            "engagements": engagement_df,
            "outcomes": outcome_df,
        }

    def test_all_entities_generated(self, full_pipeline_data):
        """Test that all entities are generated."""
        assert len(full_pipeline_data["hcps"]) > 0
        assert len(full_pipeline_data["patients"]) > 0
        # Treatment and outcome only for initiated patients
        initiated_count = (full_pipeline_data["patients"]["treatment_initiated"] == 1).sum()
        if initiated_count > 0:
            assert len(full_pipeline_data["treatments"]) > 0
            assert len(full_pipeline_data["outcomes"]) > 0
        assert len(full_pipeline_data["engagements"]) > 0

    def test_referential_integrity_chain(self, full_pipeline_data):
        """Test referential integrity across all entities."""
        hcp_ids = set(full_pipeline_data["hcps"]["hcp_id"])
        set(full_pipeline_data["patients"]["patient_id"])
        initiated_ids = set(
            full_pipeline_data["patients"][
                full_pipeline_data["patients"]["treatment_initiated"] == 1
            ]["patient_id"]
        )

        # Engagements reference valid HCPs
        for hcp_id in full_pipeline_data["engagements"]["hcp_id"].unique():
            assert hcp_id in hcp_ids

        # Treatments reference valid initiated patients
        if len(full_pipeline_data["treatments"]) > 0:
            for patient_id in full_pipeline_data["treatments"]["patient_id"].unique():
                assert patient_id in initiated_ids

        # Outcomes reference valid initiated patients
        if len(full_pipeline_data["outcomes"]) > 0:
            for patient_id in full_pipeline_data["outcomes"]["patient_id"].unique():
                assert patient_id in initiated_ids

    def test_data_splits_present(self, full_pipeline_data):
        """Test that event entities have data splits."""
        # Only event entities need splits (not static entities like HCPs)
        event_entities = ["patients", "treatments", "engagements", "outcomes"]

        for entity_name in event_entities:
            df = full_pipeline_data[entity_name]
            if len(df) > 0:
                assert "data_split" in df.columns, f"{entity_name} missing data_split"
                assert df["data_split"].notna().all(), f"{entity_name} has null splits"
