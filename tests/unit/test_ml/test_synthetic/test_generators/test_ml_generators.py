"""
Tests for ML Pipeline Generators (Prediction, Trigger).

Tests ML entity generation with referential integrity and business logic.
"""

import pytest
import numpy as np
import pandas as pd

from src.ml.synthetic.generators import (
    HCPGenerator,
    PatientGenerator,
    PredictionGenerator,
    TriggerGenerator,
    GeneratorConfig,
)
from src.ml.synthetic.config import Brand, DGPType


class TestPredictionGenerator:
    """Test suite for PredictionGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a PredictionGenerator instance."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
        )
        return PredictionGenerator(config)

    @pytest.fixture
    def linked_generator(self):
        """Create generator with linked patient data."""
        # Generate patients
        patient_config = GeneratorConfig(
            seed=42,
            n_records=50,
            dgp_type=DGPType.CONFOUNDED,
        )
        patient_df = PatientGenerator(patient_config).generate()

        # Create prediction generator
        config = GeneratorConfig(seed=42, n_records=200)
        return PredictionGenerator(config, patient_df=patient_df), patient_df

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_records == 100
        assert generator.entity_type == "ml_predictions"

    def test_generate_correct_columns(self, generator):
        """Test that generated DataFrame has correct columns."""
        df = generator.generate()

        required_columns = [
            "prediction_id",
            "patient_journey_id",
            "patient_id",
            "brand",
            "prediction_type",
            "prediction_value",
            "confidence_score",
            "uncertainty",
            "model_version",
            "prediction_date",
            "data_split",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_prediction_ids_unique(self, generator):
        """Test that prediction IDs are unique."""
        df = generator.generate()
        assert df["prediction_id"].nunique() == len(df)

    def test_prediction_types_valid(self, generator):
        """Test that prediction types are from valid set."""
        df = generator.generate()
        # Note: Types MUST match Supabase prediction_type ENUM:
        # {trigger, propensity, risk, churn, next_best_action}
        valid_types = {
            "propensity",
            "risk",
            "trigger",
            "churn",
            "next_best_action",
        }

        for pred_type in df["prediction_type"].unique():
            assert pred_type in valid_types, f"Invalid type: {pred_type}"

    def test_confidence_score_range(self, generator):
        """Test that confidence scores are within 0.6-0.95."""
        df = generator.generate()
        assert df["confidence_score"].min() >= 0.6
        assert df["confidence_score"].max() <= 0.95

    def test_uncertainty_complements_confidence(self, generator):
        """Test that uncertainty = 1 - confidence."""
        df = generator.generate()
        for _, row in df.iterrows():
            expected_uncertainty = 1 - row["confidence_score"]
            assert abs(row["uncertainty"] - expected_uncertainty) < 0.001

    def test_prediction_value_ranges(self, generator):
        """Test that prediction values are within type-specific ranges."""
        df = generator.generate()

        # All prediction types are probability-based (0-1 range)
        # Types: propensity, risk, trigger, churn, next_best_action
        assert df["prediction_value"].min() >= 0, "Prediction values should be >= 0"
        assert df["prediction_value"].max() <= 1, "Prediction values should be <= 1"

    def test_model_version_format(self, generator):
        """Test that model versions have correct format."""
        df = generator.generate()

        for version in df["model_version"].unique():
            assert version.startswith("v"), f"Version should start with 'v': {version}"
            # Should be like v1.5, v2.3, etc.
            parts = version[1:].split(".")
            assert len(parts) == 2, f"Version should have major.minor format: {version}"

    def test_linked_generation_referential_integrity(self, linked_generator):
        """Test that linked generation maintains referential integrity."""
        pred_gen, patient_df = linked_generator
        prediction_df = pred_gen.generate()

        if len(prediction_df) > 0:
            valid_patient_ids = set(patient_df["patient_id"])
            for patient_id in prediction_df["patient_id"].unique():
                assert patient_id in valid_patient_ids

    def test_brand_consistency_with_config(self):
        """Test that brand matches config when specified."""
        config = GeneratorConfig(seed=42, n_records=50, brand=Brand.KISQALI)
        generator = PredictionGenerator(config)
        df = generator.generate()

        assert all(df["brand"] == Brand.KISQALI.value)


class TestTriggerGenerator:
    """Test suite for TriggerGenerator."""

    @pytest.fixture
    def generator(self):
        """Create a TriggerGenerator instance."""
        config = GeneratorConfig(
            seed=42,
            n_records=100,
        )
        return TriggerGenerator(config)

    @pytest.fixture
    def linked_generator(self):
        """Create generator with linked patient and HCP data."""
        # Generate HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=20)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Generate patients
        patient_config = GeneratorConfig(
            seed=42,
            n_records=50,
            dgp_type=DGPType.CONFOUNDED,
        )
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        # Create trigger generator
        config = GeneratorConfig(seed=42, n_records=150)
        return TriggerGenerator(config, patient_df=patient_df, hcp_df=hcp_df), patient_df, hcp_df

    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.n_records == 100
        assert generator.entity_type == "triggers"

    def test_generate_correct_columns(self, generator):
        """Test that generated DataFrame has correct columns."""
        df = generator.generate()

        required_columns = [
            "trigger_id",
            "patient_id",
            "hcp_id",
            "trigger_timestamp",
            "trigger_type",
            "priority",
            "confidence_score",
            "lead_time_days",
            "expiration_date",
            "delivery_channel",
            "delivery_status",
            "acceptance_status",
            "data_split",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_trigger_ids_unique(self, generator):
        """Test that trigger IDs are unique."""
        df = generator.generate()
        assert df["trigger_id"].nunique() == len(df)

    def test_trigger_types_valid(self, generator):
        """Test that trigger types are from valid set."""
        df = generator.generate()
        valid_types = {
            "prescription_opportunity",
            "adherence_risk",
            "churn_prevention",
            "cross_sell",
            "engagement_gap",
            "competitive_threat",
            "treatment_switch",
            "reactivation",
        }

        for trigger_type in df["trigger_type"].unique():
            assert trigger_type in valid_types, f"Invalid type: {trigger_type}"

    def test_priority_values_valid(self, generator):
        """Test that priority values match schema enum."""
        df = generator.generate()
        valid_priorities = {"critical", "high", "medium", "low"}

        for priority in df["priority"].unique():
            assert priority in valid_priorities, f"Invalid priority: {priority}"

    def test_confidence_score_range(self, generator):
        """Test that confidence scores are within 0.5-0.99."""
        df = generator.generate()
        assert df["confidence_score"].min() >= 0.50
        assert df["confidence_score"].max() <= 0.99

    def test_lead_time_days_positive(self, generator):
        """Test that lead time days are positive."""
        df = generator.generate()
        assert df["lead_time_days"].min() > 0

    def test_delivery_channels_valid(self, generator):
        """Test that delivery channels are valid."""
        df = generator.generate()
        valid_channels = {"email", "crm", "mobile", "portal", "rep_alert"}

        for channel in df["delivery_channel"].unique():
            assert channel in valid_channels, f"Invalid channel: {channel}"

    def test_delivery_status_valid(self, generator):
        """Test that delivery status values are valid."""
        df = generator.generate()
        valid_statuses = {"pending", "delivered", "viewed", "failed"}

        for status in df["delivery_status"].unique():
            assert status in valid_statuses, f"Invalid status: {status}"

    def test_acceptance_status_valid(self, generator):
        """Test that acceptance status values are valid."""
        df = generator.generate()
        valid_statuses = {"pending", "accepted", "rejected", "expired"}

        for status in df["acceptance_status"].unique():
            assert status in valid_statuses, f"Invalid status: {status}"

    def test_causal_chain_structure(self, generator):
        """Test that causal chain has expected structure."""
        df = generator.generate()

        for chain in df["causal_chain"]:
            assert isinstance(chain, dict)
            # Should have at least root_cause or confidence
            assert "root_cause" in chain or "confidence" in chain

    def test_supporting_evidence_structure(self, generator):
        """Test that supporting evidence has expected structure."""
        df = generator.generate()

        for evidence in df["supporting_evidence"]:
            assert isinstance(evidence, dict)
            assert "model_version" in evidence
            assert "data_sources" in evidence

    def test_trigger_reason_not_empty(self, generator):
        """Test that trigger reasons are populated."""
        df = generator.generate()
        assert df["trigger_reason"].notna().all()
        assert all(len(str(reason)) > 10 for reason in df["trigger_reason"])

    def test_recommended_action_not_empty(self, generator):
        """Test that recommended actions are populated."""
        df = generator.generate()
        assert df["recommended_action"].notna().all()
        assert all(len(str(action)) > 10 for action in df["recommended_action"])

    def test_linked_generation_referential_integrity(self, linked_generator):
        """Test that linked generation maintains referential integrity."""
        trigger_gen, patient_df, hcp_df = linked_generator
        trigger_df = trigger_gen.generate()

        if len(trigger_df) > 0:
            valid_patient_ids = set(patient_df["patient_id"])
            for patient_id in trigger_df["patient_id"].unique():
                assert patient_id in valid_patient_ids


class TestMLGeneratorIntegration:
    """Test integration between ML generators and other entities."""

    @pytest.fixture
    def full_pipeline_data(self):
        """Generate complete pipeline data including ML entities."""
        # HCPs
        hcp_config = GeneratorConfig(seed=42, n_records=30)
        hcp_df = HCPGenerator(hcp_config).generate()

        # Patients
        patient_config = GeneratorConfig(
            seed=42,
            n_records=100,
            dgp_type=DGPType.CONFOUNDED,
        )
        patient_df = PatientGenerator(patient_config, hcp_df=hcp_df).generate()

        # Predictions
        prediction_config = GeneratorConfig(seed=42, n_records=200)
        prediction_df = PredictionGenerator(
            prediction_config, patient_df=patient_df
        ).generate()

        # Triggers
        trigger_config = GeneratorConfig(seed=42, n_records=150)
        trigger_df = TriggerGenerator(
            trigger_config, patient_df=patient_df, hcp_df=hcp_df
        ).generate()

        return {
            "hcps": hcp_df,
            "patients": patient_df,
            "predictions": prediction_df,
            "triggers": trigger_df,
        }

    def test_all_ml_entities_generated(self, full_pipeline_data):
        """Test that all ML entities are generated."""
        assert len(full_pipeline_data["hcps"]) > 0
        assert len(full_pipeline_data["patients"]) > 0
        assert len(full_pipeline_data["predictions"]) > 0
        assert len(full_pipeline_data["triggers"]) > 0

    def test_referential_integrity_chain(self, full_pipeline_data):
        """Test referential integrity across all ML entities."""
        patient_ids = set(full_pipeline_data["patients"]["patient_id"])

        # Predictions reference valid patients
        for patient_id in full_pipeline_data["predictions"]["patient_id"].unique():
            assert patient_id in patient_ids

        # Triggers reference valid patients
        for patient_id in full_pipeline_data["triggers"]["patient_id"].unique():
            assert patient_id in patient_ids

    def test_data_splits_present(self, full_pipeline_data):
        """Test that ML entities have data splits."""
        ml_entities = ["patients", "predictions", "triggers"]

        for entity_name in ml_entities:
            df = full_pipeline_data[entity_name]
            if len(df) > 0:
                assert "data_split" in df.columns, f"{entity_name} missing data_split"
                assert df["data_split"].notna().all(), f"{entity_name} has null splits"

    def test_brand_consistency(self, full_pipeline_data):
        """Test that brands are consistent across entities."""
        # Get all brands used
        all_brands = set()
        for entity_name in ["patients", "predictions", "triggers"]:
            df = full_pipeline_data[entity_name]
            if "brand" in df.columns:
                all_brands.update(df["brand"].unique())

        # All brands should be valid
        valid_brands = {b.value for b in Brand}
        for brand in all_brands:
            assert brand in valid_brands
