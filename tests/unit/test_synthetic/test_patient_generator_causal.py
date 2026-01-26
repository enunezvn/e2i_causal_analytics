"""Tests for Patient Generator Causal Columns.

Tests causal columns are generated, engagement_score distribution,
treatment_initiated binary values, and TRUE_ATE causal effect embedding.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import pearsonr

from src.ml.synthetic.config import DGPType
from src.ml.synthetic.generators import (
    GeneratorConfig,
    HCPGenerator,
    PatientGenerator,
)


class TestCausalColumnsGenerated:
    """Test causal columns are generated."""

    def test_disease_severity_column_exists(self):
        """Test disease_severity column is generated."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "disease_severity" in df.columns

    def test_academic_hcp_column_exists(self):
        """Test academic_hcp column is generated."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "academic_hcp" in df.columns

    def test_engagement_score_column_exists(self):
        """Test engagement_score column is generated."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "engagement_score" in df.columns

    def test_treatment_initiated_column_exists(self):
        """Test treatment_initiated column is generated."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "treatment_initiated" in df.columns

    def test_days_to_treatment_column_exists(self):
        """Test days_to_treatment column is generated."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "days_to_treatment" in df.columns

    def test_age_at_diagnosis_column_exists(self):
        """Test age_at_diagnosis column is generated."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "age_at_diagnosis" in df.columns

    def test_all_causal_columns_present(self):
        """Test all causal columns are present together."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        causal_columns = [
            "disease_severity",
            "academic_hcp",
            "engagement_score",
            "treatment_initiated",
            "days_to_treatment",
            "age_at_diagnosis",
        ]
        for col in causal_columns:
            assert col in df.columns, f"Missing causal column: {col}"


class TestEngagementScoreDistribution:
    """Test engagement_score distribution."""

    def test_engagement_score_range(self):
        """Test engagement_score is within 0-10 range."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df["engagement_score"].min() >= 0.0
        assert df["engagement_score"].max() <= 10.0

    def test_engagement_score_distribution_reasonable(self):
        """Test engagement_score has reasonable distribution."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        mean = df["engagement_score"].mean()
        std = df["engagement_score"].std()

        # With confounding, mean is biased towards higher values (6-10 range)
        assert 5.0 < mean < 10.0
        # Should have some variance
        assert std > 0.5

    def test_engagement_score_variation(self):
        """Test engagement_score has sufficient variation."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        # With confounding and expit(), range is compressed but still has variation
        score_range = df["engagement_score"].max() - df["engagement_score"].min()
        assert score_range > 2.5

    def test_engagement_score_not_constant(self):
        """Test engagement_score is not constant."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        unique_scores = df["engagement_score"].nunique()
        assert unique_scores > 10  # Should have variety


class TestTreatmentInitiatedBinary:
    """Test treatment_initiated binary values."""

    def test_treatment_initiated_is_binary(self):
        """Test treatment_initiated has only 0 and 1 values."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        unique_values = df["treatment_initiated"].unique()
        assert set(unique_values).issubset({0, 1})

    def test_treatment_initiated_has_both_values(self):
        """Test treatment_initiated has both 0 and 1."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        unique_values = set(df["treatment_initiated"].unique())
        assert 0 in unique_values
        assert 1 in unique_values

    def test_treatment_initiated_reasonable_proportion(self):
        """Test treatment_initiated has reasonable proportion of 1s."""
        config = GeneratorConfig(n_records=2000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        proportion = df["treatment_initiated"].mean()
        # With confounded DGP, higher engagement leads to higher treatment rates
        assert 0.1 < proportion < 0.99

    def test_days_to_treatment_only_when_initiated(self):
        """Test days_to_treatment only has values when treatment_initiated=1."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        # For non-initiated, days_to_treatment should be null
        non_initiated = df[df["treatment_initiated"] == 0]
        if len(non_initiated) > 0:
            assert non_initiated["days_to_treatment"].isna().all() or \
                   (non_initiated["days_to_treatment"] == None).all()

    def test_days_to_treatment_range_when_initiated(self):
        """Test days_to_treatment is in valid range for initiated."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        initiated = df[df["treatment_initiated"] == 1]
        valid_days = initiated["days_to_treatment"].dropna()
        if len(valid_days) > 0:
            assert valid_days.min() >= 7
            assert valid_days.max() <= 90


class TestDiseaseSeverityDistribution:
    """Test disease_severity distribution."""

    def test_disease_severity_range(self):
        """Test disease_severity is within 0-10 range."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df["disease_severity"].min() >= 0.0
        assert df["disease_severity"].max() <= 10.0

    def test_disease_severity_mean_around_5(self):
        """Test disease_severity mean is around 5."""
        config = GeneratorConfig(n_records=2000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        mean = df["disease_severity"].mean()
        # Should be around 5.0 (configured mean)
        assert 4.0 < mean < 6.0


class TestAcademicHCPDistribution:
    """Test academic_hcp distribution."""

    def test_academic_hcp_is_binary(self):
        """Test academic_hcp has only 0 and 1 values."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        unique_values = df["academic_hcp"].unique()
        assert set(unique_values).issubset({0, 1})

    def test_academic_hcp_proportion_around_30_percent(self):
        """Test academic_hcp is around 30% (as configured)."""
        config = GeneratorConfig(n_records=5000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        proportion = df["academic_hcp"].mean()
        # Should be around 0.30
        assert 0.20 < proportion < 0.40


class TestTrueATEEmbedding:
    """Test TRUE_ATE causal effect embedding."""

    def test_true_ate_stored_in_attrs(self):
        """Test TRUE_ATE is stored in DataFrame attrs."""
        config = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "true_ate" in df.attrs
        assert isinstance(df.attrs["true_ate"], (int, float))

    def test_dgp_type_stored_in_attrs(self):
        """Test DGP type is stored in DataFrame attrs."""
        config = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "dgp_type" in df.attrs
        assert df.attrs["dgp_type"] == "confounded"

    def test_confounders_stored_in_attrs(self):
        """Test confounders are stored in DataFrame attrs."""
        config = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert "confounders" in df.attrs
        assert isinstance(df.attrs["confounders"], list)

    def test_true_ate_value_for_confounded_dgp(self):
        """Test TRUE_ATE value for confounded DGP."""
        config = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        # Confounded DGP should have TRUE_ATE of 0.25 (from DGP_CONFIGS)
        assert df.attrs["true_ate"] == 0.25

    def test_treatment_outcome_correlation(self):
        """Test treatment (engagement) is correlated with outcome."""
        config = GeneratorConfig(n_records=5000, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        # Engagement score should be correlated with treatment_initiated
        corr, p_value = pearsonr(df["engagement_score"], df["treatment_initiated"])

        # Should have positive correlation (TRUE causal effect)
        assert corr > 0
        # Should be statistically significant
        assert p_value < 0.05

    def test_confounder_treatment_correlation(self):
        """Test confounders are correlated with treatment."""
        config = GeneratorConfig(n_records=5000, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        # Disease severity should be correlated with engagement (confounding)
        corr, p_value = pearsonr(df["disease_severity"], df["engagement_score"])

        # Should have positive correlation (confounding structure)
        assert corr > 0

    def test_confounder_outcome_correlation(self):
        """Test confounders are correlated with outcome."""
        config = GeneratorConfig(n_records=5000, seed=42, dgp_type=DGPType.CONFOUNDED)
        gen = PatientGenerator(config)

        df = gen.generate()

        # Disease severity should be correlated with treatment_initiated
        corr, p_value = pearsonr(df["disease_severity"], df["treatment_initiated"])

        # Should have positive correlation (confounding structure)
        assert corr > 0


class TestDGPTypes:
    """Test different DGP types."""

    def test_simple_linear_dgp(self):
        """Test SIMPLE_LINEAR DGP generates valid data."""
        config = GeneratorConfig(n_records=500, seed=42, dgp_type=DGPType.SIMPLE_LINEAR)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df.attrs["dgp_type"] == "simple_linear"
        assert "engagement_score" in df.columns
        assert "treatment_initiated" in df.columns

    def test_heterogeneous_dgp(self):
        """Test HETEROGENEOUS DGP generates valid data."""
        config = GeneratorConfig(n_records=500, seed=42, dgp_type=DGPType.HETEROGENEOUS)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df.attrs["dgp_type"] == "heterogeneous"

    def test_time_series_dgp(self):
        """Test TIME_SERIES DGP generates valid data."""
        config = GeneratorConfig(n_records=500, seed=42, dgp_type=DGPType.TIME_SERIES)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df.attrs["dgp_type"] == "time_series"

    def test_selection_bias_dgp(self):
        """Test SELECTION_BIAS DGP generates valid data."""
        config = GeneratorConfig(n_records=500, seed=42, dgp_type=DGPType.SELECTION_BIAS)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df.attrs["dgp_type"] == "selection_bias"

    def test_different_dgps_have_different_ates(self):
        """Test different DGPs have potentially different TRUE_ATEs."""
        dgp_ates = {}
        for dgp_type in [DGPType.SIMPLE_LINEAR, DGPType.CONFOUNDED, DGPType.SELECTION_BIAS]:
            config = GeneratorConfig(n_records=100, seed=42, dgp_type=dgp_type)
            gen = PatientGenerator(config)
            df = gen.generate()
            dgp_ates[dgp_type.value] = df.attrs["true_ate"]

        # All should have ATEs stored
        assert len(dgp_ates) == 3
        for ate in dgp_ates.values():
            assert ate is not None


class TestHCPIntegration:
    """Test HCP DataFrame integration."""

    @pytest.fixture
    def hcp_df(self):
        """Create HCP DataFrame."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = HCPGenerator(config)
        return gen.generate()

    def test_hcp_ids_from_hcp_df(self, hcp_df):
        """Test hcp_ids come from provided HCP DataFrame."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = PatientGenerator(config, hcp_df=hcp_df)

        df = gen.generate()

        # All hcp_ids should be from hcp_df
        valid_hcp_ids = set(hcp_df["hcp_id"])
        patient_hcp_ids = set(df["hcp_id"])

        assert patient_hcp_ids.issubset(valid_hcp_ids)

    def test_academic_hcp_matching(self, hcp_df):
        """Test academic patients tend to be matched with academic HCPs."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config, hcp_df=hcp_df)

        df = gen.generate()

        # Merge to get HCP academic status
        df_merged = df.merge(
            hcp_df[["hcp_id", "academic_hcp"]],
            on="hcp_id",
            suffixes=("_patient", "_hcp"),
        )

        # Check correlation between patient academic flag and HCP academic flag
        # (when academic HCPs are available)
        if hcp_df["academic_hcp"].sum() > 0:
            academic_patients = df_merged[df_merged["academic_hcp_patient"] == 1]
            if len(academic_patients) > 10:
                # Academic patients should more often have academic HCPs
                prop_academic_hcp = academic_patients["academic_hcp_hcp"].mean()
                assert prop_academic_hcp > 0.2  # At least 20% matching


class TestReproducibility:
    """Test reproducibility with seed."""

    def test_same_seed_same_results(self):
        """Test same seed produces same results."""
        config1 = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)
        config2 = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)

        gen1 = PatientGenerator(config1)
        gen2 = PatientGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Should be identical
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_different_results(self):
        """Test different seeds produce different results."""
        config1 = GeneratorConfig(n_records=100, seed=42, dgp_type=DGPType.CONFOUNDED)
        config2 = GeneratorConfig(n_records=100, seed=123, dgp_type=DGPType.CONFOUNDED)

        gen1 = PatientGenerator(config1)
        gen2 = PatientGenerator(config2)

        df1 = gen1.generate()
        df2 = gen2.generate()

        # Values should be different
        assert not df1["engagement_score"].equals(df2["engagement_score"])


class TestAgeAtDiagnosis:
    """Test age_at_diagnosis values."""

    def test_age_at_diagnosis_range(self):
        """Test age_at_diagnosis is in valid range."""
        config = GeneratorConfig(n_records=1000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        assert df["age_at_diagnosis"].min() >= 18
        assert df["age_at_diagnosis"].max() <= 85

    def test_age_at_diagnosis_is_integer(self):
        """Test age_at_diagnosis values are integers."""
        config = GeneratorConfig(n_records=100, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        # All values should be integers (or float representation of integers)
        for age in df["age_at_diagnosis"]:
            assert float(age) == int(age)


class TestDataSplit:
    """Test data_split assignment."""

    def test_data_split_values_valid(self):
        """Test data_split has valid values."""
        config = GeneratorConfig(n_records=500, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        valid_splits = ["train", "validation", "test", "holdout"]
        for split in df["data_split"].unique():
            assert split in valid_splits

    def test_data_split_approximate_ratios(self):
        """Test data_split follows approximate ratios."""
        config = GeneratorConfig(n_records=2000, seed=42)
        gen = PatientGenerator(config)

        df = gen.generate()

        split_counts = df["data_split"].value_counts(normalize=True)

        # Allow 15% tolerance
        assert abs(split_counts.get("train", 0) - 0.70) < 0.15
        assert abs(split_counts.get("validation", 0) - 0.15) < 0.10
        assert abs(split_counts.get("test", 0) - 0.15) < 0.10
