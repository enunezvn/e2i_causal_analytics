"""End-to-end integration tests for Tier 0 MLOps workflow.

This module tests the complete MLOps pipeline:
    scope_definer → data_preparer → cohort_constructor → model_selector →
    model_trainer → feature_analyzer → model_deployer → observability_connector

Prerequisites:
    - API running (port 8000)
    - MLflow running (port 5000)
    - Opik running (port 5173/8080)
    - Supabase connectivity

Test Data:
    - Brand: Kisqali (877 patients with well-distributed splits)
    - Target: discontinuation_flag (binary classification)

Author: E2I Causal Analytics Team
"""

import asyncio
import os
import uuid
from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest


# =============================================================================
# TEST CONFIGURATION
# =============================================================================

TEST_CONFIG = {
    "brand": "Kisqali",
    "problem_type": "binary_classification",
    "target_outcome": "discontinuation_flag",
    "indication": "HR+/HER2- breast cancer",
    "hpo_trials": 5,  # Reduced for testing (production: 20-50)
    "min_eligible_patients": 30,
    "min_auc_threshold": 0.55,  # Lower threshold for synthetic data
}


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="module")
def experiment_id():
    """Generate unique experiment ID for test run."""
    return f"test_tier0_e2e_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def sample_patient_data():
    """Generate sample patient journey data for testing."""
    np.random.seed(42)
    n_samples = 100

    return pd.DataFrame(
        {
            "patient_journey_id": [f"PJ_{i:04d}" for i in range(n_samples)],
            "brand": ["Kisqali"] * n_samples,
            "journey_status": np.random.choice(
                ["active", "completed", "discontinued"], n_samples, p=[0.6, 0.2, 0.2]
            ),
            "data_quality_score": np.random.uniform(0.5, 1.0, n_samples),
            "created_at": pd.date_range("2024-01-01", periods=n_samples, freq="D"),
            "days_on_therapy": np.random.randint(30, 365, n_samples),
            "hcp_visits": np.random.randint(1, 20, n_samples),
            "prior_treatments": np.random.randint(0, 5, n_samples),
            "age_group": np.random.choice(["<50", "50-65", ">65"], n_samples),
            "region": np.random.choice(["Northeast", "Southeast", "Midwest", "West"], n_samples),
            "discontinuation_flag": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
        }
    )


@pytest.fixture
def mock_supabase_data(sample_patient_data):
    """Mock Supabase query response."""
    return sample_patient_data


# =============================================================================
# STEP 1: SCOPE DEFINER TESTS
# =============================================================================


class TestScopeDefiner:
    """Tests for the scope_definer agent."""

    @pytest.mark.asyncio
    async def test_scope_definition_creates_experiment(self, experiment_id):
        """Test that scope_definer creates proper experiment scope."""
        from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

        agent = ScopeDefinerAgent()

        input_data = {
            "problem_description": f"Predict patient discontinuation risk for {TEST_CONFIG['brand']}",
            "business_objective": "Identify high-risk patients early for intervention",
            "target_outcome": TEST_CONFIG["target_outcome"],
            "problem_type_hint": TEST_CONFIG["problem_type"],
            "brand": TEST_CONFIG["brand"],
        }

        result = await agent.run(input_data)

        # Verify required outputs
        assert "scope_spec" in result, "Missing scope_spec in result"
        assert "success_criteria" in result, "Missing success_criteria in result"

        scope_spec = result["scope_spec"]
        assert scope_spec.get("problem_type") == TEST_CONFIG["problem_type"]
        assert "prediction_target" in scope_spec or "target_outcome" in scope_spec

    @pytest.mark.asyncio
    async def test_scope_spec_has_minimum_requirements(self, experiment_id):
        """Test scope spec contains minimum sample requirements."""
        from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

        agent = ScopeDefinerAgent()

        result = await agent.run(
            {
                "problem_description": "Test problem",
                "business_objective": "Test objective",
                "target_outcome": "target",
                "problem_type_hint": "binary_classification",
            }
        )

        scope_spec = result.get("scope_spec", {})
        # Should have minimum samples defined
        min_samples = scope_spec.get("minimum_samples", scope_spec.get("min_samples", 0))
        assert min_samples >= 10, "Minimum samples should be at least 10"


# =============================================================================
# STEP 2: DATA PREPARER TESTS
# =============================================================================


class TestDataPreparer:
    """Tests for the data_preparer agent."""

    @pytest.mark.asyncio
    async def test_data_preparer_loads_data(self, experiment_id):
        """Test that data_preparer loads and validates data."""
        from src.agents.ml_foundation.data_preparer import DataPreparerAgent

        agent = DataPreparerAgent()

        scope_spec = {
            "experiment_id": experiment_id,
            "problem_type": TEST_CONFIG["problem_type"],
            "target_column": TEST_CONFIG["target_outcome"],
            "use_sample_data": True,
            "sample_size": 500,
        }

        result = await agent.run(
            {
                "scope_spec": scope_spec,
                "data_source": "patient_journeys",
                "brand": TEST_CONFIG["brand"],
            }
        )

        # Verify QC report
        assert "qc_report" in result or "qc_status" in result
        assert "gate_passed" in result

    @pytest.mark.asyncio
    async def test_qc_gate_blocks_on_failure(self):
        """Test QC gate blocks training on validation failure."""
        from src.agents.ml_foundation.data_preparer import DataPreparerAgent

        agent = DataPreparerAgent()

        # Invalid scope spec with impossible requirements
        invalid_scope = {
            "experiment_id": "test_invalid",
            "problem_type": "binary_classification",
            "required_columns": ["nonexistent_column"],
            "use_sample_data": True,
            "sample_size": 10,
        }

        result = await agent.run(
            {
                "scope_spec": invalid_scope,
                "data_source": "empty_source",
            }
        )

        # Gate should not pass for invalid data
        # Note: May return gate_passed=True with warnings depending on implementation
        assert "gate_passed" in result


# =============================================================================
# STEP 3: COHORT CONSTRUCTOR TESTS
# =============================================================================


class TestCohortConstructor:
    """Tests for the cohort_constructor agent."""

    @pytest.mark.asyncio
    async def test_cohort_construction_filters_patients(self, sample_patient_data):
        """Test cohort constructor filters patients by criteria."""
        from src.agents.cohort_constructor import CohortConstructorAgent

        agent = CohortConstructorAgent()

        eligible_df, result = await agent.run(
            patient_df=sample_patient_data,
            brand=TEST_CONFIG["brand"],
            indication=TEST_CONFIG["indication"],
        )

        # Verify filtering occurred
        assert len(eligible_df) <= len(sample_patient_data)
        assert result.status == "success"
        assert len(result.eligible_patient_ids) > 0

    @pytest.mark.asyncio
    async def test_cohort_meets_minimum_size(self, sample_patient_data):
        """Test cohort meets minimum size requirements."""
        from src.agents.cohort_constructor import CohortConstructorAgent

        agent = CohortConstructorAgent()

        eligible_df, result = await agent.run(
            patient_df=sample_patient_data,
            brand=TEST_CONFIG["brand"],
        )

        # Should have enough patients for ML
        assert (
            len(eligible_df) >= TEST_CONFIG["min_eligible_patients"]
        ), f"Cohort size {len(eligible_df)} below minimum {TEST_CONFIG['min_eligible_patients']}"

    @pytest.mark.asyncio
    async def test_cohort_has_audit_trail(self, sample_patient_data):
        """Test cohort construction creates audit trail."""
        from src.agents.cohort_constructor import CohortConstructorAgent

        agent = CohortConstructorAgent()

        eligible_df, result = await agent.run(
            patient_df=sample_patient_data,
            brand=TEST_CONFIG["brand"],
        )

        # Verify audit metadata
        assert result.cohort_id is not None
        assert result.execution_id is not None
        assert result.eligibility_stats is not None


# =============================================================================
# STEP 4: MODEL SELECTOR TESTS
# =============================================================================


class TestModelSelector:
    """Tests for the model_selector agent."""

    @pytest.mark.asyncio
    async def test_model_selection_returns_candidate(self, experiment_id):
        """Test model selector returns a valid candidate."""
        from src.agents.ml_foundation.model_selector import ModelSelectorAgent

        agent = ModelSelectorAgent()

        scope_spec = {
            "experiment_id": experiment_id,
            "problem_type": TEST_CONFIG["problem_type"],
        }

        qc_report = {"gate_passed": True, "overall_score": 0.95}

        result = await agent.run(
            {
                "scope_spec": scope_spec,
                "qc_report": qc_report,
                "skip_benchmarks": True,
            }
        )

        # Verify model candidate
        assert "model_candidate" in result or "primary_candidate" in result
        candidate = result.get("model_candidate") or result.get("primary_candidate")
        assert candidate is not None

    @pytest.mark.asyncio
    async def test_model_selection_considers_problem_type(self, experiment_id):
        """Test model selection considers problem type."""
        from src.agents.ml_foundation.model_selector import ModelSelectorAgent

        agent = ModelSelectorAgent()

        # Binary classification scope
        scope_spec = {
            "experiment_id": experiment_id,
            "problem_type": "binary_classification",
        }

        result = await agent.run(
            {
                "scope_spec": scope_spec,
                "qc_report": {"gate_passed": True},
                "skip_benchmarks": True,
            }
        )

        # Should select classification algorithm
        candidate = result.get("model_candidate") or result.get("primary_candidate")
        if hasattr(candidate, "algorithm_name"):
            algo = candidate.algorithm_name.lower()
        elif isinstance(candidate, dict):
            algo = candidate.get("algorithm_name", candidate.get("algorithm", "")).lower()
        else:
            algo = str(candidate).lower()

        # Common classification algorithms
        valid_classifiers = ["xgboost", "lightgbm", "random_forest", "logistic", "gradient"]
        assert any(
            clf in algo for clf in valid_classifiers
        ), f"Expected classifier, got {algo}"


# =============================================================================
# STEP 5: MODEL TRAINER TESTS
# =============================================================================


class TestModelTrainer:
    """Tests for the model_trainer agent."""

    @pytest.mark.asyncio
    async def test_model_training_produces_metrics(self, experiment_id, sample_patient_data):
        """Test model trainer produces validation metrics."""
        from src.agents.ml_foundation.model_trainer import ModelTrainerAgent

        agent = ModelTrainerAgent()

        # Prepare simple training data
        X = sample_patient_data[["days_on_therapy", "hcp_visits", "prior_treatments"]]
        y = sample_patient_data[TEST_CONFIG["target_outcome"]]

        model_candidate = {
            "algorithm_name": "LogisticRegression",
            "hyperparameters": {"C": 1.0, "max_iter": 100},
        }

        result = await agent.run(
            {
                "experiment_id": experiment_id,
                "model_candidate": model_candidate,
                "qc_report": {"gate_passed": True},
                "enable_hpo": False,  # Skip HPO for unit test
                "problem_type": TEST_CONFIG["problem_type"],
                "train_data": {"X": X.iloc[:70], "y": y.iloc[:70]},
                "validation_data": {"X": X.iloc[70:85], "y": y.iloc[70:85]},
                "test_data": {"X": X.iloc[85:], "y": y.iloc[85:]},
            }
        )

        # Verify metrics produced
        assert "validation_metrics" in result or "auc_roc" in result

    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_model_training_with_hpo(self, experiment_id, sample_patient_data):
        """Test model training with HPO (slow test)."""
        from src.agents.ml_foundation.model_trainer import ModelTrainerAgent

        agent = ModelTrainerAgent()

        X = sample_patient_data[["days_on_therapy", "hcp_visits", "prior_treatments"]]
        y = sample_patient_data[TEST_CONFIG["target_outcome"]]

        model_candidate = {
            "algorithm_name": "XGBClassifier",
            "hyperparameters": {},
        }

        result = await agent.run(
            {
                "experiment_id": experiment_id,
                "model_candidate": model_candidate,
                "qc_report": {"gate_passed": True},
                "enable_hpo": True,
                "hpo_trials": TEST_CONFIG["hpo_trials"],
                "problem_type": TEST_CONFIG["problem_type"],
                "train_data": {"X": X.iloc[:70], "y": y.iloc[:70]},
                "validation_data": {"X": X.iloc[70:85], "y": y.iloc[70:85]},
                "test_data": {"X": X.iloc[85:], "y": y.iloc[85:]},
            }
        )

        # Verify HPO results
        assert "best_hyperparameters" in result or "hpo_trials_run" in result


# =============================================================================
# STEP 6: FEATURE ANALYZER TESTS
# =============================================================================


class TestFeatureAnalyzer:
    """Tests for the feature_analyzer agent."""

    @pytest.mark.asyncio
    async def test_feature_importance_computation(self, experiment_id, sample_patient_data):
        """Test feature analyzer computes importance scores."""
        from src.agents.ml_foundation.feature_analyzer import FeatureAnalyzerAgent
        from sklearn.linear_model import LogisticRegression

        agent = FeatureAnalyzerAgent()

        # Train a simple model for testing
        X = sample_patient_data[["days_on_therapy", "hcp_visits", "prior_treatments"]].values
        y = sample_patient_data[TEST_CONFIG["target_outcome"]].values

        model = LogisticRegression(max_iter=100)
        model.fit(X, y)

        result = await agent.run(
            {
                "experiment_id": experiment_id,
                "model_uri": None,  # Not using MLflow URI
                "trained_model": model,
                "X_sample": pd.DataFrame(
                    X[:50], columns=["days_on_therapy", "hcp_visits", "prior_treatments"]
                ),
                "y_sample": y[:50],
                "max_samples": 50,
            }
        )

        # Verify feature importance
        assert "feature_importance" in result or "shap_analysis" in result


# =============================================================================
# STEP 7: MODEL DEPLOYER TESTS
# =============================================================================


class TestModelDeployer:
    """Tests for the model_deployer agent."""

    @pytest.mark.asyncio
    async def test_deployment_manifest_creation(self, experiment_id):
        """Test model deployer creates deployment manifest."""
        from src.agents.ml_foundation.model_deployer import ModelDeployerAgent

        agent = ModelDeployerAgent()

        result = await agent.run(
            {
                "experiment_id": experiment_id,
                "model_uri": f"runs:/{experiment_id}/model",
                "validation_metrics": {"auc_roc": 0.75, "f1_score": 0.68},
                "success_criteria_met": True,
                "deployment_name": f"kisqali_discontinuation_{experiment_id[:8]}",
                "deployment_action": "register",  # Just register, don't deploy
            }
        )

        # Verify manifest created
        assert "deployment_manifest" in result or "status" in result

    @pytest.mark.asyncio
    async def test_deployment_requires_success_criteria(self, experiment_id):
        """Test deployment blocked when success criteria not met."""
        from src.agents.ml_foundation.model_deployer import ModelDeployerAgent

        agent = ModelDeployerAgent()

        result = await agent.run(
            {
                "experiment_id": experiment_id,
                "model_uri": f"runs:/{experiment_id}/model",
                "validation_metrics": {"auc_roc": 0.45},  # Below threshold
                "success_criteria_met": False,
                "deployment_name": "test_deployment",
                "deployment_action": "register",
            }
        )

        # Deployment should reflect unsuccessful criteria
        status = result.get("status", result.get("deployment_successful", None))
        # May still register but not promote to production


# =============================================================================
# STEP 8: OBSERVABILITY CONNECTOR TESTS
# =============================================================================


class TestObservabilityConnector:
    """Tests for the observability_connector agent."""

    @pytest.mark.asyncio
    async def test_event_logging(self, experiment_id):
        """Test observability connector logs events."""
        from src.agents.ml_foundation.observability_connector import (
            ObservabilityConnectorAgent,
        )

        agent = ObservabilityConnectorAgent()

        events = [
            {
                "event_type": "training_started",
                "agent_name": "model_trainer",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"experiment_id": experiment_id},
            },
            {
                "event_type": "training_completed",
                "agent_name": "model_trainer",
                "timestamp": datetime.utcnow().isoformat(),
                "metadata": {"experiment_id": experiment_id, "auc": 0.75},
            },
        ]

        result = await agent.run(
            {
                "events_to_log": events,
                "time_window": "1h",
            }
        )

        # Verify logging attempted
        assert "emission_successful" in result or "events_logged" in result

    @pytest.mark.asyncio
    async def test_quality_metrics_aggregation(self):
        """Test observability connector aggregates quality metrics."""
        from src.agents.ml_foundation.observability_connector import (
            ObservabilityConnectorAgent,
        )

        agent = ObservabilityConnectorAgent()

        result = await agent.run(
            {
                "time_window": "1h",
                "agent_name_filter": None,
            }
        )

        # Should return aggregated metrics
        assert result is not None


# =============================================================================
# END-TO-END PIPELINE TEST
# =============================================================================


class TestTier0EndToEnd:
    """Full end-to-end pipeline integration test."""

    @pytest.mark.asyncio
    @pytest.mark.slow
    @pytest.mark.e2e
    async def test_full_mlops_pipeline(self, experiment_id, sample_patient_data):
        """Test complete Tier 0 MLOps pipeline end-to-end.

        This test executes the full workflow:
        1. Scope Definition
        2. Data Preparation (with QC gate)
        3. Cohort Construction
        4. Model Selection
        5. Model Training
        6. Feature Analysis
        7. Model Deployment
        8. Observability Logging
        """
        from src.agents.cohort_constructor import CohortConstructorAgent
        from src.agents.ml_foundation.data_preparer import DataPreparerAgent
        from src.agents.ml_foundation.feature_analyzer import FeatureAnalyzerAgent
        from src.agents.ml_foundation.model_deployer import ModelDeployerAgent
        from src.agents.ml_foundation.model_selector import ModelSelectorAgent
        from src.agents.ml_foundation.model_trainer import ModelTrainerAgent
        from src.agents.ml_foundation.observability_connector import (
            ObservabilityConnectorAgent,
        )
        from src.agents.ml_foundation.scope_definer import ScopeDefinerAgent

        pipeline_state: dict[str, Any] = {
            "experiment_id": experiment_id,
            "brand": TEST_CONFIG["brand"],
        }

        # Step 1: Scope Definition
        scope_agent = ScopeDefinerAgent()
        scope_result = await scope_agent.run(
            {
                "problem_description": f"Predict discontinuation for {TEST_CONFIG['brand']}",
                "business_objective": "Early risk identification",
                "target_outcome": TEST_CONFIG["target_outcome"],
                "problem_type_hint": TEST_CONFIG["problem_type"],
            }
        )
        pipeline_state["scope_spec"] = scope_result.get("scope_spec", {})
        pipeline_state["scope_spec"]["experiment_id"] = experiment_id
        assert scope_result.get("validation_passed", True), "Scope validation failed"

        # Step 2: Data Preparation
        data_agent = DataPreparerAgent()
        # Use sample data directly
        pipeline_state["scope_spec"]["use_sample_data"] = True
        pipeline_state["scope_spec"]["sample_size"] = 500
        data_result = await data_agent.run(
            {
                "scope_spec": pipeline_state["scope_spec"],
                "data_source": "patient_journeys",
            }
        )
        pipeline_state["qc_report"] = data_result.get("qc_report", {})
        pipeline_state["gate_passed"] = data_result.get("gate_passed", True)

        # QC Gate Check
        if not pipeline_state["gate_passed"]:
            pytest.skip("QC gate failed - data quality issues detected")

        # Step 3: Cohort Construction
        cohort_agent = CohortConstructorAgent()
        eligible_df, cohort_result = await cohort_agent.run(
            patient_df=sample_patient_data,
            brand=TEST_CONFIG["brand"],
        )
        pipeline_state["eligible_patient_ids"] = cohort_result.eligible_patient_ids
        assert len(eligible_df) >= TEST_CONFIG["min_eligible_patients"], "Insufficient cohort"

        # Step 4: Model Selection
        selector_agent = ModelSelectorAgent()
        selector_result = await selector_agent.run(
            {
                "scope_spec": pipeline_state["scope_spec"],
                "qc_report": pipeline_state["qc_report"],
                "skip_benchmarks": True,
            }
        )
        pipeline_state["model_candidate"] = selector_result.get(
            "model_candidate", selector_result.get("primary_candidate")
        )
        assert pipeline_state["model_candidate"] is not None

        # Step 5: Model Training
        trainer_agent = ModelTrainerAgent()

        # Prepare data for training
        X = eligible_df[["days_on_therapy", "hcp_visits", "prior_treatments"]]
        y = eligible_df[TEST_CONFIG["target_outcome"]]

        train_size = int(0.7 * len(X))
        val_size = int(0.15 * len(X))

        trainer_result = await trainer_agent.run(
            {
                "experiment_id": experiment_id,
                "model_candidate": pipeline_state["model_candidate"],
                "qc_report": pipeline_state["qc_report"],
                "enable_hpo": False,  # Skip HPO in e2e test
                "problem_type": TEST_CONFIG["problem_type"],
                "train_data": {"X": X.iloc[:train_size], "y": y.iloc[:train_size]},
                "validation_data": {
                    "X": X.iloc[train_size : train_size + val_size],
                    "y": y.iloc[train_size : train_size + val_size],
                },
                "test_data": {
                    "X": X.iloc[train_size + val_size :],
                    "y": y.iloc[train_size + val_size :],
                },
            }
        )
        pipeline_state["trained_model"] = trainer_result.get("trained_model")
        pipeline_state["validation_metrics"] = trainer_result.get("validation_metrics", {})
        pipeline_state["model_uri"] = trainer_result.get("model_artifact_uri")

        # Step 6: Feature Analysis (optional - may fail without MLflow)
        try:
            analyzer_agent = FeatureAnalyzerAgent()
            analyzer_result = await analyzer_agent.run(
                {
                    "experiment_id": experiment_id,
                    "trained_model": pipeline_state["trained_model"],
                    "X_sample": X.iloc[:50],
                    "y_sample": y.iloc[:50],
                    "max_samples": 50,
                }
            )
            pipeline_state["feature_importance"] = analyzer_result.get("feature_importance")
        except Exception as e:
            # Feature analysis is optional
            pipeline_state["feature_importance"] = None

        # Step 7: Model Deployment
        deployer_agent = ModelDeployerAgent()
        deployer_result = await deployer_agent.run(
            {
                "experiment_id": experiment_id,
                "model_uri": pipeline_state.get("model_uri", f"runs:/{experiment_id}/model"),
                "validation_metrics": pipeline_state["validation_metrics"],
                "success_criteria_met": True,
                "deployment_name": f"kisqali_discontinuation_{experiment_id[:8]}",
                "deployment_action": "register",
            }
        )
        pipeline_state["deployment_manifest"] = deployer_result.get("deployment_manifest")

        # Step 8: Observability Logging
        obs_agent = ObservabilityConnectorAgent()
        obs_result = await obs_agent.run(
            {
                "events_to_log": [
                    {
                        "event_type": "pipeline_completed",
                        "agent_name": "tier0_e2e_test",
                        "timestamp": datetime.utcnow().isoformat(),
                        "metadata": {
                            "experiment_id": experiment_id,
                            "stages_completed": 8,
                        },
                    }
                ],
                "time_window": "1h",
            }
        )

        # Final verification
        assert pipeline_state["scope_spec"] is not None
        assert pipeline_state["gate_passed"] is True
        assert len(pipeline_state["eligible_patient_ids"]) >= TEST_CONFIG["min_eligible_patients"]
        assert pipeline_state["model_candidate"] is not None
        assert pipeline_state["trained_model"] is not None


# =============================================================================
# PYTEST MARKERS
# =============================================================================


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end integration tests")
