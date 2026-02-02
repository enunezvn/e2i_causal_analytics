"""End-to-End Tests for ML Foundation Pipeline.

Tests the complete Tier 0 pipeline orchestration:
- Full pipeline execution with all 7 agents
- QC gate enforcement (data quality validation)
- Resume from specific stages
- Handoff protocol validation
- Error handling and recovery
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.tier_0 import MLFoundationPipeline, PipelineConfig, PipelineStage
from src.agents.tier_0.handoff_protocols import (
    validate_data_to_selector_handoff,
    validate_scope_to_data_handoff,
    validate_selector_to_trainer_handoff,
    validate_trainer_to_deployer_handoff,
)

# =============================================================================
# TEST FIXTURES
# =============================================================================


@pytest.fixture
def sample_input_data():
    """Sample input data for pipeline execution."""
    return {
        "problem_description": "Predict HCP conversion probability",
        "business_objective": "Increase market share for Kisqali",
        "target_outcome": "hcp_conversion",
        "brand": "Kisqali",
        "region": "northeast",
        "data_source": "business_metrics",
        "use_sample_data": True,
    }


@pytest.fixture
def mock_scope_definer_output():
    """Mock output from ScopeDefinerAgent."""
    return {
        "status": "completed",
        "experiment_id": "exp_test_001",
        "experiment_name": "hcp_conversion_prediction",
        "scope_spec": {
            "experiment_id": "exp_test_001",
            "experiment_name": "hcp_conversion_prediction",
            "problem_type": "binary_classification",
            "prediction_target": "hcp_conversion",
            "prediction_horizon_days": 30,
            "target_population": "HCPs in northeast region",
            "inclusion_criteria": ["active_hcp", "kisqali_relevant"],
            "exclusion_criteria": ["inactive_6_months"],
            "required_features": ["engagement_score", "visit_count", "territory"],
            "excluded_features": ["pii_fields"],
            "feature_categories": ["engagement", "demographics", "activity"],
            "regulatory_constraints": [],
            "ethical_constraints": ["fair_ml"],
            "technical_constraints": ["latency_100ms"],
            "minimum_samples": 1000,
            "brand": "Kisqali",
            "region": "northeast",
            "use_case": "hcp_conversion",
            "use_sample_data": True,
            "created_by": "scope_definer",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "success_criteria": {
            "experiment_id": "exp_test_001",
            "min_auc": 0.75,
            "min_precision": 0.70,
            "min_recall": 0.70,
            "max_inference_latency_ms": 100,
            "min_samples": 1000,
            "max_model_size_mb": 100,
            "min_improvement_over_baseline": 0.05,
        },
    }


@pytest.fixture
def mock_data_preparer_output_passed():
    """Mock output from DataPreparerAgent with QC passed."""
    return {
        "status": "completed",
        "experiment_id": "exp_test_001",
        "gate_passed": True,  # CRITICAL: Pipeline checks this at top level
        "qc_report": {
            "report_id": "qc_test_001",
            "experiment_id": "exp_test_001",
            "status": "passed",
            "overall_score": 0.92,
            "qc_passed": True,
            "completeness_score": 0.95,
            "validity_score": 0.90,
            "consistency_score": 0.88,
            "uniqueness_score": 0.98,
            "timeliness_score": 0.85,
            "expectation_results": [],
            "failed_expectations": [],
            "warnings": [],
            "remediation_steps": [],
            "blocking_issues": [],
            "row_count": 5000,
            "column_count": 25,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        },
        "baseline_metrics": {
            "experiment_id": "exp_test_001",
            "split_type": "train",
            "feature_stats": {
                "engagement_score": {"mean": 0.65, "std": 0.15, "min": 0.1, "max": 1.0},
                "visit_count": {"mean": 5.2, "std": 2.1, "min": 0, "max": 20},
            },
            "target_rate": 0.23,
            "target_distribution": {"0": 0.77, "1": 0.23},
            "training_samples": 3500,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        },
    }


@pytest.fixture
def mock_data_preparer_output_failed():
    """Mock output from DataPreparerAgent with QC failed."""
    return {
        "status": "completed",
        "experiment_id": "exp_test_001",
        "gate_passed": False,  # CRITICAL: Pipeline checks this at top level
        "qc_report": {
            "report_id": "qc_test_002",
            "experiment_id": "exp_test_001",
            "status": "failed",
            "overall_score": 0.45,
            "qc_passed": False,
            "completeness_score": 0.40,
            "validity_score": 0.50,
            "consistency_score": 0.45,
            "uniqueness_score": 0.90,
            "timeliness_score": 0.35,
            "expectation_results": [],
            "failed_expectations": ["expect_column_values_to_not_be_null"],
            "warnings": ["High null rate in engagement_score"],
            "remediation_steps": ["Impute missing values"],
            "blocking_issues": ["40% missing data in required features"],
            "row_count": 1200,
            "column_count": 25,
            "validated_at": datetime.now(timezone.utc).isoformat(),
        },
        "baseline_metrics": None,
    }


@pytest.fixture
def mock_model_selector_output():
    """Mock output from ModelSelectorAgent."""
    return {
        "status": "completed",
        "experiment_id": "exp_test_001",
        "model_candidate": {
            "algorithm_name": "XGBoost",
            "algorithm_class": "xgboost.XGBClassifier",
            "algorithm_family": "tree",
            "default_hyperparameters": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            "hyperparameter_search_space": {
                "n_estimators": {"type": "int", "low": 50, "high": 300},
                "max_depth": {"type": "int", "low": 3, "high": 10},
                "learning_rate": {"type": "float", "low": 0.01, "high": 0.3},
            },
            "expected_performance": {"auc": 0.82, "precision": 0.78, "recall": 0.75},
            "training_time_estimate_hours": 0.5,
            "estimated_inference_latency_ms": 15,
            "memory_requirement_gb": 2.0,
            "interpretability_score": 0.75,
            "scalability_score": 0.85,
            "selection_score": 0.82,
        },
        "selection_rationale": "XGBoost selected for high interpretability and fast inference",
    }


@pytest.fixture
def mock_model_trainer_output():
    """Mock output from ModelTrainerAgent."""
    return {
        "status": "completed",
        "experiment_id": "exp_test_001",
        "training_run_id": "run_test_001",
        "model_id": "model_test_001",
        "model_uri": "mlflow://models/hcp_conversion/1",
        "trained_model": MagicMock(),  # Mock model object
        "train_metrics": {"auc": 0.88, "precision": 0.82, "recall": 0.79},
        "validation_metrics": {"auc": 0.85, "precision": 0.80, "recall": 0.77},
        "test_metrics": {"auc": 0.84, "precision": 0.79, "recall": 0.76},
        "auc_roc": 0.84,
        "precision": 0.79,
        "recall": 0.76,
        "f1_score": 0.775,
        "success_criteria_met": True,
        "success_criteria_results": {
            "min_auc": True,
            "min_precision": True,
            "min_recall": True,
        },
        "best_hyperparameters": {
            "n_estimators": 150,
            "max_depth": 7,
            "learning_rate": 0.08,
        },
        "hpo_completed": True,
        "hpo_trials_run": 20,
        "mlflow_run_id": "mlflow_run_001",
        "model_artifact_uri": "s3://models/exp_test_001/model",
        "training_duration_seconds": 1250.5,
        "algorithm_name": "XGBoost",
    }


@pytest.fixture
def mock_feature_analyzer_output():
    """Mock output from FeatureAnalyzerAgent."""
    return {
        "status": "completed",
        "experiment_id": "exp_test_001",
        "model_version": "1",
        "shap_analysis": {
            "experiment_id": "exp_test_001",
            "model_version": "1",
            "shap_analysis_id": "shap_test_001",
            "feature_importance": [
                {"feature": "engagement_score", "importance": 0.35, "rank": 1},
                {"feature": "visit_count", "importance": 0.25, "rank": 2},
                {"feature": "territory", "importance": 0.15, "rank": 3},
            ],
            "interactions": [{"features": ["engagement_score", "visit_count"], "strength": 0.12}],
            "top_features": ["engagement_score", "visit_count", "territory"],
            "interpretation": "Engagement score is the strongest predictor of HCP conversion",
            "executive_summary": "Model relies primarily on HCP engagement metrics",
            "key_insights": [
                "Engagement score drives 35% of predictions",
                "Visit frequency is second most important",
            ],
            "recommendations": [
                "Focus on improving engagement score tracking",
                "Consider adding more engagement-related features",
            ],
            "samples_analyzed": 1000,
            "computation_time_seconds": 45.2,
        },
        "feature_importance": [
            {"feature": "engagement_score", "importance": 0.35, "rank": 1},
            {"feature": "visit_count", "importance": 0.25, "rank": 2},
        ],
    }


@pytest.fixture
def mock_model_deployer_output():
    """Mock output from ModelDeployerAgent."""
    return {
        "status": "completed",
        "deployment_successful": True,
        "health_check_passed": True,
        "deployment_manifest": {
            "deployment_id": "deploy_test_001",
            "experiment_id": "exp_test_001",
            "model_uri": "mlflow://models/hcp_conversion/1",
            "environment": "staging",
            "endpoint_url": "https://staging.e2i.io/predict/hcp_conversion",
            "resources": {"cpu": "2", "memory": "4Gi"},
        },
        "version_record": {
            "version": 1,
            "stage": "Staging",
            "promoted_at": datetime.now(timezone.utc).isoformat(),
        },
        "bentoml_tag": "e2i_exp_test_001_model:v1",
        "endpoint_url": "https://staging.e2i.io/predict/hcp_conversion",
        "endpoint_name": "hcp_conversion_v1",
        "rollback_available": False,
        "previous_deployment_id": None,
    }


# =============================================================================
# MOCK AGENT FACTORY
# =============================================================================


def create_mock_agent(output):
    """Create a mock agent that returns the given output."""
    agent = MagicMock()
    agent.run = AsyncMock(return_value=output)
    return agent


# =============================================================================
# PIPELINE EXECUTION TESTS
# =============================================================================


class TestMLFoundationPipelineExecution:
    """Tests for complete pipeline execution."""

    @pytest.mark.asyncio
    async def test_full_pipeline_success(
        self,
        sample_input_data,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
        mock_model_trainer_output,
        mock_feature_analyzer_output,
        mock_model_deployer_output,
    ):
        """Test successful execution of complete pipeline."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            # Configure mock agents
            mock_get_agent.side_effect = lambda name: {
                "scope_definer": create_mock_agent(mock_scope_definer_output),
                "data_preparer": create_mock_agent(mock_data_preparer_output_passed),
                "model_selector": create_mock_agent(mock_model_selector_output),
                "model_trainer": create_mock_agent(mock_model_trainer_output),
                "feature_analyzer": create_mock_agent(mock_feature_analyzer_output),
                "model_deployer": create_mock_agent(mock_model_deployer_output),
            }[name]

            pipeline = MLFoundationPipeline()
            result = await pipeline.run(sample_input_data)

            # Verify pipeline completed
            assert result.status == "completed"
            assert result.current_stage == PipelineStage.COMPLETED
            assert len(result.errors) == 0

            # Verify all stage outputs present
            assert result.scope_spec is not None
            assert result.qc_report is not None
            assert result.model_candidate is not None
            assert result.training_result is not None
            assert result.shap_analysis is not None
            assert result.deployment_result is not None

            # Verify experiment ID propagated
            assert result.experiment_id == "exp_test_001"

            # Verify timing recorded
            assert len(result.stage_timings) > 0
            assert result.total_duration_seconds is not None

    @pytest.mark.asyncio
    async def test_pipeline_skip_deployment(
        self,
        sample_input_data,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
        mock_model_trainer_output,
        mock_feature_analyzer_output,
    ):
        """Test pipeline with deployment skipped."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            mock_get_agent.side_effect = lambda name: {
                "scope_definer": create_mock_agent(mock_scope_definer_output),
                "data_preparer": create_mock_agent(mock_data_preparer_output_passed),
                "model_selector": create_mock_agent(mock_model_selector_output),
                "model_trainer": create_mock_agent(mock_model_trainer_output),
                "feature_analyzer": create_mock_agent(mock_feature_analyzer_output),
            }[name]

            config = PipelineConfig(skip_deployment=True)
            pipeline = MLFoundationPipeline(config=config)
            result = await pipeline.run(sample_input_data)

            assert result.status == "completed"
            assert result.deployment_result is None


# =============================================================================
# QC GATE TESTS
# =============================================================================


class TestQCGateEnforcement:
    """Tests for QC gate behavior."""

    @pytest.mark.asyncio
    async def test_qc_gate_blocks_pipeline_on_failure(
        self,
        sample_input_data,
        mock_scope_definer_output,
        mock_data_preparer_output_failed,
    ):
        """Test that pipeline stops when QC fails."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            mock_get_agent.side_effect = lambda name: {
                "scope_definer": create_mock_agent(mock_scope_definer_output),
                "data_preparer": create_mock_agent(mock_data_preparer_output_failed),
            }[name]

            pipeline = MLFoundationPipeline()

            # Pipeline should either raise RuntimeError or return failed status
            try:
                result = await pipeline.run(sample_input_data)
                # If no exception, check for failed status
                assert result.status == "failed" or result.current_stage == PipelineStage.FAILED
                assert len(result.errors) > 0
            except RuntimeError as e:
                # Verify QC gate error
                assert "QC" in str(e) or "gate" in str(e).lower() or "failed" in str(e).lower()

    @pytest.mark.asyncio
    async def test_qc_gate_passes_with_good_data(
        self,
        sample_input_data,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
        mock_model_trainer_output,
        mock_feature_analyzer_output,
        mock_model_deployer_output,
    ):
        """Test that pipeline continues when QC passes."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            mock_get_agent.side_effect = lambda name: {
                "scope_definer": create_mock_agent(mock_scope_definer_output),
                "data_preparer": create_mock_agent(mock_data_preparer_output_passed),
                "model_selector": create_mock_agent(mock_model_selector_output),
                "model_trainer": create_mock_agent(mock_model_trainer_output),
                "feature_analyzer": create_mock_agent(mock_feature_analyzer_output),
                "model_deployer": create_mock_agent(mock_model_deployer_output),
            }[name]

            pipeline = MLFoundationPipeline()
            result = await pipeline.run(sample_input_data)

            # Verify QC passed
            assert result.qc_report["qc_passed"] is True
            assert result.status == "completed"


# =============================================================================
# RESUME FROM STAGE TESTS
# =============================================================================


class TestResumeFromStage:
    """Tests for resume from specific stage functionality."""

    @pytest.mark.asyncio
    async def test_resume_from_model_training(
        self,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
        mock_model_trainer_output,
        mock_feature_analyzer_output,
        mock_model_deployer_output,
    ):
        """Test resuming pipeline from model training stage."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            mock_get_agent.side_effect = lambda name: {
                "model_trainer": create_mock_agent(mock_model_trainer_output),
                "feature_analyzer": create_mock_agent(mock_feature_analyzer_output),
                "model_deployer": create_mock_agent(mock_model_deployer_output),
            }[name]

            # Create previous result up to model selection
            from src.agents.tier_0.pipeline import PipelineResult

            previous_result = PipelineResult(
                pipeline_run_id="run_prev",
                status="partial",
                current_stage=PipelineStage.MODEL_SELECTION,
                experiment_id="exp_test_001",
                scope_spec=mock_scope_definer_output["scope_spec"],
                success_criteria=mock_scope_definer_output["success_criteria"],
                qc_report=mock_data_preparer_output_passed["qc_report"],
                baseline_metrics=mock_data_preparer_output_passed["baseline_metrics"],
                model_candidate=mock_model_selector_output["model_candidate"],
                stages_completed=["scope_definition", "data_preparation", "model_selection"],
            )

            pipeline = MLFoundationPipeline()
            result = await pipeline.run_from_stage(
                stage=PipelineStage.MODEL_TRAINING,
                previous_result=previous_result,
                input_data={},
            )

            # Verify resumed from training
            assert result.status == "completed"
            assert result.training_result is not None
            assert result.shap_analysis is not None
            assert result.deployment_result is not None

            # Verify previous outputs preserved
            assert result.scope_spec == mock_scope_definer_output["scope_spec"]
            assert result.qc_report == mock_data_preparer_output_passed["qc_report"]

    @pytest.mark.asyncio
    async def test_resume_from_feature_analysis(
        self,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
        mock_model_trainer_output,
        mock_feature_analyzer_output,
        mock_model_deployer_output,
    ):
        """Test resuming pipeline from feature analysis stage."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            mock_get_agent.side_effect = lambda name: {
                "feature_analyzer": create_mock_agent(mock_feature_analyzer_output),
                "model_deployer": create_mock_agent(mock_model_deployer_output),
            }[name]

            from src.agents.tier_0.pipeline import PipelineResult

            previous_result = PipelineResult(
                pipeline_run_id="run_prev",
                status="partial",
                current_stage=PipelineStage.MODEL_TRAINING,
                experiment_id="exp_test_001",
                scope_spec=mock_scope_definer_output["scope_spec"],
                qc_report=mock_data_preparer_output_passed["qc_report"],
                model_candidate=mock_model_selector_output["model_candidate"],
                training_result=mock_model_trainer_output,
                stages_completed=[
                    "scope_definition",
                    "data_preparation",
                    "model_selection",
                    "model_training",
                ],
            )

            pipeline = MLFoundationPipeline()
            result = await pipeline.run_from_stage(
                stage=PipelineStage.FEATURE_ANALYSIS,
                previous_result=previous_result,
                input_data={},
            )

            assert result.status == "completed"
            assert result.shap_analysis is not None
            assert result.training_result == mock_model_trainer_output


# =============================================================================
# HANDOFF PROTOCOL VALIDATION TESTS
# =============================================================================


class TestHandoffProtocolValidation:
    """Tests for handoff protocol validation functions."""

    def test_validate_scope_to_data_handoff_valid(self, mock_scope_definer_output):
        """Test valid scope to data handoff."""
        handoff_data = {
            "scope_spec": mock_scope_definer_output["scope_spec"],
            "experiment_id": mock_scope_definer_output["experiment_id"],
            "success_criteria": mock_scope_definer_output["success_criteria"],
        }

        is_valid, errors = validate_scope_to_data_handoff(handoff_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_scope_to_data_handoff_invalid_problem_type(self, mock_scope_definer_output):
        """Test invalid problem type in scope spec."""
        scope_spec = mock_scope_definer_output["scope_spec"].copy()
        scope_spec["problem_type"] = "invalid_type"

        handoff_data = {
            "scope_spec": scope_spec,
            "experiment_id": mock_scope_definer_output["experiment_id"],
        }

        is_valid, errors = validate_scope_to_data_handoff(handoff_data)

        assert is_valid is False
        assert any("Invalid problem_type" in e for e in errors)

    def test_validate_scope_to_data_handoff_mismatched_experiment_id(
        self, mock_scope_definer_output
    ):
        """Test mismatched experiment IDs."""
        handoff_data = {
            "scope_spec": mock_scope_definer_output["scope_spec"],
            "experiment_id": "different_experiment_id",
        }

        is_valid, errors = validate_scope_to_data_handoff(handoff_data)

        assert is_valid is False
        assert any("experiment_id must match" in e for e in errors)

    def test_validate_data_to_selector_handoff_qc_passed(
        self, mock_scope_definer_output, mock_data_preparer_output_passed
    ):
        """Test valid data to selector handoff with QC passed."""
        handoff_data = {
            "scope_spec": mock_scope_definer_output["scope_spec"],
            "qc_report": mock_data_preparer_output_passed["qc_report"],
        }

        is_valid, errors = validate_data_to_selector_handoff(handoff_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_data_to_selector_handoff_qc_failed(
        self, mock_scope_definer_output, mock_data_preparer_output_failed
    ):
        """Test data to selector handoff with QC failed."""
        handoff_data = {
            "scope_spec": mock_scope_definer_output["scope_spec"],
            "qc_report": mock_data_preparer_output_failed["qc_report"],
        }

        is_valid, errors = validate_data_to_selector_handoff(handoff_data)

        assert is_valid is False
        assert any("QC GATE FAILED" in e for e in errors)

    def test_validate_data_to_selector_handoff_blocking_issues(
        self, mock_scope_definer_output, mock_data_preparer_output_failed
    ):
        """Test data to selector handoff with blocking issues."""
        handoff_data = {
            "scope_spec": mock_scope_definer_output["scope_spec"],
            "qc_report": mock_data_preparer_output_failed["qc_report"],
        }

        is_valid, errors = validate_data_to_selector_handoff(handoff_data)

        assert is_valid is False
        assert any("blocking issues" in e for e in errors)

    def test_validate_selector_to_trainer_handoff_valid(
        self,
        mock_model_selector_output,
        mock_data_preparer_output_passed,
    ):
        """Test valid selector to trainer handoff."""
        handoff_data = {
            "model_candidate": mock_model_selector_output["model_candidate"],
            "qc_report": mock_data_preparer_output_passed["qc_report"],
            "experiment_id": "exp_test_001",
        }

        is_valid, errors = validate_selector_to_trainer_handoff(handoff_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_selector_to_trainer_handoff_missing_algorithm(self):
        """Test selector to trainer handoff with missing algorithm."""
        handoff_data = {
            "model_candidate": {
                "algorithm_family": "tree",
                # Missing algorithm_name and algorithm_class
            },
            "qc_report": {"qc_passed": True},
            "experiment_id": "exp_test_001",
        }

        is_valid, errors = validate_selector_to_trainer_handoff(handoff_data)

        assert is_valid is False
        assert any("algorithm_name" in e for e in errors)

    def test_validate_trainer_to_deployer_handoff_staging(self, mock_model_trainer_output):
        """Test valid trainer to deployer handoff for staging."""
        handoff_data = {
            "model_uri": mock_model_trainer_output["model_uri"],
            "experiment_id": mock_model_trainer_output["experiment_id"],
            "validation_metrics": mock_model_trainer_output["validation_metrics"],
            "success_criteria_met": mock_model_trainer_output["success_criteria_met"],
            "target_environment": "staging",
        }

        is_valid, errors = validate_trainer_to_deployer_handoff(handoff_data)

        assert is_valid is True
        assert len(errors) == 0

    def test_validate_trainer_to_deployer_handoff_production_missing_shadow(
        self, mock_model_trainer_output
    ):
        """Test production deployment without sufficient shadow mode."""
        handoff_data = {
            "model_uri": mock_model_trainer_output["model_uri"],
            "experiment_id": mock_model_trainer_output["experiment_id"],
            "validation_metrics": mock_model_trainer_output["validation_metrics"],
            "success_criteria_met": True,
            "target_environment": "production",
            "shadow_mode_duration_hours": 10,  # Less than required 24
            "shadow_mode_requests": 500,  # Less than required 1000
        }

        is_valid, errors = validate_trainer_to_deployer_handoff(handoff_data)

        assert is_valid is False
        assert any("shadow_mode_duration_hours" in e for e in errors)
        assert any("shadow_mode_requests" in e for e in errors)

    def test_validate_trainer_to_deployer_handoff_production_valid(self, mock_model_trainer_output):
        """Test valid production deployment with sufficient shadow mode."""
        handoff_data = {
            "model_uri": mock_model_trainer_output["model_uri"],
            "experiment_id": mock_model_trainer_output["experiment_id"],
            "validation_metrics": mock_model_trainer_output["validation_metrics"],
            "success_criteria_met": True,
            "target_environment": "production",
            "shadow_mode_duration_hours": 48,
            "shadow_mode_requests": 5000,
        }

        is_valid, errors = validate_trainer_to_deployer_handoff(handoff_data)

        assert is_valid is True
        assert len(errors) == 0


# =============================================================================
# ERROR HANDLING TESTS
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in pipeline."""

    @pytest.mark.asyncio
    async def test_scope_definer_failure(self, sample_input_data):
        """Test handling of scope definer failure.

        The pipeline catches exceptions internally and returns a failed result
        with error details in the errors list.
        """
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            failed_agent = MagicMock()
            failed_agent.run = AsyncMock(side_effect=ValueError("Invalid problem description"))
            mock_get_agent.return_value = failed_agent

            pipeline = MLFoundationPipeline()
            result = await pipeline.run(sample_input_data)

            # Pipeline catches exceptions and returns failed result
            assert result.status == "failed"
            # current_stage reflects where error occurred
            assert result.current_stage == PipelineStage.SCOPE_DEFINITION
            assert len(result.errors) > 0

            # Verify error details captured
            error = result.errors[0]
            assert "Invalid problem description" in error.get("error", "")
            assert error.get("error_type") == "ValueError"
            assert error.get("stage") == "scope_definition"

    @pytest.mark.asyncio
    async def test_model_trainer_failure_with_partial_result(
        self,
        sample_input_data,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
    ):
        """Test handling of model trainer failure with partial results.

        When trainer fails, pipeline should preserve partial results from
        earlier stages while recording the error.
        """
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            failed_trainer = MagicMock()
            failed_trainer.run = AsyncMock(side_effect=RuntimeError("Training failed"))

            mock_get_agent.side_effect = lambda name: {
                "scope_definer": create_mock_agent(mock_scope_definer_output),
                "data_preparer": create_mock_agent(mock_data_preparer_output_passed),
                "model_selector": create_mock_agent(mock_model_selector_output),
                "model_trainer": failed_trainer,
            }[name]

            pipeline = MLFoundationPipeline()
            result = await pipeline.run(sample_input_data)

            # Pipeline catches exceptions and returns failed result
            assert result.status == "failed"
            assert len(result.errors) > 0

            # Verify error details captured
            error = result.errors[0]
            assert "Training failed" in error.get("error", "")
            assert error.get("error_type") == "RuntimeError"
            assert error.get("stage") == "model_training"

            # Verify partial results from completed stages are preserved
            assert result.scope_spec is not None
            assert result.scope_spec.get("experiment_id") == "exp_test_001"
            assert result.qc_report is not None
            assert result.qc_report.get("qc_passed") is True
            assert result.model_candidate is not None
            assert result.model_candidate.get("algorithm_name") == "XGBoost"

            # Verify completed stages were recorded
            assert "scope_definition" in result.stages_completed
            assert "data_preparation" in result.stages_completed
            assert "model_selection" in result.stages_completed


# =============================================================================
# CONFIGURATION TESTS
# =============================================================================


class TestPipelineConfiguration:
    """Tests for pipeline configuration options."""

    def test_default_configuration(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()

        assert config.skip_deployment is False
        assert config.enable_hpo is True
        assert config.hpo_trials == 50
        assert config.target_environment == "staging"

    def test_custom_configuration(self):
        """Test custom pipeline configuration."""
        config = PipelineConfig(
            skip_deployment=True,
            enable_hpo=False,
            hpo_trials=50,
            target_environment="production",
        )

        assert config.skip_deployment is True
        assert config.enable_hpo is False
        assert config.hpo_trials == 50
        assert config.target_environment == "production"

    @pytest.mark.asyncio
    async def test_hpo_disabled_configuration(
        self,
        sample_input_data,
        mock_scope_definer_output,
        mock_data_preparer_output_passed,
        mock_model_selector_output,
        mock_model_trainer_output,
        mock_feature_analyzer_output,
        mock_model_deployer_output,
    ):
        """Test pipeline with HPO disabled."""
        with patch("src.agents.tier_0.pipeline.MLFoundationPipeline._get_agent") as mock_get_agent:
            mock_trainer = create_mock_agent(mock_model_trainer_output)

            mock_get_agent.side_effect = lambda name: {
                "scope_definer": create_mock_agent(mock_scope_definer_output),
                "data_preparer": create_mock_agent(mock_data_preparer_output_passed),
                "model_selector": create_mock_agent(mock_model_selector_output),
                "model_trainer": mock_trainer,
                "feature_analyzer": create_mock_agent(mock_feature_analyzer_output),
                "model_deployer": create_mock_agent(mock_model_deployer_output),
            }[name]

            config = PipelineConfig(enable_hpo=False)
            pipeline = MLFoundationPipeline(config=config)
            result = await pipeline.run(sample_input_data)

            assert result.status == PipelineStage.COMPLETED

            # Verify trainer was called with HPO disabled
            trainer_call = mock_trainer.run.call_args
            assert trainer_call[0][0].get("enable_hpo") is False


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for pipeline components."""

    def test_pipeline_stage_enum(self):
        """Test pipeline stage enum values."""
        assert PipelineStage.SCOPE_DEFINITION.value == "scope_definition"
        assert PipelineStage.DATA_PREPARATION.value == "data_preparation"
        assert PipelineStage.MODEL_SELECTION.value == "model_selection"
        assert PipelineStage.MODEL_TRAINING.value == "model_training"
        assert PipelineStage.FEATURE_ANALYSIS.value == "feature_analysis"
        assert PipelineStage.MODEL_DEPLOYMENT.value == "model_deployment"
        assert PipelineStage.COMPLETED.value == "completed"
        assert PipelineStage.FAILED.value == "failed"

    def test_pipeline_result_dataclass(self):
        """Test PipelineResult dataclass."""
        from src.agents.tier_0.pipeline import PipelineResult

        result = PipelineResult(
            pipeline_run_id="test_run",
            status="completed",
            current_stage=PipelineStage.COMPLETED,
        )

        assert result.pipeline_run_id == "test_run"
        assert result.status == "completed"
        assert result.current_stage == PipelineStage.COMPLETED
        assert result.scope_spec is None
        assert len(result.errors) == 0

    def test_pipeline_result_with_outputs(self, mock_scope_definer_output):
        """Test PipelineResult with stage outputs."""
        from src.agents.tier_0.pipeline import PipelineResult

        result = PipelineResult(
            pipeline_run_id="test_run",
            status="in_progress",
            current_stage=PipelineStage.SCOPE_DEFINITION,
            scope_spec=mock_scope_definer_output["scope_spec"],
            stage_timings={"scope_definition": 1.5},
        )

        assert result.scope_spec == mock_scope_definer_output["scope_spec"]
        assert result.stage_timings["scope_definition"] == 1.5
