"""Tests for BentoML service templates."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestClassificationServiceTemplate:
    """Test ClassificationServiceTemplate."""

    @patch("src.mlops.bentoml_templates.classification_service.BENTOML_AVAILABLE", False)
    def test_raises_without_bentoml(self):
        """Should raise ImportError when BentoML not available."""
        from src.mlops.bentoml_templates import ClassificationServiceTemplate

        with pytest.raises(ImportError, match="BentoML is not installed"):
            ClassificationServiceTemplate.create(
                model_tag="test_model:latest",
            )

    def test_import_available(self):
        """Should be importable."""
        from src.mlops.bentoml_templates import ClassificationServiceTemplate

        assert ClassificationServiceTemplate is not None

    def test_request_model_import(self):
        """Should have ClassificationInput model."""
        from src.mlops.bentoml_templates.classification_service import ClassificationInput

        # Test model creation
        input_data = ClassificationInput(
            features=[[1.0, 2.0, 3.0]],
            threshold=0.5,
        )

        assert input_data.features == [[1.0, 2.0, 3.0]]
        assert input_data.threshold == 0.5
        assert input_data.return_all_classes is False

    def test_response_model_import(self):
        """Should have ClassificationOutput model."""
        from src.mlops.bentoml_templates.classification_service import ClassificationOutput

        output_data = ClassificationOutput(
            predictions=[0, 1, 0],
            probabilities=[0.3, 0.8, 0.4],
            confidence_scores=[0.7, 0.8, 0.6],
            model_id="test_model:v1",
            prediction_time_ms=10.5,
        )

        assert output_data.predictions == [0, 1, 0]
        assert output_data.probabilities == [0.3, 0.8, 0.4]
        assert output_data.all_probabilities is None

    def test_batch_input_model(self):
        """Should have BatchClassificationInput model."""
        from src.mlops.bentoml_templates.classification_service import BatchClassificationInput

        batch_input = BatchClassificationInput(
            batch_id="batch_001",
            features=[[1.0, 2.0], [3.0, 4.0]],
        )

        assert batch_input.batch_id == "batch_001"
        assert len(batch_input.features) == 2

    def test_batch_output_model(self):
        """Should have BatchClassificationOutput model."""
        from src.mlops.bentoml_templates.classification_service import BatchClassificationOutput

        batch_output = BatchClassificationOutput(
            batch_id="batch_001",
            total_samples=2,
            predictions=[0, 1],
            probabilities=[0.3, 0.8],
            processing_time_ms=25.0,
        )

        assert batch_output.batch_id == "batch_001"
        assert batch_output.total_samples == 2


class TestRegressionServiceTemplate:
    """Test RegressionServiceTemplate."""

    @patch("src.mlops.bentoml_templates.regression_service.BENTOML_AVAILABLE", False)
    def test_raises_without_bentoml(self):
        """Should raise ImportError when BentoML not available."""
        from src.mlops.bentoml_templates import RegressionServiceTemplate

        with pytest.raises(ImportError, match="BentoML is not installed"):
            RegressionServiceTemplate.create(
                model_tag="test_model:latest",
            )

    def test_import_available(self):
        """Should be importable."""
        from src.mlops.bentoml_templates import RegressionServiceTemplate

        assert RegressionServiceTemplate is not None

    def test_request_model_import(self):
        """Should have RegressionInput model."""
        from src.mlops.bentoml_templates.regression_service import RegressionInput

        input_data = RegressionInput(
            features=[[1.0, 2.0, 3.0]],
            return_intervals=True,
            confidence_level=0.95,
        )

        assert input_data.features == [[1.0, 2.0, 3.0]]
        assert input_data.return_intervals is True
        assert input_data.confidence_level == 0.95

    def test_response_model_import(self):
        """Should have RegressionOutput model."""
        from src.mlops.bentoml_templates.regression_service import RegressionOutput

        output_data = RegressionOutput(
            predictions=[10.5, 20.3, 15.7],
            model_id="test_model:v1",
            prediction_time_ms=8.2,
        )

        assert output_data.predictions == [10.5, 20.3, 15.7]
        assert output_data.lower_bounds is None
        assert output_data.upper_bounds is None

    def test_response_with_intervals(self):
        """Should support prediction intervals."""
        from src.mlops.bentoml_templates.regression_service import RegressionOutput

        output_data = RegressionOutput(
            predictions=[10.5, 20.3],
            lower_bounds=[8.0, 17.0],
            upper_bounds=[13.0, 23.5],
            model_id="test_model:v1",
            prediction_time_ms=12.5,
        )

        assert output_data.lower_bounds == [8.0, 17.0]
        assert output_data.upper_bounds == [13.0, 23.5]

    def test_batch_regression_models(self):
        """Should have batch models."""
        from src.mlops.bentoml_templates.regression_service import (
            BatchRegressionInput,
            BatchRegressionOutput,
        )

        batch_input = BatchRegressionInput(
            batch_id="batch_001",
            features=[[1.0], [2.0]],
        )

        batch_output = BatchRegressionOutput(
            batch_id="batch_001",
            total_samples=2,
            predictions=[10.5, 20.3],
            mean_prediction=15.4,
            std_prediction=4.9,
            processing_time_ms=15.0,
        )

        assert batch_input.batch_id == "batch_001"
        assert batch_output.mean_prediction == 15.4


class TestCausalInferenceServiceTemplate:
    """Test CausalInferenceServiceTemplate."""

    @patch("src.mlops.bentoml_templates.causal_service.BENTOML_AVAILABLE", False)
    def test_raises_without_bentoml(self):
        """Should raise ImportError when BentoML not available."""
        from src.mlops.bentoml_templates import CausalInferenceServiceTemplate

        with pytest.raises(ImportError, match="BentoML is not installed"):
            CausalInferenceServiceTemplate.create(
                model_tag="test_model:latest",
            )

    def test_import_available(self):
        """Should be importable."""
        from src.mlops.bentoml_templates import CausalInferenceServiceTemplate

        assert CausalInferenceServiceTemplate is not None

    def test_causal_input_model(self):
        """Should have CausalInput model."""
        from src.mlops.bentoml_templates.causal_service import CausalInput

        input_data = CausalInput(
            features=[[1.0, 2.0]],
            treatment=[1],
            return_intervals=True,
        )

        assert input_data.features == [[1.0, 2.0]]
        assert input_data.treatment == [1]
        assert input_data.return_intervals is True
        assert input_data.confidence_level == 0.95

    def test_cate_output_model(self):
        """Should have CATEOutput model."""
        from src.mlops.bentoml_templates.causal_service import CATEOutput

        output_data = CATEOutput(
            cate=[0.5, 0.3, -0.2],
            ate=0.2,
            ate_std=0.35,
            model_id="cate_model:v1",
            prediction_time_ms=25.0,
        )

        assert output_data.cate == [0.5, 0.3, -0.2]
        assert output_data.ate == 0.2
        assert output_data.lower_bounds is None

    def test_cate_output_with_intervals(self):
        """Should support confidence intervals."""
        from src.mlops.bentoml_templates.causal_service import CATEOutput

        output_data = CATEOutput(
            cate=[0.5, 0.3],
            lower_bounds=[0.2, -0.1],
            upper_bounds=[0.8, 0.7],
            ate=0.4,
            ate_std=0.25,
            model_id="cate_model:v1",
            prediction_time_ms=30.0,
        )

        assert output_data.lower_bounds == [0.2, -0.1]
        assert output_data.upper_bounds == [0.8, 0.7]

    def test_treatment_effect_models(self):
        """Should have treatment effect models."""
        from src.mlops.bentoml_templates.causal_service import (
            TreatmentEffectInput,
            TreatmentEffectOutput,
        )

        te_input = TreatmentEffectInput(
            features=[[1.0, 2.0]],
            treatment_values=[0, 1],
        )

        te_output = TreatmentEffectOutput(
            potential_outcomes={"T=0": [10.0], "T=1": [12.5]},
            treatment_effects=[2.5],
            mean_effect=2.5,
            effect_std=0.0,
        )

        assert te_input.treatment_values == [0, 1]
        assert te_output.treatment_effects == [2.5]

    def test_batch_causal_models(self):
        """Should have batch models."""
        from src.mlops.bentoml_templates.causal_service import (
            BatchCausalInput,
            BatchCausalOutput,
        )

        batch_input = BatchCausalInput(
            batch_id="batch_001",
            features=[[1.0], [2.0], [3.0]],
        )

        batch_output = BatchCausalOutput(
            batch_id="batch_001",
            total_samples=3,
            cate=[0.5, 0.3, -0.2],
            ate=0.2,
            ate_std=0.35,
            positive_effect_ratio=0.67,
            processing_time_ms=40.0,
        )

        assert batch_input.batch_id == "batch_001"
        assert batch_output.positive_effect_ratio == 0.67


class TestTemplateModuleInit:
    """Test bentoml_templates module initialization."""

    def test_exports_all_templates(self):
        """Should export all template classes."""
        from src.mlops.bentoml_templates import (
            ClassificationServiceTemplate,
            RegressionServiceTemplate,
            CausalInferenceServiceTemplate,
        )

        assert ClassificationServiceTemplate is not None
        assert RegressionServiceTemplate is not None
        assert CausalInferenceServiceTemplate is not None

    def test_all_list(self):
        """Should have __all__ list."""
        from src.mlops import bentoml_templates

        assert hasattr(bentoml_templates, "__all__")
        assert "ClassificationServiceTemplate" in bentoml_templates.__all__
        assert "RegressionServiceTemplate" in bentoml_templates.__all__
        assert "CausalInferenceServiceTemplate" in bentoml_templates.__all__
