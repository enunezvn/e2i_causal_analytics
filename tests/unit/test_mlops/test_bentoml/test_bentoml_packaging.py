"""Tests for BentoML packaging utilities."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.mlops.bentoml_packaging import (
    BentoConfig,
    ContainerConfig,
    register_model_auto,
    generate_service_file,
    generate_docker_compose,
    create_bentofile,
)


class TestBentoConfig:
    """Test BentoConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = BentoConfig(
            service_name="test_service",
        )

        assert config.service_name == "test_service"
        assert config.service_file == "service.py"
        assert config.python_version == "3.11"
        assert "*.py" in config.include
        assert "__pycache__/" in config.exclude

    def test_to_yaml_basic(self):
        """Should convert to YAML format."""
        config = BentoConfig(
            service_name="test_service",
            description="Test description",
        )

        yaml_content = config.to_yaml()
        assert "service" in yaml_content
        assert "test_service" in yaml_content

    def test_to_yaml_with_packages(self):
        """Should include Python packages in YAML."""
        config = BentoConfig(
            service_name="test_service",
            python_packages=["numpy", "pandas"],
        )

        yaml_content = config.to_yaml()
        assert "python" in yaml_content
        assert "packages" in yaml_content

    def test_to_yaml_with_docker(self):
        """Should include Docker config when specified."""
        config = BentoConfig(
            service_name="test_service",
            docker_base_image="python:3.11-slim",
            docker_env={"LOG_LEVEL": "INFO"},
        )

        yaml_content = config.to_yaml()
        assert "docker" in yaml_content


class TestContainerConfig:
    """Test ContainerConfig dataclass."""

    def test_default_values(self):
        """Should have sensible defaults."""
        config = ContainerConfig(
            image_name="test-image",
        )

        assert config.image_tag == "latest"
        assert config.port == 3000
        assert config.workers == 1

    def test_full_image_name_without_registry(self):
        """Should format image name without registry."""
        config = ContainerConfig(
            image_name="test-image",
            image_tag="v1.0",
        )

        assert config.full_image_name == "test-image:v1.0"

    def test_full_image_name_with_registry(self):
        """Should format image name with registry."""
        config = ContainerConfig(
            image_name="test-image",
            image_tag="v1.0",
            registry="docker.io/myorg",
        )

        assert config.full_image_name == "docker.io/myorg/test-image:v1.0"


class TestRegisterModelAuto:
    """Test auto model registration."""

    @patch("src.mlops.bentoml_packaging.BENTOML_AVAILABLE", False)
    def test_raises_without_bentoml(self):
        """Should raise ImportError when BentoML not available."""
        with pytest.raises(ImportError, match="BentoML is not installed"):
            register_model_auto(
                model=MagicMock(),
                model_name="test_model",
            )

    @patch("src.mlops.bentoml_packaging.BENTOML_AVAILABLE", True)
    @patch("src.mlops.bentoml_packaging.register_sklearn_model")
    def test_detects_sklearn(self, mock_register):
        """Should detect sklearn models."""
        mock_model = MagicMock()
        mock_model.__class__.__module__ = "sklearn.ensemble._forest"
        mock_register.return_value = "test_model:v1"

        result = register_model_auto(
            model=mock_model,
            model_name="test_model",
        )

        mock_register.assert_called_once()
        assert result == "test_model:v1"

    @patch("src.mlops.bentoml_packaging.BENTOML_AVAILABLE", True)
    @patch("src.mlops.bentoml_packaging.register_xgboost_model")
    def test_detects_xgboost(self, mock_register):
        """Should detect XGBoost models."""
        mock_model = MagicMock()
        mock_model.__class__.__module__ = "xgboost.core"
        mock_register.return_value = "test_model:v1"

        result = register_model_auto(
            model=mock_model,
            model_name="test_model",
        )

        mock_register.assert_called_once()
        assert result == "test_model:v1"

    @patch("src.mlops.bentoml_packaging.BENTOML_AVAILABLE", True)
    @patch("src.mlops.bentoml_packaging.register_causal_model")
    def test_explicit_framework(self, mock_register):
        """Should use explicit framework when specified."""
        mock_model = MagicMock()
        mock_register.return_value = "test_model:v1"

        result = register_model_auto(
            model=mock_model,
            model_name="test_model",
            framework="econml",
        )

        mock_register.assert_called_once()
        assert result == "test_model:v1"


class TestGenerateServiceFile:
    """Test service file generation."""

    def test_generate_classification_service(self, tmp_path):
        """Should generate classification service file."""
        output_path = tmp_path / "service.py"

        result = generate_service_file(
            model_tag="churn_model:latest",
            service_type="classification",
            output_path=output_path,
            service_name="churn_service",
        )

        assert result.exists()
        content = result.read_text()
        assert "ClassificationServiceTemplate" in content
        assert "churn_model:latest" in content
        assert "churn_service" in content

    def test_generate_regression_service(self, tmp_path):
        """Should generate regression service file."""
        output_path = tmp_path / "service.py"

        result = generate_service_file(
            model_tag="sales_model:latest",
            service_type="regression",
            output_path=output_path,
        )

        assert result.exists()
        content = result.read_text()
        assert "RegressionServiceTemplate" in content
        assert "sales_model:latest" in content

    def test_generate_causal_service(self, tmp_path):
        """Should generate causal inference service file."""
        output_path = tmp_path / "service.py"

        result = generate_service_file(
            model_tag="cate_model:latest",
            service_type="causal",
            output_path=output_path,
        )

        assert result.exists()
        content = result.read_text()
        assert "CausalInferenceServiceTemplate" in content
        assert "cate_model:latest" in content

    def test_invalid_service_type(self, tmp_path):
        """Should raise error for invalid service type."""
        with pytest.raises(ValueError, match="Unknown service type"):
            generate_service_file(
                model_tag="model:latest",
                service_type="invalid",
                output_path=tmp_path / "service.py",
            )


class TestGenerateDockerCompose:
    """Test docker-compose generation."""

    def test_generate_single_service(self, tmp_path):
        """Should generate compose file for single service."""
        services = [
            {
                "name": "classification-service",
                "image": "e2i-classification:latest",
                "port": 3001,
            }
        ]

        result = generate_docker_compose(
            services=services,
            output_path=tmp_path / "docker-compose.yaml",
        )

        assert result.exists()
        content = result.read_text()
        assert "classification-service" in content

    def test_generate_multiple_services(self, tmp_path):
        """Should generate compose file for multiple services."""
        services = [
            {"name": "svc1", "image": "img1:latest", "port": 3001},
            {"name": "svc2", "image": "img2:latest", "port": 3002},
        ]

        result = generate_docker_compose(
            services=services,
            output_path=tmp_path / "docker-compose.yaml",
        )

        assert result.exists()
        content = result.read_text()
        assert "svc1" in content
        assert "svc2" in content


class TestCreateBentofile:
    """Test bentofile creation."""

    def test_creates_bentofile(self, tmp_path):
        """Should create bentofile.yaml."""
        config = BentoConfig(
            service_name="test_service",
            description="Test service",
        )

        result = create_bentofile(tmp_path, config)

        assert result.exists()
        assert result.name == "bentofile.yaml"

    def test_creates_directory_if_missing(self, tmp_path):
        """Should create output directory if it doesn't exist."""
        config = BentoConfig(service_name="test_service")
        output_dir = tmp_path / "nested" / "dir"

        result = create_bentofile(output_dir, config)

        assert output_dir.exists()
        assert result.exists()
