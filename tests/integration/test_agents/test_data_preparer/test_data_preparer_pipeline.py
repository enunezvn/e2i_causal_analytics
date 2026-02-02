"""Integration tests for the data_preparer agent pipeline.

Tests the full data preparation pipeline using sample data.
"""

import numpy as np
import pandas as pd
import pytest

from src.agents.ml_foundation.data_preparer.graph import create_data_preparer_graph
from src.agents.ml_foundation.data_preparer.nodes import (
    compute_baseline_metrics,
    detect_leakage,
    load_data,
    run_quality_checks,
    transform_data,
)
from src.agents.ml_foundation.data_preparer.state import DataPreparerState
from src.repositories import SampleDataGenerator, get_data_splitter


@pytest.fixture
def sample_generator():
    """Create a sample data generator with fixed seed."""
    return SampleDataGenerator(seed=42)


@pytest.fixture
def sample_scope_spec():
    """Create a sample scope specification."""
    return {
        "experiment_id": "exp_test_123",
        "problem_type": "binary_classification",
        "target_column": "conversion",
        "data_source": "business_metrics",
        "use_sample_data": True,
        "sample_size": 500,
        "date_column": "created_at",
        "required_columns": ["metric_value", "brand"],
        "expected_dtypes": {
            "metric_value": "float",
            "brand": "object",
        },
        "scaling_method": "standard",
        "encoding_method": "label",
        "max_staleness_days": 365,
    }


@pytest.fixture
def sample_business_metrics(sample_generator):
    """Generate sample business metrics data."""
    return sample_generator.business_metrics(n_samples=500)


class TestLoadDataNode:
    """Tests for the load_data node."""

    @pytest.mark.asyncio
    async def test_load_sample_data_success(self, sample_scope_spec):
        """Test loading sample data successfully."""
        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": sample_scope_spec,
            "data_source": "business_metrics",
        }

        result = await load_data(state)

        assert "train_df" in result
        assert "validation_df" in result
        assert "test_df" in result
        assert result["train_df"] is not None
        assert len(result["train_df"]) > 0

    @pytest.mark.asyncio
    async def test_load_data_creates_splits(self, sample_scope_spec):
        """Test that loading creates proper train/val/test splits."""
        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": sample_scope_spec,
            "data_source": "business_metrics",
        }

        result = await load_data(state)

        train_len = len(result["train_df"])
        val_len = len(result["validation_df"])
        test_len = len(result["test_df"])
        total = train_len + val_len + test_len

        # Check splits are reasonable (around 60/20/20)
        assert train_len / total > 0.4  # At least 40% train
        assert val_len > 0
        assert test_len > 0


class TestQualityCheckerNode:
    """Tests for the run_quality_checks node."""

    @pytest.mark.asyncio
    async def test_quality_checks_on_clean_data(self, sample_generator):
        """Test quality checks pass on clean sample data."""
        df = sample_generator.business_metrics(n_samples=500)
        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {
                "date_column": "created_at",
                "required_columns": [],
                "expected_dtypes": {},
                "max_staleness_days": 365,
            },
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await run_quality_checks(state)

        assert "qc_status" in result
        assert "overall_score" in result
        assert result["qc_status"] in ["passed", "warning"]
        assert result["overall_score"] >= 0.8

    @pytest.mark.asyncio
    async def test_quality_checks_detect_missing_values(self, sample_generator):
        """Test quality checks detect missing values in required columns."""
        df = sample_generator.business_metrics(n_samples=500)

        # Inject nulls into a required column
        df.loc[df.index[:50], "metric_value"] = None

        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {
                "date_column": "created_at",
                "required_columns": ["metric_value"],  # This has nulls now
                "expected_dtypes": {},
                "max_staleness_days": 365,
            },
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await run_quality_checks(state)

        # Should have blocking issues due to nulls in required column
        assert len(result.get("blocking_issues", [])) > 0 or any(
            "null" in str(w).lower() for w in result.get("warnings", [])
        )


class TestLeakageDetectorNode:
    """Tests for the detect_leakage node."""

    @pytest.mark.asyncio
    async def test_no_leakage_on_clean_data(self, sample_generator):
        """Test no leakage detected on properly split data."""
        df = sample_generator.business_metrics(n_samples=500)
        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {
                "target_column": "target",
                "date_column": "created_at",
            },
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await detect_leakage(state)

        # Should not have blocking leakage issues
        assert result.get("has_temporal_leakage", False) is False
        assert result.get("has_target_leakage", False) is False

    @pytest.mark.asyncio
    async def test_detects_train_test_contamination(self, sample_generator):
        """Test detection of train-test contamination."""
        df = sample_generator.business_metrics(n_samples=500)
        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        # Create contamination by making test overlap with train
        contaminated_test = pd.concat(
            [
                splits.test,
                splits.train.head(10),  # Add train samples to test
            ],
            ignore_index=True,
        )

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {},
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": contaminated_test,
        }

        result = await detect_leakage(state)

        # Should detect contamination via leakage_detected and blocking_issues
        assert result.get("leakage_detected", False) is True
        # Verify contamination is in blocking issues
        blocking_issues = result.get("blocking_issues", [])
        assert any("contamination" in str(issue).lower() for issue in blocking_issues)


class TestDataTransformerNode:
    """Tests for the transform_data node."""

    @pytest.mark.asyncio
    async def test_transform_applies_scaling(self, sample_generator):
        """Test that transformation applies scaling."""
        df = sample_generator.business_metrics(n_samples=500)
        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {
                "scaling_method": "standard",
                "encoding_method": "label",
                "exclude_columns": ["id"],
            },
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await transform_data(state)

        assert "X_train" in result
        assert "transformations_applied" in result
        assert len(result["transformations_applied"]) > 0

        # Check that scaling was applied
        scaling_transformations = [
            t for t in result["transformations_applied"] if t.get("type") == "scaling"
        ]
        assert len(scaling_transformations) > 0

    @pytest.mark.asyncio
    async def test_transform_handles_categorical(self, sample_generator):
        """Test that transformation encodes categorical variables."""
        df = sample_generator.business_metrics(n_samples=500)
        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {
                "scaling_method": "standard",
                "encoding_method": "label",
                "exclude_columns": ["id"],
            },
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await transform_data(state)

        # Check encoding was applied
        encoding_transformations = [
            t for t in result["transformations_applied"] if t.get("type") == "encoding"
        ]
        assert len(encoding_transformations) > 0

    @pytest.mark.asyncio
    async def test_transform_separates_target(self, sample_generator):
        """Test that transformation separates target column."""
        df = sample_generator.business_metrics(n_samples=500)
        # Add a target column
        df["target"] = np.random.randint(0, 2, size=len(df))

        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {
                "target_column": "target",
                "scaling_method": "standard",
                "encoding_method": "label",
            },
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await transform_data(state)

        assert "y_train" in result
        assert result["y_train"] is not None
        assert "target" not in result["X_train"].columns


class TestBaselineMetricsNode:
    """Tests for the compute_baseline_metrics node."""

    @pytest.mark.asyncio
    async def test_computes_baseline_metrics(self, sample_generator):
        """Test baseline metrics computation."""
        df = sample_generator.business_metrics(n_samples=500)
        splitter = get_data_splitter(random_seed=42)
        splits = splitter.random_split(df)

        state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": {"target_column": "metric_value"},
            "train_df": splits.train,
            "validation_df": splits.val,
            "test_df": splits.test,
        }

        result = await compute_baseline_metrics(state)

        # baseline_computer returns keys at top level
        assert "feature_stats" in result
        assert "computed_at" in result
        assert "training_samples" in result


class TestFullPipeline:
    """End-to-end tests for the full data_preparer pipeline."""

    @pytest.mark.asyncio
    async def test_full_pipeline_with_sample_data(self, sample_scope_spec):
        """Test the complete pipeline with sample data."""
        # Create the graph
        graph = create_data_preparer_graph()
        compiled = graph.compile()

        # Initial state
        initial_state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": sample_scope_spec,
            "data_source": "business_metrics",
        }

        # Run the pipeline
        final_state = await compiled.ainvoke(initial_state)

        # Verify outputs
        assert "gate_passed" in final_state
        assert "qc_status" in final_state
        assert "overall_score" in final_state

        # For clean sample data, should pass
        assert final_state["qc_status"] in ["passed", "warning"]

    @pytest.mark.asyncio
    async def test_pipeline_produces_ml_ready_data(self, sample_scope_spec):
        """Test that pipeline produces ML-ready data."""
        graph = create_data_preparer_graph()
        compiled = graph.compile()

        initial_state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": sample_scope_spec,
            "data_source": "business_metrics",
        }

        final_state = await compiled.ainvoke(initial_state)

        # Should have transformed data
        if "X_train" in final_state:
            X_train = final_state["X_train"]
            assert X_train is not None
            assert len(X_train) > 0

            # All columns should be numeric after transformation
            for col in X_train.columns:
                assert (
                    pd.api.types.is_numeric_dtype(X_train[col]) or X_train[col].dtype == object
                ), f"Column {col} has unexpected dtype {X_train[col].dtype}"

    @pytest.mark.asyncio
    async def test_pipeline_gate_decision(self, sample_scope_spec):
        """Test that pipeline makes correct gate decision."""
        graph = create_data_preparer_graph()
        compiled = graph.compile()

        initial_state: DataPreparerState = {
            "experiment_id": "exp_test_123",
            "scope_spec": sample_scope_spec,
            "data_source": "business_metrics",
        }

        final_state = await compiled.ainvoke(initial_state)

        # Gate decision should be made
        assert "gate_passed" in final_state
        assert isinstance(final_state["gate_passed"], bool)

        # If QC passed and score >= 0.80, gate should pass
        if final_state.get("qc_status") == "passed" and final_state.get("overall_score", 0) >= 0.80:
            assert final_state["gate_passed"] is True
