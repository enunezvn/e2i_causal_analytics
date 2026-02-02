"""Tests for Data Lineage Tracking.

Version: 1.0.0
Tests the data lineage tracking module for ML pipeline traceability.

Coverage:
- SourceType, TransformationType, SplitType enums
- DataSource, TransformationStep, SplitRecord dataclasses
- LineageGraph dataclass
- LineageTracker main class
- MLflow integration (mocked)
- JSON serialization/deserialization
- Validation and reporting
"""

import json
import os
import tempfile
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.mlops.data_lineage import (
    DataSource,
    LineageGraph,
    LineageTracker,
    SourceType,
    SplitRecord,
    SplitType,
    TransformationStep,
    TransformationType,
    get_lineage_tracker,
)

# ============================================================================
# SOURCETYPE ENUM TESTS
# ============================================================================


class TestSourceType:
    """Test SourceType enum."""

    def test_all_source_types_defined(self):
        """Test all expected source types are defined."""
        expected_types = [
            "supabase",
            "csv",
            "parquet",
            "feature_store",
            "api",
            "s3",
            "cached",
            "generated",
        ]
        actual_types = [s.value for s in SourceType]
        assert set(actual_types) == set(expected_types)

    def test_source_type_is_string_enum(self):
        """Test that source types can be used as strings."""
        assert SourceType.SUPABASE == "supabase"
        assert SourceType.CSV == "csv"
        assert SourceType.PARQUET == "parquet"

    def test_source_type_from_string(self):
        """Test creating SourceType from string."""
        assert SourceType("supabase") == SourceType.SUPABASE
        assert SourceType("csv") == SourceType.CSV


# ============================================================================
# TRANSFORMATIONTYPE ENUM TESTS
# ============================================================================


class TestTransformationType:
    """Test TransformationType enum."""

    def test_all_transformation_types_defined(self):
        """Test all expected transformation types are defined."""
        expected_types = [
            "preprocessing",
            "feature_engineering",
            "imputation",
            "encoding",
            "scaling",
            "filtering",
            "aggregation",
            "splitting",
            "sampling",
            "augmentation",
            "merge",
            "custom",
        ]
        actual_types = [t.value for t in TransformationType]
        assert set(actual_types) == set(expected_types)

    def test_transformation_type_is_string_enum(self):
        """Test that transformation types can be used as strings."""
        assert TransformationType.PREPROCESSING == "preprocessing"
        assert TransformationType.SCALING == "scaling"

    def test_transformation_type_from_string(self):
        """Test creating TransformationType from string."""
        assert TransformationType("preprocessing") == TransformationType.PREPROCESSING
        assert TransformationType("encoding") == TransformationType.ENCODING


# ============================================================================
# SPLITTYPE ENUM TESTS
# ============================================================================


class TestSplitType:
    """Test SplitType enum."""

    def test_all_split_types_defined(self):
        """Test all expected split types are defined."""
        expected_types = ["random", "temporal", "stratified", "entity", "combined"]
        actual_types = [s.value for s in SplitType]
        assert set(actual_types) == set(expected_types)

    def test_split_type_is_string_enum(self):
        """Test that split types can be used as strings."""
        assert SplitType.RANDOM == "random"
        assert SplitType.TEMPORAL == "temporal"

    def test_split_type_from_string(self):
        """Test creating SplitType from string."""
        assert SplitType("temporal") == SplitType.TEMPORAL
        assert SplitType("entity") == SplitType.ENTITY


# ============================================================================
# DATASOURCE DATACLASS TESTS
# ============================================================================


class TestDataSource:
    """Test DataSource dataclass."""

    def test_minimal_creation(self):
        """Test creating DataSource with minimal fields."""
        source = DataSource(
            source_id="src_12345678",
            source_type=SourceType.SUPABASE,
            source_name="patient_metrics",
        )
        assert source.source_id == "src_12345678"
        assert source.source_type == SourceType.SUPABASE
        assert source.source_name == "patient_metrics"
        assert source.query is None
        assert source.row_count is None

    def test_full_creation(self):
        """Test creating DataSource with all fields."""
        source = DataSource(
            source_id="src_12345678",
            source_type=SourceType.SUPABASE,
            source_name="patient_metrics",
            query="SELECT * FROM patient_metrics",
            table_name="patient_metrics",
            row_count=10000,
            column_count=25,
            column_names=["id", "metric1", "metric2"],
            schema={"id": "int64", "metric1": "float64"},
            data_hash="abc123def456",
            metadata={"brand": "remibrutinib"},
        )
        assert source.row_count == 10000
        assert source.column_count == 25
        assert source.data_hash == "abc123def456"

    def test_to_dict(self):
        """Test converting DataSource to dictionary."""
        source = DataSource(
            source_id="src_12345678",
            source_type=SourceType.CSV,
            source_name="training_data",
            row_count=5000,
        )
        result = source.to_dict()

        assert result["source_id"] == "src_12345678"
        assert result["source_type"] == "csv"
        assert result["source_name"] == "training_data"
        assert result["row_count"] == 5000
        assert "created_at" in result

    def test_created_at_default(self):
        """Test that created_at is set automatically."""
        source = DataSource(
            source_id="src_test",
            source_type=SourceType.PARQUET,
            source_name="test",
        )
        assert source.created_at is not None
        assert isinstance(source.created_at, datetime)


# ============================================================================
# TRANSFORMATIONSTEP DATACLASS TESTS
# ============================================================================


class TestTransformationStep:
    """Test TransformationStep dataclass."""

    def test_minimal_creation(self):
        """Test creating TransformationStep with minimal fields."""
        step = TransformationStep(
            step_id="step_12345678",
            source_id="src_12345678",
            transformation_type=TransformationType.PREPROCESSING,
            transformation_name="fit_scaler",
        )
        assert step.step_id == "step_12345678"
        assert step.source_id == "src_12345678"
        assert step.transformation_type == TransformationType.PREPROCESSING
        assert step.validation_passed is True

    def test_full_creation(self):
        """Test creating TransformationStep with all fields."""
        step = TransformationStep(
            step_id="step_12345678",
            source_id="src_12345678",
            transformation_type=TransformationType.SCALING,
            transformation_name="standard_scaler",
            input_shape=(10000, 25),
            output_shape=(10000, 25),
            parameters={"with_mean": True, "with_std": True},
            rows_affected=10000,
            columns_modified=["metric1", "metric2"],
            validation_passed=True,
        )
        assert step.input_shape == (10000, 25)
        assert step.output_shape == (10000, 25)
        assert step.parameters["with_mean"] is True

    def test_to_dict(self):
        """Test converting TransformationStep to dictionary."""
        step = TransformationStep(
            step_id="step_test",
            source_id="src_test",
            transformation_type=TransformationType.ENCODING,
            transformation_name="one_hot_encoder",
            input_shape=(100, 10),
            output_shape=(100, 15),
        )
        result = step.to_dict()

        assert result["step_id"] == "step_test"
        assert result["transformation_type"] == "encoding"
        assert result["input_shape"] == (100, 10)
        assert result["output_shape"] == (100, 15)

    def test_validation_errors(self):
        """Test TransformationStep with validation errors."""
        step = TransformationStep(
            step_id="step_failed",
            source_id="src_test",
            transformation_type=TransformationType.CUSTOM,
            transformation_name="failed_transform",
            validation_passed=False,
            validation_errors=["Missing required columns", "Invalid data types"],
        )
        assert step.validation_passed is False
        assert len(step.validation_errors) == 2


# ============================================================================
# SPLITRECORD DATACLASS TESTS
# ============================================================================


class TestSplitRecord:
    """Test SplitRecord dataclass."""

    def test_minimal_creation(self):
        """Test creating SplitRecord with minimal fields."""
        split = SplitRecord(
            split_id="split_12345678",
            source_id="src_12345678",
            split_type=SplitType.RANDOM,
        )
        assert split.split_id == "split_12345678"
        assert split.split_type == SplitType.RANDOM
        assert split.leakage_checked is False
        assert split.leakage_detected is False

    def test_full_creation(self):
        """Test creating SplitRecord with all fields."""
        split = SplitRecord(
            split_id="split_12345678",
            source_id="src_12345678",
            split_type=SplitType.TEMPORAL,
            ratios={"train": 0.6, "val": 0.2, "test": 0.15, "holdout": 0.05},
            split_column="date",
            random_seed=42,
            train_size=6000,
            val_size=2000,
            test_size=1500,
            holdout_size=500,
            leakage_checked=True,
            leakage_detected=False,
        )
        assert split.ratios["train"] == 0.6
        assert split.train_size == 6000
        assert split.leakage_checked is True

    def test_to_dict(self):
        """Test converting SplitRecord to dictionary."""
        split = SplitRecord(
            split_id="split_test",
            source_id="src_test",
            split_type=SplitType.STRATIFIED,
            stratify_column="target",
            train_size=800,
            test_size=200,
        )
        result = split.to_dict()

        assert result["split_id"] == "split_test"
        assert result["split_type"] == "stratified"
        assert result["stratify_column"] == "target"
        assert result["train_size"] == 800

    def test_leakage_detection(self):
        """Test SplitRecord with leakage detected."""
        split = SplitRecord(
            split_id="split_leaked",
            source_id="src_test",
            split_type=SplitType.ENTITY,
            leakage_checked=True,
            leakage_detected=True,
            leakage_details={
                "entity_overlap": 50,
                "overlap_percentage": 5.0,
            },
        )
        assert split.leakage_detected is True
        assert split.leakage_details["entity_overlap"] == 50


# ============================================================================
# LINEAGEGRAPH DATACLASS TESTS
# ============================================================================


class TestLineageGraph:
    """Test LineageGraph dataclass."""

    def test_empty_creation(self):
        """Test creating empty LineageGraph."""
        graph = LineageGraph(graph_id="lineage_test")
        assert graph.graph_id == "lineage_test"
        assert len(graph.sources) == 0
        assert len(graph.transformations) == 0
        assert len(graph.splits) == 0

    def test_with_sources(self):
        """Test LineageGraph with sources."""
        source = DataSource(
            source_id="src_1",
            source_type=SourceType.CSV,
            source_name="data.csv",
        )
        graph = LineageGraph(
            graph_id="lineage_test",
            sources=[source],
        )
        assert len(graph.sources) == 1
        assert graph.sources[0].source_name == "data.csv"

    def test_to_dict(self):
        """Test converting LineageGraph to dictionary."""
        source = DataSource(
            source_id="src_1",
            source_type=SourceType.SUPABASE,
            source_name="metrics",
        )
        transform = TransformationStep(
            step_id="step_1",
            source_id="src_1",
            transformation_type=TransformationType.PREPROCESSING,
            transformation_name="preprocess",
        )
        graph = LineageGraph(
            graph_id="lineage_test",
            sources=[source],
            transformations=[transform],
            mlflow_run_id="run_123",
        )
        result = graph.to_dict()

        assert result["graph_id"] == "lineage_test"
        assert len(result["sources"]) == 1
        assert len(result["transformations"]) == 1
        assert result["mlflow_run_id"] == "run_123"

    def test_get_source(self):
        """Test getting a source by ID."""
        source1 = DataSource(
            source_id="src_1",
            source_type=SourceType.CSV,
            source_name="data1.csv",
        )
        source2 = DataSource(
            source_id="src_2",
            source_type=SourceType.PARQUET,
            source_name="data2.parquet",
        )
        graph = LineageGraph(
            graph_id="test",
            sources=[source1, source2],
        )

        found = graph.get_source("src_2")
        assert found is not None
        assert found.source_name == "data2.parquet"

        not_found = graph.get_source("src_999")
        assert not_found is None

    def test_get_transformation_chain(self):
        """Test getting transformation chain."""
        source = DataSource(
            source_id="src_1",
            source_type=SourceType.CSV,
            source_name="data.csv",
        )
        transform1 = TransformationStep(
            step_id="step_1",
            source_id="src_1",
            transformation_type=TransformationType.IMPUTATION,
            transformation_name="impute",
        )
        transform2 = TransformationStep(
            step_id="step_2",
            source_id="step_1",
            transformation_type=TransformationType.SCALING,
            transformation_name="scale",
        )
        graph = LineageGraph(
            graph_id="test",
            sources=[source],
            transformations=[transform1, transform2],
        )

        chain = graph.get_transformation_chain("src_1")
        assert len(chain) == 2
        assert chain[0].transformation_name == "impute"
        assert chain[1].transformation_name == "scale"


# ============================================================================
# LINEAGETRACKER TESTS
# ============================================================================


class TestLineageTracker:
    """Test LineageTracker class."""

    def test_initialization_default_id(self):
        """Test initialization with auto-generated ID."""
        tracker = LineageTracker()
        assert tracker.graph_id.startswith("lineage_")
        assert len(tracker.graph_id) == len("lineage_") + 12

    def test_initialization_custom_id(self):
        """Test initialization with custom ID."""
        tracker = LineageTracker(graph_id="my_lineage")
        assert tracker.graph_id == "my_lineage"

    def test_graph_property(self):
        """Test graph property."""
        tracker = LineageTracker(graph_id="test")
        assert tracker.graph.graph_id == "test"
        assert isinstance(tracker.graph, LineageGraph)


class TestLineageTrackerRecordSource:
    """Test LineageTracker.record_source method."""

    def test_record_source_basic(self):
        """Test recording a basic source."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.SUPABASE,
            source_name="patient_metrics",
        )

        assert source_id.startswith("src_")
        assert len(tracker.graph.sources) == 1
        assert tracker.graph.sources[0].source_name == "patient_metrics"

    def test_record_source_with_query(self):
        """Test recording source with SQL query."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type="supabase",
            source_name="metrics",
            query="SELECT * FROM metrics WHERE brand='remibrutinib'",
            table_name="metrics",
            row_count=10000,
            column_count=25,
        )

        source = tracker.graph.get_source(source_id)
        assert source.query.startswith("SELECT")
        assert source.row_count == 10000

    def test_record_source_string_type(self):
        """Test recording source with string type."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type="csv",
            source_name="data.csv",
        )

        source = tracker.graph.get_source(source_id)
        assert source.source_type == SourceType.CSV

    def test_record_source_with_schema(self):
        """Test recording source with schema."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.PARQUET,
            source_name="features.parquet",
            column_names=["id", "feature1", "feature2"],
            schema={"id": "int64", "feature1": "float64", "feature2": "float64"},
        )

        source = tracker.graph.get_source(source_id)
        assert source.schema["id"] == "int64"
        assert len(source.column_names) == 3

    def test_record_source_with_metadata(self):
        """Test recording source with metadata."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.API,
            source_name="external_api",
            metadata={"api_version": "v2", "endpoint": "/data"},
        )

        source = tracker.graph.get_source(source_id)
        assert source.metadata["api_version"] == "v2"


class TestLineageTrackerRecordSourceFromDataFrame:
    """Test LineageTracker.record_source_from_dataframe method."""

    def test_record_from_dataframe(self):
        """Test recording source from DataFrame."""
        tracker = LineageTracker()
        df = pd.DataFrame(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [10.0, 20.0, 30.0, 40.0, 50.0],
                "category": ["A", "B", "A", "B", "A"],
            }
        )

        source_id = tracker.record_source_from_dataframe(
            df=df,
            source_name="test_data",
        )

        source = tracker.graph.get_source(source_id)
        assert source.row_count == 5
        assert source.column_count == 3
        assert "id" in source.column_names
        assert source.schema["id"] == "int64"

    def test_record_from_dataframe_with_hash(self):
        """Test recording source from DataFrame with hash."""
        tracker = LineageTracker()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

        source_id = tracker.record_source_from_dataframe(
            df=df,
            source_name="hashed_data",
            compute_hash=True,
        )

        source = tracker.graph.get_source(source_id)
        assert source.data_hash is not None
        assert len(source.data_hash) == 16

    def test_record_from_dataframe_without_hash(self):
        """Test recording source from DataFrame without hash."""
        tracker = LineageTracker()
        df = pd.DataFrame({"x": [1, 2, 3]})

        source_id = tracker.record_source_from_dataframe(
            df=df,
            source_name="no_hash_data",
            compute_hash=False,
        )

        source = tracker.graph.get_source(source_id)
        assert source.data_hash is None

    def test_record_from_dataframe_invalid_input(self):
        """Test recording from non-DataFrame raises error."""
        tracker = LineageTracker()

        with pytest.raises(ValueError, match="must be a pandas DataFrame"):
            tracker.record_source_from_dataframe(
                df={"not": "a dataframe"},
                source_name="invalid",
            )

    def test_record_from_dataframe_custom_type(self):
        """Test recording from DataFrame with custom source type."""
        tracker = LineageTracker()
        df = pd.DataFrame({"col": [1, 2, 3]})

        source_id = tracker.record_source_from_dataframe(
            df=df,
            source_name="generated_data",
            source_type=SourceType.GENERATED,
        )

        source = tracker.graph.get_source(source_id)
        assert source.source_type == SourceType.GENERATED


class TestLineageTrackerRecordTransformation:
    """Test LineageTracker.record_transformation method."""

    def test_record_transformation_basic(self):
        """Test recording a basic transformation."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )

        step_id = tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.PREPROCESSING,
            transformation_name="initial_preprocess",
        )

        assert step_id.startswith("step_")
        assert len(tracker.graph.transformations) == 1
        assert tracker.graph.transformations[0].source_id == source_id

    def test_record_transformation_with_shapes(self):
        """Test recording transformation with shape changes."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.PARQUET,
            source_name="features.parquet",
        )

        tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.ENCODING,
            transformation_name="one_hot_encode",
            input_shape=(1000, 10),
            output_shape=(1000, 25),
            columns_added=["cat_A", "cat_B", "cat_C"],
        )

        transform = tracker.graph.transformations[0]
        assert transform.input_shape == (1000, 10)
        assert transform.output_shape == (1000, 25)
        assert len(transform.columns_added) == 3

    def test_record_transformation_chain(self):
        """Test recording a chain of transformations."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.SUPABASE,
            source_name="raw_data",
        )

        step1_id = tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.IMPUTATION,
            transformation_name="fill_missing",
        )

        step2_id = tracker.record_transformation(
            source_id=step1_id,
            transformation_type=TransformationType.SCALING,
            transformation_name="standardize",
        )

        tracker.record_transformation(
            source_id=step2_id,
            transformation_type=TransformationType.FEATURE_ENGINEERING,
            transformation_name="create_features",
        )

        assert len(tracker.graph.transformations) == 3
        chain = tracker.graph.get_transformation_chain(source_id)
        assert len(chain) == 3

    def test_record_transformation_string_type(self):
        """Test recording transformation with string type."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )

        tracker.record_transformation(
            source_id=source_id,
            transformation_type="scaling",
            transformation_name="min_max_scale",
        )

        transform = tracker.graph.transformations[0]
        assert transform.transformation_type == TransformationType.SCALING

    def test_record_transformation_with_validation_errors(self):
        """Test recording transformation with validation errors."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )

        tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.CUSTOM,
            transformation_name="custom_transform",
            validation_passed=False,
            validation_errors=["Column 'x' contains NaN", "Invalid dtype for 'y'"],
        )

        transform = tracker.graph.transformations[0]
        assert transform.validation_passed is False
        assert len(transform.validation_errors) == 2


class TestLineageTrackerRecordSplit:
    """Test LineageTracker.record_split method."""

    def test_record_split_basic(self):
        """Test recording a basic split."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )

        split_id = tracker.record_split(
            source_id=source_id,
            split_type=SplitType.RANDOM,
        )

        assert split_id.startswith("split_")
        assert len(tracker.graph.splits) == 1

    def test_record_split_with_ratios(self):
        """Test recording split with ratios."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.SUPABASE,
            source_name="metrics",
        )

        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.TEMPORAL,
            ratios={"train": 0.6, "val": 0.2, "test": 0.15, "holdout": 0.05},
            split_column="date",
            train_size=6000,
            val_size=2000,
            test_size=1500,
            holdout_size=500,
        )

        split = tracker.graph.splits[0]
        assert split.ratios["train"] == 0.6
        assert split.train_size == 6000
        assert split.split_column == "date"

    def test_record_split_with_leakage(self):
        """Test recording split with leakage detection."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )

        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.ENTITY,
            entity_column="patient_id",
            leakage_checked=True,
            leakage_detected=True,
            leakage_details={
                "entity_overlap": ["P001", "P002"],
                "overlap_count": 2,
            },
        )

        split = tracker.graph.splits[0]
        assert split.leakage_checked is True
        assert split.leakage_detected is True
        assert split.leakage_details["overlap_count"] == 2

    def test_record_split_string_type(self):
        """Test recording split with string type."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.PARQUET,
            source_name="data.parquet",
        )

        tracker.record_split(
            source_id=source_id,
            split_type="stratified",
            stratify_column="target",
        )

        split = tracker.graph.splits[0]
        assert split.split_type == SplitType.STRATIFIED


class TestLineageTrackerMLflowIntegration:
    """Test LineageTracker MLflow integration."""

    def test_associate_mlflow_run(self):
        """Test associating with MLflow run."""
        tracker = LineageTracker()
        tracker.associate_mlflow_run(
            run_id="abc123",
            experiment_name="my_experiment",
        )

        assert tracker.graph.mlflow_run_id == "abc123"
        assert tracker.graph.experiment_name == "my_experiment"

    @pytest.mark.asyncio
    async def test_log_to_mlflow_disabled(self):
        """Test logging when MLflow is disabled."""
        tracker = LineageTracker()
        tracker.record_source(
            source_type=SourceType.CSV,
            source_name="test.csv",
        )

        mock_connector = MagicMock()
        mock_connector.enabled = False

        result = await tracker.log_to_mlflow(mlflow_connector=mock_connector)
        assert result is False

    @pytest.mark.asyncio
    async def test_log_to_mlflow_no_run(self):
        """Test logging when no run is associated."""
        tracker = LineageTracker()
        tracker.record_source(
            source_type=SourceType.CSV,
            source_name="test.csv",
        )

        mock_connector = MagicMock()
        mock_connector.enabled = True

        result = await tracker.log_to_mlflow(mlflow_connector=mock_connector)
        assert result is False


class TestLineageTrackerValidation:
    """Test LineageTracker validation methods."""

    def test_validate_lineage_clean(self):
        """Test validation with clean lineage."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )
        tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.PREPROCESSING,
            transformation_name="preprocess",
        )
        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.RANDOM,
            leakage_checked=True,
            leakage_detected=False,
        )

        issues = tracker.validate_lineage()
        assert len(issues) == 0

    def test_validate_lineage_orphan_transformation(self):
        """Test validation detects orphan transformations."""
        tracker = LineageTracker()
        # Add transformation without source
        tracker.record_transformation(
            source_id="nonexistent_source",
            transformation_type=TransformationType.SCALING,
            transformation_name="scale",
        )

        issues = tracker.validate_lineage()
        assert len(issues) >= 1
        assert any("unknown source" in issue for issue in issues)

    def test_validate_lineage_unchecked_split(self):
        """Test validation warns about unchecked splits."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )
        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.RANDOM,
            leakage_checked=False,
        )

        issues = tracker.validate_lineage()
        assert len(issues) >= 1
        assert any("not checked for leakage" in issue for issue in issues)

    def test_validate_lineage_detected_leakage(self):
        """Test validation reports detected leakage as critical."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )
        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.ENTITY,
            leakage_checked=True,
            leakage_detected=True,
        )

        issues = tracker.validate_lineage()
        assert len(issues) >= 1
        assert any("CRITICAL" in issue for issue in issues)

    def test_validate_lineage_validation_failures(self):
        """Test validation reports transformation failures."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
        )
        tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.CUSTOM,
            transformation_name="bad_transform",
            validation_passed=False,
            validation_errors=["Error 1", "Error 2"],
        )

        issues = tracker.validate_lineage()
        assert len(issues) >= 1
        assert any("failed validation" in issue for issue in issues)


class TestLineageTrackerSerialization:
    """Test LineageTracker serialization methods."""

    def test_to_json(self):
        """Test converting to JSON."""
        tracker = LineageTracker(graph_id="test_graph")
        tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
            row_count=100,
        )

        json_str = tracker.to_json()
        data = json.loads(json_str)

        assert data["graph_id"] == "test_graph"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["source_name"] == "data.csv"

    def test_save_and_load(self):
        """Test saving and loading lineage."""
        tracker = LineageTracker(graph_id="persistent_graph")
        source_id = tracker.record_source(
            source_type=SourceType.SUPABASE,
            source_name="metrics",
            row_count=5000,
            column_names=["id", "value"],
        )
        tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.SCALING,
            transformation_name="standardize",
            input_shape=(5000, 2),
            output_shape=(5000, 2),
        )
        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.RANDOM,
            ratios={"train": 0.8, "test": 0.2},
            train_size=4000,
            test_size=1000,
            leakage_checked=True,
        )
        tracker.associate_mlflow_run("run_123", "experiment_1")

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            tracker.save(temp_path)
            loaded = LineageTracker.load(temp_path)

            assert loaded.graph_id == "persistent_graph"
            assert len(loaded.graph.sources) == 1
            assert len(loaded.graph.transformations) == 1
            assert len(loaded.graph.splits) == 1
            assert loaded.graph.mlflow_run_id == "run_123"

            # Verify source details
            source = loaded.graph.sources[0]
            assert source.source_name == "metrics"
            assert source.row_count == 5000

            # Verify transformation details
            transform = loaded.graph.transformations[0]
            assert transform.transformation_name == "standardize"
            assert transform.input_shape == (5000, 2)

            # Verify split details
            split = loaded.graph.splits[0]
            assert split.train_size == 4000
            assert split.leakage_checked is True

        finally:
            os.unlink(temp_path)


class TestLineageTrackerSummary:
    """Test LineageTracker.get_summary method."""

    def test_get_summary_empty(self):
        """Test summary of empty tracker."""
        tracker = LineageTracker(graph_id="empty_graph")
        summary = tracker.get_summary()

        assert summary["graph_id"] == "empty_graph"
        assert summary["num_sources"] == 0
        assert summary["num_transformations"] == 0
        assert summary["num_splits"] == 0

    def test_get_summary_populated(self):
        """Test summary of populated tracker."""
        tracker = LineageTracker()
        source_id = tracker.record_source(
            source_type=SourceType.CSV,
            source_name="data.csv",
            row_count=1000,
        )
        tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.PREPROCESSING,
            transformation_name="preprocess",
            input_shape=(1000, 10),
            output_shape=(1000, 10),
        )
        tracker.record_split(
            source_id=source_id,
            split_type=SplitType.RANDOM,
            train_size=800,
            leakage_checked=True,
            leakage_detected=False,
        )

        summary = tracker.get_summary()

        assert summary["num_sources"] == 1
        assert summary["num_transformations"] == 1
        assert summary["num_splits"] == 1
        assert len(summary["sources"]) == 1
        assert summary["sources"][0]["rows"] == 1000
        assert summary["splits"][0]["leakage_detected"] is False


# ============================================================================
# MODULE FUNCTION TESTS
# ============================================================================


class TestModuleFunctions:
    """Test module-level functions."""

    def test_get_lineage_tracker_default(self):
        """Test get_lineage_tracker with default ID."""
        tracker = get_lineage_tracker()
        assert tracker.graph_id.startswith("lineage_")

    def test_get_lineage_tracker_custom_id(self):
        """Test get_lineage_tracker with custom ID."""
        tracker = get_lineage_tracker(graph_id="custom_lineage")
        assert tracker.graph_id == "custom_lineage"


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestLineageTrackerIntegration:
    """Integration tests for LineageTracker."""

    def test_full_pipeline_lineage(self):
        """Test tracking a complete ML pipeline."""
        tracker = LineageTracker(graph_id="ml_pipeline")

        # Record data source
        source_id = tracker.record_source(
            source_type=SourceType.SUPABASE,
            source_name="patient_metrics",
            query="SELECT * FROM metrics WHERE brand='remibrutinib'",
            row_count=10000,
            column_count=25,
            column_names=["patient_id", "metric1", "metric2", "target"],
        )

        # Record preprocessing
        step1_id = tracker.record_transformation(
            source_id=source_id,
            transformation_type=TransformationType.IMPUTATION,
            transformation_name="mean_imputation",
            input_shape=(10000, 25),
            output_shape=(10000, 25),
            parameters={"strategy": "mean"},
        )

        # Record scaling
        step2_id = tracker.record_transformation(
            source_id=step1_id,
            transformation_type=TransformationType.SCALING,
            transformation_name="standard_scaler",
            input_shape=(10000, 25),
            output_shape=(10000, 25),
            parameters={"with_mean": True, "with_std": True},
        )

        # Record encoding
        step3_id = tracker.record_transformation(
            source_id=step2_id,
            transformation_type=TransformationType.ENCODING,
            transformation_name="one_hot_encoder",
            input_shape=(10000, 25),
            output_shape=(10000, 40),
            columns_added=["cat_A", "cat_B", "cat_C"],
        )

        # Record split
        tracker.record_split(
            source_id=step3_id,
            split_type=SplitType.TEMPORAL,
            ratios={"train": 0.6, "val": 0.2, "test": 0.15, "holdout": 0.05},
            split_column="date",
            train_size=6000,
            val_size=2000,
            test_size=1500,
            holdout_size=500,
            leakage_checked=True,
            leakage_detected=False,
        )

        # Associate with MLflow
        tracker.associate_mlflow_run("run_abc123", "e2i_experiment")

        # Validate
        issues = tracker.validate_lineage()
        assert len(issues) == 0

        # Get summary
        summary = tracker.get_summary()
        assert summary["num_sources"] == 1
        assert summary["num_transformations"] == 3
        assert summary["num_splits"] == 1
        assert summary["mlflow_run_id"] == "run_abc123"

        # Verify transformation chain
        chain = tracker.graph.get_transformation_chain(source_id)
        assert len(chain) == 3
        assert chain[0].transformation_name == "mean_imputation"
        assert chain[2].transformation_name == "one_hot_encoder"

    def test_multi_source_lineage(self):
        """Test tracking lineage with multiple sources."""
        tracker = LineageTracker(graph_id="multi_source")

        # Record multiple sources
        source1_id = tracker.record_source(
            source_type=SourceType.SUPABASE,
            source_name="patient_data",
            row_count=5000,
        )
        source2_id = tracker.record_source(
            source_type=SourceType.FEATURE_STORE,
            source_name="historical_features",
            row_count=5000,
        )

        # Record merge transformation
        tracker.record_transformation(
            source_id=source1_id,
            transformation_type=TransformationType.MERGE,
            transformation_name="join_features",
            input_shape=(5000, 10),
            output_shape=(5000, 20),
            metadata={"join_key": "patient_id", "other_source": source2_id},
        )

        summary = tracker.get_summary()
        assert summary["num_sources"] == 2
        assert summary["num_transformations"] == 1

    def test_dataframe_pipeline_lineage(self):
        """Test tracking lineage with actual DataFrames."""
        tracker = LineageTracker(graph_id="df_pipeline")

        # Create sample DataFrame
        df = pd.DataFrame(
            {
                "patient_id": range(1, 101),
                "age": [25 + i % 50 for i in range(100)],
                "metric1": [10.0 + i * 0.5 for i in range(100)],
                "category": ["A", "B", "C", "D"] * 25,
            }
        )

        # Record from DataFrame
        source_id = tracker.record_source_from_dataframe(
            df=df,
            source_name="sample_patients",
            compute_hash=True,
        )

        source = tracker.graph.get_source(source_id)
        assert source.row_count == 100
        assert source.column_count == 4
        assert source.data_hash is not None
        assert "patient_id" in source.column_names
