"""
Data Lineage Tracking for E2I Causal Analytics.

This module provides comprehensive data lineage tracking to trace data
from source through transformations to model training. It integrates
with MLflow for artifact storage and provides full auditability.

Features:
- Track data sources with metadata
- Record all data transformations
- Track train/val/test/holdout splits
- Integrate with MLflow artifacts
- Generate lineage graphs and reports

Usage:
    from src.mlops.data_lineage import LineageTracker

    tracker = LineageTracker()

    # Record data source
    source_id = tracker.record_source(
        source_type="supabase",
        source_name="patient_metrics",
        query="SELECT * FROM patient_metrics WHERE brand='remibrutinib'",
        row_count=10000,
        column_count=25,
    )

    # Record transformation
    tracker.record_transformation(
        source_id=source_id,
        transformation_type="preprocessing",
        transformation_name="fit_preprocessing",
        input_shape=(10000, 25),
        output_shape=(10000, 25),
        parameters={"scaling_method": "standard"},
    )

    # Record split
    tracker.record_split(
        source_id=source_id,
        split_type="temporal",
        ratios={"train": 0.6, "val": 0.2, "test": 0.15, "holdout": 0.05},
        split_column="date",
    )

Author: E2I Causal Analytics Team
Version: 1.0.0
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMS
# ============================================================================


class SourceType(str, Enum):
    """Types of data sources."""

    SUPABASE = "supabase"
    CSV = "csv"
    PARQUET = "parquet"
    FEATURE_STORE = "feature_store"
    API = "api"
    S3 = "s3"
    CACHED = "cached"
    GENERATED = "generated"  # Synthetic data


class TransformationType(str, Enum):
    """Types of data transformations."""

    PREPROCESSING = "preprocessing"
    FEATURE_ENGINEERING = "feature_engineering"
    IMPUTATION = "imputation"
    ENCODING = "encoding"
    SCALING = "scaling"
    FILTERING = "filtering"
    AGGREGATION = "aggregation"
    SPLITTING = "splitting"
    SAMPLING = "sampling"
    AUGMENTATION = "augmentation"
    MERGE = "merge"
    CUSTOM = "custom"


class SplitType(str, Enum):
    """Types of data splits."""

    RANDOM = "random"
    TEMPORAL = "temporal"
    STRATIFIED = "stratified"
    ENTITY = "entity"
    COMBINED = "combined"


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class DataSource:
    """Represents a data source in the lineage graph."""

    source_id: str
    source_type: SourceType
    source_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Source details
    query: Optional[str] = None
    file_path: Optional[str] = None
    table_name: Optional[str] = None

    # Data statistics
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    column_names: Optional[List[str]] = None
    schema: Optional[Dict[str, str]] = None

    # Checksum for data integrity
    data_hash: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "source_name": self.source_name,
            "created_at": self.created_at.isoformat(),
            "query": self.query,
            "file_path": self.file_path,
            "table_name": self.table_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "column_names": self.column_names,
            "schema": self.schema,
            "data_hash": self.data_hash,
            "metadata": self.metadata,
        }


@dataclass
class TransformationStep:
    """Represents a transformation step in the lineage graph."""

    step_id: str
    source_id: str  # Parent source/step
    transformation_type: TransformationType
    transformation_name: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Shape changes
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None

    # Configuration
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Statistics
    rows_affected: Optional[int] = None
    columns_added: Optional[List[str]] = None
    columns_removed: Optional[List[str]] = None
    columns_modified: Optional[List[str]] = None

    # Validation
    validation_passed: bool = True
    validation_errors: List[str] = field(default_factory=list)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "step_id": self.step_id,
            "source_id": self.source_id,
            "transformation_type": self.transformation_type.value,
            "transformation_name": self.transformation_name,
            "created_at": self.created_at.isoformat(),
            "input_shape": self.input_shape,
            "output_shape": self.output_shape,
            "parameters": self.parameters,
            "rows_affected": self.rows_affected,
            "columns_added": self.columns_added,
            "columns_removed": self.columns_removed,
            "columns_modified": self.columns_modified,
            "validation_passed": self.validation_passed,
            "validation_errors": self.validation_errors,
            "metadata": self.metadata,
        }


@dataclass
class SplitRecord:
    """Records a data split operation."""

    split_id: str
    source_id: str
    split_type: SplitType
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Split configuration
    ratios: Dict[str, float] = field(default_factory=dict)
    split_column: Optional[str] = None
    stratify_column: Optional[str] = None
    entity_column: Optional[str] = None
    random_seed: Optional[int] = None

    # Split results
    train_size: Optional[int] = None
    val_size: Optional[int] = None
    test_size: Optional[int] = None
    holdout_size: Optional[int] = None

    # Leakage detection
    leakage_checked: bool = False
    leakage_detected: bool = False
    leakage_details: Dict[str, Any] = field(default_factory=dict)

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "split_id": self.split_id,
            "source_id": self.source_id,
            "split_type": self.split_type.value,
            "created_at": self.created_at.isoformat(),
            "ratios": self.ratios,
            "split_column": self.split_column,
            "stratify_column": self.stratify_column,
            "entity_column": self.entity_column,
            "random_seed": self.random_seed,
            "train_size": self.train_size,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "holdout_size": self.holdout_size,
            "leakage_checked": self.leakage_checked,
            "leakage_detected": self.leakage_detected,
            "leakage_details": self.leakage_details,
            "metadata": self.metadata,
        }


@dataclass
class LineageGraph:
    """Complete lineage graph for a data pipeline."""

    graph_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    sources: List[DataSource] = field(default_factory=list)
    transformations: List[TransformationStep] = field(default_factory=list)
    splits: List[SplitRecord] = field(default_factory=list)

    # Associated MLflow run
    mlflow_run_id: Optional[str] = None
    experiment_name: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "graph_id": self.graph_id,
            "created_at": self.created_at.isoformat(),
            "sources": [s.to_dict() for s in self.sources],
            "transformations": [t.to_dict() for t in self.transformations],
            "splits": [s.to_dict() for s in self.splits],
            "mlflow_run_id": self.mlflow_run_id,
            "experiment_name": self.experiment_name,
            "metadata": self.metadata,
        }

    def get_source(self, source_id: str) -> Optional[DataSource]:
        """Get a source by ID."""
        for source in self.sources:
            if source.source_id == source_id:
                return source
        return None

    def get_transformation_chain(self, source_id: str) -> List[TransformationStep]:
        """Get all transformations applied to a source."""
        chain = []
        current_id = source_id
        for transform in self.transformations:
            if transform.source_id == current_id:
                chain.append(transform)
                current_id = transform.step_id
        return chain


# ============================================================================
# LINEAGE TRACKER
# ============================================================================


class LineageTracker:
    """
    Tracks data lineage through ML pipelines.

    This class provides methods to record data sources, transformations,
    and splits for complete pipeline traceability.
    """

    def __init__(self, graph_id: Optional[str] = None):
        """Initialize lineage tracker.

        Args:
            graph_id: Optional graph ID (auto-generated if not provided)
        """
        self.graph = LineageGraph(
            graph_id=graph_id or f"lineage_{uuid.uuid4().hex[:12]}"
        )
        logger.info(f"Initialized lineage tracker: {self.graph.graph_id}")

    @property
    def graph_id(self) -> str:
        """Get the graph ID."""
        return self.graph.graph_id

    # ========================================================================
    # DATA SOURCE TRACKING
    # ========================================================================

    def record_source(
        self,
        source_type: Union[SourceType, str],
        source_name: str,
        query: Optional[str] = None,
        file_path: Optional[str] = None,
        table_name: Optional[str] = None,
        row_count: Optional[int] = None,
        column_count: Optional[int] = None,
        column_names: Optional[List[str]] = None,
        schema: Optional[Dict[str, str]] = None,
        data_hash: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a data source.

        Args:
            source_type: Type of data source
            source_name: Name/identifier for the source
            query: SQL query if applicable
            file_path: File path if applicable
            table_name: Table name if applicable
            row_count: Number of rows
            column_count: Number of columns
            column_names: List of column names
            schema: Column type schema
            data_hash: Hash of data for integrity check
            metadata: Additional metadata

        Returns:
            Source ID
        """
        if isinstance(source_type, str):
            source_type = SourceType(source_type)

        source_id = f"src_{uuid.uuid4().hex[:8]}"

        source = DataSource(
            source_id=source_id,
            source_type=source_type,
            source_name=source_name,
            query=query,
            file_path=file_path,
            table_name=table_name,
            row_count=row_count,
            column_count=column_count,
            column_names=column_names,
            schema=schema,
            data_hash=data_hash,
            metadata=metadata or {},
        )

        self.graph.sources.append(source)
        logger.info(f"Recorded source: {source_name} ({source_type.value}) -> {source_id}")

        return source_id

    def record_source_from_dataframe(
        self,
        df: Any,  # pd.DataFrame
        source_name: str,
        source_type: Union[SourceType, str] = SourceType.CACHED,
        compute_hash: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a data source from a DataFrame.

        Args:
            df: Pandas DataFrame
            source_name: Name for the source
            source_type: Type of source
            compute_hash: Whether to compute data hash
            metadata: Additional metadata

        Returns:
            Source ID
        """
        import pandas as pd

        if not isinstance(df, pd.DataFrame):
            raise ValueError("df must be a pandas DataFrame")

        data_hash = None
        if compute_hash:
            data_hash = self._compute_dataframe_hash(df)

        # Extract schema
        schema = {col: str(df[col].dtype) for col in df.columns}

        return self.record_source(
            source_type=source_type,
            source_name=source_name,
            row_count=len(df),
            column_count=len(df.columns),
            column_names=list(df.columns),
            schema=schema,
            data_hash=data_hash,
            metadata=metadata,
        )

    def _compute_dataframe_hash(self, df: Any) -> str:
        """Compute hash of DataFrame for integrity checking."""
        import pandas as pd

        # Hash the shape and column names
        shape_str = f"{df.shape}"
        cols_str = ",".join(sorted(df.columns))

        # Sample some data for hash (full hash is expensive)
        sample_size = min(1000, len(df))
        sample = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df

        # Convert sample to string and hash
        data_str = sample.to_csv(index=False)
        combined = f"{shape_str}|{cols_str}|{data_str}"

        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    # ========================================================================
    # TRANSFORMATION TRACKING
    # ========================================================================

    def record_transformation(
        self,
        source_id: str,
        transformation_type: Union[TransformationType, str],
        transformation_name: str,
        input_shape: Optional[Tuple[int, ...]] = None,
        output_shape: Optional[Tuple[int, ...]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        rows_affected: Optional[int] = None,
        columns_added: Optional[List[str]] = None,
        columns_removed: Optional[List[str]] = None,
        columns_modified: Optional[List[str]] = None,
        validation_passed: bool = True,
        validation_errors: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a transformation step.

        Args:
            source_id: ID of the source/previous step
            transformation_type: Type of transformation
            transformation_name: Name of the transformation
            input_shape: Input data shape
            output_shape: Output data shape
            parameters: Transformation parameters
            rows_affected: Number of rows affected
            columns_added: Columns added
            columns_removed: Columns removed
            columns_modified: Columns modified
            validation_passed: Whether validation passed
            validation_errors: List of validation errors
            metadata: Additional metadata

        Returns:
            Step ID
        """
        if isinstance(transformation_type, str):
            transformation_type = TransformationType(transformation_type)

        step_id = f"step_{uuid.uuid4().hex[:8]}"

        step = TransformationStep(
            step_id=step_id,
            source_id=source_id,
            transformation_type=transformation_type,
            transformation_name=transformation_name,
            input_shape=input_shape,
            output_shape=output_shape,
            parameters=parameters or {},
            rows_affected=rows_affected,
            columns_added=columns_added,
            columns_removed=columns_removed,
            columns_modified=columns_modified,
            validation_passed=validation_passed,
            validation_errors=validation_errors or [],
            metadata=metadata or {},
        )

        self.graph.transformations.append(step)
        logger.info(
            f"Recorded transformation: {transformation_name} "
            f"({transformation_type.value}) -> {step_id}"
        )

        return step_id

    # ========================================================================
    # SPLIT TRACKING
    # ========================================================================

    def record_split(
        self,
        source_id: str,
        split_type: Union[SplitType, str],
        ratios: Optional[Dict[str, float]] = None,
        split_column: Optional[str] = None,
        stratify_column: Optional[str] = None,
        entity_column: Optional[str] = None,
        random_seed: Optional[int] = None,
        train_size: Optional[int] = None,
        val_size: Optional[int] = None,
        test_size: Optional[int] = None,
        holdout_size: Optional[int] = None,
        leakage_checked: bool = False,
        leakage_detected: bool = False,
        leakage_details: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a data split operation.

        Args:
            source_id: ID of the source being split
            split_type: Type of split
            ratios: Split ratios
            split_column: Column used for splitting (e.g., date)
            stratify_column: Column used for stratification
            entity_column: Column used for entity splitting
            random_seed: Random seed for reproducibility
            train_size: Size of training set
            val_size: Size of validation set
            test_size: Size of test set
            holdout_size: Size of holdout set
            leakage_checked: Whether leakage was checked
            leakage_detected: Whether leakage was detected
            leakage_details: Details of detected leakage
            metadata: Additional metadata

        Returns:
            Split ID
        """
        if isinstance(split_type, str):
            split_type = SplitType(split_type)

        split_id = f"split_{uuid.uuid4().hex[:8]}"

        split = SplitRecord(
            split_id=split_id,
            source_id=source_id,
            split_type=split_type,
            ratios=ratios or {},
            split_column=split_column,
            stratify_column=stratify_column,
            entity_column=entity_column,
            random_seed=random_seed,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            holdout_size=holdout_size,
            leakage_checked=leakage_checked,
            leakage_detected=leakage_detected,
            leakage_details=leakage_details or {},
            metadata=metadata or {},
        )

        self.graph.splits.append(split)
        logger.info(f"Recorded split: {split_type.value} -> {split_id}")

        return split_id

    def record_split_from_result(
        self,
        source_id: str,
        split_result: Any,  # SplitResult from data_splitter
        leakage_report: Optional[Any] = None,  # LeakageReport
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Record a split from a SplitResult object.

        Args:
            source_id: ID of the source being split
            split_result: SplitResult from data_splitter
            leakage_report: Optional LeakageReport from leakage detection
            metadata: Additional metadata

        Returns:
            Split ID
        """
        # Map split type
        split_type_map = {
            "random": SplitType.RANDOM,
            "temporal": SplitType.TEMPORAL,
            "stratified": SplitType.STRATIFIED,
            "entity": SplitType.ENTITY,
            "combined": SplitType.COMBINED,
        }

        split_type = split_type_map.get(
            split_result.config.split_type, SplitType.RANDOM
        )

        # Get sizes
        train_size = len(split_result.train) if split_result.train is not None else None
        val_size = len(split_result.validation) if split_result.validation is not None else None
        test_size = len(split_result.test) if split_result.test is not None else None
        holdout_size = len(split_result.holdout) if split_result.holdout is not None else None

        # Leakage info
        leakage_checked = leakage_report is not None
        leakage_detected = leakage_report.has_leakage if leakage_report else False
        leakage_details = {}
        if leakage_report and leakage_report.has_leakage:
            leakage_details = {
                "entity_leakage": leakage_report.entity_leakage,
                "temporal_leakage": leakage_report.temporal_leakage,
                "feature_leakage": leakage_report.feature_leakage,
                "recommendations": leakage_report.recommendations,
            }

        return self.record_split(
            source_id=source_id,
            split_type=split_type,
            ratios={
                "train": split_result.config.train_ratio,
                "val": split_result.config.val_ratio,
                "test": split_result.config.test_ratio,
                "holdout": split_result.config.holdout_ratio,
            },
            split_column=split_result.config.date_column,
            stratify_column=split_result.config.stratify_column,
            entity_column=split_result.config.entity_column,
            random_seed=split_result.config.random_seed,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            holdout_size=holdout_size,
            leakage_checked=leakage_checked,
            leakage_detected=leakage_detected,
            leakage_details=leakage_details,
            metadata=metadata,
        )

    # ========================================================================
    # MLFLOW INTEGRATION
    # ========================================================================

    def associate_mlflow_run(
        self,
        run_id: str,
        experiment_name: Optional[str] = None,
    ) -> None:
        """Associate lineage with an MLflow run.

        Args:
            run_id: MLflow run ID
            experiment_name: Experiment name
        """
        self.graph.mlflow_run_id = run_id
        self.graph.experiment_name = experiment_name
        logger.info(f"Associated lineage with MLflow run: {run_id}")

    async def log_to_mlflow(
        self,
        mlflow_connector: Any = None,  # MLflowConnector
    ) -> bool:
        """Log lineage as MLflow artifact.

        Args:
            mlflow_connector: Optional MLflowConnector instance

        Returns:
            True if successful
        """
        import json
        import tempfile
        import os

        if mlflow_connector is None:
            from src.mlops.mlflow_connector import get_mlflow_connector

            mlflow_connector = get_mlflow_connector()

        if not mlflow_connector.enabled or not self.graph.mlflow_run_id:
            logger.warning("MLflow not enabled or no run associated")
            return False

        try:
            # Create temporary file with lineage JSON
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(self.graph.to_dict(), f, indent=2)
                temp_path = f.name

            # Log as artifact
            await mlflow_connector._log_artifact(
                self.graph.mlflow_run_id,
                temp_path,
                "lineage",
            )

            # Cleanup
            os.unlink(temp_path)

            logger.info(f"Logged lineage to MLflow: {self.graph.graph_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to log lineage to MLflow: {e}")
            return False

    # ========================================================================
    # REPORTING
    # ========================================================================

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the lineage graph.

        Returns:
            Summary dictionary
        """
        return {
            "graph_id": self.graph.graph_id,
            "created_at": self.graph.created_at.isoformat(),
            "num_sources": len(self.graph.sources),
            "num_transformations": len(self.graph.transformations),
            "num_splits": len(self.graph.splits),
            "mlflow_run_id": self.graph.mlflow_run_id,
            "sources": [
                {
                    "id": s.source_id,
                    "type": s.source_type.value,
                    "name": s.source_name,
                    "rows": s.row_count,
                }
                for s in self.graph.sources
            ],
            "transformations": [
                {
                    "id": t.step_id,
                    "type": t.transformation_type.value,
                    "name": t.transformation_name,
                    "input_shape": t.input_shape,
                    "output_shape": t.output_shape,
                }
                for t in self.graph.transformations
            ],
            "splits": [
                {
                    "id": s.split_id,
                    "type": s.split_type.value,
                    "train_size": s.train_size,
                    "leakage_detected": s.leakage_detected,
                }
                for s in self.graph.splits
            ],
        }

    def validate_lineage(self) -> List[str]:
        """Validate the lineage graph for issues.

        Returns:
            List of validation warnings/errors
        """
        issues = []

        # Check for orphan transformations
        source_ids = {s.source_id for s in self.graph.sources}
        step_ids = {t.step_id for t in self.graph.transformations}
        all_ids = source_ids | step_ids

        for transform in self.graph.transformations:
            if transform.source_id not in all_ids:
                issues.append(
                    f"Transformation {transform.step_id} references unknown source: "
                    f"{transform.source_id}"
                )

        # Check for splits without leakage checks
        for split in self.graph.splits:
            if not split.leakage_checked:
                issues.append(
                    f"Split {split.split_id} was not checked for leakage"
                )

        # Check for detected leakage
        for split in self.graph.splits:
            if split.leakage_detected:
                issues.append(
                    f"CRITICAL: Split {split.split_id} has detected data leakage"
                )

        # Check for validation failures
        for transform in self.graph.transformations:
            if not transform.validation_passed:
                issues.append(
                    f"Transformation {transform.step_id} failed validation: "
                    f"{transform.validation_errors}"
                )

        return issues

    def to_json(self, indent: int = 2) -> str:
        """Convert lineage graph to JSON.

        Args:
            indent: JSON indentation

        Returns:
            JSON string
        """
        return json.dumps(self.graph.to_dict(), indent=indent)

    def save(self, file_path: str) -> None:
        """Save lineage to a JSON file.

        Args:
            file_path: Path to save the file
        """
        with open(file_path, "w") as f:
            f.write(self.to_json())
        logger.info(f"Saved lineage to: {file_path}")

    @classmethod
    def load(cls, file_path: str) -> "LineageTracker":
        """Load lineage from a JSON file.

        Args:
            file_path: Path to the file

        Returns:
            LineageTracker instance
        """
        with open(file_path, "r") as f:
            data = json.load(f)

        tracker = cls(graph_id=data["graph_id"])

        # Load sources
        for source_data in data.get("sources", []):
            tracker.graph.sources.append(
                DataSource(
                    source_id=source_data["source_id"],
                    source_type=SourceType(source_data["source_type"]),
                    source_name=source_data["source_name"],
                    created_at=datetime.fromisoformat(source_data["created_at"]),
                    query=source_data.get("query"),
                    file_path=source_data.get("file_path"),
                    table_name=source_data.get("table_name"),
                    row_count=source_data.get("row_count"),
                    column_count=source_data.get("column_count"),
                    column_names=source_data.get("column_names"),
                    schema=source_data.get("schema"),
                    data_hash=source_data.get("data_hash"),
                    metadata=source_data.get("metadata", {}),
                )
            )

        # Load transformations
        for step_data in data.get("transformations", []):
            tracker.graph.transformations.append(
                TransformationStep(
                    step_id=step_data["step_id"],
                    source_id=step_data["source_id"],
                    transformation_type=TransformationType(step_data["transformation_type"]),
                    transformation_name=step_data["transformation_name"],
                    created_at=datetime.fromisoformat(step_data["created_at"]),
                    input_shape=tuple(step_data["input_shape"]) if step_data.get("input_shape") else None,
                    output_shape=tuple(step_data["output_shape"]) if step_data.get("output_shape") else None,
                    parameters=step_data.get("parameters", {}),
                    rows_affected=step_data.get("rows_affected"),
                    columns_added=step_data.get("columns_added"),
                    columns_removed=step_data.get("columns_removed"),
                    columns_modified=step_data.get("columns_modified"),
                    validation_passed=step_data.get("validation_passed", True),
                    validation_errors=step_data.get("validation_errors", []),
                    metadata=step_data.get("metadata", {}),
                )
            )

        # Load splits
        for split_data in data.get("splits", []):
            tracker.graph.splits.append(
                SplitRecord(
                    split_id=split_data["split_id"],
                    source_id=split_data["source_id"],
                    split_type=SplitType(split_data["split_type"]),
                    created_at=datetime.fromisoformat(split_data["created_at"]),
                    ratios=split_data.get("ratios", {}),
                    split_column=split_data.get("split_column"),
                    stratify_column=split_data.get("stratify_column"),
                    entity_column=split_data.get("entity_column"),
                    random_seed=split_data.get("random_seed"),
                    train_size=split_data.get("train_size"),
                    val_size=split_data.get("val_size"),
                    test_size=split_data.get("test_size"),
                    holdout_size=split_data.get("holdout_size"),
                    leakage_checked=split_data.get("leakage_checked", False),
                    leakage_detected=split_data.get("leakage_detected", False),
                    leakage_details=split_data.get("leakage_details", {}),
                    metadata=split_data.get("metadata", {}),
                )
            )

        tracker.graph.mlflow_run_id = data.get("mlflow_run_id")
        tracker.graph.experiment_name = data.get("experiment_name")
        tracker.graph.metadata = data.get("metadata", {})

        logger.info(f"Loaded lineage from: {file_path}")
        return tracker


# ============================================================================
# MODULE-LEVEL ACCESS
# ============================================================================


def get_lineage_tracker(graph_id: Optional[str] = None) -> LineageTracker:
    """Get a new lineage tracker.

    Args:
        graph_id: Optional graph ID

    Returns:
        LineageTracker instance
    """
    return LineageTracker(graph_id=graph_id)
