"""
Split Validator for Synthetic Data

Validates ML-compliant data splits:
- Patient-level isolation (no patient in multiple splits)
- Temporal ordering (chronological splits)
- Correct split ratios
- No data leakage
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pandas as pd

from ..config import SyntheticDataConfig

logger = logging.getLogger(__name__)


@dataclass
class LeakageInfo:
    """Information about a detected data leakage."""

    leakage_type: str
    severity: str  # "critical", "warning", "info"
    description: str
    affected_entities: int = 0
    affected_splits: List[str] = field(default_factory=list)
    sample_ids: List[str] = field(default_factory=list)


@dataclass
class SplitValidationResult:
    """Result of split validation."""

    is_valid: bool
    total_records: int
    split_counts: Dict[str, int] = field(default_factory=dict)
    split_ratios: Dict[str, float] = field(default_factory=dict)
    expected_ratios: Dict[str, float] = field(default_factory=dict)
    ratio_errors: Dict[str, float] = field(default_factory=dict)
    leakages: List[LeakageInfo] = field(default_factory=list)
    temporal_violations: int = 0
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def add_leakage(self, leakage: LeakageInfo) -> None:
        """Add a detected leakage."""
        self.leakages.append(leakage)
        if leakage.severity == "critical":
            self.is_valid = False

    def has_critical_leakage(self) -> bool:
        """Check if any critical leakage was detected."""
        return any(l.severity == "critical" for l in self.leakages)


class SplitValidator:
    """
    Validates ML data splits for leakage and correct distribution.

    Integrates with the existing LeakageDetector from data_splitter.py
    when available, otherwise uses built-in checks.

    Usage:
        validator = SplitValidator()
        result = validator.validate(
            df=patient_df,
            entity_column="patient_id",
            date_column="journey_start_date",
            split_column="data_split"
        )
        if not result.is_valid:
            for leakage in result.leakages:
                print(f"{leakage.severity}: {leakage.description}")
    """

    def __init__(
        self,
        config: Optional[SyntheticDataConfig] = None,
        ratio_tolerance: float = 0.05,
    ):
        """
        Initialize split validator.

        Args:
            config: Synthetic data config with split boundaries
            ratio_tolerance: Maximum allowed deviation from expected ratios
        """
        self.config = config or SyntheticDataConfig()
        self.ratio_tolerance = ratio_tolerance
        self._leakage_detector = self._get_leakage_detector()

    def _get_leakage_detector(self):
        """Try to get the LeakageDetector from data_splitter module."""
        try:
            from src.repositories.data_splitter import LeakageDetector

            return LeakageDetector()
        except ImportError:
            logger.debug("LeakageDetector not available, using built-in checks")
            return None

    def validate(
        self,
        df: pd.DataFrame,
        entity_column: str = "patient_id",
        date_column: str = "journey_start_date",
        split_column: str = "data_split",
        target_column: Optional[str] = None,
    ) -> SplitValidationResult:
        """
        Validate data splits for leakage and correct distribution.

        Args:
            df: DataFrame to validate
            entity_column: Column containing entity IDs (patient, HCP, etc.)
            date_column: Column containing dates for temporal validation
            split_column: Column containing split assignments
            target_column: Optional target column for target leakage check

        Returns:
            SplitValidationResult with validation details
        """
        result = SplitValidationResult(
            is_valid=True,
            total_records=len(df),
            expected_ratios={
                "train": self.config.split_boundaries.train_ratio,
                "validation": self.config.split_boundaries.validation_ratio,
                "test": self.config.split_boundaries.test_ratio,
                "holdout": self.config.split_boundaries.holdout_ratio,
            },
        )

        # Check required columns
        required_cols = [entity_column, split_column]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            result.errors.append(f"Missing columns: {missing}")
            result.is_valid = False
            return result

        # 1. Calculate split distribution
        self._validate_split_distribution(df, split_column, result)

        # 2. Check entity-level isolation
        self._validate_entity_isolation(df, entity_column, split_column, result)

        # 3. Check temporal ordering
        if date_column in df.columns:
            self._validate_temporal_ordering(df, date_column, split_column, result)

        # 4. Check for target leakage if target provided
        if target_column and target_column in df.columns:
            self._validate_target_leakage(df, entity_column, split_column, target_column, result)

        # 5. Use LeakageDetector if available
        if self._leakage_detector:
            self._run_leakage_detector(
                df, entity_column, date_column, split_column, target_column, result
            )

        return result

    def _validate_split_distribution(
        self,
        df: pd.DataFrame,
        split_column: str,
        result: SplitValidationResult,
    ) -> None:
        """Validate that split ratios match expectations."""
        split_counts = df[split_column].value_counts().to_dict()
        total = len(df)

        for split in ["train", "validation", "test", "holdout"]:
            count = split_counts.get(split, 0)
            ratio = count / total if total > 0 else 0.0
            expected = result.expected_ratios.get(split, 0.0)
            error = abs(ratio - expected)

            result.split_counts[split] = count
            result.split_ratios[split] = ratio
            result.ratio_errors[split] = error

            if error > self.ratio_tolerance:
                result.warnings.append(
                    f"Split '{split}' ratio {ratio:.2%} differs from expected "
                    f"{expected:.2%} by {error:.2%}"
                )

    def _validate_entity_isolation(
        self,
        df: pd.DataFrame,
        entity_column: str,
        split_column: str,
        result: SplitValidationResult,
    ) -> None:
        """Check that no entity appears in multiple splits."""
        entity_splits = df.groupby(entity_column)[split_column].nunique()
        multi_split_entities = entity_splits[entity_splits > 1]

        if len(multi_split_entities) > 0:
            # Get sample of problematic entities
            sample_ids = multi_split_entities.head(5).index.tolist()

            result.add_leakage(
                LeakageInfo(
                    leakage_type="entity_overlap",
                    severity="critical",
                    description=(
                        f"{len(multi_split_entities)} entities appear in multiple splits. "
                        "This violates patient-level isolation."
                    ),
                    affected_entities=len(multi_split_entities),
                    sample_ids=sample_ids,
                )
            )

    def _validate_temporal_ordering(
        self,
        df: pd.DataFrame,
        date_column: str,
        split_column: str,
        result: SplitValidationResult,
    ) -> None:
        """Check that splits are temporally ordered."""
        # Convert dates if needed
        dates = pd.to_datetime(df[date_column])

        # Get date ranges per split
        split_dates = {}
        for split in ["train", "validation", "test", "holdout"]:
            mask = df[split_column] == split
            if mask.any():
                split_dates[split] = {
                    "min": dates[mask].min(),
                    "max": dates[mask].max(),
                }

        # Check for temporal overlap
        splits_order = ["train", "validation", "test", "holdout"]
        violations = 0

        for i, split1 in enumerate(splits_order[:-1]):
            if split1 not in split_dates:
                continue
            for split2 in splits_order[i + 1 :]:
                if split2 not in split_dates:
                    continue

                # Check if earlier split has dates after later split
                if split_dates[split1]["max"] > split_dates[split2]["min"]:
                    violations += 1
                    result.add_leakage(
                        LeakageInfo(
                            leakage_type="temporal_overlap",
                            severity="warning",
                            description=(
                                f"Temporal overlap between '{split1}' "
                                f"(max: {split_dates[split1]['max']}) and '{split2}' "
                                f"(min: {split_dates[split2]['min']})"
                            ),
                            affected_splits=[split1, split2],
                        )
                    )

        result.temporal_violations = violations

    def _validate_target_leakage(
        self,
        df: pd.DataFrame,
        entity_column: str,
        split_column: str,
        target_column: str,
        result: SplitValidationResult,
    ) -> None:
        """Check for target leakage (future information in features)."""
        # This is a simplified check - full target leakage detection
        # requires understanding feature-target relationships
        train_targets = df[df[split_column] == "train"][target_column]
        test_targets = df[df[split_column] == "test"][target_column]

        # Check if train targets have suspiciously high correlation with test
        # This is a heuristic - real leakage detection is more complex
        if len(train_targets) > 0 and len(test_targets) > 0:
            train_mean = train_targets.mean()
            test_mean = test_targets.mean()

            # If distributions are very different, might indicate issues
            if abs(train_mean - test_mean) > 0.3 * train_targets.std():
                result.warnings.append(
                    f"Target distribution differs significantly between train "
                    f"(mean={train_mean:.3f}) and test (mean={test_mean:.3f})"
                )

    def _run_leakage_detector(
        self,
        df: pd.DataFrame,
        entity_column: str,
        date_column: str,
        split_column: str,
        target_column: Optional[str],
        result: SplitValidationResult,
    ) -> None:
        """Run the LeakageDetector from data_splitter if available."""
        if not self._leakage_detector:
            return

        try:
            # Split dataframes by split type
            train_df = df[df[split_column] == "train"].copy()
            val_df = df[df[split_column] == "validation"].copy()
            test_df = df[df[split_column] == "test"].copy()
            holdout_df = df[df[split_column] == "holdout"].copy()

            # Run detector
            leakage_report = self._leakage_detector.detect_leakage(
                train_df=train_df,
                val_df=val_df,
                test_df=test_df,
                entity_column=entity_column,
                date_column=date_column if date_column in df.columns else None,
                target_column=target_column,
                holdout_df=holdout_df if len(holdout_df) > 0 else None,
            )

            # Process leakage report
            if hasattr(leakage_report, "has_leakage") and leakage_report.has_leakage:
                if hasattr(leakage_report, "leakage_types"):
                    for lt in leakage_report.leakage_types:
                        result.add_leakage(
                            LeakageInfo(
                                leakage_type=lt.get("type", "unknown"),
                                severity="critical",
                                description=lt.get("description", "Leakage detected"),
                            )
                        )

        except Exception as e:
            result.warnings.append(f"LeakageDetector check failed: {str(e)}")

    def validate_multiple_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        entity_column: str = "patient_id",
        date_column: str = "journey_start_date",
        split_column: str = "data_split",
    ) -> Dict[str, SplitValidationResult]:
        """
        Validate splits across multiple related datasets.

        Args:
            datasets: Dict of table_name -> DataFrame
            entity_column: Column containing entity IDs
            date_column: Column containing dates
            split_column: Column containing split assignments

        Returns:
            Dict of table_name -> SplitValidationResult
        """
        results = {}

        for table_name, df in datasets.items():
            if split_column not in df.columns:
                result = SplitValidationResult(
                    is_valid=True,
                    total_records=len(df),
                )
                result.warnings.append(f"No split column in {table_name}")
                results[table_name] = result
            else:
                results[table_name] = self.validate(
                    df=df,
                    entity_column=entity_column,
                    date_column=date_column,
                    split_column=split_column,
                )

        # Check cross-dataset consistency
        self._validate_cross_dataset_consistency(datasets, results, entity_column, split_column)

        return results

    def _validate_cross_dataset_consistency(
        self,
        datasets: Dict[str, pd.DataFrame],
        results: Dict[str, SplitValidationResult],
        entity_column: str,
        split_column: str,
    ) -> None:
        """Check that entity splits are consistent across datasets."""
        # Collect entity -> split mapping from first dataset with both columns
        entity_splits: Dict[str, str] = {}
        source_table = None

        for table_name, df in datasets.items():
            if entity_column in df.columns and split_column in df.columns:
                for _, row in df.iterrows():
                    entity = row[entity_column]
                    split = row[split_column]
                    if entity not in entity_splits:
                        entity_splits[entity] = split
                        source_table = table_name
                break

        if not entity_splits or not source_table:
            return

        # Check other datasets for consistency
        for table_name, df in datasets.items():
            if table_name == source_table:
                continue
            if entity_column not in df.columns or split_column not in df.columns:
                continue

            inconsistent = 0
            for _, row in df.iterrows():
                entity = row[entity_column]
                split = row[split_column]
                if entity in entity_splits and entity_splits[entity] != split:
                    inconsistent += 1

            if inconsistent > 0:
                results[table_name].add_leakage(
                    LeakageInfo(
                        leakage_type="cross_dataset_inconsistency",
                        severity="critical",
                        description=(
                            f"{inconsistent} entities have different splits in "
                            f"{table_name} vs {source_table}"
                        ),
                        affected_entities=inconsistent,
                    )
                )

    def get_validation_summary(self, results: Dict[str, SplitValidationResult]) -> Dict[str, Any]:
        """Get a summary of split validation results."""
        all_leakages = []
        for name, result in results.items():
            for leakage in result.leakages:
                all_leakages.append(
                    {
                        "table": name,
                        "type": leakage.leakage_type,
                        "severity": leakage.severity,
                        "description": leakage.description,
                    }
                )

        return {
            "all_valid": all(r.is_valid for r in results.values()),
            "total_tables": len(results),
            "valid_tables": sum(1 for r in results.values() if r.is_valid),
            "total_leakages": len(all_leakages),
            "critical_leakages": sum(1 for l in all_leakages if l["severity"] == "critical"),
            "leakages": all_leakages,
            "tables": {
                name: {
                    "is_valid": r.is_valid,
                    "records": r.total_records,
                    "split_ratios": r.split_ratios,
                    "leakages": len(r.leakages),
                }
                for name, r in results.items()
            },
        }
