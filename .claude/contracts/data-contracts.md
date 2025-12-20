# Data Layer Contracts

**Purpose**: Define interfaces and expectations for all data-related components to prevent integration issues.

---

## Input Data Contract

### Raw Data Schema

```python
# Expected schema for incoming data
RAW_DATA_SCHEMA = {
    "id": "string",  # Unique identifier
    "timestamp": "datetime",  # Event timestamp
    "feature_1": "float",
    "feature_2": "int",
    "feature_3": "categorical",
    "target": "float"  # For training data only
}

# Required fields (must be present and non-null)
REQUIRED_FIELDS = ["id", "timestamp", "feature_1"]

# Optional fields (can be null)
OPTIONAL_FIELDS = ["feature_2", "feature_3"]
```

### Data Quality Constraints

```python
DATA_QUALITY_RULES = {
    "id": {
        "unique": True,
        "format": r"^[A-Z0-9]{10}$"
    },
    "timestamp": {
        "min_date": "2020-01-01",
        "max_date": "today + 1day",  # Allow slight future dates for timezone issues
        "timezone": "UTC"
    },
    "feature_1": {
        "min_value": 0,
        "max_value": 100,
        "allow_null": False
    },
    "feature_2": {
        "min_value": 0,
        "max_value": None,
        "allow_null": True
    },
    "feature_3": {
        "allowed_values": ["A", "B", "C", "D"],
        "allow_null": True
    }
}
```

---

## Processed Data Contract

### Feature Schema

```python
# Schema after feature engineering
FEATURE_SCHEMA = {
    "id": "string",
    "timestamp": "datetime",

    # Original features (cleaned)
    "feature_1_clean": "float",
    "feature_2_imputed": "float",
    "feature_3_encoded": "int",

    # Engineered features
    "feature_1_squared": "float",
    "feature_lag_7d": "float",
    "feature_rolling_mean_30d": "float",

    # Time-based features
    "day_of_week": "int",
    "month": "int",
    "is_weekend": "bool",

    # Target (for training)
    "target": "float"
}

# Feature groups for easier management
FEATURE_GROUPS = {
    "original": ["feature_1_clean", "feature_2_imputed", "feature_3_encoded"],
    "engineered": ["feature_1_squared", "feature_lag_7d", "feature_rolling_mean_30d"],
    "temporal": ["day_of_week", "month", "is_weekend"]
}
```

### Feature Store Contract

```python
# Contract for feature store interactions
class FeatureStoreContract:
    """Contract for feature store operations."""

    def get_features(
        self,
        entity_ids: List[str],
        feature_names: List[str],
        timestamp: datetime = None
    ) -> pd.DataFrame:
        """
        Retrieve features with point-in-time correctness.

        Args:
            entity_ids: List of entity IDs
            feature_names: List of feature names to retrieve
            timestamp: Point-in-time for historical features (None = latest)

        Returns:
            DataFrame with requested features

        Raises:
            ValueError: If feature names don't exist
            ValueError: If timestamp is in the future
        """
        pass

    def write_features(
        self,
        features_df: pd.DataFrame,
        feature_group: str,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Write features to feature store.

        Args:
            features_df: DataFrame containing features
            feature_group: Name of the feature group
            metadata: Metadata about features (version, author, etc.)

        Raises:
            ValueError: If schema doesn't match expected
            ValueError: If feature group doesn't exist
        """
        pass
```

---

## Training Data Contract

### Train/Test Split Requirements

```python
SPLIT_REQUIREMENTS = {
    "method": "time_based",  # or "random", "stratified"
    "train_ratio": 0.7,
    "validation_ratio": 0.15,
    "test_ratio": 0.15,

    # For time-based splits
    "train_end_date": "2024-06-30",
    "validation_end_date": "2024-09-30",
    "test_end_date": "2024-12-31",

    # Leakage prevention
    "ensure_no_overlap": True,
    "check_temporal_order": True,

    # Stratification (if applicable)
    "stratify_column": "target_category",
    "min_samples_per_stratum": 100
}
```

### Data Leakage Checks

```python
# Mandatory checks before training
LEAKAGE_CHECKS = {
    "no_id_overlap": {
        "description": "Ensure no entity IDs appear in multiple splits",
        "critical": True
    },
    "temporal_order": {
        "description": "Ensure train dates < validation dates < test dates",
        "critical": True
    },
    "no_future_features": {
        "description": "Ensure no features derived from future data",
        "critical": True
    },
    "scaler_fit_on_train_only": {
        "description": "Ensure scalers/encoders only fit on training data",
        "critical": True
    }
}
```

---

## Data Versioning Contract

### Dataset Versioning

```python
# Required metadata for all datasets
DATASET_METADATA = {
    "version": "string",  # Semantic version (e.g., "1.2.0")
    "created_at": "datetime",
    "created_by": "string",
    "source_data_hash": "string",  # Hash of raw data
    "processing_script_version": "string",
    "row_count": "int",
    "column_count": "int",
    "quality_score": "float",  # Overall quality metric
    "notes": "string"
}

# Version naming convention
VERSION_FORMAT = "{major}.{minor}.{patch}"
# Major: Breaking schema changes
# Minor: New features or significant changes
# Patch: Bug fixes or minor updates
```

---

## Data Access Contract

### Data Loading Interface

```python
class DataLoaderContract:
    """Contract for data loading operations."""

    def load_raw_data(
        self,
        start_date: datetime,
        end_date: datetime,
        filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Load raw data within date range.

        Args:
            start_date: Start of date range (inclusive)
            end_date: End of date range (inclusive)
            filters: Optional filters to apply

        Returns:
            DataFrame conforming to RAW_DATA_SCHEMA

        Raises:
            ValueError: If date range is invalid
            ConnectionError: If data source unavailable
        """
        pass

    def load_processed_features(
        self,
        dataset_version: str,
        split: str  # "train", "validation", "test"
    ) -> pd.DataFrame:
        """
        Load processed features for a specific split.

        Args:
            dataset_version: Version of dataset to load
            split: Which split to load

        Returns:
            DataFrame conforming to FEATURE_SCHEMA

        Raises:
            ValueError: If version doesn't exist
            ValueError: If split is invalid
        """
        pass
```

---

## Data Validation Contract

### Validation Rules

```python
# Validation to run on all data before processing
VALIDATION_RULES = {
    "schema_validation": {
        "check": "Schema matches expected",
        "action_on_fail": "reject"
    },
    "null_check": {
        "check": "Required fields have no nulls",
        "action_on_fail": "reject"
    },
    "range_check": {
        "check": "Values within expected ranges",
        "action_on_fail": "warn_and_clip"
    },
    "duplicate_check": {
        "check": "No duplicate IDs",
        "action_on_fail": "reject"
    },
    "freshness_check": {
        "check": "Data not older than 7 days",
        "action_on_fail": "warn"
    }
}

class DataValidator:
    """Contract for data validation."""

    def validate_schema(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data schema matches expected."""
        pass

    def validate_quality(self, df: pd.DataFrame) -> ValidationResult:
        """Validate data quality constraints."""
        pass

    def validate_distribution(
        self,
        df: pd.DataFrame,
        reference_df: pd.DataFrame
    ) -> ValidationResult:
        """Validate distribution hasn't drifted significantly."""
        pass
```

---

## Change Management

### Breaking Changes
Changes to this contract that require code updates:
1. Adding required fields to schemas
2. Changing data types
3. Removing fields
4. Changing validation rules from "warn" to "reject"

### Non-Breaking Changes
Changes that don't require code updates:
1. Adding optional fields
2. Relaxing validation rules
3. Adding new feature groups
4. Updating documentation

### Deprecation Process
1. Announce deprecation with 30-day notice
2. Add warnings to deprecated functionality
3. Provide migration guide
4. Remove after grace period

---

**Last Updated**: [Date]
**Version**: 1.0
**Owner**: [Team/Person]

**Instructions**:
1. All data processing code MUST conform to these contracts
2. Validate contracts in CI/CD pipeline
3. Update version when making changes
4. Notify all stakeholders of breaking changes
