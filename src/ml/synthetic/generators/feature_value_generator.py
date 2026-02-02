"""
Feature Value Generator.

Generates time-series feature values for the feature store.
Used by Drift Monitor agent for drift detection.
"""

import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseGenerator, GeneratorConfig


class FeatureValueGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for feature values time-series data.

    Creates time-series values for each feature with:
    - Proper event_timestamp for drift detection windows
    - Entity values as JSONB (hcp_id, patient_id, brand)
    - Freshness status tracking

    Requires features_df from FeatureStoreSeeder to map feature IDs.
    """

    # Value generation configurations by feature name
    VALUE_CONFIGS: Dict[str, Dict] = {
        # HCP demographics
        "specialty_encoded": {
            "type": "categorical",
            "options": ["oncology", "dermatology", "rheumatology", "hematology", "neurology"],
            "weights": [0.30, 0.25, 0.20, 0.15, 0.10],
        },
        "years_experience": {
            "type": "int",
            "mean": 15,
            "std": 8,
            "min": 1,
            "max": 45,
        },
        "academic_flag": {
            "type": "binary",
            "p": 0.30,
        },
        "patient_volume_monthly": {
            "type": "int",
            "mean": 120,
            "std": 40,
            "min": 20,
            "max": 300,
        },
        # Patient features
        "disease_severity": {
            "type": "float",
            "mean": 5.0,
            "std": 2.0,
            "min": 0,
            "max": 10,
        },
        "age_at_diagnosis": {
            "type": "int",
            "mean": 55,
            "std": 15,
            "min": 18,
            "max": 85,
        },
        "comorbidity_count": {
            "type": "int",
            "mean": 2,
            "std": 1.5,
            "min": 0,
            "max": 8,
        },
        "prior_treatment_count": {
            "type": "int",
            "mean": 1.5,
            "std": 1,
            "min": 0,
            "max": 5,
        },
        "insurance_tier": {
            "type": "categorical",
            "options": ["high", "medium", "low"],
            "weights": [0.30, 0.50, 0.20],
        },
        # Brand performance
        "trx_30d": {
            "type": "int",
            "mean": 5000,
            "std": 1500,
            "min": 500,
            "max": 15000,
        },
        "nrx_30d": {
            "type": "int",
            "mean": 1000,
            "std": 400,
            "min": 100,
            "max": 3000,
        },
        "conversion_rate": {
            "type": "float",
            "mean": 0.15,
            "std": 0.05,
            "min": 0.0,
            "max": 1.0,
        },
        # Causal features
        "engagement_score": {
            "type": "float",
            "mean": 5.5,
            "std": 2.0,
            "min": 0,
            "max": 10,
        },
        "treatment_propensity": {
            "type": "float",
            "mean": 0.35,
            "std": 0.15,
            "min": 0.0,
            "max": 1.0,
        },
        "outcome_probability": {
            "type": "float",
            "mean": 0.65,
            "std": 0.15,
            "min": 0.0,
            "max": 1.0,
        },
    }

    # Entity key types for generating appropriate entity values
    ENTITY_CONFIGS: Dict[str, str] = {
        "hcp_id": "hcp",
        "patient_id": "patient",
        "brand": "brand",
        "region": "region",
    }

    BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali"]
    REGIONS = ["northeast", "south", "midwest", "west"]

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "feature_values"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        features_df: Optional[pd.DataFrame] = None,
        patient_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the feature value generator.

        Args:
            config: Generator configuration.
            features_df: Features DataFrame from FeatureStoreSeeder (required).
            patient_df: Optional patient DataFrame for entity consistency.
        """
        super().__init__(config)
        self.features_df = features_df
        self.patient_df = patient_df

        # Extract entity IDs from patient_df if available
        self._patient_ids = None
        self._hcp_ids = None
        if patient_df is not None:
            if "patient_id" in patient_df.columns:
                self._patient_ids = patient_df["patient_id"].unique().tolist()
            if "hcp_id" in patient_df.columns:
                self._hcp_ids = patient_df["hcp_id"].unique().tolist()

    def generate(self) -> pd.DataFrame:
        """
        Generate feature values time-series.

        Returns:
            DataFrame with feature values matching schema.
        """
        if self.features_df is None or len(self.features_df) == 0:
            raise ValueError("features_df is required for FeatureValueGenerator")

        n = self.config.n_records
        self._log(f"Generating {n} feature values records")

        # Get feature info
        features = self.features_df.to_dict("records")
        n_features = len(features)

        # Calculate values per feature
        values_per_feature = max(1, n // n_features)

        records = []
        for feature in features:
            feature_records = self._generate_feature_values(
                feature_id=feature["id"],
                feature_name=feature["name"],
                entity_keys=feature["entity_keys"],
                n_values=values_per_feature,
            )
            records.extend(feature_records)

        df = pd.DataFrame(records)

        # Trim to requested size
        if len(df) > n:
            df = df.sample(n=n, random_state=self.config.seed)
            df = df.reset_index(drop=True)

        self._log(f"Generated {len(df)} feature values records")
        return df

    def _generate_feature_values(
        self,
        feature_id: str,
        feature_name: str,
        entity_keys: List[str],
        n_values: int,
    ) -> List[Dict]:
        """
        Generate values for a single feature.

        Args:
            feature_id: UUID of the feature.
            feature_name: Name of the feature.
            entity_keys: List of entity keys (e.g., ["hcp_id"], ["brand", "region"]).
            n_values: Number of values to generate.

        Returns:
            List of feature value records.
        """
        records = []
        value_config = self.VALUE_CONFIGS.get(
            feature_name, {"type": "float", "mean": 0.5, "std": 0.2}
        )

        # Generate timestamps across the date range
        timestamps = self._generate_timestamps(n_values)

        for i in range(n_values):
            # Generate entity values
            entity_values = self._generate_entity_values(entity_keys)

            # Generate feature value
            value = self._generate_value(value_config)

            # Determine freshness based on timestamp age
            ts = timestamps[i]
            age_hours = (datetime.now() - ts).total_seconds() / 3600
            if age_hours < 24:
                freshness = "fresh"
            elif age_hours < 168:  # 7 days
                freshness = "stale"
            else:
                freshness = "expired"

            records.append(
                {
                    "id": str(uuid.uuid4()),
                    "feature_id": feature_id,
                    "entity_values": entity_values,
                    "value": {"value": value},  # JSONB format
                    "event_timestamp": ts.isoformat(),
                    "freshness_status": freshness,
                }
            )

        return records

    def _generate_timestamps(self, n: int) -> List[datetime]:
        """Generate timestamps across the date range."""
        start = datetime.combine(self.config.start_date, datetime.min.time())
        end = datetime.combine(self.config.end_date, datetime.max.time())

        # Bias towards more recent dates (exponential distribution)
        total_seconds = (end - start).total_seconds()
        offsets = self._rng.exponential(total_seconds / 3, n)
        offsets = np.clip(offsets, 0, total_seconds)

        timestamps = [end - timedelta(seconds=float(offset)) for offset in offsets]
        return sorted(timestamps)

    def _generate_entity_values(self, entity_keys: List[str]) -> Dict:
        """Generate entity values for given keys."""
        entity_values = {}

        for key in entity_keys:
            if key == "hcp_id":
                if self._hcp_ids:
                    entity_values[key] = self._rng.choice(self._hcp_ids)
                else:
                    entity_values[key] = f"hcp_{self._rng.integers(1, 5000):05d}"
            elif key == "patient_id":
                if self._patient_ids:
                    entity_values[key] = self._rng.choice(self._patient_ids)
                else:
                    entity_values[key] = f"pt_{self._rng.integers(1, 25000):06d}"
            elif key == "brand":
                entity_values[key] = self._rng.choice(self.BRANDS)
            elif key == "region":
                entity_values[key] = self._rng.choice(self.REGIONS)
            else:
                entity_values[key] = f"{key}_{self._rng.integers(1, 1000):04d}"

        return entity_values

    def _generate_value(self, config: Dict):
        """Generate a single value based on config."""
        value_type = config.get("type", "float")

        if value_type == "categorical":
            options = config["options"]
            weights = config.get("weights")
            return self._rng.choice(options, p=weights)

        elif value_type == "binary":
            p = config.get("p", 0.5)
            return int(self._rng.random() < p)

        elif value_type == "int":
            mean = config.get("mean", 50)
            std = config.get("std", 10)
            min_val = config.get("min", 0)
            max_val = config.get("max", 100)
            value = self._rng.normal(mean, std)
            return int(np.clip(value, min_val, max_val))

        elif value_type == "float":
            mean = config.get("mean", 0.5)
            std = config.get("std", 0.2)
            min_val = config.get("min", 0.0)
            max_val = config.get("max", 1.0)
            value = self._rng.normal(mean, std)
            return round(float(np.clip(value, min_val, max_val)), 4)

        else:
            return self._rng.random()
