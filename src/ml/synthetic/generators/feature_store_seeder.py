"""
Feature Store Seeder.

Seeds feature groups and features for the feature store.
Supports drift detection and feature monitoring.
"""

import uuid
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .base import BaseGenerator, GeneratorConfig


class FeatureStoreSeeder(BaseGenerator[pd.DataFrame]):
    """
    Seeder for feature store metadata (groups and features).

    Creates the comprehensive feature set (~15 features across 4 domains):
    1. hcp_demographics (4 features)
    2. patient_features (5 features)
    3. brand_performance (3 features)
    4. causal_features (3 features)
    """

    # Feature group definitions
    FEATURE_GROUPS: Dict[str, Dict] = {
        "hcp_demographics": {
            "description": "Healthcare provider demographic and profile features",
            "owner": "data-team",
            "source_table": "hcp_profiles",
            "expected_update_frequency_hours": 168,  # Weekly
            "max_age_hours": 720,  # 30 days
            "tags": ["demographics", "hcp", "core"],
        },
        "patient_features": {
            "description": "Patient journey and demographic features",
            "owner": "data-team",
            "source_table": "patient_journeys",
            "expected_update_frequency_hours": 24,  # Daily
            "max_age_hours": 168,  # 7 days
            "tags": ["patient", "journey", "core"],
        },
        "brand_performance": {
            "description": "Brand-level performance and prescription metrics",
            "owner": "analytics-team",
            "source_table": "business_metrics",
            "expected_update_frequency_hours": 24,  # Daily
            "max_age_hours": 168,  # 7 days
            "tags": ["brand", "performance", "metrics"],
        },
        "causal_features": {
            "description": "Features used in causal inference and treatment effect estimation",
            "owner": "ml-team",
            "source_table": "patient_journeys",
            "expected_update_frequency_hours": 24,  # Daily
            "max_age_hours": 72,  # 3 days for fresher causal data
            "tags": ["causal", "treatment", "outcome"],
        },
    }

    # Feature definitions by group
    FEATURES: Dict[str, List[Dict]] = {
        "hcp_demographics": [
            {
                "name": "specialty_encoded",
                "description": "Encoded primary specialty of HCP",
                "value_type": "string",
                "entity_keys": ["hcp_id"],
                "drift_threshold": 0.15,
                "tags": ["categorical"],
            },
            {
                "name": "years_experience",
                "description": "Number of years HCP has been in practice",
                "value_type": "int64",
                "entity_keys": ["hcp_id"],
                "drift_threshold": 0.10,
                "tags": ["numerical"],
            },
            {
                "name": "academic_flag",
                "description": "Whether HCP is affiliated with academic institution (0/1)",
                "value_type": "int64",
                "entity_keys": ["hcp_id"],
                "drift_threshold": 0.05,
                "tags": ["binary"],
            },
            {
                "name": "patient_volume_monthly",
                "description": "Average monthly patient volume for HCP",
                "value_type": "int64",
                "entity_keys": ["hcp_id"],
                "drift_threshold": 0.15,
                "tags": ["numerical", "volume"],
            },
        ],
        "patient_features": [
            {
                "name": "disease_severity",
                "description": "Disease severity score (0-10 scale)",
                "value_type": "float64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.10,
                "tags": ["numerical", "clinical"],
            },
            {
                "name": "age_at_diagnosis",
                "description": "Patient age at diagnosis in years",
                "value_type": "int64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.10,
                "tags": ["numerical", "demographic"],
            },
            {
                "name": "comorbidity_count",
                "description": "Number of comorbid conditions",
                "value_type": "int64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.15,
                "tags": ["numerical", "clinical"],
            },
            {
                "name": "prior_treatment_count",
                "description": "Number of prior treatments attempted",
                "value_type": "int64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.10,
                "tags": ["numerical", "history"],
            },
            {
                "name": "insurance_tier",
                "description": "Insurance coverage tier (high/medium/low)",
                "value_type": "string",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.10,
                "tags": ["categorical", "coverage"],
            },
        ],
        "brand_performance": [
            {
                "name": "trx_30d",
                "description": "Total prescriptions in last 30 days",
                "value_type": "int64",
                "entity_keys": ["brand", "region"],
                "drift_threshold": 0.20,
                "tags": ["aggregate", "rx"],
            },
            {
                "name": "nrx_30d",
                "description": "New prescriptions in last 30 days",
                "value_type": "int64",
                "entity_keys": ["brand", "region"],
                "drift_threshold": 0.20,
                "tags": ["aggregate", "rx"],
            },
            {
                "name": "conversion_rate",
                "description": "HCP conversion rate (0-1)",
                "value_type": "float64",
                "entity_keys": ["brand", "region"],
                "drift_threshold": 0.10,
                "tags": ["rate", "performance"],
            },
        ],
        "causal_features": [
            {
                "name": "engagement_score",
                "description": "HCP engagement score (0-10) - treatment variable",
                "value_type": "float64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.08,
                "tags": ["treatment", "causal"],
            },
            {
                "name": "treatment_propensity",
                "description": "Predicted propensity to initiate treatment (0-1)",
                "value_type": "float64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.10,
                "tags": ["predicted", "causal"],
            },
            {
                "name": "outcome_probability",
                "description": "Predicted probability of treatment success (0-1)",
                "value_type": "float64",
                "entity_keys": ["patient_id"],
                "drift_threshold": 0.10,
                "tags": ["predicted", "outcome"],
            },
        ],
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "feature_store_metadata"

    def __init__(self, config: Optional[GeneratorConfig] = None):
        """
        Initialize the feature store seeder.

        Args:
            config: Generator configuration.
        """
        super().__init__(config)

    def generate(self) -> pd.DataFrame:
        """
        Generate both feature groups and features.

        Returns:
            DataFrame with features (feature_groups returned via seed() method).
        """
        _, features_df = self.seed()
        return features_df

    def seed(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Seed feature groups and features.

        Returns:
            Tuple of (feature_groups_df, features_df).
        """
        self._log("Seeding feature groups and features")

        # Generate feature groups
        feature_groups_records = []
        group_id_map = {}

        for group_name, group_config in self.FEATURE_GROUPS.items():
            group_id = str(uuid.uuid4())
            group_id_map[group_name] = group_id

            feature_groups_records.append(
                {
                    "id": group_id,
                    "name": group_name,
                    "description": group_config["description"],
                    "owner": group_config["owner"],
                    "tags": group_config["tags"],
                    "source_table": group_config["source_table"],
                    "expected_update_frequency_hours": group_config[
                        "expected_update_frequency_hours"
                    ],
                    "max_age_hours": group_config["max_age_hours"],
                }
            )

        feature_groups_df = pd.DataFrame(feature_groups_records)

        # Generate features
        features_records = []

        for group_name, features in self.FEATURES.items():
            group_id = group_id_map[group_name]

            for feature_config in features:
                features_records.append(
                    {
                        "id": str(uuid.uuid4()),
                        "feature_group_id": group_id,
                        "name": feature_config["name"],
                        "description": feature_config["description"],
                        "value_type": feature_config["value_type"],
                        "entity_keys": feature_config["entity_keys"],
                        "owner": self.FEATURE_GROUPS[group_name]["owner"],
                        "tags": feature_config["tags"],
                        "drift_threshold": feature_config["drift_threshold"],
                    }
                )

        features_df = pd.DataFrame(features_records)

        self._log(f"Seeded {len(feature_groups_df)} groups, {len(features_df)} features")
        return feature_groups_df, features_df

    def get_feature_ids(self) -> Dict[str, str]:
        """
        Get mapping of feature names to IDs.

        Returns:
            Dictionary mapping feature_name -> feature_id.
        """
        _, features_df = self.seed()
        return dict(zip(features_df["name"], features_df["id"], strict=False))
