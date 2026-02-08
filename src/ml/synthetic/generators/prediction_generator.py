"""
ML Prediction Generator.

Generates synthetic ML predictions for patient journeys.
"""

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config import Brand
from .base import BaseGenerator, GeneratorConfig


class PredictionGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for ML predictions.

    Generates prediction records with:
    - Model predictions (propensity, risk, value)
    - Confidence scores and uncertainty
    - Feature importance summaries
    """

    # Prediction types with typical value ranges
    # Note: Keys MUST match Supabase prediction_type ENUM:
    # {trigger, propensity, risk, churn, next_best_action}
    PREDICTION_TYPES = {
        "propensity": {"min": 0.0, "max": 1.0, "mean": 0.35, "std": 0.20},
        "risk": {"min": 0.0, "max": 1.0, "mean": 0.65, "std": 0.18},
        "trigger": {"min": 0.0, "max": 1.0, "mean": 0.40, "std": 0.22},
        "churn": {"min": 0.0, "max": 1.0, "mean": 0.20, "std": 0.15},
        "next_best_action": {"min": 0.0, "max": 1.0, "mean": 0.50, "std": 0.25},
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "ml_predictions"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        patient_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the prediction generator.

        Args:
            config: Generator configuration.
            patient_df: Patient DataFrame for foreign key integrity.
        """
        super().__init__(config)
        self.patient_df = patient_df

    def generate(self) -> pd.DataFrame:
        """
        Generate ML predictions.

        Returns:
            DataFrame with ML predictions matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} ML predictions...")

        if self.patient_df is not None:
            # Generate predictions for patients
            predictions_per_patient = max(1, n // len(self.patient_df))
            records = []

            for _, patient in self.patient_df.iterrows():
                n_preds = self._rng.integers(1, predictions_per_patient + 2)
                for _ in range(n_preds):
                    record = self._generate_prediction_record(patient)
                    records.append(record)

            df = pd.DataFrame(records)
        else:
            df = self._generate_standalone_predictions(n)

        # Add IDs
        df["prediction_id"] = self._generate_ids("pred", len(df))

        # Assign splits based on prediction dates
        if "prediction_date" in df.columns:
            df["data_split"] = self._assign_splits(df["prediction_date"].tolist())

        self._log(f"Generated {len(df)} ML predictions")
        return df

    def _generate_prediction_record(self, patient: pd.Series) -> Dict:
        """Generate a prediction record linked to patient."""
        # Select prediction type
        prediction_type = self._rng.choice(list(self.PREDICTION_TYPES.keys()))
        config = self.PREDICTION_TYPES[prediction_type]

        # Generate prediction value based on patient features
        base_value = self._rng.normal(config["mean"], config["std"])

        # Adjust based on patient engagement and severity
        engagement_score = patient.get("engagement_score", 5.0)
        patient.get("disease_severity", 5.0)

        # Higher engagement improves positive predictions
        if prediction_type in ["propensity", "risk", "trigger"]:
            adjustment = (engagement_score - 5) * 0.05
            base_value += adjustment * config["std"]
        elif prediction_type == "churn":
            # Lower engagement increases churn risk
            adjustment = (5 - engagement_score) * 0.05
            base_value += adjustment * config["std"]

        # Clip to valid range
        prediction_value = np.clip(base_value, config["min"], config["max"])

        # Generate confidence and uncertainty
        confidence = self._rng.uniform(0.6, 0.95)
        uncertainty = 1 - confidence

        # Model version
        model_version = f"v{self._rng.integers(1, 5)}.{self._rng.integers(0, 10)}"

        # Prediction date (after journey start)
        journey_start = pd.to_datetime(patient.get("journey_start_date", "2023-01-01"))
        days_offset = self._rng.integers(0, 90)
        prediction_date = journey_start + pd.Timedelta(days=int(days_offset))

        return {
            "patient_journey_id": patient.get("patient_journey_id", ""),
            "patient_id": patient.get("patient_id", ""),
            "hcp_id": patient.get("hcp_id", ""),
            "brand": patient.get("brand", Brand.REMIBRUTINIB.value),
            "prediction_type": prediction_type,
            "prediction_value": round(prediction_value, 4),
            "confidence_score": round(confidence, 3),
            "uncertainty": round(uncertainty, 3),
            "model_version": model_version,
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        }

    def _generate_standalone_predictions(self, n: int) -> pd.DataFrame:
        """Generate predictions without patient linkage."""
        patient_ids = self._generate_ids("pt", n, width=6)
        journey_ids = self._generate_ids("patient", n, width=6)
        hcp_ids = self._generate_ids("hcp", max(100, n // 10))

        # Generate prediction types
        prediction_types = self._random_choice(
            list(self.PREDICTION_TYPES.keys()),
            n,
        ).tolist()

        # Generate values for each type
        prediction_values = []
        for pred_type in prediction_types:
            config = self.PREDICTION_TYPES[pred_type]
            value = self._rng.normal(config["mean"], config["std"])
            value = np.clip(value, config["min"], config["max"])
            prediction_values.append(round(value, 4))

        # Confidence and uncertainty
        confidences = self._random_float(0.6, 0.95, n)
        uncertainties = 1 - confidences

        # Model versions
        model_versions = [
            f"v{self._rng.integers(1, 5)}.{self._rng.integers(0, 10)}" for _ in range(n)
        ]

        # Dates and brands
        prediction_dates = self._random_dates(n)
        if self.config.brand:
            brands = [self.config.brand.value] * n
        else:
            brands = self._random_choice([b.value for b in Brand], n).tolist()

        return pd.DataFrame(
            {
                "patient_journey_id": journey_ids,
                "patient_id": patient_ids,
                "hcp_id": self._random_choice(hcp_ids, n).tolist(),
                "brand": brands,
                "prediction_type": prediction_types,
                "prediction_value": prediction_values,
                "confidence_score": np.round(confidences, 3),
                "uncertainty": np.round(uncertainties, 3),
                "model_version": model_versions,
                "prediction_date": prediction_dates,
            }
        )
