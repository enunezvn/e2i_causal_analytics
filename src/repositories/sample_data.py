"""
Sample Data Generator - Phase 1: Data Loading Foundation

Generate realistic test data for ML pipelines:
- Match production table schemas
- Configurable sample sizes
- Reproducible via random seeds
- Support for all ML-relevant tables

Version: 1.0.0
"""

import logging
import random
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# E2I Brand and Region Enums
BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali"]
REGIONS = ["US", "EU", "APAC", "LATAM", "JP"]

# KPI Names (subset of the 46+ KPIs)
KPIS = [
    "TRx_volume",
    "NRx_volume",
    "NBRx_volume",
    "market_share",
    "patient_volume",
    "hcp_reach",
    "conversion_rate",
    "compliance_rate",
    "refill_rate",
    "time_to_first_fill",
    "abandonment_rate",
    "switch_rate",
    "voice_of_share",
    "digital_engagement",
    "rep_coverage",
]

# Agent Names
AGENT_NAMES = [
    "orchestrator",
    "causal_impact",
    "gap_analyzer",
    "drift_monitor",
    "experiment_designer",
    "prediction_synthesizer",
    "resource_optimizer",
    "explainer",
    "feedback_learner",
]


class SampleDataGenerator:
    """
    Generate realistic sample data for E2I tables.

    Supports:
    - business_metrics: KPI snapshots
    - predictions: ML predictions with confidence
    - triggers: HCP triggers
    - patient_journeys: Patient journey events
    - agent_activities: Agent analysis outputs

    Example:
        gen = SampleDataGenerator(seed=42)

        # Generate business metrics
        df = gen.business_metrics(n_samples=1000)

        # Generate predictions
        df = gen.predictions(n_samples=500)

        # Generate full dataset for ML
        datasets = gen.generate_ml_dataset(
            n_samples=1000,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
    """

    def __init__(self, seed: int = 42):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

    def _random_date(
        self,
        start: datetime,
        end: datetime,
    ) -> datetime:
        """Generate random datetime between start and end."""
        delta = end - start
        random_seconds = random.randint(0, int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    def _random_uuid(self) -> str:
        """Generate random UUID."""
        return str(uuid.uuid4())

    def business_metrics(
        self,
        n_samples: int = 1000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        brands: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Generate business_metrics sample data.

        Args:
            n_samples: Number of samples
            start_date: Start date (defaults to 1 year ago)
            end_date: End date (defaults to today)
            brands: List of brands (defaults to all)

        Returns:
            DataFrame with business metrics data
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=365)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        brands = brands or BRANDS

        data = []
        for _ in range(n_samples):
            brand = random.choice(brands)
            kpi = random.choice(KPIS)
            region = random.choice(REGIONS)

            # Generate realistic values based on KPI type
            if "rate" in kpi or "share" in kpi:
                value = np.random.beta(5, 2) * 100  # 0-100%
                target = np.random.uniform(60, 90)
            elif "volume" in kpi:
                value = np.random.exponential(scale=1000)
                target = value * np.random.uniform(0.8, 1.2)
            elif "time" in kpi:
                value = np.random.exponential(scale=30)  # days
                target = 30
            else:
                value = np.random.exponential(scale=500)
                target = value * np.random.uniform(0.9, 1.1)

            achievement_rate = value / target if target > 0 else 1.0
            roi = np.random.uniform(0.5, 5.0) if "volume" in kpi else None

            data.append(
                {
                    "metric_id": self._random_uuid(),
                    "metric_date": self._random_date(start, end).date().isoformat(),
                    "metric_name": kpi,
                    "brand": brand,
                    "region": region,
                    "value": round(value, 2),
                    "target": round(target, 2),
                    "achievement_rate": round(achievement_rate, 4),
                    "roi": round(roi, 2) if roi else None,
                    "created_at": self._random_date(start, end).isoformat(),
                }
            )

        return pd.DataFrame(data)

    def predictions(
        self,
        n_samples: int = 500,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate predictions sample data.

        Args:
            n_samples: Number of samples
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with predictions data
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=365)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        prediction_types = ["churn", "response", "conversion", "value", "risk"]
        model_versions = ["v1.0", "v1.1", "v2.0", "v2.1"]

        data = []
        for _ in range(n_samples):
            pred_type = random.choice(prediction_types)
            brand = random.choice(BRANDS)

            # Generate prediction value based on type
            if pred_type in ["churn", "response", "conversion"]:
                predicted_value = np.random.beta(2, 5)  # Most predictions low
                threshold = 0.5
            elif pred_type == "value":
                predicted_value = np.random.exponential(scale=10000)
                threshold = None
            else:  # risk
                predicted_value = np.random.beta(2, 8)
                threshold = 0.3

            confidence = np.random.beta(5, 2)  # High confidence
            rank = random.randint(1, 100)

            # Actual outcome (with some noise vs prediction)
            if threshold:
                actual = 1 if np.random.random() < predicted_value + np.random.normal(0, 0.1) else 0
            else:
                actual = predicted_value * np.random.uniform(0.7, 1.3)

            data.append(
                {
                    "prediction_id": self._random_uuid(),
                    "entity_id": self._random_uuid(),
                    "entity_type": "hcp" if random.random() > 0.3 else "patient",
                    "prediction_type": pred_type,
                    "brand": brand,
                    "predicted_value": round(predicted_value, 4),
                    "confidence": round(confidence, 4),
                    "rank": rank,
                    "threshold": threshold,
                    "actual_outcome": round(actual, 4) if isinstance(actual, float) else actual,
                    "model_version": random.choice(model_versions),
                    "prediction_date": self._random_date(start, end).isoformat(),
                    "created_at": self._random_date(start, end).isoformat(),
                }
            )

        return pd.DataFrame(data)

    def triggers(
        self,
        n_samples: int = 500,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate triggers sample data.

        Args:
            n_samples: Number of samples
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with triggers data
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=365)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        trigger_types = [
            "high_prescriber_decline",
            "competitor_switch",
            "new_patient_surge",
            "compliance_drop",
            "market_share_loss",
            "hcp_disengagement",
        ]
        severities = ["low", "medium", "high", "critical"]

        data = []
        for _ in range(n_samples):
            trigger_type = random.choice(trigger_types)
            brand = random.choice(BRANDS)
            region = random.choice(REGIONS)
            severity = random.choice(severities)

            # Generate change metrics
            baseline_value = np.random.exponential(scale=100)
            current_value = baseline_value * np.random.uniform(0.6, 1.4)
            change_pct = ((current_value - baseline_value) / baseline_value) * 100

            data.append(
                {
                    "trigger_id": self._random_uuid(),
                    "trigger_type": trigger_type,
                    "brand": brand,
                    "region": region,
                    "severity": severity,
                    "baseline_value": round(baseline_value, 2),
                    "current_value": round(current_value, 2),
                    "change_percentage": round(change_pct, 2),
                    "detected_at": self._random_date(start, end).isoformat(),
                    "resolved_at": (
                        self._random_date(start, end).isoformat() if random.random() > 0.4 else None
                    ),
                    "created_at": self._random_date(start, end).isoformat(),
                }
            )

        return pd.DataFrame(data)

    def patient_journeys(
        self,
        n_patients: int = 200,
        n_events_per_patient: int = 5,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate patient_journeys sample data.

        Args:
            n_patients: Number of unique patients
            n_events_per_patient: Average events per patient
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with patient journey data
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=365)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        event_types = [
            "diagnosis",
            "prescription",
            "refill",
            "lab_test",
            "office_visit",
            "switch",
            "discontinuation",
            "adverse_event",
        ]

        data = []
        for _ in range(n_patients):
            patient_id = self._random_uuid()
            brand = random.choice(BRANDS)
            n_events = max(1, int(np.random.poisson(n_events_per_patient)))

            # Generate events in chronological order
            patient_start = self._random_date(start, end)
            event_dates = sorted(
                [patient_start + timedelta(days=random.randint(0, 180)) for _ in range(n_events)]
            )

            for event_date in event_dates:
                event_type = random.choice(event_types)

                data.append(
                    {
                        "journey_id": self._random_uuid(),
                        "patient_id": patient_id,
                        "brand": brand,
                        "event_type": event_type,
                        "event_date": event_date.isoformat(),
                        "days_since_start": (event_date - patient_start).days,
                        "hcp_id": self._random_uuid() if event_type != "refill" else None,
                        "region": random.choice(REGIONS),
                        "created_at": event_date.isoformat(),
                    }
                )

        return pd.DataFrame(data)

    def agent_activities(
        self,
        n_samples: int = 300,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate agent_activities sample data.

        Args:
            n_samples: Number of samples
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with agent activities data
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=90)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        activity_types = ["analysis", "prediction", "recommendation", "alert"]
        statuses = ["completed", "failed", "in_progress"]

        data = []
        for _ in range(n_samples):
            agent = random.choice(AGENT_NAMES)
            brand = random.choice(BRANDS)
            activity_type = random.choice(activity_types)
            status = random.choices(statuses, weights=[0.85, 0.05, 0.10])[0]

            duration_ms = int(np.random.exponential(scale=5000))
            confidence = np.random.beta(5, 2) if status == "completed" else None

            data.append(
                {
                    "activity_id": self._random_uuid(),
                    "agent_name": agent,
                    "activity_type": activity_type,
                    "brand": brand,
                    "status": status,
                    "duration_ms": duration_ms,
                    "confidence": round(confidence, 4) if confidence else None,
                    "input_tokens": random.randint(100, 5000),
                    "output_tokens": random.randint(50, 2000),
                    "created_at": self._random_date(start, end).isoformat(),
                }
            )

        return pd.DataFrame(data)

    def causal_paths(
        self,
        n_samples: int = 200,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate causal_paths sample data.

        Args:
            n_samples: Number of samples
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with causal paths data
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=180)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        causes = [
            "rep_visit",
            "digital_engagement",
            "conference_attendance",
            "sample_delivery",
            "peer_influence",
            "market_access",
        ]
        effects = [
            "prescription_increase",
            "market_share_gain",
            "patient_volume",
            "brand_preference",
            "trial_initiation",
        ]

        data = []
        for _ in range(n_samples):
            brand = random.choice(BRANDS)
            cause = random.choice(causes)
            effect = random.choice(effects)

            # Generate causal effect metrics
            ate = np.random.normal(0.2, 0.1)  # Average treatment effect
            p_value = np.random.exponential(scale=0.1)
            confidence_interval = [ate - 0.1, ate + 0.1]

            data.append(
                {
                    "path_id": self._random_uuid(),
                    "brand": brand,
                    "cause": cause,
                    "effect": effect,
                    "average_treatment_effect": round(ate, 4),
                    "p_value": round(min(p_value, 1.0), 4),
                    "confidence_interval_lower": round(confidence_interval[0], 4),
                    "confidence_interval_upper": round(confidence_interval[1], 4),
                    "sample_size": random.randint(100, 5000),
                    "method": random.choice(["dowhy", "econml", "causalml"]),
                    "created_at": self._random_date(start, end).isoformat(),
                }
            )

        return pd.DataFrame(data)

    def ml_patients(
        self,
        n_patients: int = 200,
        target_rate: float = 0.3,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate ML-ready patient data with classification target.

        This method produces patient-level features suitable for
        binary classification (e.g., discontinuation prediction).
        Output matches PatientJourneysSchema for Pandera validation.

        Args:
            n_patients: Number of patients
            target_rate: Approximate rate of positive class (discontinuation)
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with patient-level features and discontinuation_flag
        """
        if start_date:
            start = datetime.fromisoformat(start_date)
        else:
            start = datetime.now() - timedelta(days=365)

        if end_date:
            end = datetime.fromisoformat(end_date)
        else:
            end = datetime.now()

        # Valid journey statuses per Pandera schema
        # E2I_JOURNEY_STATUSES = ["active", "stable", "transitioning", "completed"]

        # Valid regions per Pandera schema (lowercase)
        # E2I_REGIONS = ["northeast", "south", "midwest", "west"]
        valid_regions = ["northeast", "south", "midwest", "west"]

        data = []
        for _ in range(n_patients):
            patient_journey_id = self._random_uuid()
            patient_id = self._random_uuid()  # Required by schema
            brand = random.choice(BRANDS)
            geographic_region = random.choice(valid_regions)  # Renamed per schema

            # Generate patient features
            days_on_therapy = np.random.randint(30, 365)
            hcp_visits = np.random.randint(1, 20)
            prior_treatments = np.random.randint(0, 5)
            age_group = random.choice(["<50", "50-65", ">65"])
            data_quality_score = np.random.uniform(0.5, 1.0)

            # Generate discontinuation flag with correlation to features
            # Higher risk with: fewer hcp_visits, more prior treatments, shorter therapy
            risk_score = (
                0.3  # Base rate
                - 0.01 * hcp_visits  # More visits = lower risk
                + 0.05 * prior_treatments  # More prior treatments = higher risk
                - 0.001 * days_on_therapy  # Longer therapy = lower risk
            )
            risk_score = max(0.05, min(0.95, risk_score + np.random.normal(0, 0.1)))
            discontinuation_flag = 1 if np.random.random() < risk_score else 0

            # Map discontinuation to valid journey status
            # completed = successfully finished, stable = ongoing well
            if discontinuation_flag:
                journey_status = "transitioning"  # About to discontinue
            else:
                journey_status = random.choice(["active", "stable", "completed"])

            journey_start = self._random_date(start, end)
            journey_end = (
                journey_start + timedelta(days=days_on_therapy)
                if journey_status == "completed"
                else None
            )

            data.append(
                {
                    "patient_journey_id": patient_journey_id,
                    "patient_id": patient_id,  # Required by schema
                    "brand": brand,
                    "geographic_region": geographic_region,  # Renamed per schema
                    "journey_status": journey_status,
                    "journey_start_date": journey_start.isoformat(),
                    "journey_end_date": journey_end.isoformat() if journey_end else None,
                    "data_quality_score": round(data_quality_score, 3),
                    "days_on_therapy": days_on_therapy,
                    "hcp_visits": hcp_visits,
                    "prior_treatments": prior_treatments,
                    "age_group": age_group,
                    "discontinuation_flag": discontinuation_flag,
                    "created_at": journey_start.isoformat(),
                }
            )

        return pd.DataFrame(data)

    def generate_ml_dataset(
        self,
        table: str = "business_metrics",
        n_samples: int = 1000,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        test_ratio: float = 0.2,
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate a complete ML dataset with splits.

        Args:
            table: Table to generate (business_metrics, predictions, etc.)
            n_samples: Total samples
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio

        Returns:
            Dict with train, val, test DataFrames
        """
        # Generate data based on table
        if table == "business_metrics":
            df = self.business_metrics(n_samples)
        elif table == "predictions":
            df = self.predictions(n_samples)
        elif table == "triggers":
            df = self.triggers(n_samples)
        elif table == "patient_journeys":
            df = self.patient_journeys(n_samples // 5, 5)
        elif table == "agent_activities":
            df = self.agent_activities(n_samples)
        elif table == "causal_paths":
            df = self.causal_paths(n_samples)
        else:
            raise ValueError(f"Unknown table: {table}")

        # Shuffle
        df = df.sample(frac=1, random_state=self.seed).reset_index(drop=True)

        # Split
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        return {
            "train": df.iloc[:train_end].reset_index(drop=True),
            "val": df.iloc[train_end:val_end].reset_index(drop=True),
            "test": df.iloc[val_end:].reset_index(drop=True),
        }


# Convenience function
def get_sample_generator(seed: int = 42) -> SampleDataGenerator:
    """Get a SampleDataGenerator instance."""
    return SampleDataGenerator(seed=seed)
