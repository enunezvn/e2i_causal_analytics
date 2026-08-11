"""
Sample Data Generator - Phase 1: Data Loading Foundation

Generate realistic test data for ML pipelines:
- Match production table schemas
- Configurable sample sizes
- Reproducible via random seeds (per-purpose instance streams — dates,
  categoricals, and numerics draw independently, so editing one field's
  draw pattern cannot shift another's; see SampleDataGenerator.__init__)
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

from src.services.enum_labels import REGION_ENUM_LABELS

logger = logging.getLogger(__name__)


# E2I Brand and Region Enums
BRANDS = ["Remibrutinib", "Fabhalta", "Kisqali"]
# region_type labels from the shared enum-label module (#1517). This list
# previously drifted to ["US", "EU", "APAC", "LATAM", "JP"] — values the
# production region_type enum can never accept and BusinessMetricsSchema
# (pandera) rejects, contradicting this module's "match production table
# schemas" purpose. Sourced (not hand-copied) so it cannot drift again.
REGIONS = list(REGION_ENUM_LABELS)

# KPI Names (subset of the defined KPIs)
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

        Draws come from per-purpose instance streams instead of the
        process-global RNGs (#1542): with a single shared stream, resizing
        any categorical constant changed how ``random.choice`` consumed it
        and shifted every subsequent date draw (#1521 -> #1524, an emptied
        temporal-split window). Dates, categoricals, and numerics therefore
        draw independently, and constructing a generator no longer reseeds
        ``random`` / ``np.random`` for the whole process. ``RandomState`` is
        used (not ``default_rng``) because its frozen bitstream matches the
        legacy global draws, preserving the realised regime statistics.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        self._rng_dates = random.Random(f"{seed}:dates")
        self._rng_categorical = random.Random(f"{seed}:categorical")
        self._rng_numeric = random.Random(f"{seed}:numeric")
        self._np_rng = np.random.RandomState(seed)

    def _random_date(
        self,
        start: datetime,
        end: datetime,
    ) -> datetime:
        """Generate random datetime between start and end."""
        delta = end - start
        random_seconds = self._rng_dates.randint(0, int(delta.total_seconds()))
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
            brand = self._rng_categorical.choice(brands)
            kpi = self._rng_categorical.choice(KPIS)
            region = self._rng_categorical.choice(REGIONS)

            # Generate realistic values based on KPI type
            if "rate" in kpi or "share" in kpi:
                value = self._np_rng.beta(5, 2) * 100  # 0-100%
                target = self._np_rng.uniform(60, 90)
            elif "volume" in kpi:
                value = self._np_rng.exponential(scale=1000)
                target = value * self._np_rng.uniform(0.8, 1.2)
            elif "time" in kpi:
                value = self._np_rng.exponential(scale=30)  # days
                target = 30
            else:
                value = self._np_rng.exponential(scale=500)
                target = value * self._np_rng.uniform(0.9, 1.1)

            achievement_rate = value / target if target > 0 else 1.0
            roi = self._np_rng.uniform(0.5, 5.0) if "volume" in kpi else None

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
            pred_type = self._rng_categorical.choice(prediction_types)
            brand = self._rng_categorical.choice(BRANDS)

            # Generate prediction value based on type
            if pred_type in ["churn", "response", "conversion"]:
                predicted_value = self._np_rng.beta(2, 5)  # Most predictions low
                threshold = 0.5
            elif pred_type == "value":
                predicted_value = self._np_rng.exponential(scale=10000)
                threshold = None
            else:  # risk
                predicted_value = self._np_rng.beta(2, 8)
                threshold = 0.3

            confidence = self._np_rng.beta(5, 2)  # High confidence
            rank = self._rng_numeric.randint(1, 100)

            # Actual outcome (with some noise vs prediction)
            actual: float
            if threshold:
                actual = (
                    1.0
                    if self._np_rng.random() < predicted_value + self._np_rng.normal(0, 0.1)
                    else 0.0
                )
            else:
                actual = predicted_value * self._np_rng.uniform(0.7, 1.3)

            data.append(
                {
                    "prediction_id": self._random_uuid(),
                    "entity_id": self._random_uuid(),
                    "entity_type": "hcp" if self._rng_categorical.random() > 0.3 else "patient",
                    "prediction_type": pred_type,
                    "brand": brand,
                    "predicted_value": round(predicted_value, 4),
                    "confidence": round(confidence, 4),
                    "rank": rank,
                    "threshold": threshold,
                    "actual_outcome": round(actual, 4) if isinstance(actual, float) else actual,
                    "model_version": self._rng_categorical.choice(model_versions),
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
            trigger_type = self._rng_categorical.choice(trigger_types)
            brand = self._rng_categorical.choice(BRANDS)
            region = self._rng_categorical.choice(REGIONS)
            severity = self._rng_categorical.choice(severities)

            # Generate change metrics
            baseline_value = self._np_rng.exponential(scale=100)
            current_value = baseline_value * self._np_rng.uniform(0.6, 1.4)
            change_pct = ((current_value - baseline_value) / baseline_value) * 100

            # Drawn unconditionally so the date stream consumes a fixed
            # number of draws per row — a categorical shift flipping the
            # resolved gate must not move later rows' dates (#1542).
            resolved_date = self._random_date(start, end)

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
                        resolved_date.isoformat() if self._rng_categorical.random() > 0.4 else None
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
            brand = self._rng_categorical.choice(BRANDS)
            n_events = max(1, int(self._np_rng.poisson(n_events_per_patient)))

            # Generate events in chronological order (event-day offsets are
            # date-shaped draws, so they live on the dates stream)
            patient_start = self._random_date(start, end)
            event_dates = sorted(
                [
                    patient_start + timedelta(days=self._rng_dates.randint(0, 180))
                    for _ in range(n_events)
                ]
            )

            for event_date in event_dates:
                event_type = self._rng_categorical.choice(event_types)

                data.append(
                    {
                        "journey_id": self._random_uuid(),
                        "patient_id": patient_id,
                        "brand": brand,
                        "event_type": event_type,
                        "event_date": event_date.isoformat(),
                        "days_since_start": (event_date - patient_start).days,
                        "hcp_id": self._random_uuid() if event_type != "refill" else None,
                        "region": self._rng_categorical.choice(REGIONS),
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
            agent = self._rng_categorical.choice(AGENT_NAMES)
            brand = self._rng_categorical.choice(BRANDS)
            activity_type = self._rng_categorical.choice(activity_types)
            status = self._rng_categorical.choices(statuses, weights=[0.85, 0.05, 0.10])[0]

            duration_ms = int(self._np_rng.exponential(scale=5000))
            confidence = self._np_rng.beta(5, 2) if status == "completed" else None

            data.append(
                {
                    "activity_id": self._random_uuid(),
                    "agent_name": agent,
                    "activity_type": activity_type,
                    "brand": brand,
                    "status": status,
                    "duration_ms": duration_ms,
                    "confidence": round(confidence, 4) if confidence else None,
                    "input_tokens": self._rng_numeric.randint(100, 5000),
                    "output_tokens": self._rng_numeric.randint(50, 2000),
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
            brand = self._rng_categorical.choice(BRANDS)
            cause = self._rng_categorical.choice(causes)
            effect = self._rng_categorical.choice(effects)

            # Generate causal effect metrics
            ate = self._np_rng.normal(0.2, 0.1)  # Average treatment effect
            p_value = self._np_rng.exponential(scale=0.1)
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
                    "sample_size": self._rng_numeric.randint(100, 5000),
                    "method": self._rng_categorical.choice(["dowhy", "econml", "causalml"]),
                    "created_at": self._random_date(start, end).isoformat(),
                }
            )

        return pd.DataFrame(data)

    def ml_patients(
        self,
        n_patients: int = 1500,
        positive_rate: float = 0.30,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        *,
        signal_strength: float = 1.0,
        noise_sd: float = 0.10,
        signalize_extra_features: bool = False,
    ) -> pd.DataFrame:
        """
        Generate ML-ready patient data with classification target.

        This method produces patient-level features suitable for
        binary classification (e.g., discontinuation prediction).
        Output matches PatientJourneysSchema for Pandera validation.

        Args:
            n_patients: Number of patients
            positive_rate: Base rate of the positive class (discontinuation).
                Drives the constant term of the risk_score generator. Use
                ``0.30`` for the default balanced regime; pass ``0.02`` for
                an adverse / extreme-imbalance regime that exercises the
                pipeline's class-imbalance remediation paths. The realised
                positive rate will diverge slightly from this value because
                it is combined with feature-driven adjustments and clipped
                to ``[0.05, 0.95]`` before Bernoulli sampling.
            start_date: Start date
            end_date: End date
            signal_strength: Multiplier on the deterministic feature
                contributions to ``risk_score``. ``1.0`` is the historical
                default (preserves the ``default``/``adverse`` regime
                behavior). Higher values widen the gap between high- and
                low-risk patients and raise the achievable val AUC. Used by
                the ``clean`` regime (Section A of pre_phase2_unblockers).
            noise_sd: Standard deviation of the Gaussian noise added to
                ``risk_score`` (further scaled by ``max(scale, 0.05)``).
                Default ``0.10`` matches the historical generator. Lower
                values raise SNR; higher values regularize the train→val
                gap.
            signalize_extra_features: When True, four additional features
                contribute to ``risk_score``: ``age_group``,
                ``geographic_region``, ``brand``, and ``data_quality_score``.
                Coefficients chosen to be similar in magnitude to the
                three existing signal features so no single feature
                dominates SHAP rankings. Default False preserves the
                original generator behavior (only ``hcp_visits``,
                ``prior_treatments``, ``days_on_therapy`` carry signal).

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

        # Valid regions per Pandera schema — the shared region_type labels
        # (#1517: sourced from enum_labels, not a second hand copy).
        valid_regions = list(REGION_ENUM_LABELS)

        data = []
        for _ in range(n_patients):
            patient_journey_id = self._random_uuid()
            patient_id = self._random_uuid()  # Required by schema
            brand = self._rng_categorical.choice(BRANDS)
            geographic_region = self._rng_categorical.choice(valid_regions)  # Renamed per schema

            # Generate patient features
            days_on_therapy = self._np_rng.randint(30, 365)
            hcp_visits = self._np_rng.randint(1, 20)
            prior_treatments = self._np_rng.randint(0, 5)
            age_group = self._rng_categorical.choice(["<50", "50-65", ">65"])
            data_quality_score = self._np_rng.uniform(0.5, 1.0)

            # Generate discontinuation flag with correlation to features
            # Higher risk with: fewer hcp_visits, more prior treatments, shorter therapy
            #
            # The feature-driven adjustments and the noise floor are scaled by
            # `positive_rate / 0.30` so that when callers pass a low base rate
            # (e.g. 0.02 for the adverse regime) the clipping floor and noise do
            # NOT dominate the final positive class share.
            #
            # When ``signalize_extra_features=True`` (clean regime — Section A
            # of pre_phase2_unblockers) four additional features contribute,
            # so the SHAP feature ranking surface is non-degenerate without
            # any one feature dominating. ``signal_strength`` uniformly
            # multiplies the deterministic component for tuning headroom.
            scale = max(positive_rate / 0.30, 0.0)

            extra_signal = 0.0
            if signalize_extra_features:
                # Coefficients tuned so the per-feature contribution span
                # matches the existing 3 features (~±0.10 across each
                # feature's range). data_quality_score has a narrow
                # uniform [0.5, 1.0] support, so its coefficient (-0.40)
                # is larger in absolute value than the binary indicators
                # (+0.10 / -0.05 / +0.08) — what matters for SHAP balance
                # is the *contribution* magnitude, not the raw coefficient.
                extra_signal = (
                    0.10 * (1 if age_group == ">65" else 0)
                    - 0.05 * (1 if geographic_region == "west" else 0)
                    + 0.08 * (1 if brand == "Kisqali" else 0)
                    - 0.40 * (data_quality_score - 0.75)
                )

            risk_score = (
                positive_rate  # Base rate (regime-controlled)
                + scale
                * signal_strength
                * (
                    -0.01 * hcp_visits  # More visits = lower risk
                    + 0.05 * prior_treatments  # More prior treatments = higher risk
                    - 0.001 * days_on_therapy  # Longer therapy = lower risk
                    + extra_signal
                )
            )
            noise = self._np_rng.normal(0, noise_sd * max(scale, 0.05))
            min_floor = min(0.05, max(positive_rate * 0.5, 0.001))
            risk_score = max(min_floor, min(0.95, risk_score + noise))
            discontinuation_flag = 1 if self._np_rng.random() < risk_score else 0

            # Map discontinuation to valid journey status
            # completed = successfully finished, stable = ongoing well
            if discontinuation_flag:
                journey_status = "transitioning"  # About to discontinue
            else:
                journey_status = self._rng_categorical.choice(["active", "stable", "completed"])

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
