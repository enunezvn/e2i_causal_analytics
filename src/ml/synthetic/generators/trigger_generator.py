"""
Trigger Generator.

Generates synthetic triggers for patient/HCP targeting actions.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .base import BaseGenerator, GeneratorConfig
from ..config import Brand


class TriggerGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for triggers.

    Generates trigger records with:
    - Priority levels (critical, high, medium, low)
    - Delivery channels and status
    - Causal chain and supporting evidence
    - Outcome tracking
    """

    # Trigger types based on agent actions
    TRIGGER_TYPES = [
        "prescription_opportunity",
        "adherence_risk",
        "churn_prevention",
        "cross_sell",
        "engagement_gap",
        "competitive_threat",
        "treatment_switch",
        "reactivation",
    ]

    # Priority distribution (weighted toward actionable)
    PRIORITY_DIST = {
        "critical": 0.10,
        "high": 0.30,
        "medium": 0.40,
        "low": 0.20,
    }

    # Delivery channels
    DELIVERY_CHANNELS = ["email", "crm", "mobile", "portal", "rep_alert"]

    # Delivery status values
    DELIVERY_STATUS_VALUES = ["pending", "delivered", "viewed", "failed"]

    # Acceptance status values
    ACCEPTANCE_STATUS_VALUES = ["pending", "accepted", "rejected", "expired"]

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "triggers"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        patient_df: Optional[pd.DataFrame] = None,
        hcp_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the trigger generator.

        Args:
            config: Generator configuration.
            patient_df: Patient DataFrame for foreign key integrity.
            hcp_df: HCP DataFrame for foreign key integrity.
        """
        super().__init__(config)
        self.patient_df = patient_df
        self.hcp_df = hcp_df

    def generate(self) -> pd.DataFrame:
        """
        Generate triggers.

        Returns:
            DataFrame with triggers matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} triggers...")

        if self.patient_df is not None and self.hcp_df is not None:
            # Generate triggers linked to patients and HCPs
            records = []
            triggers_per_patient = max(1, n // len(self.patient_df))

            for _, patient in self.patient_df.iterrows():
                n_triggers = self._rng.integers(1, triggers_per_patient + 2)
                for _ in range(n_triggers):
                    record = self._generate_trigger_record(patient)
                    records.append(record)

            df = pd.DataFrame(records)
        else:
            df = self._generate_standalone_triggers(n)

        # Add IDs
        df["trigger_id"] = self._generate_ids("trg", len(df))

        # Assign splits based on trigger timestamps
        if "trigger_timestamp" in df.columns:
            df["data_split"] = self._assign_splits(df["trigger_timestamp"].tolist())

        self._log(f"Generated {len(df)} triggers")
        return df

    def _generate_trigger_record(self, patient: pd.Series) -> Dict:
        """Generate a trigger record linked to patient."""
        # Select trigger type based on patient state
        engagement_score = patient.get("engagement_score", 5.0)
        treatment_initiated = patient.get("treatment_initiated", 0)

        trigger_type = self._select_trigger_type(engagement_score, treatment_initiated)

        # Priority based on engagement and treatment status
        priority = self._select_priority(engagement_score, treatment_initiated)

        # Confidence score (higher for clear-cut cases)
        confidence = self._calculate_confidence(engagement_score, trigger_type)

        # Timestamps
        journey_start = pd.to_datetime(patient.get("journey_start_date", "2023-01-01"))
        days_offset = self._rng.integers(7, 90)
        trigger_timestamp = journey_start + pd.Timedelta(days=int(days_offset))

        # Lead time and expiration
        lead_time_days = self._rng.integers(3, 30)
        expiration_date = trigger_timestamp + pd.Timedelta(days=int(lead_time_days))

        # Delivery information
        delivery_channel = self._rng.choice(self.DELIVERY_CHANNELS)
        delivery_status = self._rng.choice(
            self.DELIVERY_STATUS_VALUES,
            p=[0.10, 0.60, 0.25, 0.05],
        )

        # Acceptance (only if delivered/viewed)
        if delivery_status in ["delivered", "viewed"]:
            acceptance_status = self._rng.choice(
                self.ACCEPTANCE_STATUS_VALUES,
                p=[0.15, 0.50, 0.20, 0.15],
            )
        else:
            acceptance_status = "pending"

        # Outcome tracking (some triggers have measured outcomes)
        outcome_tracked = self._rng.random() < 0.40
        outcome_value = None
        if outcome_tracked and acceptance_status == "accepted":
            # Positive outcome more likely with high engagement
            outcome_value = round(
                self._rng.beta(2 + engagement_score / 5, 3) * 1.0, 3
            )

        # Generate causal chain and evidence
        causal_chain = self._generate_causal_chain(trigger_type, engagement_score)
        supporting_evidence = self._generate_supporting_evidence(trigger_type)

        return {
            "patient_id": patient.get("patient_id", ""),
            "hcp_id": patient.get("hcp_id", ""),
            "trigger_timestamp": trigger_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "trigger_type": trigger_type,
            "priority": priority,
            "confidence_score": round(confidence, 3),
            "lead_time_days": lead_time_days,
            "expiration_date": expiration_date.strftime("%Y-%m-%d"),
            "delivery_channel": delivery_channel,
            "delivery_status": delivery_status,
            "acceptance_status": acceptance_status,
            "outcome_tracked": outcome_tracked,
            "outcome_value": outcome_value,
            "trigger_reason": self._generate_trigger_reason(trigger_type),
            "causal_chain": causal_chain,
            "supporting_evidence": supporting_evidence,
            "recommended_action": self._generate_recommended_action(trigger_type),
            "brand": patient.get("brand", Brand.REMIBRUTINIB.value),
        }

    def _select_trigger_type(
        self,
        engagement_score: float,
        treatment_initiated: int,
    ) -> str:
        """Select trigger type based on patient state."""
        if treatment_initiated == 1:
            # Patient already on treatment - focus on adherence/retention
            probs = {
                "adherence_risk": 0.35,
                "churn_prevention": 0.25,
                "cross_sell": 0.15,
                "treatment_switch": 0.10,
                "engagement_gap": 0.10,
                "reactivation": 0.05,
            }
        else:
            # Not yet on treatment - focus on acquisition
            probs = {
                "prescription_opportunity": 0.40,
                "engagement_gap": 0.25,
                "competitive_threat": 0.15,
                "cross_sell": 0.10,
                "reactivation": 0.10,
            }

        # Adjust for engagement
        if engagement_score < 4:
            # Low engagement → more gap/reactivation triggers
            if "engagement_gap" in probs:
                probs["engagement_gap"] *= 1.5
            if "reactivation" in probs:
                probs["reactivation"] *= 1.5

        # Normalize
        options = list(probs.keys())
        weights = np.array(list(probs.values()))
        weights = weights / weights.sum()

        return self._rng.choice(options, p=weights)

    def _select_priority(
        self,
        engagement_score: float,
        treatment_initiated: int,
    ) -> str:
        """Select priority based on patient state."""
        # Base distribution
        probs = list(self.PRIORITY_DIST.values())

        # High-value patients (low engagement + not initiated) get higher priority
        if engagement_score < 4 and treatment_initiated == 0:
            probs = [0.20, 0.40, 0.30, 0.10]  # Shift toward higher priority
        elif engagement_score > 7 and treatment_initiated == 1:
            probs = [0.05, 0.20, 0.45, 0.30]  # Lower priority (already engaged)

        return self._rng.choice(list(self.PRIORITY_DIST.keys()), p=probs)

    def _calculate_confidence(
        self,
        engagement_score: float,
        trigger_type: str,
    ) -> float:
        """Calculate confidence score for trigger."""
        # Base confidence
        base_confidence = 0.70

        # Clearer signal for certain trigger types
        type_adjustment = {
            "adherence_risk": 0.10,
            "prescription_opportunity": 0.08,
            "churn_prevention": 0.05,
            "engagement_gap": 0.03,
        }.get(trigger_type, 0.0)

        # Engagement extremes are clearer signals
        engagement_factor = abs(engagement_score - 5) / 10.0 * 0.15

        confidence = base_confidence + type_adjustment + engagement_factor
        noise = self._rng.normal(0, 0.05)

        return np.clip(confidence + noise, 0.50, 0.99)

    def _generate_causal_chain(
        self,
        trigger_type: str,
        engagement_score: float,
    ) -> Dict:
        """Generate causal chain JSON."""
        chains = {
            "prescription_opportunity": {
                "root_cause": "high_prescriber_fit",
                "intermediate_factors": ["engagement_pattern", "treatment_gap"],
                "confidence": round(0.7 + engagement_score * 0.02, 2),
            },
            "adherence_risk": {
                "root_cause": "declining_engagement",
                "intermediate_factors": ["refill_pattern", "support_interaction"],
                "confidence": round(0.65 + (10 - engagement_score) * 0.03, 2),
            },
            "churn_prevention": {
                "root_cause": "competitor_activity",
                "intermediate_factors": ["price_sensitivity", "satisfaction_score"],
                "confidence": round(0.60 + self._rng.random() * 0.2, 2),
            },
        }
        return chains.get(trigger_type, {"root_cause": "model_prediction", "confidence": 0.70})

    def _generate_supporting_evidence(self, trigger_type: str) -> Dict:
        """Generate supporting evidence JSON."""
        return {
            "data_sources": ["claims_data", "engagement_logs", "prescription_history"],
            "model_version": f"v{self._rng.integers(1, 4)}.{self._rng.integers(0, 10)}",
            "feature_importance": {
                "engagement_recency": round(self._rng.uniform(0.1, 0.4), 2),
                "prescription_history": round(self._rng.uniform(0.1, 0.3), 2),
                "hcp_relationship": round(self._rng.uniform(0.05, 0.2), 2),
            },
        }

    def _generate_trigger_reason(self, trigger_type: str) -> str:
        """Generate human-readable trigger reason."""
        reasons = {
            "prescription_opportunity": "High prescriber fit score with treatment gap identified",
            "adherence_risk": "Declining engagement pattern suggests potential non-adherence",
            "churn_prevention": "Competitor activity detected in territory",
            "cross_sell": "Patient profile matches additional indication criteria",
            "engagement_gap": "Below-average engagement compared to similar HCPs",
            "competitive_threat": "Recent competitive detailing detected",
            "treatment_switch": "Patient may benefit from therapy adjustment",
            "reactivation": "Lapsed patient with high historical value",
        }
        return reasons.get(trigger_type, "Model-generated recommendation")

    def _generate_recommended_action(self, trigger_type: str) -> str:
        """Generate recommended action text."""
        actions = {
            "prescription_opportunity": "Schedule detail visit to discuss treatment benefits",
            "adherence_risk": "Initiate patient support program outreach",
            "churn_prevention": "Deploy competitive positioning materials",
            "cross_sell": "Present expanded indication data",
            "engagement_gap": "Increase touchpoint frequency",
            "competitive_threat": "Prioritize relationship-building activities",
            "treatment_switch": "Discuss alternative treatment options with HCP",
            "reactivation": "Re-engage with updated clinical evidence",
        }
        return actions.get(trigger_type, "Follow up with appropriate engagement")

    def _generate_standalone_triggers(self, n: int) -> pd.DataFrame:
        """Generate triggers without patient/HCP linkage."""
        patient_ids = self._generate_ids("pt", n, width=6)
        hcp_ids = self._generate_ids("hcp", max(100, n // 10))

        # Generate engagement scores for trigger selection
        engagement_scores = self._random_normal(5.0, 2.0, n, clip_min=0, clip_max=10)
        treatment_initiated = (engagement_scores > 5).astype(int)

        # Generate trigger types
        trigger_types = [
            self._select_trigger_type(eng, treat)
            for eng, treat in zip(engagement_scores, treatment_initiated)
        ]

        # Generate priorities
        priorities = [
            self._select_priority(eng, treat)
            for eng, treat in zip(engagement_scores, treatment_initiated)
        ]

        # Confidence scores
        confidences = [
            self._calculate_confidence(eng, ttype)
            for eng, ttype in zip(engagement_scores, trigger_types)
        ]

        # Timestamps and dates
        trigger_timestamps = self._random_dates(n)
        lead_times = self._rng.integers(3, 30, size=n)
        expiration_dates = [
            (pd.to_datetime(ts) + pd.Timedelta(days=int(lt))).strftime("%Y-%m-%d")
            for ts, lt in zip(trigger_timestamps, lead_times)
        ]

        # Delivery information
        delivery_channels = self._random_choice(self.DELIVERY_CHANNELS, n).tolist()
        delivery_statuses = self._random_choice(
            self.DELIVERY_STATUS_VALUES,
            n,
            p=[0.10, 0.60, 0.25, 0.05],
        ).tolist()

        # Acceptance statuses
        acceptance_statuses = []
        for ds in delivery_statuses:
            if ds in ["delivered", "viewed"]:
                acceptance_statuses.append(
                    self._rng.choice(self.ACCEPTANCE_STATUS_VALUES, p=[0.15, 0.50, 0.20, 0.15])
                )
            else:
                acceptance_statuses.append("pending")

        # Outcome tracking
        outcome_tracked = self._rng.random(n) < 0.40
        outcome_values = [
            round(self._rng.beta(3, 3) * 1.0, 3) if tracked and acc == "accepted" else None
            for tracked, acc in zip(outcome_tracked, acceptance_statuses)
        ]

        # Brands
        if self.config.brand:
            brands = [self.config.brand.value] * n
        else:
            brands = self._random_choice([b.value for b in Brand], n)

        return pd.DataFrame({
            "patient_id": patient_ids,
            "hcp_id": self._random_choice(hcp_ids, n).tolist(),
            "trigger_timestamp": trigger_timestamps,
            "trigger_type": trigger_types,
            "priority": priorities,
            "confidence_score": np.round(confidences, 3),
            "lead_time_days": lead_times,
            "expiration_date": expiration_dates,
            "delivery_channel": delivery_channels,
            "delivery_status": delivery_statuses,
            "acceptance_status": acceptance_statuses,
            "outcome_tracked": outcome_tracked,
            "outcome_value": outcome_values,
            "trigger_reason": [self._generate_trigger_reason(tt) for tt in trigger_types],
            "causal_chain": [
                self._generate_causal_chain(tt, eng)
                for tt, eng in zip(trigger_types, engagement_scores)
            ],
            "supporting_evidence": [self._generate_supporting_evidence(tt) for tt in trigger_types],
            "recommended_action": [self._generate_recommended_action(tt) for tt in trigger_types],
            "brand": brands,
        })
