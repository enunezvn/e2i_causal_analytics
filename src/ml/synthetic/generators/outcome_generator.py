"""
Business Outcome Generator.

Generates synthetic business outcomes linked to patient journeys.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.special import expit

from .base import BaseGenerator, GeneratorConfig
from ..config import Brand, DGPType, DGP_CONFIGS


class OutcomeGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for business outcomes.

    Generates outcomes with:
    - Links to patient journeys
    - Causal relationship with treatment/engagement
    - Business KPIs (conversion, revenue, etc.)
    """

    # Outcome types
    OUTCOME_TYPES = [
        "prescription_written",
        "prescription_filled",
        "treatment_adherent",
        "treatment_switch",
        "treatment_discontinue",
    ]

    # Outcome type probabilities (conditional on treatment initiation)
    OUTCOME_PROBS = {
        "prescription_written": 0.95,  # Almost all initiated get Rx
        "prescription_filled": 0.80,   # 80% fill rate
        "treatment_adherent": 0.65,    # 65% adherent at 6 months
        "treatment_switch": 0.15,      # 15% switch therapy
        "treatment_discontinue": 0.20, # 20% discontinue
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "business_outcomes"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        patient_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the outcome generator.

        Args:
            config: Generator configuration.
            patient_df: Patient DataFrame with treatment info.
        """
        super().__init__(config)
        self.patient_df = patient_df
        self._dgp_config = None
        if self.config.dgp_type:
            self._dgp_config = DGP_CONFIGS.get(self.config.dgp_type)

    def generate(self) -> pd.DataFrame:
        """
        Generate business outcomes.

        Returns:
            DataFrame with business outcomes matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} business outcomes...")

        if self.patient_df is not None:
            # Generate outcomes for initiated patients
            initiated = self.patient_df[
                self.patient_df["treatment_initiated"] == 1
            ].copy()

            if len(initiated) == 0:
                self._log("Warning: No patients initiated treatment")
                return pd.DataFrame()

            # Each initiated patient can have multiple outcomes
            outcomes_per_patient = max(1, n // len(initiated))
            records = []

            for _, patient in initiated.iterrows():
                # Number of outcomes for this patient
                n_outcomes = self._rng.integers(1, min(5, outcomes_per_patient + 1))

                for _ in range(n_outcomes):
                    record = self._generate_outcome_record(patient)
                    records.append(record)

            df = pd.DataFrame(records)
        else:
            # Generate standalone outcomes
            df = self._generate_standalone_outcomes(n)

        # Add IDs and dates
        df["outcome_id"] = self._generate_ids("out", len(df))

        # Assign splits based on outcome dates
        if "outcome_date" in df.columns:
            df["data_split"] = self._assign_splits(df["outcome_date"].tolist())

        self._log(f"Generated {len(df)} business outcomes")
        return df

    def _generate_outcome_record(self, patient: pd.Series) -> Dict:
        """Generate a single outcome record linked to patient."""
        # Base outcome probability from patient engagement
        engagement_score = patient.get("engagement_score", 5.0)
        disease_severity = patient.get("disease_severity", 5.0)

        # Select outcome type with causal relationship to engagement
        outcome_type = self._select_outcome_type(engagement_score)

        # Calculate outcome value/revenue
        outcome_value = self._calculate_outcome_value(
            outcome_type,
            engagement_score,
            disease_severity,
        )

        # Generate outcome date (after journey start)
        journey_start = pd.to_datetime(patient.get("journey_start_date", "2023-01-01"))
        days_offset = self._rng.integers(30, 180)  # 1-6 months after start
        outcome_date = journey_start + pd.Timedelta(days=int(days_offset))

        return {
            "patient_journey_id": patient.get("patient_journey_id", ""),
            "patient_id": patient.get("patient_id", ""),
            "hcp_id": patient.get("hcp_id", ""),
            "brand": patient.get("brand", Brand.REMIBRUTINIB.value),
            "outcome_type": outcome_type,
            "outcome_date": outcome_date.strftime("%Y-%m-%d"),
            "outcome_value": round(outcome_value, 2),
            "engagement_at_outcome": round(engagement_score, 1),
            "disease_severity_at_outcome": round(disease_severity, 1),
        }

    def _select_outcome_type(self, engagement_score: float) -> str:
        """Select outcome type with engagement-dependent probabilities."""
        # Higher engagement → better outcomes
        engagement_factor = engagement_score / 10.0  # 0-1 scale

        # Adjust probabilities based on engagement
        probs = []
        for outcome_type in self.OUTCOME_TYPES:
            base_prob = self.OUTCOME_PROBS[outcome_type]

            if outcome_type in ["prescription_written", "prescription_filled", "treatment_adherent"]:
                # Positive outcomes increase with engagement
                adjusted_prob = base_prob * (0.7 + 0.3 * engagement_factor)
            elif outcome_type in ["treatment_switch", "treatment_discontinue"]:
                # Negative outcomes decrease with engagement
                adjusted_prob = base_prob * (1.3 - 0.3 * engagement_factor)
            else:
                adjusted_prob = base_prob

            probs.append(adjusted_prob)

        # Normalize probabilities
        probs = np.array(probs)
        probs = probs / probs.sum()

        return self._rng.choice(self.OUTCOME_TYPES, p=probs)

    def _calculate_outcome_value(
        self,
        outcome_type: str,
        engagement_score: float,
        disease_severity: float,
    ) -> float:
        """Calculate business value of outcome."""
        # Base values by outcome type (in dollars)
        base_values = {
            "prescription_written": 0,  # No direct revenue yet
            "prescription_filled": 5000,  # First fill value
            "treatment_adherent": 15000,  # 6-month value
            "treatment_switch": 2500,  # Partial value
            "treatment_discontinue": 0,  # Lost value
        }

        base = base_values.get(outcome_type, 0)

        # Adjust for engagement (higher engagement = better adherence = more value)
        engagement_factor = 0.8 + 0.4 * (engagement_score / 10.0)

        # Adjust for severity (higher severity = longer treatment = more value)
        severity_factor = 0.9 + 0.2 * (disease_severity / 10.0)

        # Add noise
        noise = self._rng.normal(1.0, 0.1)

        return base * engagement_factor * severity_factor * noise

    def _generate_standalone_outcomes(self, n: int) -> pd.DataFrame:
        """Generate outcomes without patient linkage."""
        # Generate basic fields
        patient_ids = self._generate_ids("pt", n, width=6)
        journey_ids = self._generate_ids("patient", n, width=6)
        hcp_ids = self._generate_ids("hcp", max(100, n // 10))

        # Generate engagement/severity for outcome calculation
        engagement_scores = self._random_normal(5.0, 2.0, n, clip_min=0, clip_max=10)
        disease_severities = self._random_normal(5.0, 2.0, n, clip_min=0, clip_max=10)

        # Generate outcome types
        outcome_types = [
            self._select_outcome_type(eng)
            for eng in engagement_scores
        ]

        # Generate outcome values
        outcome_values = [
            self._calculate_outcome_value(ot, eng, sev)
            for ot, eng, sev in zip(outcome_types, engagement_scores, disease_severities)
        ]

        # Generate dates
        outcome_dates = self._random_dates(n)

        # Determine brand
        if self.config.brand:
            brands = [self.config.brand.value] * n
        else:
            brands = self._random_choice([b.value for b in Brand], n)

        return pd.DataFrame({
            "patient_journey_id": journey_ids,
            "patient_id": patient_ids,
            "hcp_id": self._random_choice(hcp_ids, n).tolist(),
            "brand": brands,
            "outcome_type": outcome_types,
            "outcome_date": outcome_dates,
            "outcome_value": outcome_values,
            "engagement_at_outcome": np.round(engagement_scores, 1),
            "disease_severity_at_outcome": np.round(disease_severities, 1),
        })
