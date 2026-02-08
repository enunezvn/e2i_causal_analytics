"""
Patient Journey Generator.

Generates synthetic patient journeys with embedded causal effects.
This is the core generator for causal validation.
"""

from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
from scipy.special import expit

from ..config import (
    DGP_CONFIGS,
    Brand,
    DGPType,
    InsuranceTypeEnum,
    RegionEnum,
)
from .base import BaseGenerator, GeneratorConfig


class PatientGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for patient journeys with embedded causal effects.

    This generator creates patient records with:
    - Confounders (disease_severity, academic_hcp)
    - Treatment (engagement_score)
    - Outcome (treatment_initiated)
    - TRUE CAUSAL EFFECT embedded per DGP

    The causal structure follows:
        Confounders → Treatment
        Confounders → Outcome
        Treatment → Outcome (TRUE CAUSAL EFFECT)
    """

    # Insurance distribution (US commercial market)
    # Only using enum values that exist in schema
    INSURANCE_DIST = {
        InsuranceTypeEnum.COMMERCIAL: 0.60,
        InsuranceTypeEnum.MEDICARE: 0.30,
        InsuranceTypeEnum.MEDICAID: 0.10,
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "patient_journeys"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        hcp_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the patient generator.

        Args:
            config: Generator configuration.
            hcp_df: Optional HCP DataFrame for foreign key integrity.
        """
        super().__init__(config)
        self.hcp_df = hcp_df
        self._dgp_config = None
        if self.config.dgp_type:
            self._dgp_config = DGP_CONFIGS.get(self.config.dgp_type)

    def generate(self) -> pd.DataFrame:
        """
        Generate patient journeys with embedded causal effects.

        Returns:
            DataFrame with patient journeys matching schema.
        """
        n = self.config.n_records
        dgp_type = self.config.dgp_type or DGPType.CONFOUNDED
        self._log(f"Generating {n} patient journeys with DGP: {dgp_type.value}")

        # Get DGP config
        dgp_config = DGP_CONFIGS.get(dgp_type)
        true_ate = dgp_config.true_ate if dgp_config else 0.25

        # Generate confounders FIRST (they affect both treatment and outcome)
        confounders = self._generate_confounders(n, dgp_type)

        # Generate treatment (engagement) based on confounders
        engagement_scores = self._generate_treatment(confounders, dgp_type)

        # Generate outcome with TRUE causal effect
        outcomes = self._generate_outcome(
            engagement_scores,
            confounders,
            true_ate,
            dgp_type,
        )

        # Generate dates and assign splits
        journey_dates = self._random_dates(n)
        data_splits = self._assign_splits(journey_dates)

        # Generate HCP assignments
        hcp_ids = self._assign_hcps(n, confounders)

        # Generate patient IDs
        patient_ids = self._generate_ids("pt", n, width=6)
        journey_ids = self._generate_ids("patient", n, width=6)

        # Determine brand
        if self.config.brand:
            brands = [self.config.brand.value] * n
        else:
            brands = self._random_choice([b.value for b in Brand], n).tolist()

        # Build DataFrame
        df = pd.DataFrame(
            {
                "patient_journey_id": journey_ids,
                "patient_id": patient_ids,
                "hcp_id": hcp_ids,
                "brand": brands,
                "journey_start_date": journey_dates,
                "data_split": data_splits,
                "disease_severity": confounders["disease_severity"],
                "academic_hcp": confounders["academic_hcp"],
                "engagement_score": engagement_scores,
                "treatment_initiated": outcomes["treatment_initiated"],
                "days_to_treatment": outcomes["days_to_treatment"],
                "geographic_region": self._random_choice(
                    [r.value for r in RegionEnum],
                    n,
                ),
                "insurance_type": self._random_choice(
                    [i.value for i in InsuranceTypeEnum],
                    n,
                    p=[self.INSURANCE_DIST[i] for i in InsuranceTypeEnum],
                ),
                "age_at_diagnosis": self._random_int(18, 85, n),
            }
        )

        # Store ground truth metadata
        df.attrs["true_ate"] = true_ate
        df.attrs["dgp_type"] = dgp_type.value
        df.attrs["confounders"] = dgp_config.confounders if dgp_config else []

        self._log(f"Generated {len(df)} patient journeys (TRUE_ATE={true_ate})")
        return df

    def _generate_confounders(
        self,
        n: int,
        dgp_type: DGPType,
    ) -> Dict[str, np.ndarray]:
        """
        Generate confounding variables.

        These affect both treatment and outcome.
        """
        # Disease severity: 0-10 scale, normally distributed
        disease_severity = self._random_normal(5.0, 2.0, n, clip_min=0, clip_max=10)

        # Academic HCP: binary, ~30% are academic
        academic_hcp = (self._rng.random(n) < 0.30).astype(int)

        # For heterogeneous DGP, add segment-specific variation
        if dgp_type == DGPType.HETEROGENEOUS:
            # Create segments based on disease severity
            # High: severity > 7, Medium: 4-7, Low: < 4
            pass  # Segment effects handled in outcome generation

        return {
            "disease_severity": disease_severity,
            "academic_hcp": academic_hcp,
        }

    def _generate_treatment(
        self,
        confounders: Dict[str, np.ndarray],
        dgp_type: DGPType,
    ) -> np.ndarray:
        """
        Generate treatment (engagement_score) with confounding.

        Treatment propensity is influenced by confounders:
        - Higher disease severity → more engagement
        - Academic HCP → more engagement
        """
        n = len(confounders["disease_severity"])

        if dgp_type == DGPType.SIMPLE_LINEAR:
            # No confounding - pure random treatment
            engagement = self._random_float(0, 10, n)
        elif dgp_type == DGPType.SELECTION_BIAS:
            # Strong selection bias based on disease severity
            propensity = (
                2.0
                + 0.8 * confounders["disease_severity"]  # Strong severity effect
                + self._rng.normal(0, 0.5, n)
            )
            engagement = expit(propensity / 3) * 10
        else:
            # Standard confounding structure
            propensity = (
                3.0
                + 0.3 * confounders["disease_severity"]
                + 2.0 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
            engagement = expit(propensity / 3) * 10

        return np.clip(engagement, 0, 10)

    def _generate_outcome(
        self,
        treatment: np.ndarray,
        confounders: Dict[str, np.ndarray],
        true_ate: float,
        dgp_type: DGPType,
    ) -> Dict[str, np.ndarray]:
        """
        Generate outcome with TRUE causal effect.

        Outcome = f(confounders) + TRUE_ATE * treatment + noise

        This is the key function for causal validation.
        """
        n = len(treatment)

        if dgp_type == DGPType.SIMPLE_LINEAR:
            # Simple linear: Y = TRUE_ATE * T + noise
            outcome_propensity = -2.0 + true_ate * treatment + self._rng.normal(0, 1, n)
        elif dgp_type == DGPType.HETEROGENEOUS:
            # Heterogeneous treatment effects by segment
            # Segment assignment based on disease severity
            segments = np.where(
                confounders["disease_severity"] > 7,
                "high",
                np.where(confounders["disease_severity"] > 4, "medium", "low"),
            )

            # CATE by segment
            cate = np.where(
                segments == "high",
                0.50,  # High severity: strong effect
                np.where(segments == "medium", 0.30, 0.15),  # Medium: moderate, Low: weak
            )

            outcome_propensity = (
                -2.0
                + cate * treatment  # Heterogeneous effect
                + 0.4 * confounders["disease_severity"]
                + 0.6 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
        elif dgp_type == DGPType.TIME_SERIES:
            # Time series: effect with lag
            # Simulated by adding temporal decay
            lag_effect = 0.85 ** np.arange(n)  # Decay over time
            effective_treatment = treatment * (0.5 + 0.5 * lag_effect)

            outcome_propensity = (
                -2.0
                + true_ate * effective_treatment
                + 0.4 * confounders["disease_severity"]
                + 0.6 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
        elif dgp_type == DGPType.SELECTION_BIAS:
            # Selection bias: outcome affected by selection mechanism
            # Higher baseline for high-severity patients
            selection_baseline = 0.3 * confounders["disease_severity"]

            outcome_propensity = (
                -2.0
                + selection_baseline
                + true_ate * treatment
                + 0.2 * confounders["disease_severity"]  # Residual confounding
                + 0.6 * confounders["academic_hcp"]
                + self._rng.normal(0, 1, n)
            )
        else:
            # Default: Confounded DGP
            outcome_propensity = (
                -2.0
                + true_ate * treatment  # TRUE CAUSAL EFFECT
                + 0.4 * confounders["disease_severity"]  # Confounding
                + 0.6 * confounders["academic_hcp"]  # Confounding
                + self._rng.normal(0, 1, n)
            )

        # Convert to binary outcome
        treatment_initiated = (expit(outcome_propensity) > 0.5).astype(int)

        # Generate days to treatment (only for those who initiated)
        days_to_treatment: Any = np.where(
            treatment_initiated == 1,
            self._random_int(7, 90, n),
            np.nan,  # Use np.nan instead of None for numpy compatibility
        )

        return {
            "treatment_initiated": treatment_initiated,
            "days_to_treatment": days_to_treatment,
        }

    def _assign_hcps(
        self,
        n: int,
        confounders: Dict[str, np.ndarray],
    ) -> List[str]:
        """
        Assign HCPs to patients.

        If HCP DataFrame provided, maintains referential integrity.
        Otherwise generates placeholder IDs.
        """
        if self.hcp_df is not None and len(self.hcp_df) > 0:
            # Match academic patients to academic HCPs when possible
            academic_hcps = self.hcp_df[self.hcp_df["academic_hcp"] == 1]["hcp_id"].values
            non_academic_hcps = self.hcp_df[self.hcp_df["academic_hcp"] == 0]["hcp_id"].values

            hcp_ids = []
            for is_academic in confounders["academic_hcp"]:
                if is_academic == 1 and len(academic_hcps) > 0:
                    hcp_ids.append(self._rng.choice(academic_hcps))
                elif len(non_academic_hcps) > 0:
                    hcp_ids.append(self._rng.choice(non_academic_hcps))
                else:
                    hcp_ids.append(self._rng.choice(self.hcp_df["hcp_id"].values))

            return hcp_ids
        else:
            # Generate placeholder HCP IDs
            n_hcps = max(100, n // 10)  # ~10 patients per HCP
            hcp_ids = self._generate_ids("hcp", n_hcps)
            return cast(List[str], self._random_choice(hcp_ids, n).tolist())
