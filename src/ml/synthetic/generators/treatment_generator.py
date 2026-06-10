"""
Treatment Event Generator.

Generates synthetic treatment events linked to patient journeys.
"""

from typing import List, Optional, cast

import numpy as np
import pandas as pd

from ..clinical_codes import brand_codes
from ..config import Brand
from .base import BaseGenerator, GeneratorConfig


class TreatmentGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for treatment events.

    Generates treatment records linked to patient journeys with:
    - Temporal ordering (treatment follows engagement)
    - Dose/frequency patterns by brand
    - Treatment response outcomes
    """

    # Treatment type distributions by brand
    # Note: Values MUST match Supabase event_type ENUM:
    # {diagnosis, prescription, lab_test, procedure, consultation, hospitalization}
    TREATMENT_TYPES = {
        Brand.REMIBRUTINIB: [
            ("prescription", 0.70),
            ("consultation", 0.15),
            ("lab_test", 0.10),
            ("procedure", 0.05),
        ],
        Brand.FABHALTA: [
            ("prescription", 0.75),
            ("consultation", 0.15),
            ("lab_test", 0.10),
        ],
        Brand.KISQALI: [
            ("prescription", 0.65),
            ("consultation", 0.15),
            ("lab_test", 0.12),
            ("procedure", 0.08),
        ],
    }

    # Default treatment types for unknown brands
    DEFAULT_TREATMENT_TYPES = [
        ("prescription", 0.70),
        ("consultation", 0.15),
        ("lab_test", 0.10),
        ("procedure", 0.05),
    ]

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "treatment_events"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        patient_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the treatment generator.

        Args:
            config: Generator configuration.
            patient_df: Patient DataFrame for foreign key integrity.
        """
        super().__init__(config)
        self.patient_df = patient_df

    def generate(self) -> pd.DataFrame:
        """
        Generate treatment events.

        Returns:
            DataFrame with treatment events matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} treatment events...")

        # Get patient data for referential integrity
        if self.patient_df is not None:
            # Only generate for patients who initiated treatment
            eligible_patients = self.patient_df[self.patient_df["treatment_initiated"] == 1].copy()

            if len(eligible_patients) == 0:
                self._log("Warning: No eligible patients for treatment events")
                return pd.DataFrame()

            # Calculate events per patient
            events_per_patient = max(1, n // len(eligible_patients))
            patient_ids = []
            journey_ids = []
            brands = []
            start_dates = []
            hcp_ids = []

            for _, patient in eligible_patients.iterrows():
                # Each patient gets multiple treatment events
                n_events = self._rng.integers(
                    max(1, events_per_patient // 2),
                    events_per_patient * 2,
                )
                for _ in range(n_events):
                    patient_ids.append(patient["patient_id"])
                    journey_ids.append(patient["patient_journey_id"])
                    brands.append(patient.get("brand", Brand.REMIBRUTINIB.value))
                    start_dates.append(patient["journey_start_date"])
                    hcp_ids.append(patient.get("hcp_id", ""))

            n_actual = len(patient_ids)
        else:
            # Generate standalone
            n_actual = n
            patient_ids = self._generate_ids("pt", n, width=6)
            journey_ids = self._generate_ids("patient", n, width=6)
            brands = self._random_choice([b.value for b in Brand], n).tolist()
            start_dates = self._random_dates(n)
            hcp_ids = self._generate_ids("hcp", max(100, n // 10))
            hcp_ids = self._random_choice(hcp_ids, n).tolist()

        # Generate treatment data
        treatment_ids = self._generate_ids("trx", n_actual)
        treatment_types = self._generate_treatment_types(brands)
        treatment_dates = self._generate_treatment_dates(start_dates)
        days_supply = self._random_int(7, 90, n_actual)
        refill_number = self._random_int(0, 12, n_actual)

        # Treatment response (adherence/efficacy)
        adherence_scores = self._random_normal(0.75, 0.15, n_actual, clip_min=0, clip_max=1)
        efficacy_scores = self._random_normal(0.65, 0.20, n_actual, clip_min=0, clip_max=1)

        # Indication-correct coding per brand (Shard 04). The DB carries the dx as
        # `icd_codes text[]` (real column) and `drug_ndc/drug_name/drug_class/
        # event_subtype` (real columns). `primary_diagnosis_code` is exposed on this
        # frame as a scalar for Shard 05/06 joins + tests, but is NOT a treatment_events
        # DB column (it lives on patient_journeys) -> the loader gates it out. Non-
        # portfolio brands fall back to a neutral antihistamine bundle so a row never
        # emits a mismatched indication.
        drug_ndc, drug_name, drug_class, primary_dx, event_subtype, icd_codes = (
            [],
            [],
            [],
            [],
            [],
            [],
        )
        for b in brands[:n_actual]:
            try:
                codes = brand_codes(b)
            except KeyError:
                codes = {
                    "icd10": ["L50.9"],
                    "drug_class": "Antihistamine",
                    "drug_name": "cetirizine",
                    "ndc": "00078-0000-00",
                }
            dx = str(self._rng.choice(cast("list[str]", codes["icd10"])))
            primary_dx.append(dx)
            icd_codes.append([dx])
            drug_ndc.append(codes["ndc"])
            drug_name.append(codes["drug_name"])
            drug_class.append(codes["drug_class"])
            event_subtype.append(str(codes["drug_class"]).lower().replace(" ", "_"))

        # Build DataFrame
        df = pd.DataFrame(
            {
                "treatment_event_id": treatment_ids,
                "patient_journey_id": journey_ids[:n_actual],
                "patient_id": patient_ids[:n_actual],
                "hcp_id": hcp_ids[:n_actual] if isinstance(hcp_ids, list) else hcp_ids,
                "brand": brands[:n_actual] if isinstance(brands, list) else brands,
                "treatment_date": treatment_dates,
                "treatment_type": treatment_types,
                "days_supply": days_supply,
                "refill_number": refill_number,
                "adherence_score": np.round(adherence_scores, 2),
                "efficacy_score": np.round(efficacy_scores, 2),
                "drug_ndc": drug_ndc,
                "drug_name": drug_name,
                "drug_class": drug_class,
                "primary_diagnosis_code": primary_dx,
                "icd_codes": icd_codes,
                "event_subtype": event_subtype,
                "data_split": self._assign_splits(treatment_dates),
            }
        )

        self._log(f"Generated {len(df)} treatment events")
        return df

    def _generate_treatment_types(self, brands: List[str]) -> List[str]:
        """Generate treatment types based on brand."""
        treatment_types = []

        for brand_str in brands:
            try:
                brand = Brand(brand_str)
                types = self.TREATMENT_TYPES.get(brand, self.DEFAULT_TREATMENT_TYPES)
            except ValueError:
                types = self.DEFAULT_TREATMENT_TYPES

            # Sample from distribution
            options = [t[0] for t in types]
            probs = [t[1] for t in types]
            treatment_type = self._rng.choice(options, p=probs)
            treatment_types.append(treatment_type)

        return treatment_types

    def _generate_treatment_dates(self, start_dates: List[str]) -> List[str]:
        """Generate treatment dates after journey start."""
        treatment_dates = []

        for start_date in start_dates:
            # Treatment starts 7-30 days after journey start
            days_offset = self._rng.integers(7, 31)
            start = pd.to_datetime(start_date)
            treatment_date = start + pd.Timedelta(days=int(days_offset))
            treatment_dates.append(treatment_date.strftime("%Y-%m-%d"))

        # Shard 04: re-anchor onto the rolling window (no-op when anchor_to_now off)
        return self._shift_dates_to_window(treatment_dates)
