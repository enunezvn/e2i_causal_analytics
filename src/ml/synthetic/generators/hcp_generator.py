"""
HCP (Healthcare Provider) Generator.

Generates synthetic HCP profiles matching Supabase schema.
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from .base import BaseGenerator, GeneratorConfig
from ..config import (
    Brand,
    SpecialtyEnum,
    PracticeTypeEnum,
    RegionEnum,
)


class HCPGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for HCP (Healthcare Provider) profiles.

    Generates HCPs with:
    - Specialty distribution matching pharma targeting
    - Practice type distribution (academic heavy for certain brands)
    - Geographic distribution
    - Experience levels with realistic ranges
    - Brand association based on specialty alignment
    """

    # Specialty distributions by brand (reflects targeting strategy)
    # Using existing SpecialtyEnum values from config
    BRAND_SPECIALTY_DIST = {
        Brand.REMIBRUTINIB: {
            SpecialtyEnum.DERMATOLOGY: 0.50,
            SpecialtyEnum.ALLERGY_IMMUNOLOGY: 0.35,  # CSU indication
            SpecialtyEnum.RHEUMATOLOGY: 0.15,
        },
        Brand.FABHALTA: {
            SpecialtyEnum.HEMATOLOGY: 0.60,  # PNH indication
            SpecialtyEnum.INTERNAL_MEDICINE: 0.30,
            SpecialtyEnum.NEUROLOGY: 0.10,  # Rare disease overlap
        },
        Brand.KISQALI: {
            SpecialtyEnum.ONCOLOGY: 1.00,  # HR+/HER2- breast cancer
        },
    }

    # Practice type distributions (academic tends to have higher engagement)
    PRACTICE_TYPE_DIST = {
        PracticeTypeEnum.ACADEMIC: 0.25,
        PracticeTypeEnum.COMMUNITY: 0.50,
        PracticeTypeEnum.PRIVATE: 0.25,
    }

    # Geographic region distribution (reflects US market)
    REGION_DIST = {
        RegionEnum.NORTHEAST: 0.22,
        RegionEnum.SOUTH: 0.38,
        RegionEnum.MIDWEST: 0.21,
        RegionEnum.WEST: 0.19,
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "hcp_profiles"

    def generate(self) -> pd.DataFrame:
        """
        Generate HCP profiles.

        Returns:
            DataFrame with HCP profiles matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} HCP profiles...")

        # Determine brand distribution
        if self.config.brand:
            brands = [self.config.brand.value] * n
        else:
            # Equal distribution across brands
            brands = self._random_choice(
                [b.value for b in Brand],
                n,
            )

        # Generate specialties based on brand
        specialties = self._generate_specialties(brands)

        # Generate practice types with academic bias for certain specialties
        practice_types = self._generate_practice_types(specialties)

        # Generate base data
        df = pd.DataFrame({
            "hcp_id": self._generate_ids("hcp", n),
            "npi": self._generate_npis(n),
            "specialty": specialties,
            "practice_type": practice_types,
            "geographic_region": self._random_choice(
                [r.value for r in RegionEnum],
                n,
                p=[self.REGION_DIST[r] for r in RegionEnum],
            ),
            "years_experience": self._random_int(2, 40, n),
            "academic_hcp": self._generate_academic_flags(practice_types),
            "total_patient_volume": self._generate_patient_volumes(practice_types),
            "brand": brands,
        })

        self._log(f"Generated {len(df)} HCP profiles")
        return df

    def _generate_specialties(self, brands: List[str]) -> List[str]:
        """Generate specialties aligned with brands."""
        specialties = []

        for brand_str in brands:
            brand = Brand(brand_str)
            dist = self.BRAND_SPECIALTY_DIST.get(brand, {})

            if dist:
                spec_options = list(dist.keys())
                spec_probs = list(dist.values())
                specialty = self._rng.choice(
                    [s.value for s in spec_options],
                    p=spec_probs,
                )
            else:
                # Fallback to random specialty
                specialty = self._rng.choice([s.value for s in SpecialtyEnum])

            specialties.append(specialty)

        return specialties

    def _generate_practice_types(self, specialties: List[str]) -> List[str]:
        """Generate practice types with specialty-based bias."""
        practice_types = []

        for specialty in specialties:
            # Academic hospitals more common for oncology
            if "oncology" in specialty.lower():
                probs = [0.40, 0.40, 0.20]  # Higher academic
            else:
                probs = [
                    self.PRACTICE_TYPE_DIST[p]
                    for p in PracticeTypeEnum
                ]

            practice_type = self._rng.choice(
                [p.value for p in PracticeTypeEnum],
                p=probs,
            )
            practice_types.append(practice_type)

        return practice_types

    def _generate_npis(self, n: int) -> List[str]:
        """Generate unique NPI numbers (10-digit)."""
        # NPIs are 10 digits starting with 1 or 2
        base = self._rng.integers(1_000_000_000, 2_999_999_999, size=n)
        return [str(npi) for npi in base]

    def _generate_academic_flags(self, practice_types: List[str]) -> np.ndarray:
        """Generate academic HCP flags based on practice type."""
        flags = np.zeros(len(practice_types), dtype=int)

        for i, pt in enumerate(practice_types):
            if pt == PracticeTypeEnum.ACADEMIC.value:
                # Most academic practice HCPs are academic (80%)
                flags[i] = 1 if self._rng.random() < 0.80 else 0
            else:
                # Some community/private HCPs have academic affiliations (15%)
                flags[i] = 1 if self._rng.random() < 0.15 else 0

        return flags

    def _generate_patient_volumes(self, practice_types: List[str]) -> np.ndarray:
        """Generate patient volumes based on practice type."""
        volumes = np.zeros(len(practice_types), dtype=int)

        for i, pt in enumerate(practice_types):
            if pt == PracticeTypeEnum.ACADEMIC.value:
                # Academic: higher volume, more variance
                volumes[i] = int(self._rng.normal(300, 100))
            elif pt == PracticeTypeEnum.COMMUNITY.value:
                # Community: moderate volume
                volumes[i] = int(self._rng.normal(200, 75))
            else:
                # Private: lower volume, less variance
                volumes[i] = int(self._rng.normal(150, 50))

        # Clip to reasonable range
        return np.clip(volumes, 50, 600)
