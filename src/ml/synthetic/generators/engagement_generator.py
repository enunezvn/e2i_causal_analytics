"""
Engagement Event Generator.

Generates synthetic engagement events (rep visits, digital, etc.).
"""

from typing import List, Optional

import numpy as np
import pandas as pd

from ..config import Brand, EngagementTypeEnum
from .base import BaseGenerator, GeneratorConfig


class EngagementGenerator(BaseGenerator[pd.DataFrame]):
    """
    Generator for engagement events.

    Generates HCP engagement records with:
    - Multiple engagement types (detail visits, digital, etc.)
    - Quality/depth scoring
    - Temporal patterns (more engagement = better outcomes)
    """

    # Engagement type distributions by practice setting
    ENGAGEMENT_TYPE_DIST = {
        "academic": {
            EngagementTypeEnum.SPEAKER_PROGRAM: 0.30,
            EngagementTypeEnum.DETAIL_VISIT: 0.25,
            EngagementTypeEnum.WEBINAR: 0.25,
            EngagementTypeEnum.DIGITAL: 0.15,
            EngagementTypeEnum.SAMPLE_REQUEST: 0.05,
        },
        "community": {
            EngagementTypeEnum.DETAIL_VISIT: 0.40,
            EngagementTypeEnum.SAMPLE_REQUEST: 0.25,
            EngagementTypeEnum.DIGITAL: 0.20,
            EngagementTypeEnum.WEBINAR: 0.10,
            EngagementTypeEnum.SPEAKER_PROGRAM: 0.05,
        },
        "private": {
            EngagementTypeEnum.SAMPLE_REQUEST: 0.35,
            EngagementTypeEnum.DETAIL_VISIT: 0.30,
            EngagementTypeEnum.DIGITAL: 0.25,
            EngagementTypeEnum.WEBINAR: 0.08,
            EngagementTypeEnum.SPEAKER_PROGRAM: 0.02,
        },
    }

    # Default distribution
    DEFAULT_DIST = {
        EngagementTypeEnum.DETAIL_VISIT: 0.35,
        EngagementTypeEnum.DIGITAL: 0.25,
        EngagementTypeEnum.SAMPLE_REQUEST: 0.20,
        EngagementTypeEnum.WEBINAR: 0.12,
        EngagementTypeEnum.SPEAKER_PROGRAM: 0.08,
    }

    @property
    def entity_type(self) -> str:
        """Return entity type."""
        return "engagement_events"

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        hcp_df: Optional[pd.DataFrame] = None,
        patient_df: Optional[pd.DataFrame] = None,
    ):
        """
        Initialize the engagement generator.

        Args:
            config: Generator configuration.
            hcp_df: HCP DataFrame for foreign key integrity.
            patient_df: Patient DataFrame for engagement-outcome linking.
        """
        super().__init__(config)
        self.hcp_df = hcp_df
        self.patient_df = patient_df

    def generate(self) -> pd.DataFrame:
        """
        Generate engagement events.

        Returns:
            DataFrame with engagement events matching schema.
        """
        n = self.config.n_records
        self._log(f"Generating {n} engagement events...")

        # Get HCP data for referential integrity
        if self.hcp_df is not None:
            # Calculate events per HCP
            events_per_hcp = max(1, n // len(self.hcp_df))
            hcp_ids = []
            practice_types = []

            for _, hcp in self.hcp_df.iterrows():
                n_events = self._rng.integers(
                    max(1, events_per_hcp // 2),
                    events_per_hcp * 2,
                )
                for _ in range(n_events):
                    hcp_ids.append(hcp["hcp_id"])
                    practice_types.append(hcp.get("practice_type", "community"))

            n_actual = len(hcp_ids)
        else:
            # Generate standalone
            n_actual = n
            n_hcps = max(100, n // 10)
            hcp_pool = self._generate_ids("hcp", n_hcps)
            hcp_ids = self._random_choice(hcp_pool, n).tolist()
            practice_types = self._random_choice(
                ["academic", "community", "private"],
                n,
                p=[0.25, 0.50, 0.25],
            ).tolist()

        # Generate engagement data
        engagement_ids = self._generate_ids("eng", n_actual)
        engagement_types = self._generate_engagement_types(practice_types)
        engagement_dates = self._random_dates(n_actual)

        # Quality metrics
        quality_scores = self._generate_quality_scores(engagement_types, n_actual)
        duration_minutes = self._generate_durations(engagement_types)

        # Rep information
        rep_ids = self._generate_ids("rep", max(50, n_actual // 20))
        rep_assignments = self._random_choice(rep_ids, n_actual).tolist()

        # Determine brand
        if self.config.brand:
            brands = [self.config.brand.value] * n_actual
        else:
            brands = self._random_choice([b.value for b in Brand], n_actual).tolist()

        # Build DataFrame
        df = pd.DataFrame(
            {
                "engagement_event_id": engagement_ids,
                "hcp_id": hcp_ids[:n_actual] if isinstance(hcp_ids, list) else hcp_ids,
                "rep_id": rep_assignments,
                "brand": brands,
                "engagement_date": engagement_dates,
                "engagement_type": engagement_types,
                "quality_score": quality_scores,
                "duration_minutes": duration_minutes,
                "data_split": self._assign_splits(engagement_dates),
            }
        )

        self._log(f"Generated {len(df)} engagement events")
        return df

    def _generate_engagement_types(self, practice_types: List[str]) -> List[str]:
        """Generate engagement types based on practice type."""
        engagement_types = []

        for practice_type in practice_types:
            dist = self.ENGAGEMENT_TYPE_DIST.get(practice_type, self.DEFAULT_DIST)
            options = [e.value for e in dist.keys()]
            probs = list(dist.values())
            engagement_type = self._rng.choice(options, p=probs)
            engagement_types.append(engagement_type)

        return engagement_types

    def _generate_quality_scores(
        self,
        engagement_types: List[str],
        n: int,
    ) -> np.ndarray:
        """Generate quality scores (0-10) based on engagement type."""
        scores = np.zeros(n)

        for i, eng_type in enumerate(engagement_types):
            # Speaker programs and webinars typically have higher quality
            if eng_type in [
                EngagementTypeEnum.SPEAKER_PROGRAM.value,
                EngagementTypeEnum.WEBINAR.value,
            ]:
                scores[i] = self._rng.normal(7.5, 1.5)
            elif eng_type == EngagementTypeEnum.DETAIL_VISIT.value:
                scores[i] = self._rng.normal(6.5, 1.8)
            elif eng_type == EngagementTypeEnum.DIGITAL.value:
                scores[i] = self._rng.normal(5.5, 2.0)
            else:
                scores[i] = self._rng.normal(5.0, 1.5)

        return np.clip(scores, 0, 10).round(1)

    def _generate_durations(self, engagement_types: List[str]) -> List[int]:
        """Generate engagement duration in minutes."""
        durations = []

        for eng_type in engagement_types:
            if eng_type == EngagementTypeEnum.SPEAKER_PROGRAM.value:
                duration = self._rng.integers(60, 180)
            elif eng_type == EngagementTypeEnum.WEBINAR.value:
                duration = self._rng.integers(30, 90)
            elif eng_type == EngagementTypeEnum.DETAIL_VISIT.value:
                duration = self._rng.integers(10, 30)
            elif eng_type == EngagementTypeEnum.DIGITAL.value:
                duration = self._rng.integers(2, 15)
            else:  # Sample request
                duration = self._rng.integers(5, 15)

            durations.append(int(duration))

        return durations
