"""Mock data connector for testing Heterogeneous Optimizer Agent.

This connector generates synthetic data with heterogeneous treatment effects
for use in unit tests and development without database access.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class MockDataConnector:
    """Mock data connector for testing.

    Generates synthetic pharma CATE data with heterogeneous treatment effects
    across different HCP segments (specialty, volume, region).
    """

    async def query(
        self, source: str, columns: List[str], filters: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """Generate mock pharma CATE data.

        Args:
            source: Data source name (ignored, always generates mock data)
            columns: List of columns to include
            filters: Optional filters (ignored in mock)

        Returns:
            DataFrame with synthetic CATE data
        """
        np.random.seed(42)
        n_samples = 1000

        # Generate heterogeneous treatment effects
        data = {
            # Segment variables
            "hcp_specialty": np.random.choice(
                ["Oncology", "Cardiology", "Primary Care", "Rheumatology"], n_samples
            ),
            "patient_volume_decile": np.random.choice(
                ["1-2", "3-4", "5-6", "7-8", "9-10"], n_samples
            ),
            "region": np.random.choice(["Northeast", "Southeast", "Midwest", "West"], n_samples),
            # Effect modifiers
            "hcp_tenure": np.random.uniform(1, 30, n_samples),
            "competitive_pressure": np.random.uniform(0, 1, n_samples),
            "formulary_status": np.random.choice([0, 1], n_samples),
            # Treatment (binary: 0 or 1)
            "hcp_engagement_frequency": np.random.choice([0, 1], n_samples),
        }

        # Generate outcome with heterogeneous treatment effects
        # Different segments have different treatment responses
        outcome = np.zeros(n_samples)
        for i in range(n_samples):
            # Base outcome
            base = 100 + np.random.normal(0, 20)

            # Heterogeneous treatment effect based on specialty
            treatment_effect: float = 0.0
            if data["hcp_engagement_frequency"][i] == 1:
                if data["hcp_specialty"][i] == "Oncology":
                    treatment_effect = 50 + np.random.normal(0, 10)  # High responder
                elif data["hcp_specialty"][i] == "Cardiology":
                    treatment_effect = 30 + np.random.normal(0, 10)  # Medium responder
                elif data["hcp_specialty"][i] == "Primary Care":
                    treatment_effect = 10 + np.random.normal(0, 10)  # Low responder
                else:  # Rheumatology
                    treatment_effect = 25 + np.random.normal(0, 10)

                # Modify by tenure
                treatment_effect *= 1 + data["hcp_tenure"][i] / 100

                # Modify by competitive pressure (negative)
                treatment_effect *= 1 - data["competitive_pressure"][i] * 0.3

            outcome[i] = base + treatment_effect

        data["trx_total"] = outcome

        df = pd.DataFrame(data)

        # Filter columns
        if columns:
            available_cols = [col for col in columns if col in df.columns]
            df = df[available_cols]

        return df
