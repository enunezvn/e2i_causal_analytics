"""
Synthetic Twin Training Data (#705 H4)
======================================

A synthetic, RWD-ready training-frame provider for the digital-twin *generative*
model. The platform runs on synthetic vendor data, so the offline training job
needs a frame that:

* matches each twin type's ``TwinGenerator.DEFAULT_FEATURES`` schema (mixed
  categorical + numeric, so the label-encoder/scaler paths are exercised), and
* carries a learnable ``outcome`` target so a real sklearn model fits with a
  finite, certifiable R².

This mirrors the H5 effect provider's synthetic-known-first / RWD-ready pattern
(``src/digital_twin/effect/provider.py``) but for the GENERATIVE model — a
distinct schema from the effect estimator's confounder frame; the two must not
be conflated. Models trained on this frame are labelled ``data_provenance =
"synthetic"`` by the training job so they are never mistaken for RWD-trained.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .models.twin_models import TwinType
from .twin_generator import TwinGenerator

# String columns → exercise the LabelEncoder path on train + generate.
_CATEGORICAL: Dict[TwinType, Dict[str, List[str]]] = {
    TwinType.HCP: {
        "specialty": ["oncology", "cardiology", "neurology", "endocrinology", "rheumatology"],
        "practice_type": ["academic", "community", "private"],
        "region": ["northeast", "south", "midwest", "west"],
        "priority_tier": ["A", "B", "C"],
        "preferred_channel": ["email", "field", "virtual"],
        "adoption_stage": [
            "innovator",
            "early_adopter",
            "early_majority",
            "late_majority",
            "laggard",
        ],
    },
    TwinType.PATIENT: {
        "age_group": ["18-34", "35-49", "50-64", "65+"],
        "gender": ["female", "male", "other"],
        "geographic_region": ["northeast", "south", "midwest", "west"],
        "primary_diagnosis_code": ["C50", "C34", "I21", "E11", "M06"],
        "insurance_type": ["commercial", "medicare", "medicaid", "self_pay"],
        "journey_stage": ["diagnosis", "treatment", "maintenance", "remission"],
    },
    TwinType.TERRITORY: {
        "region": ["northeast", "south", "midwest", "west"],
    },
}

# Integer numeric ranges (inclusive).
_NUMERIC_INT: Dict[TwinType, Dict[str, Tuple[int, int]]] = {
    TwinType.HCP: {
        "years_experience": (1, 40),
        "practice_size": (1, 50),
        "decile": (1, 10),
        "total_patient_volume": (50, 5000),
        "target_patient_volume": (10, 1000),
        "last_interaction_days": (0, 365),
    },
    TwinType.PATIENT: {
        "comorbidity_count": (0, 10),
        "journey_duration_days": (0, 720),
        "treatment_line": (1, 5),
        "insurance_coverage_flag": (0, 1),
    },
    TwinType.TERRITORY: {
        "state_count": (1, 50),
        "zip_count": (1, 2000),
        "total_hcps": (0, 5000),
        "covered_hcps": (0, 5000),
        "total_patient_volume": (1000, 500000),
    },
}

# Float numeric ranges.
_NUMERIC_FLOAT: Dict[TwinType, Dict[str, Tuple[float, float]]] = {
    TwinType.HCP: {
        "digital_engagement_score": (0.0, 1.0),
        "interaction_frequency": (0.0, 10.0),
        "peer_influence_score": (0.0, 1.0),
    },
    TwinType.PATIENT: {
        "socioeconomic_index": (0.0, 1.0),
        "risk_score": (0.0, 1.0),
        "journey_complexity_score": (0.0, 1.0),
    },
    TwinType.TERRITORY: {
        "coverage_rate": (0.0, 1.0),
        "market_share": (0.0, 1.0),
        "growth_rate": (-0.5, 0.5),
        "competitor_presence": (0.0, 1.0),
    },
}

DEFAULT_TARGET_COLUMN = "outcome"


def synthetic_training_frame(
    twin_type: TwinType,
    *,
    n_rows: int = 2000,
    target_col: str = DEFAULT_TARGET_COLUMN,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic training frame matching ``twin_type``'s feature schema.

    Args:
        twin_type: Which twin type's ``DEFAULT_FEATURES`` schema to generate.
        n_rows: Number of rows (default comfortably above
            ``TwinGenerator.MIN_TRAINING_SAMPLES``).
        target_col: Name of the learnable outcome column to add.
        seed: RNG seed — same seed ⇒ identical frame (deterministic).

    Returns:
        A ``pd.DataFrame`` with every ``DEFAULT_FEATURES[twin_type]`` column plus
        a learnable ``target_col`` (weighted sum of standardised numeric features
        + small noise).
    """
    rng = np.random.default_rng(seed)
    features = TwinGenerator.DEFAULT_FEATURES.get(twin_type, [])
    categorical = _CATEGORICAL.get(twin_type, {})
    ints = _NUMERIC_INT.get(twin_type, {})
    floats = _NUMERIC_FLOAT.get(twin_type, {})

    cols: Dict[str, np.ndarray] = {}
    for col in features:
        if col in categorical:
            cols[col] = rng.choice(categorical[col], size=n_rows)
        elif col in ints:
            lo, hi = ints[col]
            cols[col] = rng.integers(lo, hi + 1, size=n_rows)
        elif col in floats:
            lo_f, hi_f = floats[col]
            cols[col] = rng.uniform(lo_f, hi_f, size=n_rows)
        else:  # pragma: no cover - fallback for any unspecified feature
            cols[col] = rng.uniform(0.0, 1.0, size=n_rows)

    df = pd.DataFrame(cols)

    # Learnable target: weighted sum of standardised numeric features + noise.
    numeric_cols = [c for c in features if c in ints or c in floats]
    target = np.zeros(n_rows, dtype=float)
    if numeric_cols:
        weights = rng.uniform(0.2, 1.0, size=len(numeric_cols))
        for weight, col in zip(weights, numeric_cols, strict=True):
            values = df[col].astype(float).to_numpy()
            spread = values.std() or 1.0
            target += weight * (values - values.mean()) / spread
        target /= len(numeric_cols)
    target += rng.normal(0.0, 0.05, size=n_rows)
    df[target_col] = target

    return df


__all__ = ["DEFAULT_TARGET_COLUMN", "synthetic_training_frame"]
