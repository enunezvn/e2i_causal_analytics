"""Synthetic twin training-data provider (#705 H4 / H6).

The digital-twin platform runs on synthetic vendor data; the generator-training
path needs a synthetic, RWD-ready frame that matches each twin type's
DEFAULT_FEATURES schema and carries a learnable target, so a real model can be
trained end-to-end without external files. This mirrors the H5 effect provider's
synthetic-known-first / RWD-ready pattern (but for the GENERATIVE model, a
distinct schema — not the effect estimator's confounder frame).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.digital_twin.models.twin_models import Brand, TwinType
from src.digital_twin.twin_generator import TwinGenerator


@pytest.mark.parametrize("twin_type", [TwinType.HCP, TwinType.PATIENT, TwinType.TERRITORY])
def test_synthetic_frame_has_all_default_features_plus_target(twin_type):
    from src.digital_twin.training_data import synthetic_training_frame

    df = synthetic_training_frame(twin_type, n_rows=1100, target_col="outcome", seed=3)

    assert len(df) == 1100
    for col in TwinGenerator.DEFAULT_FEATURES[twin_type]:
        assert col in df.columns, f"missing feature {col} for {twin_type}"
    assert "outcome" in df.columns


def test_synthetic_frame_is_deterministic_for_seed():
    from src.digital_twin.training_data import synthetic_training_frame

    a = synthetic_training_frame(TwinType.HCP, n_rows=1050, seed=11)
    b = synthetic_training_frame(TwinType.HCP, n_rows=1050, seed=11)
    assert a.equals(b)


@pytest.mark.parametrize("twin_type", [TwinType.HCP, TwinType.PATIENT, TwinType.TERRITORY])
def test_synthetic_frame_trains_a_real_finite_model(twin_type):
    """The frame must train a real sklearn model with a finite (certifiable) R²."""
    from src.digital_twin.training_data import synthetic_training_frame

    df = synthetic_training_frame(twin_type, n_rows=1200, target_col="outcome", seed=5)
    generator = TwinGenerator(twin_type=twin_type, brand=Brand.KISQALI)
    metrics = generator.train(df, target_col="outcome")

    assert generator.model is not None
    assert np.isfinite(metrics.r2_score)
    # A real (mixed categorical/numeric) frame exercises the label encoders.
    pop = generator.generate(n=5, seed=1)
    assert pop.size == 5
