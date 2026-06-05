"""
Round-trip tests for digital-twin model persistence (#705 H4).

The twin persistence layer must save a trained TwinGenerator to MLflow and
reload it into a *fresh* generator such that generation is bit-for-bit
reproducible. This is the anti-mock contract that replaces the fabricated
``_save_to_mlflow`` stub (which returned ``models:/twin_<type>_<brand>/latest``
without ever logging an artifact).

These tests are hermetic: they point MLflow at a ``file://`` tracking store in
a tmp dir, so no MLflow server/container is required (CI has no ``mlflow:5000``).

The faithful round-trip against the real proxied-artifacts ``http://mlflow:5000``
server is NOT a suite test (the server is unreachable from CI / the host venv).
It is verified manually on the droplet, where the server IS reachable:

    docker exec -i e2i_api python scripts/train_twin_model.py \
        --twin-type hcp --brand Remibrutinib --synthetic
    # then POST /digital-twin/simulate and confirm a 200 (not 503).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.digital_twin.models.twin_models import Brand, TwinType
from src.digital_twin.twin_generator import TwinGenerator


def _synthetic_hcp_frame(n: int = 1200, seed: int = 7) -> pd.DataFrame:
    """Build a training frame with the HCP default feature schema + a target."""
    rng = np.random.default_rng(seed)
    specialties = ["oncology", "cardiology", "neurology", "endocrinology"]
    practice_types = ["academic", "community", "private"]
    regions = ["northeast", "south", "midwest", "west"]
    channels = ["email", "field", "virtual"]
    adoption = ["early", "majority", "laggard"]
    tiers = ["A", "B", "C"]

    decile = rng.integers(1, 11, size=n)
    engagement = rng.uniform(0, 1, size=n)
    df = pd.DataFrame(
        {
            "specialty": rng.choice(specialties, size=n),
            "years_experience": rng.integers(1, 40, size=n),
            "practice_type": rng.choice(practice_types, size=n),
            "practice_size": rng.integers(1, 50, size=n),
            "region": rng.choice(regions, size=n),
            "decile": decile,
            "priority_tier": rng.choice(tiers, size=n),
            "total_patient_volume": rng.integers(50, 5000, size=n),
            "target_patient_volume": rng.integers(10, 1000, size=n),
            "digital_engagement_score": engagement,
            "preferred_channel": rng.choice(channels, size=n),
            "last_interaction_days": rng.integers(0, 365, size=n),
            "interaction_frequency": rng.uniform(0, 10, size=n),
            "adoption_stage": rng.choice(adoption, size=n),
            "peer_influence_score": rng.uniform(0, 1, size=n),
        }
    )
    # A learnable target so the model is non-degenerate.
    df["outcome"] = (
        0.4 * df["digital_engagement_score"]
        + 0.03 * df["decile"]
        + 0.0001 * df["total_patient_volume"]
        + rng.normal(0, 0.05, size=n)
    )
    return df


def _trained_generator() -> TwinGenerator:
    g = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
    g.train(_synthetic_hcp_frame(), target_col="outcome")
    return g


@pytest.fixture()
def file_tracking(tmp_path, monkeypatch):
    uri = f"file://{tmp_path}/mlruns"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def test_save_returns_real_run_and_uri_not_fabricated(file_tracking):
    from src.digital_twin import twin_persistence

    g = _trained_generator()
    ref = twin_persistence.save_twin_artifacts(g, experiment="twin_unit")

    assert ref.run_id, "a real MLflow run_id must be returned"
    assert ref.model_uri, "a real model_uri must be returned"
    # Anti-mock: must NOT be the old fabricated stub pattern.
    assert ref.model_uri != f"models:/twin_{TwinType.HCP.value}_{Brand.KISQALI.value}/latest"
    assert ref.model_uri.startswith(("models:/", "runs:/"))


def test_save_rejects_untrained_generator(file_tracking):
    from src.digital_twin import twin_persistence

    g = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
    with pytest.raises(ValueError):
        twin_persistence.save_twin_artifacts(g, experiment="twin_unit")


def test_round_trip_predictions_identical(file_tracking):
    """A reloaded generator must reproduce predictions bit-for-bit.

    This catches the round-trip hazards the audit flagged: lost scaler /
    label_encoders, feature-column order drift, missing feature stats.
    """
    from src.digital_twin import twin_persistence

    g = _trained_generator()
    pop_before = g.generate(n=8, seed=123)
    before = [t.baseline_outcome for t in pop_before.twins]

    ref = twin_persistence.save_twin_artifacts(g, experiment="twin_unit")

    fresh = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
    assert fresh.model is None
    ok = twin_persistence.hydrate_generator(fresh, ref.model_uri, ref.run_id)
    assert ok is True
    assert fresh.model is not None
    # State fully restored.
    assert fresh.feature_columns == g.feature_columns
    assert fresh.target_column == g.target_column
    assert fresh.scaler is not None
    assert set(fresh.label_encoders) == set(g.label_encoders)

    pop_after = fresh.generate(n=8, seed=123)
    after = [t.baseline_outcome for t in pop_after.twins]
    assert after == pytest.approx(before, rel=1e-9, abs=1e-9)


def test_hydrate_returns_false_on_missing_artifact(file_tracking):
    from src.digital_twin import twin_persistence

    fresh = TwinGenerator(twin_type=TwinType.HCP, brand=Brand.KISQALI)
    ok = twin_persistence.hydrate_generator(
        fresh, "runs:/0000000000000000/model", "0000000000000000"
    )
    assert ok is False
    assert fresh.model is None
