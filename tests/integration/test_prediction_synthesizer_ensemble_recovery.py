"""Shard 08 T5 — faithful prediction_synthesizer >=2-client recovery.

Proves the assumption that >=2 manifest models -> confidence escapes the
single-model 0.30 cap, model_agreement>0.5, and the prediction is NOT
CANNOT_ASSESS (ensemble_combiner.py caps confidence at min(0.30,...) and sets
risk_level=CANNOT_ASSESS only when models_succeeded < 2).

Gated E2I_DB_INTEGRATION=1, -n0, OOM-safe. Builds a real 2-model manifest from a
synthetic frame in tmp_path and runs the real agent in-process (no DB needed —
the gate flag keeps this off the default unit lane).

STALE-PLAN CORRECTION (verified against source, do not guess):
PredictionSynthesizerOutput has NO ``risk_assessment`` attribute — that name does
not exist on the Output contract (agent.py:52) nor in state.py. The risk level
lives only as text inside ``prediction_summary`` ("Risk Assessment: <LEVEL>").
The robust escape signals are ``ensemble_prediction['confidence']`` /
``['model_agreement']`` (an EnsemblePrediction TypedDict — subscript, not attr),
``models_succeeded``, and the absence of "CANNOT_ASSESS" in prediction_summary.
include_context=False uses the simple graph (no context store) so status is a
clean "completed"; the ensemble escape holds on either graph.
"""

import os

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("E2I_DB_INTEGRATION") != "1",
    reason="faithful integration: set E2I_DB_INTEGRATION=1",
)


@pytest.mark.asyncio
async def test_two_model_manifest_escapes_cannot_assess(tmp_path):
    from src.agents.prediction_synthesizer.agent import PredictionSynthesizerAgent
    from src.agents.prediction_synthesizer.clients.inproc_model_client import (
        load_clients_from_deployment_manifest,
    )
    from src.ml.synthetic.artifacts.ensemble_trainer import (
        build_deployment_manifest,
        train_ensemble_for_cohort_brand,
    )

    rng = np.random.default_rng(1)
    X = pd.DataFrame(
        {
            "disease_severity": rng.integers(0, 3, 500).astype(float),
            "age_at_diagnosis": rng.normal(55, 10, 500),
            "treatment_arm": rng.integers(0, 2, 500),  # canonical column
        }
    )
    y = (0.3 + 0.4 * X["treatment_arm"] > rng.uniform(0, 1, 500)).astype(int)
    paths = train_ensemble_for_cohort_brand(
        X, y, cohort="initiation", brand="Kisqali", out_dir=tmp_path
    )
    manifest = build_deployment_manifest({("initiation", "Kisqali"): paths})
    clients = load_clients_from_deployment_manifest(manifest)
    assert len(clients) >= 2

    agent = PredictionSynthesizerAgent(
        model_clients=clients, enable_memory=False, enable_dspy=False, enable_opik=False
    )
    out = await agent.synthesize(
        entity_id="HCP_synth_1",
        prediction_target="conversion",
        features={"disease_severity": 2.0, "age_at_diagnosis": 60.0, "treatment_arm": 1},
        ensemble_method="weighted",
        include_context=False,
    )
    assert out.models_succeeded >= 2, f"only {out.models_succeeded} model(s) succeeded"
    ep = out.ensemble_prediction
    assert ep is not None
    assert ep["confidence"] > 0.4, f"confidence {ep['confidence']} still capped — single-model path"
    assert ep["model_agreement"] > 0.5
    # risk_level lives only inside the summary text; >=2 models must not yield CANNOT_ASSESS.
    assert "CANNOT_ASSESS" not in (out.prediction_summary or "")
