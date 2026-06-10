"""Unit tests for the prediction_synthesizer deploy plumbing (#840).

These cover the train -> serialize -> manifest path with REAL sklearn models
(no mocks): the deploy module must produce >=2 distinct fitted models for the
CSU treatment-initiation target and serialize them into a deployment manifest
that ``load_clients_from_deployment_manifest_file`` can load back into working
prediction clients. The DB-registration + end-to-end synthesize path is
covered by the faithful integration test (gated E2I_DB_INTEGRATION).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.agents.prediction_synthesizer.clients.inproc_model_client import (
    load_clients_from_deployment_manifest_file,
)
from src.mlops.prediction_synthesizer_deploy import (
    TrainedModel,
    serialize_and_write_manifest,
    train_target_models,
)

# Small cohort keeps the unit test fast + low-memory; the CLI deploy uses 6000.
_SMALL_N = 800


def test_train_target_models_returns_distinct_real_models():
    models = train_target_models(n_total=_SMALL_N, seed=7)
    assert len(models) >= 2, "ensemble needs >=2 models for a non-degenerate prediction"
    for m in models:
        assert isinstance(m, TrainedModel)
        assert m.model_name
        assert m.feature_names, "feature_names must be preserved for the client"
        assert 0.5 < m.auc <= 1.0, f"AUC {m.auc} not sane for {m.model_name}"
        assert hasattr(m.model, "predict_proba")
    # genuinely distinct fits (not the same object/coefficients twice)
    names = [m.model_name for m in models]
    assert len(set(names)) == len(names), "model_names must be unique"
    coefs = [np.asarray(m.model.coef_) for m in models]
    assert not np.allclose(coefs[0], coefs[1]), "the two models must be distinct fits"


@pytest.mark.asyncio
async def test_serialize_and_manifest_roundtrips_to_loadable_clients(tmp_path: Path):
    models = train_target_models(n_total=_SMALL_N, seed=7)
    artifact_dir = tmp_path / "artifacts"
    manifest_path = tmp_path / "deployment_manifest.json"

    uri_map = serialize_and_write_manifest(models, artifact_dir, manifest_path)

    # every model serialized to a real on-disk pickle
    assert set(uri_map.keys()) == {m.model_name for m in models}
    for path in uri_map.values():
        assert Path(path).exists(), f"artifact not written: {path}"

    # the manifest the factory reads loads back into working clients keyed by model_name
    clients = load_clients_from_deployment_manifest_file(str(manifest_path))
    assert set(clients.keys()) == {m.model_name for m in models}

    # the loaded client actually predicts (real inference, not a stub)
    feats = dict.fromkeys(models[0].feature_names, 1.0)
    out = await clients[models[0].model_name].predict("ENTITY_1", feats, "30d")
    assert "prediction" in out
    assert 0.0 <= float(out["prediction"]) <= 1.0
