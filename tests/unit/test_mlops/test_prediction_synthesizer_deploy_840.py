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
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_MANIFEST_PATH,
    TrainedModel,
    serialize_and_write_manifest,
    train_target_models,
)

# Small cohort keeps the unit test fast + low-memory; the CLI deploy uses 6000.
_SMALL_N = 800


def test_default_paths_target_writable_ml_artifacts_volume():
    """#857 Gap 2: deploy-CLI defaults must write under ``data/ml_artifacts/``.

    In the prod api container ``/app/data`` is a READ-ONLY image dir; only named
    volumes mounted under it (e.g. ``data/ml_artifacts``, the ``e2i_ml_artifacts``
    volume) are writable. The old defaults (``data/model_artifacts`` and
    ``data/deployment_manifest.json``) live at the read-only ``data/`` root, so
    the documented runbook ``python -m src.mlops.prediction_synthesizer_deploy``
    failed with ``OSError: Read-only file system``. The manifest must also share
    the artifact volume so it persists across redeploys and the factory can read
    it back.
    """
    assert DEFAULT_ARTIFACT_DIR.parts[:2] == ("data", "ml_artifacts"), (
        f"artifact dir must be under the writable data/ml_artifacts volume, "
        f"got {DEFAULT_ARTIFACT_DIR}"
    )
    assert DEFAULT_MANIFEST_PATH.parts[:2] == ("data", "ml_artifacts"), (
        f"manifest must be under the writable data/ml_artifacts volume, got {DEFAULT_MANIFEST_PATH}"
    )


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


@pytest.mark.asyncio
async def test_register_model_row_refuses_production_without_training_provenance():
    """#2259: a production row of unknown provenance is refused before any I/O.

    No client and no artifact are needed: the refusal precedes both, so nothing is written and
    the artifact check cannot mask it. The real-server twin is in
    tests/unit/test_database/learning_loop/test_ml_registry_promotion_gate_realdb.py.
    """
    from src.mlops.prediction_synthesizer_deploy import register_model_row

    with pytest.raises(ValueError, match="training_provenance"):
        await register_model_row(
            None,
            experiment_id="exp",
            model_name="m",
            model_version="1.0",
            algorithm="logistic_regression",
            artifact_path="/nonexistent/m.pkl",
            auc=0.7,
            feature_count=1,
            stage="production",
        )
