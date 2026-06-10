"""Phase 3 (G5): factory must inject model clients into prediction_synthesizer.

When `create_agent_registry()` instantiates `PredictionSynthesizerAgent`, it
should call a best-effort loader that returns either real clients (from a
deployment manifest) or an empty dict (when no manifest is configured).

The agent itself already tolerates `model_clients={}` by entering UNVALIDATED
mode (`src/testing/agent_quality_gates.py:_validate_prediction_synthesizer`),
so the contract here is: never raise on missing manifest, always return a
dict, never silently drop a manifest that IS present.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest


class _FakeProbaModel:
    def predict_proba(self, X):  # noqa: D401, ANN001
        import numpy as np

        return np.tile([0.4, 0.6], (X.shape[0], 1))

    def predict(self, X):  # noqa: ANN001
        import numpy as np

        return np.zeros(X.shape[0])


def test_try_load_prod_model_clients_returns_dict_when_no_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No manifest env var, no default file => {} (never raises)."""
    monkeypatch.delenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", raising=False)
    # Pretend the default path is somewhere empty.
    monkeypatch.chdir(tmp_path)

    from src.agents import factory as factory_mod

    result = factory_mod._try_load_prod_model_clients()
    assert result == {}


def test_default_read_path_is_writable_ml_artifacts_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#857 Gap 2: the factory default manifest READ path must be under the
    writable ``data/ml_artifacts/`` volume, not the read-only ``data/`` root.

    The deploy CLI writes the manifest to ``data/ml_artifacts/deployment_manifest.json``
    (the only writable location in the prod container). With no env override,
    ``_try_load_prod_model_clients`` must find it THERE — otherwise the factory
    silently loads ``{}`` clients and the agent fails closed even after a
    successful deploy.
    """
    monkeypatch.delenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    model_path = tmp_path / "viavol.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(_FakeProbaModel(), fh)

    new_dir = tmp_path / "data" / "ml_artifacts"
    new_dir.mkdir(parents=True)
    (new_dir / "deployment_manifest.json").write_text(
        json.dumps({"metadata": {"name": "viavol"}, "model_uri": str(model_path)})
    )

    from src.agents import factory as factory_mod

    clients = factory_mod._try_load_prod_model_clients()
    assert "viavol" in clients, (
        "factory must read the default manifest from data/ml_artifacts/ "
        f"(writable volume); got clients={list(clients)}"
    )


def test_try_load_prod_model_clients_loads_manifest_when_env_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When E2I_MODEL_DEPLOYMENT_MANIFEST_PATH points to a JSON file, load it."""
    model_path = tmp_path / "churn.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(_FakeProbaModel(), fh)

    manifest = {
        "metadata": {"name": "churn_v1"},
        "model_uri": str(model_path),
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", str(manifest_path))

    from src.agents import factory as factory_mod

    clients = factory_mod._try_load_prod_model_clients()
    assert "churn_v1" in clients


def test_try_load_prod_model_clients_swallows_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A broken manifest must not raise; returns {}."""
    monkeypatch.setenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", str(tmp_path / "does-not-exist.json"))
    from src.agents import factory as factory_mod

    assert factory_mod._try_load_prod_model_clients() == {}


def test_create_agent_registry_injects_model_clients_into_synthesizer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """End-to-end: factory passes model_clients= kwarg into the agent."""
    model_path = tmp_path / "m.pkl"
    with model_path.open("wb") as fh:
        pickle.dump(_FakeProbaModel(), fh)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {"name": "m1"}, "model_uri": str(model_path)}))
    monkeypatch.setenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", str(manifest_path))

    from src.agents.factory import create_agent_registry

    registry = create_agent_registry(include_agents=["prediction_synthesizer"])

    agent = registry.get("prediction_synthesizer")
    assert agent is not None, "prediction_synthesizer must be in registry"
    # Agent stores model_clients on the instance.
    assert hasattr(agent, "model_clients")
    assert "m1" in agent.model_clients


def test_create_agent_registry_still_works_with_no_manifest(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Backwards compatibility: empty dict on the agent when no manifest."""
    monkeypatch.delenv("E2I_MODEL_DEPLOYMENT_MANIFEST_PATH", raising=False)
    monkeypatch.chdir(tmp_path)

    from src.agents.factory import create_agent_registry

    registry = create_agent_registry(include_agents=["prediction_synthesizer"])
    agent = registry.get("prediction_synthesizer")
    assert agent is not None
    assert agent.model_clients == {}
