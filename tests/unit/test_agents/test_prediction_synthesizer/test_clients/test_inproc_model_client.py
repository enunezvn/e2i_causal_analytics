"""Unit tests for the in-process model client (Phase 3 / G5).

The InProcessModelClient is the promoted form of `Tier0ModelClient` from
`scripts/run_tier1_5_test.py`. It wraps a trained scikit-learn-style model
(anything exposing ``predict`` and optionally ``predict_proba``) so the
prediction_synthesizer agent can call it via the standard
``async predict(entity_id, features, time_horizon) -> dict`` protocol.

Also covers ``load_clients_from_deployment_manifest`` which is the bridge
between model_deployer output and prediction_synthesizer.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Test fixtures: lightweight fake models
# ---------------------------------------------------------------------------


class _FakeBinaryProbaModel:
    """Mimics a fitted sklearn binary classifier with predict_proba."""

    feature_names_in_ = np.array(["age", "income", "tenure"])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        # Deterministic: positive prob = clipped mean of features / 10
        means = X.mean(axis=1)
        pos = np.clip(means / 10.0, 0.0, 1.0)
        return np.column_stack([1 - pos, pos])

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class _FakeRegressionModel:
    """Mimics a regression model (no predict_proba)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X.sum(axis=1).astype(float)


# ---------------------------------------------------------------------------
# InProcessModelClient
# ---------------------------------------------------------------------------


class TestInProcessModelClient:
    @pytest.mark.asyncio
    async def test_predict_with_predict_proba_returns_canonical_shape(self) -> None:
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            InProcessModelClient,
        )

        model = _FakeBinaryProbaModel()
        client = InProcessModelClient(model=model, model_id="churn_v1")

        result = await client.predict(
            entity_id="patient_42",
            features={"age": 5.0, "income": 5.0, "tenure": 5.0},
            time_horizon="90d",
        )

        assert result["model_id"] == "churn_v1"
        assert 0.0 <= result["prediction"] <= 1.0
        assert isinstance(result["proba"], list)
        assert len(result["proba"]) == 2
        assert pytest.approx(sum(result["proba"]), abs=1e-6) == 1.0
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["features_used"] == ["age", "income", "tenure"]
        assert "latency_ms" in result

    @pytest.mark.asyncio
    async def test_predict_with_regression_model_returns_scalar_no_proba(self) -> None:
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            InProcessModelClient,
        )

        client = InProcessModelClient(
            model=_FakeRegressionModel(),
            model_id="ltv_v1",
            feature_names=["a", "b"],
        )

        result = await client.predict(
            entity_id="e1", features={"a": 1.5, "b": 2.5}, time_horizon="30d"
        )

        assert result["prediction"] == pytest.approx(4.0)
        assert result["proba"] is None

    @pytest.mark.asyncio
    async def test_predict_handles_nan_inputs_gracefully(self) -> None:
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            InProcessModelClient,
        )

        client = InProcessModelClient(model=_FakeBinaryProbaModel(), model_id="m1")

        result = await client.predict(
            entity_id="e1",
            features={"age": float("nan"), "income": float("inf"), "tenure": 1.0},
            time_horizon="30d",
        )

        # NaN/Inf coerced to 0.0 — model still returns a finite prediction.
        assert np.isfinite(result["prediction"])

    @pytest.mark.asyncio
    async def test_predict_returns_error_shape_on_failure(self) -> None:
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            InProcessModelClient,
        )

        class _Boom:
            def predict(self, X: np.ndarray) -> np.ndarray:
                raise RuntimeError("model broken")

        client = InProcessModelClient(model=_Boom(), model_id="broken", feature_names=["x"])

        result = await client.predict(entity_id="e1", features={"x": 1.0}, time_horizon="30d")

        assert "error" in result
        assert result["prediction"] == 0.5  # safe default
        assert result["confidence"] < 0.5


# ---------------------------------------------------------------------------
# load_clients_from_deployment_manifest
# ---------------------------------------------------------------------------


class TestLoadClientsFromDeploymentManifest:
    def test_empty_manifest_returns_empty_dict(self) -> None:
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            load_clients_from_deployment_manifest,
        )

        assert load_clients_from_deployment_manifest({}) == {}

    def test_loads_local_pickle_when_uri_is_file_path(self, tmp_path: Path) -> None:
        """Manifest with a local pickle path should yield a usable client.

        We deliberately exercise the file:// / pickle path because MLflow
        registry mocking is environment-heavy; the loader's contract is
        "given a URI, return a client" — pickle is the offline fallback.
        """
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            InProcessModelClient,
            load_clients_from_deployment_manifest,
        )

        model_path = tmp_path / "churn.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(_FakeBinaryProbaModel(), fh)

        manifest = {
            "metadata": {"name": "churn_v1"},
            "spec": {
                "models": {
                    "churn_v1": {"model_uri": str(model_path)},
                }
            },
        }
        clients = load_clients_from_deployment_manifest(manifest)

        assert set(clients) == {"churn_v1"}
        assert isinstance(clients["churn_v1"], InProcessModelClient)

    def test_loads_top_level_model_uri_shape(self, tmp_path: Path) -> None:
        """Manifest with a flat `model_uri` key (single-model) is supported."""
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            load_clients_from_deployment_manifest,
        )

        model_path = tmp_path / "ltv.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(_FakeRegressionModel(), fh)

        manifest = {
            "metadata": {"name": "ltv_v1"},
            "model_uri": str(model_path),
        }
        clients = load_clients_from_deployment_manifest(manifest)

        assert "ltv_v1" in clients

    def test_invalid_uri_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """A bad URI must not crash the loader; it logs + returns {}."""
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            load_clients_from_deployment_manifest,
        )

        manifest = {
            "metadata": {"name": "missing"},
            "model_uri": str(tmp_path / "does_not_exist.pkl"),
        }
        clients = load_clients_from_deployment_manifest(manifest)
        assert clients == {}

    def test_manifest_from_json_file_round_trip(self, tmp_path: Path) -> None:
        """`load_clients_from_deployment_manifest_file` reads a JSON manifest."""
        from src.agents.prediction_synthesizer.clients.inproc_model_client import (
            load_clients_from_deployment_manifest_file,
        )

        model_path = tmp_path / "churn.pkl"
        with model_path.open("wb") as fh:
            pickle.dump(_FakeBinaryProbaModel(), fh)

        manifest = {
            "metadata": {"name": "churn_v1"},
            "model_uri": str(model_path),
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        clients = load_clients_from_deployment_manifest_file(manifest_path)
        assert "churn_v1" in clients
