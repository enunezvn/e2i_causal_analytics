"""Tests for the SHAP flavor-agnostic MLflow loader.

These tests round-trip real models through a SQLite-backed MLflow tracking
URI so we exercise the actual flavor resolution path (xgboost / lightgbm /
sklearn / pyfunc) rather than mocking it. The compute_shap entry point is
also exercised under three failure modes:

1. **Native flavor** — every supported algorithm logs and reloads.
2. **In-memory passthrough** — when ``state["loaded_model"]`` is set, every
   ``load_model`` shim must remain unused (we wire monkeypatched loaders to
   raise to prove this).
3. **Stale URI / unreachable tracking** — ``get_model_info`` raising
   ``MlflowException`` falls through to ``pyfunc.load_model``.

Section C verification gate from ``.claude/plans/pre_phase2_unblockers.md``.
"""

from __future__ import annotations

from typing import Any, Tuple

import mlflow
import numpy as np
import pandas as pd
import pytest
from mlflow.exceptions import MlflowException
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.feature_analyzer.nodes import shap_computer
from src.agents.ml_foundation.feature_analyzer.nodes.shap_computer import (
    _load_model_flavor_agnostic,
    compute_shap,
)


@pytest.fixture(scope="module")
def mlflow_sqlite_tracking(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Module-scoped MLflow tracking URI backed by SQLite.

    A live MLflow server is not required — this URI is local and ephemeral.
    The previous tracking URI is restored at module teardown so other test
    modules running in the same session are unaffected.
    """
    tmp_dir = tmp_path_factory.mktemp("mlflow_tracking")
    db_path = tmp_dir / "mlflow.db"
    artifact_dir = tmp_dir / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    tracking_uri = f"sqlite:///{db_path}"

    prior_uri = mlflow.get_tracking_uri()
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("shap_flavor_loader_tests")
    yield tracking_uri
    mlflow.set_tracking_uri(prior_uri)


def _make_toy_dataset(n_samples: int = 50) -> Tuple[pd.DataFrame, pd.Series]:
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    columns = [f"feat_{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=columns), pd.Series(y, name="target")


def _train_and_log_sklearn(X: pd.DataFrame, y: pd.Series) -> str:
    model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model", input_example=X.iloc[:2])
        return f"runs:/{run.info.run_id}/model"


def _train_and_log_logistic(X: pd.DataFrame, y: pd.Series) -> str:
    model = LogisticRegression(max_iter=200, random_state=42).fit(X, y)
    with mlflow.start_run() as run:
        mlflow.sklearn.log_model(model, name="model", input_example=X.iloc[:2])
        return f"runs:/{run.info.run_id}/model"


def _train_and_log_xgboost(X: pd.DataFrame, y: pd.Series) -> str:
    import xgboost as xgb

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=2,
        random_state=42,
        eval_metric="logloss",
    ).fit(X, y)
    with mlflow.start_run() as run:
        mlflow.xgboost.log_model(model, name="model", input_example=X.iloc[:2])
        return f"runs:/{run.info.run_id}/model"


def _train_and_log_lightgbm(X: pd.DataFrame, y: pd.Series) -> str:
    import lightgbm as lgb

    model = lgb.LGBMClassifier(
        n_estimators=10,
        max_depth=2,
        random_state=42,
        verbose=-1,
    ).fit(X, y)
    with mlflow.start_run() as run:
        mlflow.lightgbm.log_model(model, name="model", input_example=X.iloc[:2])
        return f"runs:/{run.info.run_id}/model"


_LOGGERS = {
    "sklearn_rf": _train_and_log_sklearn,
    "logistic_regression": _train_and_log_logistic,
    "xgboost": _train_and_log_xgboost,
    "lightgbm": _train_and_log_lightgbm,
}


class TestFlavorAgnosticLoader:
    """Direct unit tests for ``_load_model_flavor_agnostic``."""

    @pytest.mark.parametrize("flavor", list(_LOGGERS.keys()))
    def test_loads_each_native_flavor(self, mlflow_sqlite_tracking: str, flavor: str) -> None:
        """Every supported native flavor must round-trip through the loader."""
        X, y = _make_toy_dataset()
        model_uri = _LOGGERS[flavor](X, y)
        loaded = _load_model_flavor_agnostic(model_uri)
        # The flavor-specific loader returns the underlying estimator (not a
        # pyfunc wrapper); .predict must accept the original DataFrame.
        preds = loaded.predict(X)
        assert preds.shape == (len(X),)

    def test_falls_back_to_pyfunc_when_get_model_info_fails(
        self,
        mlflow_sqlite_tracking: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stale URI / unreachable tracking → fall through to pyfunc.

        We monkeypatch ``get_model_info`` to raise ``MlflowException``; the
        loader must then call ``pyfunc.load_model`` and surface its result.
        ``pyfunc`` here is also monkeypatched so we don't depend on the URI
        actually resolving — we just confirm the fallback path is taken.
        """

        def _raise(_uri: str) -> Any:
            raise MlflowException("simulated stale URI")

        sentinel = object()

        def _fake_pyfunc_load(_uri: str) -> Any:
            return sentinel

        monkeypatch.setattr(mlflow.models, "get_model_info", _raise)
        monkeypatch.setattr(mlflow.pyfunc, "load_model", _fake_pyfunc_load)

        result = _load_model_flavor_agnostic("runs:/does-not-exist/model")
        assert result is sentinel


@pytest.mark.asyncio
class TestComputeSHAPFlavorPaths:
    """compute_shap exercised end-to-end through the flavor-agnostic loader."""

    @pytest.mark.parametrize("flavor", list(_LOGGERS.keys()))
    async def test_compute_shap_logged_model(
        self, mlflow_sqlite_tracking: str, flavor: str
    ) -> None:
        """SHAP must produce non-empty output for every supported flavor."""
        X, y = _make_toy_dataset(n_samples=50)
        model_uri = _LOGGERS[flavor](X, y)
        state = {
            "model_uri": model_uri,
            "experiment_id": f"exp_{flavor}",
            "max_samples": 50,
            "X_sample": X.values,
            "feature_columns": list(X.columns),
        }
        result = await compute_shap(state)
        assert "error" not in result, f"{flavor} compute_shap error: {result.get('error')}"
        assert result.get("samples_analyzed", 0) > 0
        assert result.get("top_features"), f"{flavor} returned empty top_features"

    async def test_compute_shap_uses_in_memory_model(
        self,
        mlflow_sqlite_tracking: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``state["loaded_model"]`` is set, no MLflow loader runs.

        We monkeypatch every loader plus ``_load_model_flavor_agnostic`` to
        raise; if any path other than the in-memory passthrough fires, the
        test fails loudly rather than silently masking a regression.
        """
        X, y = _make_toy_dataset(n_samples=50)
        # Train a real RF directly — no MLflow round-trip.
        model = RandomForestClassifier(n_estimators=10, random_state=42).fit(X, y)

        def _explode(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("loader was called despite in-memory model on state")

        monkeypatch.setattr(shap_computer, "_load_model_flavor_agnostic", _explode)
        monkeypatch.setattr(mlflow.sklearn, "load_model", _explode)
        monkeypatch.setattr(mlflow.xgboost, "load_model", _explode)
        monkeypatch.setattr(mlflow.lightgbm, "load_model", _explode)
        monkeypatch.setattr(mlflow.pyfunc, "load_model", _explode)

        # ``mlflow.get_run`` is also called by compute_shap when the URI is
        # ``runs:/...`` to fetch run metadata. Stub it so the in-memory path
        # doesn't have to round-trip the (non-existent) run.
        monkeypatch.setattr(
            mlflow,
            "get_run",
            lambda _run_id: type(
                "R",
                (),
                {
                    "info": type("I", (), {"run_id": "memrun"})(),
                    "data": type("D", (), {"params": {}})(),
                },
            )(),
        )

        state = {
            "model_uri": "runs:/memrun/model",
            "experiment_id": "exp_in_memory",
            "max_samples": 50,
            "X_sample": X.values,
            "feature_columns": list(X.columns),
            "loaded_model": model,
        }
        result = await compute_shap(state)
        assert "error" not in result, f"compute_shap error: {result.get('error')}"
        assert result.get("samples_analyzed", 0) > 0
        # The result must surface the same model object (passthrough).
        assert result.get("loaded_model") is model


@pytest.fixture(autouse=True)
def _seed_numpy() -> None:
    """Pin numpy's global RNG for stable assertion ordering across tests."""
    np.random.seed(42)
