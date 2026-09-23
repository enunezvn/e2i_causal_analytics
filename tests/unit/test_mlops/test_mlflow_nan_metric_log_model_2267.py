"""#2267: a NaN run metric made MLflow 3.x ``log_model`` fail, and the trainer
reported the failure as success.

MLflow 3.x ``log_model`` links the run's metrics to the new LoggedModel by
re-logging every run metric via ``log_batch`` (``Model.log`` ->
``log_model_metrics_for_step``). The SQL store dedups a re-logged metric by
entity equality; ``nan != nan``, so a NaN metric row is re-inserted and trips
the ``metrics`` primary key (``UNIQUE constraint failed ... metrics.is_nan``).
The evaluator emits NaN by design for "undefined" (the calibration slope /
intercept stability guard needs n_pos >= 30 and n_neg >= 30), so every small
test split hit this.

Real MLflow on a file-backed sqlite store (CI's heavy job runs ``mlflow server
--backend-store-uri sqlite:///mlflow.db``); nothing under test is mocked.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, Iterator

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

pytestmark = pytest.mark.timeout(300)


@pytest.fixture
def real_sqlite_mlflow(tmp_path, monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    """A fresh ``MLflowConnector`` singleton on a file-backed sqlite store."""
    import mlflow

    from src.mlops.mlflow_connector import MLflowConnector

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("MLFLOW_ARTIFACT_URI", str(tmp_path / "artifacts"))
    prior_uri = mlflow.get_tracking_uri()
    MLflowConnector._instance = None  # type: ignore[assignment]
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.set_tracking_uri(tracking_uri)

    yield tracking_uri

    while mlflow.active_run() is not None:
        mlflow.end_run()
    MLflowConnector._instance = None  # type: ignore[assignment]
    mlflow.set_tracking_uri(prior_uri)


def _fitted_model() -> LogisticRegression:
    rng = np.random.RandomState(0)
    X = rng.randn(40, 3)
    y = (X[:, 0] > 0).astype(int)
    return LogisticRegression().fit(X, y)


class _UnpicklableModel(LogisticRegression):
    """A fitted estimator MLflow cannot serialise (holds a lock)."""

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()


@pytest.mark.asyncio
async def test_log_model_succeeds_after_a_nan_metric_on_the_run(real_sqlite_mlflow: str) -> None:
    import mlflow
    from mlflow.tracking import MlflowClient

    from src.mlops.mlflow_connector import get_mlflow_connector

    conn = get_mlflow_connector()
    assert conn.enabled, "connector must be on the real sqlite store, not degraded"
    experiment_id = await conn.get_or_create_experiment("nan_metric_2267")

    async with conn.start_run(experiment_id=experiment_id, run_name="nan_2267") as run:
        await run.log_metrics(
            {
                "test_roc_auc": 0.71,
                "test_calibration_slope": float("nan"),
                "test_calibration_intercept": float("nan"),
            }
        )
        model_uri = await run.log_model(model=_fitted_model(), name="model", flavor="sklearn")
        run_id = run.run_id

    assert model_uri is not None, "log_model must not fail because the run carries a NaN metric"
    loaded = mlflow.sklearn.load_model(model_uri)
    assert loaded.predict(np.zeros((1, 3))).shape == (1,)

    logged = MlflowClient(real_sqlite_mlflow).get_run(run_id).data.metrics
    assert logged["test_roc_auc"] == pytest.approx(0.71)
    # NaN is the evaluator's "undefined" marker: recorded as absent, like None.
    assert "test_calibration_slope" not in logged
    assert "test_calibration_intercept" not in logged


def _logger_state(model: Any) -> Dict[str, Any]:
    return {
        "trained_model": model,
        "experiment_id": "exp_2267",
        "experiment_name": "trainer_nan_2267",
        "algorithm_name": "LogisticRegression",
        "problem_type": "binary_classification",
        "framework": "sklearn",
        "best_hyperparameters": {"C": 1.0},
        "enable_mlflow": True,
        "register_model": False,
        "evaluation_metrics": {
            "train_metrics": {"roc_auc": 0.80},
            "validation_metrics": {"roc_auc": 0.75},
            "test_metrics": {
                "roc_auc": 0.71,
                "calibration_slope": float("nan"),
                "calibration_slope_deviation": float("nan"),
            },
        },
    }


@pytest.mark.asyncio
async def test_trainer_logger_returns_a_model_uri_when_test_metrics_carry_nan(
    real_sqlite_mlflow: str,
) -> None:
    """The #2267 CI path: NaN calibration metrics, then the model artifact."""
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import log_to_mlflow

    result = await log_to_mlflow(_logger_state(_fitted_model()))

    assert result["mlflow_status"] == "success"
    assert result["mlflow_model_uri"] is not None


@pytest.mark.asyncio
async def test_trainer_logger_does_not_report_success_when_the_model_is_not_logged(
    real_sqlite_mlflow: str, caplog: pytest.LogCaptureFixture
) -> None:
    """A real model-logging failure is reported as a failure, never as success."""
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import log_to_mlflow

    model = _UnpicklableModel()
    model.fit(np.random.RandomState(0).randn(20, 3), np.array([0, 1] * 10))

    with caplog.at_level(logging.INFO):
        result = await log_to_mlflow(_logger_state(model))

    assert result["mlflow_model_uri"] is None
    assert result["mlflow_status"] != "success"
    assert result["mlflow_run_id"], "the run itself (params, metrics) was still logged"
    assert not any("Successfully logged model" in r.getMessage() for r in caplog.records), (
        "a None model URI must never be logged as a success"
    )
