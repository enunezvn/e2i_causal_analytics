"""#2280: a calibrated sklearn model must survive MLflow log_model -> load_model.

mlflow 3.15.1 (the worker's version) serializes sklearn models with skops by default,
and skops 0.14.0 refuses the private calibration classes a ``CalibratedClassifierCV``
holds unless they are passed as ``skops_trusted_types``. Every calibrated pipeline model
failed to log, and the trainer reported "Successfully logged model: None".

These tests use a real local MLflow store (sqlite + file artifacts in tmp_path) and real
calibrated models built by the trainer's own ``apply_post_hoc_calibration``. The
connector now logs sklearn models in the skops format explicitly (so every mlflow version
behaves like the worker's), trusting only the exact sklearn calibration classes skops
reports for THIS model. mlflow records the trusted list in the model's flavor config, so
``mlflow.sklearn.load_model`` and ``mlflow.pyfunc.load_model`` load it back unchanged.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
import yaml

mlflow = pytest.importorskip("mlflow")
skops_io = pytest.importorskip("skops.io")

from sklearn.linear_model import LogisticRegression  # noqa: E402

from src.agents.ml_foundation.model_trainer.nodes.advanced_validation import (  # noqa: E402
    apply_post_hoc_calibration,
)
from src.mlops.mlflow_connector import MLflowConnector  # noqa: E402
from src.mlops.skops_trust import (  # noqa: E402
    TRUSTED_SKLEARN_CALIBRATION_TYPES,
    _default_serialization_format,
    skops_trusted_types_for,
)

# The worker/CI run mlflow 3.15.x, whose sklearn default IS skops. An older local mlflow
# defaults to cloudpickle; there the caller asks for skops explicitly, which exercises the
# same trust path.
_DEFAULT_IS_SKOPS = _default_serialization_format(mlflow) == "skops"
_SKOPS = {} if _DEFAULT_IS_SKOPS else {"serialization_format": "skops"}


def _frame(n: int = 600, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "severity": rng.normal(size=n),
            "age": rng.normal(60, 10, size=n),
            "flag": rng.integers(0, 2, n),
        }
    ).astype(float)
    logit = 1.4 * X["severity"] + 0.03 * (X["age"] - 60) + 0.5 * X["flag"] - 0.6
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int))
    return X, y


def _calibrated(method: str):
    X, y = _frame()
    base = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X[:400], y[:400])
    model, info = apply_post_hoc_calibration(base, X[400:500], y[400:500], method=method)
    assert info["calibration_applied"] and info["calibration_method_resolved"] == method
    return model, X[500:]


class _Unlisted:
    """A type skops reports as untrusted and that is NOT on the allowlist."""

    def __init__(self):
        self.value = 1


# ---------------------------------------------------------------------------
# which types are trusted
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_trusted_types_are_exactly_what_skops_reports_for_the_model(method):
    model, _ = _calibrated(method)
    reported = skops_io.get_untrusted_types(data=skops_io.dumps(model))
    assert reported, "the calibrated model must need trusting (the #2280 failure)"
    assert set(reported) <= TRUSTED_SKLEARN_CALIBRATION_TYPES
    assert skops_trusted_types_for(model) == sorted(reported)


@pytest.mark.unit
def test_an_uncalibrated_model_needs_no_trust():
    X, y = _frame()
    assert skops_trusted_types_for(LogisticRegression(max_iter=1000).fit(X, y)) == []


@pytest.mark.unit
def test_a_type_outside_the_allowlist_is_never_trusted():
    model, _ = _calibrated("sigmoid")
    model.stray_ = _Unlisted()
    reported = skops_io.get_untrusted_types(data=skops_io.dumps(model))
    assert any(t.endswith("_Unlisted") for t in reported)
    assert skops_trusted_types_for(model) == []  # all-or-nothing: mlflow then refuses loudly


@pytest.mark.unit
def test_the_allowlist_is_only_sklearn_calibration_internals():
    assert TRUSTED_SKLEARN_CALIBRATION_TYPES == frozenset(
        {"sklearn.calibration._CalibratedClassifier", "sklearn.calibration._SigmoidCalibration"}
    )


# ---------------------------------------------------------------------------
# the real round-trip through the connector (local store, never prod MLflow)
# ---------------------------------------------------------------------------


@pytest.fixture
def local_connector(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    monkeypatch.setattr(MLflowConnector, "_instance", None)
    conn = MLflowConnector(tracking_uri=f"sqlite:///{tmp_path}/mlflow.db")
    assert conn._enabled
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")
    exp = mlflow.create_experiment("lane2280", artifact_location=f"file://{tmp_path}/artifacts")
    mlflow.set_experiment(experiment_id=exp)
    yield conn
    monkeypatch.setattr(MLflowConnector, "_instance", None)


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
async def test_a_calibrated_model_logs_and_loads_back_identically(local_connector, method):
    model, X_test = _calibrated(method)
    with mlflow.start_run() as run:
        uri = await local_connector._log_model(run.info.run_id, model, "model", "sklearn", **_SKOPS)
    assert uri, "log_model must succeed for a calibrated model"

    local = mlflow.artifacts.download_artifacts(uri)
    flavor = yaml.safe_load(open(f"{local}/MLmodel"))["flavors"]["sklearn"]
    assert flavor["serialization_format"] == "skops"
    assert sorted(flavor["skops_trusted_types"]) == skops_trusted_types_for(model)

    expected = model.predict_proba(X_test)
    loaded = await local_connector.load_model(uri, flavor="sklearn")
    np.testing.assert_array_equal(loaded.predict_proba(X_test), expected)
    np.testing.assert_array_equal(
        np.asarray(mlflow.pyfunc.load_model(uri).predict(X_test)), model.predict(X_test)
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_explicit_caller_format_is_respected(local_connector):
    model, X_test = _calibrated("sigmoid")
    with mlflow.start_run() as run:
        uri = await local_connector._log_model(
            run.info.run_id, model, "model", "sklearn", serialization_format="cloudpickle"
        )
    local = mlflow.artifacts.download_artifacts(uri)
    flavor = yaml.safe_load(open(f"{local}/MLmodel"))["flavors"]["sklearn"]
    assert flavor["serialization_format"] == "cloudpickle"


# ---------------------------------------------------------------------------
# honest failure reporting in the trainer's logger
# ---------------------------------------------------------------------------


class _RunThatFailsToLog:
    """The connector swallows the mlflow error and returns None (it logs it)."""

    async def log_model(self, **kwargs):
        return None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_a_failed_model_log_is_reported_as_a_failure(caplog):
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import _log_model_artifact

    with caplog.at_level(logging.INFO):
        uri = await _log_model_artifact(
            _RunThatFailsToLog(), object(), "LogisticRegression", "sklearn"
        )
    assert uri is None
    messages = [r.getMessage() for r in caplog.records]
    assert not any("Successfully logged model" in m for m in messages), messages
    assert any(
        r.levelno >= logging.ERROR and "NOT logged" in r.getMessage() for r in caplog.records
    )


# ---------------------------------------------------------------------------
# the other direct log site of a calibrated estimator: the risk_score trainer
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_risk_score_trainer_logs_its_calibrated_estimator(tmp_path, monkeypatch, method):
    from src.agents.prediction_synthesizer.risk_score.risk_score_trainer import RiskScoreTrainer

    db = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db)
    mlflow.set_tracking_uri(db)
    mlflow.create_experiment("risk2280", artifact_location=f"file://{tmp_path}/artifacts")
    model, X_test = _calibrated(method)

    run_id = RiskScoreTrainer(enable_mlflow=True)._mlflow_log(
        experiment="risk2280",
        run_name=None,
        model_type="logistic_regression",
        best_params={},
        metrics={},
        tags={},
        reliability_png=None,
        feature_importance={},
        honest_failures=[],
        estimator=model,
    )
    assert run_id
    local = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
    flavor = yaml.safe_load(open(f"{local}/MLmodel"))["flavors"]["sklearn"]
    if _DEFAULT_IS_SKOPS:  # the worker's mlflow: skops, with the calibration trust
        assert flavor["serialization_format"] == "skops"
        assert sorted(flavor["skops_trusted_types"]) == skops_trusted_types_for(model)
    loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    np.testing.assert_array_equal(loaded.predict_proba(X_test), model.predict_proba(X_test))


@pytest.mark.unit
@pytest.mark.asyncio
async def test_the_trainer_status_says_the_model_was_not_logged():
    """log_to_mlflow must not report ``success`` with ``mlflow_model_uri=None``."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import log_to_mlflow

    run = AsyncMock()
    run.run_id = "run_2280"
    run.log_model = AsyncMock(return_value=None)  # what the connector returns on failure
    run.__aenter__ = AsyncMock(return_value=run)
    run.__aexit__ = AsyncMock(return_value=None)
    conn = AsyncMock()
    conn.get_or_create_experiment = AsyncMock(return_value="exp_2280")
    conn.start_run = MagicMock(return_value=run)
    state = {
        "trained_model": object(),
        "experiment_id": "exp_2280",
        "algorithm_name": "LogisticRegression",
        "problem_type": "binary_classification",
        "framework": "sklearn",
        "best_hyperparameters": {},
        "evaluation_metrics": {"test_metrics": {"roc_auc": 0.83}},
        "enable_mlflow": True,
        "register_model": False,
    }
    with patch("src.mlops.mlflow_connector.get_mlflow_connector", return_value=conn):
        result = await log_to_mlflow(state)
    assert result["mlflow_model_uri"] is None
    assert result["mlflow_status"] == "model_not_logged"
    assert result["mlflow_run_id"] == "run_2280"


# ---------------------------------------------------------------------------
# codex r1 HIGH: the in-process prediction client must serve the calibrated
# probability, not pyfunc's class label
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_inproc_client_serves_the_calibrated_probability(local_connector):
    from src.agents.prediction_synthesizer.clients.inproc_model_client import (
        InProcessModelClient,
        _load_model_from_uri,
    )

    model, X_test = _calibrated("sigmoid")
    with mlflow.start_run() as run:
        uri = await local_connector._log_model(run.info.run_id, model, "model", "sklearn", **_SKOPS)
    loaded = _load_model_from_uri(uri)
    assert hasattr(loaded, "predict_proba")
    row = X_test.iloc[0]
    out = await InProcessModelClient(loaded, feature_names=list(X_test.columns)).predict(
        "e1", row.to_dict(), "30d"
    )
    expected = float(model.predict_proba(X_test.iloc[[0]])[0][1])
    assert out["prediction"] == pytest.approx(expected)
    assert 0.0 < out["prediction"] < 1.0  # a probability, not a 0/1 label


@pytest.mark.unit
def test_a_cloudpickle_caller_is_left_alone():
    """codex r1 MED: the NGBoost/MAPIE wrappers are documented to rely on cloudpickle
    (_get_mlflow_flavor); the helper never changes a caller's format."""
    from src.mlops.skops_trust import sklearn_log_model_kwargs

    model, _ = _calibrated("sigmoid")
    assert sklearn_log_model_kwargs(mlflow, model, {"serialization_format": "cloudpickle"}) == {
        "serialization_format": "cloudpickle"
    }
    assert sklearn_log_model_kwargs(mlflow, model, {"serialization_format": "skops"}) == {
        "serialization_format": "skops",
        "skops_trusted_types": skops_trusted_types_for(model),
    }
    explicit = {"serialization_format": "skops", "skops_trusted_types": ["x.Y"]}
    assert sklearn_log_model_kwargs(mlflow, model, dict(explicit)) == explicit


# ---------------------------------------------------------------------------
# codex r2: a calibrated booster is a CalibratedClassifierCV, not a native booster
# ---------------------------------------------------------------------------


class _RecordingRun:
    def __init__(self):
        self.flavors = []

    async def log_model(self, *, model, name, flavor, **kwargs):
        self.flavors.append(flavor)
        return "models:/m-recorded"


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm, framework", [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")])
async def test_a_calibrated_booster_is_logged_with_the_sklearn_flavor(algorithm, framework):
    """mlflow.xgboost.log_model(CalibratedClassifierCV) fails ('no attribute save_model',
    measured on 3.15.1): the deployed object decides the flavor, not the algorithm name."""
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import _log_model_artifact

    model, _ = _calibrated("sigmoid")  # the wrapper type is what matters here
    run = _RecordingRun()
    assert await _log_model_artifact(run, model, algorithm, framework) == "models:/m-recorded"
    assert run.flavors == ["sklearn"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_an_uncalibrated_booster_keeps_its_native_flavor():
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import _log_model_artifact

    run = _RecordingRun()
    await _log_model_artifact(run, object(), "XGBoost", "xgboost")
    assert run.flavors == ["xgboost"]


@pytest.mark.unit
def test_only_calibrated_models_are_inspected(monkeypatch):
    """codex r2 MED: no second full skops serialization of every sklearn model."""
    import skops.io as sio

    calls = []
    real = sio.dumps
    monkeypatch.setattr(sio, "dumps", lambda m: calls.append(type(m).__name__) or real(m))
    X, y = _frame()
    assert skops_trusted_types_for(LogisticRegression(max_iter=1000).fit(X, y)) == []
    assert calls == []
    model, _ = _calibrated("sigmoid")
    assert skops_trusted_types_for(model)
    assert calls == ["CalibratedClassifierCV"]


@pytest.mark.unit
def test_a_calibrated_booster_is_not_trusted_with_its_booster_classes():
    """Measured: skops also reports xgboost.core.Booster / xgboost.sklearn.XGBClassifier.
    Those are outside the allowlist, so nothing is trusted and mlflow refuses loudly."""
    xgboost = pytest.importorskip("xgboost")
    X, y = _frame()
    base = xgboost.XGBClassifier(n_estimators=5, max_depth=2, verbosity=0).fit(X[:400], y[:400])
    model, _ = apply_post_hoc_calibration(base, X[400:500], y[400:500], method="sigmoid")
    reported = skops_io.get_untrusted_types(data=skops_io.dumps(model))
    assert "xgboost.sklearn.XGBClassifier" in reported
    assert skops_trusted_types_for(model) == []
