"""Calibrated boosters (skops trust) and the NGBoost/MAPIE wrappers (cloudpickle) must
survive MLflow log_model -> load_model on the worker's mlflow (3.15.x). Part of #2207.

A. Owner decision after #2280: a calibrated XGBoost / LightGBM is logged with the sklearn
   flavor (the native flavors cannot save a ``CalibratedClassifierCV``), and skops reports
   its booster classes as untrusted. Measured 2026-09-28 (mlflow 3.15.1, skops 0.14.0,
   sklearn 1.6.1, xgboost 3.1.2, lightgbm 4.6.0), identical for sigmoid/isotonic, numpy or
   DataFrame input, with or without early stopping:
     XGBoost:  xgboost.core.Booster, xgboost.sklearn.XGBClassifier
     LightGBM: lightgbm.basic.Booster, lightgbm.sklearn.LGBMClassifier,
               collections.OrderedDict (the default_factory of Booster.best_score)
   Those exact types are trusted; ``collections.OrderedDict`` only together with a
   LightGBM booster. Any other untrusted type still means nothing is trusted.

B. The NGBoost / MAPIE conformal wrappers are documented to serialize via cloudpickle
   (``_get_mlflow_flavor``). Measured on mlflow 3.15.1 through the trainer's own
   ``_log_model_artifact``: nobody asked for cloudpickle, mlflow's default is skops, skops
   refuses the wrapper classes, and the artifact is NOT logged (``model_uri=None``) for all
   four registry entries. The trainer now asks for cloudpickle, as documented.

Real local MLflow stores (sqlite + file artifacts in tmp_path, never prod MLflow), real
fitted models, the real connector. No mocks.
"""

from __future__ import annotations

import collections
from datetime import datetime, timezone

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
from src.mlops.mlflow_connector import MLflowConnector, MLflowRun  # noqa: E402
from src.mlops.skops_trust import (  # noqa: E402
    _default_serialization_format,
    skops_trusted_types_for,
)

# Each round-trip runs mlflow's model-requirements inference (a subprocess that loads the
# model). Measured up to ~47 s (setup + call) on the loaded droplet, against the heavy lane's
# 60 s --timeout, whose thread-method kill takes the whole xdist worker down.
pytestmark = pytest.mark.timeout(180)

_DEFAULT_IS_SKOPS = _default_serialization_format(mlflow) == "skops"
_SKOPS = {} if _DEFAULT_IS_SKOPS else {"serialization_format": "skops"}

XGB_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]
LGBM_TYPES = [
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
]


def _frame(n: int = 700, seed: int = 0):
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


def _booster(kind: str, X, y):
    if kind == "xgboost":
        xgboost = pytest.importorskip("xgboost")
        return xgboost.XGBClassifier(n_estimators=20, max_depth=3, verbosity=0).fit(X, y)
    lightgbm = pytest.importorskip("lightgbm")
    return lightgbm.LGBMClassifier(n_estimators=20, verbose=-1).fit(X, y)


def _calibrated_booster(kind: str, method: str):
    X, y = _frame()
    model, info = apply_post_hoc_calibration(
        _booster(kind, X[:400], y[:400]), X[400:550], y[400:550], method=method
    )
    assert info["calibration_applied"] and info["calibration_method_resolved"] == method
    return model, X[550:]


class _Unlisted:
    """A type skops reports as untrusted and that is NOT on the allowlist."""

    def __init__(self):
        self.value = 1


@pytest.fixture
def local_connector(tmp_path, monkeypatch):
    db = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", db)
    monkeypatch.setattr(MLflowConnector, "_instance", None)
    conn = MLflowConnector(tracking_uri=db)
    assert conn._enabled
    mlflow.set_tracking_uri(db)
    exp = mlflow.create_experiment("boosters", artifact_location=f"file://{tmp_path}/artifacts")
    mlflow.set_experiment(experiment_id=exp)
    yield conn
    monkeypatch.setattr(MLflowConnector, "_instance", None)


def _flavor(uri: str) -> dict:
    local = mlflow.artifacts.download_artifacts(uri)
    with open(f"{local}/MLmodel") as f:
        return yaml.safe_load(f)["flavors"]["sklearn"]


# ---------------------------------------------------------------------------
# A. which types are trusted for a calibrated booster
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("kind, booster_types", [("xgboost", XGB_TYPES), ("lightgbm", LGBM_TYPES)])
def test_a_calibrated_booster_trusts_exactly_what_skops_reports(kind, booster_types, method):
    model, _ = _calibrated_booster(kind, method)
    reported = sorted(skops_io.get_untrusted_types(data=skops_io.dumps(model)))
    assert set(booster_types) <= set(reported)  # the measured types, still what skops reports
    assert skops_trusted_types_for(model) == reported


@pytest.mark.unit
@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
def test_a_calibrated_booster_with_an_unlisted_type_is_trusted_with_nothing(kind):
    model, _ = _calibrated_booster(kind, "sigmoid")
    model.stray_ = _Unlisted()
    reported = skops_io.get_untrusted_types(data=skops_io.dumps(model))
    assert any(t.endswith("_Unlisted") for t in reported)
    assert skops_trusted_types_for(model) == []


@pytest.mark.unit
def test_ordered_dict_is_not_trusted_without_a_lightgbm_booster():
    """OrderedDict is trusted only as part of a LightGBM booster's state.

    skops serializes an OrderedDict *instance* natively; it reports the name only for a
    reference to the class itself, e.g. the ``default_factory`` of LightGBM's
    ``Booster.best_score = defaultdict(OrderedDict)``. Plant that same reference elsewhere.
    """
    model, _ = _calibrated_booster("xgboost", "sigmoid")
    model.stray_ = collections.defaultdict(collections.OrderedDict)
    assert "collections.OrderedDict" in skops_io.get_untrusted_types(data=skops_io.dumps(model))
    assert skops_trusted_types_for(model) == []

    X, y = _frame()
    lr, _ = apply_post_hoc_calibration(
        LogisticRegression(max_iter=1000).fit(X[:400], y[:400]), X[400:550], y[400:550], "sigmoid"
    )
    lr.stray_ = collections.defaultdict(collections.OrderedDict)
    assert "collections.OrderedDict" in skops_io.get_untrusted_types(data=skops_io.dumps(lr))
    assert skops_trusted_types_for(lr) == []


# ---------------------------------------------------------------------------
# A. the real round-trip through the connector (local store)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
@pytest.mark.parametrize("kind", ["xgboost", "lightgbm"])
async def test_a_calibrated_booster_logs_and_loads_back_identically(local_connector, kind, method):
    model, X_test = _calibrated_booster(kind, method)
    with mlflow.start_run() as run:
        uri = await local_connector._log_model(run.info.run_id, model, "model", "sklearn", **_SKOPS)
    assert uri and uri.startswith("models:/"), "log_model must succeed for a calibrated booster"

    flavor = _flavor(uri)
    assert flavor["serialization_format"] == "skops"
    assert sorted(flavor["skops_trusted_types"]) == skops_trusted_types_for(model)

    expected = model.predict_proba(X_test)
    loaded = await local_connector.load_model(uri, flavor="sklearn")
    assert float(np.abs(loaded.predict_proba(X_test) - expected).max()) == 0.0
    assert (
        float(np.abs(mlflow.sklearn.load_model(uri).predict_proba(X_test) - expected).max()) == 0.0
    )


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm, kind", [("XGBoost", "xgboost"), ("LightGBM", "lightgbm")])
async def test_the_trainer_logs_a_calibrated_booster(local_connector, algorithm, kind):
    """The trainer's own path: a calibrated booster -> sklearn flavor -> mlflow's default
    format (skops on the worker's mlflow) with the booster trust."""
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import _log_model_artifact

    model, X_test = _calibrated_booster(kind, "sigmoid")
    with mlflow.start_run() as r:
        run = _run(local_connector, r)
        uri = await _log_model_artifact(run, model, algorithm, kind)
    assert uri
    flavor = _flavor(uri)
    if _DEFAULT_IS_SKOPS:
        assert flavor["serialization_format"] == "skops"
        assert sorted(flavor["skops_trusted_types"]) == skops_trusted_types_for(model)
    loaded = mlflow.sklearn.load_model(uri)
    assert float(np.abs(loaded.predict_proba(X_test) - model.predict_proba(X_test)).max()) == 0.0


# ---------------------------------------------------------------------------
# B. the NGBoost / MAPIE wrappers log with cloudpickle, as documented
# ---------------------------------------------------------------------------


def _run(conn: MLflowConnector, active) -> MLflowRun:
    return MLflowRun(
        run_id=active.info.run_id,
        experiment_id=active.info.experiment_id,
        run_name="lane",
        start_time=datetime.now(timezone.utc),
        connector=conn,
    )


def _wrapper(algorithm: str):
    """Built like the trainer builds it: its model-class resolution + registry framework."""
    pytest.importorskip("ngboost")
    pytest.importorskip("mapie")
    from src.agents.ml_foundation.model_selector.nodes.algorithm_registry import (
        ALGORITHM_REGISTRY,
    )
    from src.mlops.optuna_optimizer import get_model_class

    params = {
        "NGBoost": {"n_estimators": 30, "learning_rate": 0.05},
        "NGBoost_Conformal": {"n_estimators": 30, "learning_rate": 0.05},
        "LightGBM_Conformal": {"n_estimators": 30, "verbose": -1},
        "LogisticRegression_Conformal": {"max_iter": 1000},
    }[algorithm]
    X, y = _frame(900)
    model = get_model_class(algorithm, "binary_classification")(**params)
    model.fit(X.to_numpy()[:700], y.to_numpy()[:700])
    return model, ALGORITHM_REGISTRY[algorithm]["framework"], X.to_numpy()[700:]


_WRAPPERS = ["NGBoost", "NGBoost_Conformal", "LightGBM_Conformal", "LogisticRegression_Conformal"]


@pytest.mark.unit
@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", _WRAPPERS)
async def test_the_trainer_logs_a_wrapper_with_cloudpickle(local_connector, algorithm):
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import _log_model_artifact

    model, framework, X_test = _wrapper(algorithm)
    with mlflow.start_run() as r:
        uri = await _log_model_artifact(_run(local_connector, r), model, algorithm, framework)
    assert uri, f"{algorithm} must be logged (mlflow 3.15 skops default refused it)"
    assert _flavor(uri)["serialization_format"] == "cloudpickle"

    expected = model.predict_proba(X_test)
    loaded = await local_connector.load_model(uri, flavor="sklearn")
    assert type(loaded) is type(model)
    assert float(np.abs(loaded.predict_proba(X_test) - expected).max()) == 0.0


@pytest.mark.unit
def test_only_the_wrappers_ask_for_cloudpickle():
    from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import (
        _sklearn_serialization_kwargs,
    )

    assert _sklearn_serialization_kwargs("ngboost") == {"serialization_format": "cloudpickle"}
    for framework in ("mapie+ngboost", "mapie+lightgbm", "mapie+sklearn"):
        assert _sklearn_serialization_kwargs(framework) == {"serialization_format": "cloudpickle"}
    for framework in ("sklearn", "xgboost", "lightgbm", "econml"):
        assert _sklearn_serialization_kwargs(framework) == {}
        assert _sklearn_serialization_kwargs(framework, LogisticRegression()) == {}

    # a wrapper whose framework label was lost is still recognised by its class
    from src.mlops.wrappers.mapie_wrapper import MapieConformalBinaryClassifier
    from src.mlops.wrappers.ngboost_wrapper import NGBoostBinaryClassifier

    for model in (NGBoostBinaryClassifier(), MapieConformalBinaryClassifier(LogisticRegression())):
        assert _sklearn_serialization_kwargs("sklearn", model) == {
            "serialization_format": "cloudpickle"
        }
