"""#2318 Lane 2: a retrain logs its exact fitted preprocessing + estimator as a serving bundle.

Before #2318 only the estimator was logged, so a retrain candidate could not be served without
a re-fit. The bundle (``sklearn_ct_v1``) is the contract the BentoML sidecar's
ColumnTransformer adapter (Lane 3) loads; it must hold sklearn/numpy/builtins only because the
sidecar cannot import ``src.*``.

The only fake here is the MLflow run boundary (``_Run`` records ``log_artifact``/``set_tags``);
the preprocessor, estimator and bundle bytes are real.
"""

import hashlib
import pickle
from contextlib import asynccontextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression

from src.agents.ml_foundation.model_trainer.nodes.mlflow_logger import (
    _log_serving_bundle,
    log_to_mlflow,
)
from src.agents.ml_foundation.model_trainer.nodes.preprocessor import ModelTrainerPreprocessor
from src.agents.ml_foundation.model_trainer.nodes.serving_bundle import (
    BUNDLE_ARTIFACT_DIR,
    BUNDLE_FILENAME,
    BUNDLE_FORMAT,
    BUNDLE_SHA_TAG,
    build_serving_bundle,
    serialize_bundle,
)

RAW = [
    "disease_severity",
    "academic_hcp",
    "geographic_region",
    "insurance_type",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "rep_detailing_high",
    "sample_dropped",
    "trigger_accepted",
]


def _frame(n=300, seed=0):
    # dtypes as measured on the live goldstd cohort: disease_severity float64,
    # geographic_region/insurance_type object, the rest int64.
    r = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "disease_severity": r.normal(5, 2, n),
            "academic_hcp": r.integers(0, 2, n),
            "geographic_region": r.choice(["northeast", "south", "midwest", "west"], n),
            "insurance_type": r.choice(["commercial", "medicare", "medicaid"], n),
            "age_at_diagnosis": r.integers(20, 80, n),
            "comorbidity_burden": r.integers(0, 5, n),
            "prior_therapy_lines": r.integers(0, 4, n),
            "rep_detailing_high": r.integers(0, 2, n),
            "sample_dropped": r.integers(0, 2, n),
            "trigger_accepted": r.integers(0, 2, n),
        }
    )


def _fitted():
    X = _frame()
    y = (X["disease_severity"] + X["academic_hcp"] > 5.5).astype(int)
    pre = ModelTrainerPreprocessor().fit(X)
    Xt = pre.transform(X)
    # The live candidate's shape (F9): CalibratedClassifierCV(sigmoid) -> FrozenEstimator -> LR.
    lr = LogisticRegression(max_iter=500).fit(Xt, y)
    model = CalibratedClassifierCV(FrozenEstimator(lr), method="sigmoid").fit(Xt, y)
    return pre, model, X


# --------------------------------------------------------------------------- Task 2.1


def test_bundle_is_sklearn_only_and_predicts_identically():
    pre, model, X = _fitted()
    bundle = build_serving_bundle(model=model, preprocessor=pre)
    assert bundle["bundle_format"] == BUNDLE_FORMAT == "sklearn_ct_v1"
    assert isinstance(bundle["preprocessor"], ColumnTransformer)  # not the src.agents wrapper
    assert bundle["keep_columns"] == RAW
    assert set(bundle["numeric_columns"]) == set(RAW) - {"geographic_region", "insurance_type"}
    assert not any(c.startswith(("num__", "cat__")) for c in bundle["feature_columns"])
    assert "geographic_region_south" in bundle["feature_columns"]
    assert len(bundle["feature_columns"]) == pre.transform(X).shape[1] == 15
    import sklearn

    assert bundle["sklearn_version"] == sklearn.__version__
    # Round-trip through bytes: predictions are bit-identical to the in-memory pair.
    blob, sha = serialize_bundle(bundle)
    assert sha == hashlib.sha256(blob).hexdigest()
    loaded = pickle.loads(blob)  # noqa: S301 - test of our own artifact
    np.testing.assert_array_equal(
        loaded["model"].predict_proba(loaded["preprocessor"].transform(X[RAW])),
        model.predict_proba(pre.transform(X)),
    )


def test_keep_columns_follow_fit_order_not_input_order():
    """The sidecar reorders a raw frame by keep_columns; a shuffled frame must score the same."""
    pre, model, X = _fitted()
    bundle = build_serving_bundle(model=model, preprocessor=pre)
    shuffled = X[list(reversed(RAW))]
    np.testing.assert_array_equal(
        bundle["preprocessor"].transform(shuffled[bundle["keep_columns"]]),
        pre.transform(X),
    )


def test_bundle_pickle_references_no_src_module():
    pre, model, _ = _fitted()
    blob, _ = serialize_bundle(build_serving_bundle(model=model, preprocessor=pre))
    assert b"src.agents" not in blob and b"src.mlops" not in blob  # F2: sidecar cannot import them


def test_refuses_an_unfitted_or_foreign_preprocessor():
    _, model, _ = _fitted()
    with pytest.raises(ValueError, match="fitted ModelTrainerPreprocessor"):
        build_serving_bundle(model=model, preprocessor=ModelTrainerPreprocessor())
    with pytest.raises(ValueError, match="fitted ModelTrainerPreprocessor"):
        build_serving_bundle(model=model, preprocessor=object())


def test_refuses_a_preprocessor_fit_without_column_names():
    """A preprocessor fit on a bare ndarray has no raw column contract to serve against."""
    X = _frame()
    y = (X["disease_severity"] > 5).astype(int)
    pre = ModelTrainerPreprocessor().fit(pd.DataFrame(X.select_dtypes("number").to_numpy()))
    model = LogisticRegression(max_iter=500).fit(
        pre.transform(pd.DataFrame(X.select_dtypes("number").to_numpy())), y
    )
    with pytest.raises(ValueError, match="raw column names"):
        build_serving_bundle(model=model, preprocessor=pre)


def test_refuses_a_model_that_does_not_match_the_preprocessor_output():
    """Pairing an estimator with a preprocessor it was not fit on must not produce a bundle."""
    pre, _, X = _fitted()
    y = (X["disease_severity"] > 5).astype(int)
    narrow = LogisticRegression(max_iter=500).fit(pre.transform(X)[:, :5], y)
    with pytest.raises(ValueError, match="expects 5 features"):
        build_serving_bundle(model=narrow, preprocessor=pre)


# --------------------------------------------------------------------------- Task 2.2


class _Run:
    """Records the MLflow run calls the logger makes (the MLflow boundary only)."""

    def __init__(self):
        self.run_id = "run_2318"
        self.artifacts, self.tags = [], {}

    async def log_artifact(self, local_path, artifact_path=None):
        with open(local_path, "rb") as fh:
            self.artifacts.append((fh.read(), artifact_path, local_path))

    async def set_tags(self, tags):
        self.tags.update(tags)

    async def log_params(self, params):
        pass

    async def log_metrics(self, metrics):
        pass

    async def log_model(self, model, name, flavor, **kwargs):
        return "models:/m-2318"


@pytest.mark.asyncio
async def test_logs_bundle_and_its_sha():
    pre, model, X = _fitted()
    run = _Run()
    sha = await _log_serving_bundle(run, {"preprocessor": pre}, model)
    ((blob, art_dir, local),) = run.artifacts
    assert art_dir == BUNDLE_ARTIFACT_DIR == "serving_bundle"
    assert local.endswith("/" + BUNDLE_FILENAME) and BUNDLE_FILENAME == "bundle.pkl"
    assert BUNDLE_SHA_TAG == "e2i.serving_bundle_sha256"
    assert run.tags[BUNDLE_SHA_TAG] == sha == hashlib.sha256(blob).hexdigest()
    loaded = pickle.loads(blob)  # noqa: S301 - test of our own artifact
    np.testing.assert_array_equal(
        loaded["model"].predict_proba(loaded["preprocessor"].transform(X[loaded["keep_columns"]])),
        model.predict_proba(pre.transform(X)),
    )


@pytest.mark.asyncio
async def test_no_preprocessor_logs_nothing_and_says_why(caplog):
    _, model, _ = _fitted()
    run = _Run()
    assert await _log_serving_bundle(run, {}, model) is None
    assert run.artifacts == [] and BUNDLE_SHA_TAG not in run.tags
    assert "not servable as-trained" in caplog.text


@pytest.mark.asyncio
async def test_a_bundle_failure_is_logged_not_raised_and_tags_nothing(caplog):
    pre, _, X = _fitted()
    narrow = LogisticRegression(max_iter=500).fit(
        pre.transform(X)[:, :5], (X["disease_severity"] > 5).astype(int)
    )
    run = _Run()
    assert await _log_serving_bundle(run, {"preprocessor": pre}, narrow) is None
    assert run.artifacts == [] and BUNDLE_SHA_TAG not in run.tags
    assert "Serving bundle NOT logged" in caplog.text


@pytest.mark.asyncio
async def test_log_to_mlflow_logs_the_bundle_of_the_deployed_model():
    """Wiring: the node logs a bundle holding the DEPLOYED (calibrated) model, in the run."""
    pre, model, X = _fitted()
    run = _Run()

    class _Conn:
        async def get_or_create_experiment(self, name, tags=None):
            return "exp_2318"

        @asynccontextmanager
        async def _run_cm(self):
            yield run

        def start_run(self, **kwargs):
            return self._run_cm()

        async def register_model(self, **kwargs):
            return None

    state = {
        "trained_model": model.estimator,  # the raw LR; the calibrated one is what is deployed
        "deployed_model": model,
        "preprocessor": pre,
        "algorithm_name": "LogisticRegression",
        "framework": "sklearn",
        "enable_mlflow": True,
        "register_model": False,
    }
    with patch("src.mlops.mlflow_connector.get_mlflow_connector", return_value=_Conn()):
        result = await log_to_mlflow(state)

    assert result["mlflow_status"] == "success", result
    bundles = [a for a in run.artifacts if a[1] == BUNDLE_ARTIFACT_DIR]
    assert len(bundles) == 1
    blob = bundles[0][0]
    assert run.tags[BUNDLE_SHA_TAG] == hashlib.sha256(blob).hexdigest()
    loaded = pickle.loads(blob)  # noqa: S301 - test of our own artifact
    assert isinstance(loaded["model"], CalibratedClassifierCV)
    np.testing.assert_array_equal(
        loaded["model"].predict_proba(loaded["preprocessor"].transform(X[RAW])),
        model.predict_proba(pre.transform(X)),
    )


# --------------------------------------------------------------------------- codex r1


class _NotSidecarImportable:
    """A class the sidecar cannot import (lives in this test module, not sklearn/numpy)."""

    def __init__(self):
        self.n_features_in_ = 15


def test_serialize_refuses_a_bundle_the_sidecar_cannot_unpickle():
    """sklearn/numpy only is enforced on the bytes, not promised: a foreign class is refused."""
    pre, model, _ = _fitted()
    bundle = build_serving_bundle(model=model, preprocessor=pre)
    bundle["model"].stray_ = _NotSidecarImportable()
    with pytest.raises(ValueError, match="not loadable in the serving sidecar"):
        serialize_bundle(bundle)


class _FailingRun(_Run):
    """The connector swallows MLflow errors and reports them as a False return."""

    def __init__(self, artifact_ok=True, tags_ok=True):
        super().__init__()
        self._artifact_ok, self._tags_ok = artifact_ok, tags_ok

    async def log_artifact(self, local_path, artifact_path=None):
        if not self._artifact_ok:
            return False
        await super().log_artifact(local_path, artifact_path)
        return True

    async def set_tags(self, tags):
        if not self._tags_ok:
            return False
        await super().set_tags(tags)
        return True


@pytest.mark.asyncio
async def test_a_failed_upload_writes_no_sha_tag(caplog):
    """A tag must never claim a bundle that is not in the run."""
    pre, model, _ = _fitted()
    run = _FailingRun(artifact_ok=False)
    assert await _log_serving_bundle(run, {"preprocessor": pre}, model) is None
    assert BUNDLE_SHA_TAG not in run.tags
    assert "Serving bundle NOT logged" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_tag_write_is_not_reported_as_logged(caplog):
    pre, model, _ = _fitted()
    run = _FailingRun(tags_ok=False)
    assert await _log_serving_bundle(run, {"preprocessor": pre}, model) is None
    assert "Serving bundle NOT logged" in caplog.text


@pytest.fixture
def local_mlflow_run(tmp_path, monkeypatch):
    """A real MLflowConnector run on a throwaway sqlite store (never the prod MLflow)."""
    import mlflow

    from src.mlops.mlflow_connector import MLflowConnector, MLflowRun

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(MLflowConnector, "_instance", None)
    conn = MLflowConnector(tracking_uri=uri)
    assert conn._enabled
    mlflow.set_tracking_uri(uri)
    exp = mlflow.create_experiment("l2318", artifact_location=f"file://{tmp_path}/artifacts")
    with mlflow.start_run(experiment_id=exp) as r:
        from datetime import datetime, timezone

        yield MLflowRun(
            run_id=r.info.run_id,
            experiment_id=exp,
            run_name="l2318",
            start_time=datetime.now(timezone.utc),
            connector=conn,
        )
    monkeypatch.setattr(MLflowConnector, "_instance", None)


@pytest.mark.asyncio
async def test_connector_reports_upload_and_tag_success_or_failure(local_mlflow_run, tmp_path):
    import mlflow

    f = tmp_path / "x.bin"
    f.write_bytes(b"x")
    assert await local_mlflow_run.log_artifact(str(f), "d") is True
    assert await local_mlflow_run.log_artifact(str(tmp_path / "missing.bin"), "d") is False
    assert await local_mlflow_run.set_tags({"k": "v"}) is True
    assert mlflow.get_run(local_mlflow_run.run_id).data.tags["k"] == "v"


@pytest.mark.asyncio
async def test_bundle_round_trips_through_a_real_mlflow_run(local_mlflow_run, tmp_path):
    """End to end on a real store: the downloaded file hashes to the run tag and scores identically."""
    import mlflow

    pre, model, X = _fitted()
    sha = await _log_serving_bundle(local_mlflow_run, {"preprocessor": pre}, model)
    assert sha is not None
    assert mlflow.get_run(local_mlflow_run.run_id).data.tags[BUNDLE_SHA_TAG] == sha
    local = mlflow.artifacts.download_artifacts(
        run_id=local_mlflow_run.run_id,
        artifact_path=f"{BUNDLE_ARTIFACT_DIR}/{BUNDLE_FILENAME}",
        dst_path=str(tmp_path / "dl"),
    )
    with open(local, "rb") as fh:
        blob = fh.read()
    assert hashlib.sha256(blob).hexdigest() == sha
    loaded = pickle.loads(blob)  # noqa: S301 - test of our own artifact
    np.testing.assert_array_equal(
        loaded["model"].predict_proba(loaded["preprocessor"].transform(X[RAW])),
        model.predict_proba(pre.transform(X)),
    )
