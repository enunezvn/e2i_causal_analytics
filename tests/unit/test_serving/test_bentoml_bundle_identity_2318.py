"""#2318 Lane 3 — sidecar bundle identity + the ``sklearn_ct_v1`` raw-covariate adapter.

* Identity: FS discovery records each bundle file's sha256 and the service reports
  it (``/model_info``, ``/predict``, ``/predict_batch``, ``/shap``), so activation
  can prove WHICH file is being served (OD-3: the comparator is the served bundle,
  identified by sha256). ``model_id`` stays the serving name — existing callers
  and tests pin it by equality.
* Duplicate names under the serving root fail CLOSED: ``os.walk`` order is not a
  contract, so two files for one name would otherwise be served at random (F1).
* ``sklearn_ct_v1`` (OD-1 = B): a retrain's bundle carries the trainer's own fitted
  ``ColumnTransformer``. The adapter gives it the same raw-covariate contract the
  FeatureBuilder bundles have, and predictions equal the bundle's own
  ``model.predict_proba(preprocessor.transform(raw))`` exactly.

The ``sklearn_ct_v1`` dict is built inline with sklearn only (no dependency on
Lane 2's builder). The ColumnTransformer mirrors
``ModelTrainerPreprocessor._build_pipeline`` (transformer names, remainder
passthrough, ``verbose_feature_names_out=False``), and the estimator is the live
candidate's shape: ``CalibratedClassifierCV(FrozenEstimator(LR), method="sigmoid")``.
Real fitted sklearn objects throughout — no mocks of business logic.
"""

from __future__ import annotations

import asyncio
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import make_patient_spec
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder

NAME = "initiation_kisqali_goldstd_lr_v1"
NUM = [
    "disease_severity",
    "academic_hcp",
    "age_at_diagnosis",
    "comorbidity_burden",
    "prior_therapy_lines",
    "rep_detailing_high",
    "sample_dropped",
    "trigger_accepted",
]
CAT = ["geographic_region", "insurance_type"]


def _raw_frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
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


def _ct_bundle(seed: int = 0) -> dict[str, Any]:
    X = _raw_frame(seed=seed)
    y = (
        (X["disease_severity"] + X["academic_hcp"] + (X["geographic_region"] == "south")) > 5.5
    ).astype(int)
    ct = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    [("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]
                ),
                NUM,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                CAT,
            ),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    ).fit(X)
    encoded = ct.transform(X)
    inner = LogisticRegression(max_iter=500).fit(encoded, y)
    model = CalibratedClassifierCV(FrozenEstimator(inner), method="sigmoid").fit(encoded, y)
    return {
        "bundle_format": "sklearn_ct_v1",
        "model": model,
        "preprocessor": ct,
        "keep_columns": [str(c) for c in ct.feature_names_in_],
        "numeric_columns": list(NUM),
        "feature_columns": [str(n).split("__", 1)[-1] for n in ct.get_feature_names_out()],
        "sklearn_version": "1.6.1",
    }


def _feature_builder_bundle(seed: int = 1) -> dict[str, Any]:
    spec = make_patient_spec("initiation", "Kisqali")
    df = _raw_frame(seed=seed)
    df[spec.label_column] = np.random.default_rng(seed).integers(0, 2, len(df))
    fb = FeatureBuilder(spec)
    X, y = fb.build_from_frame(df)
    return {
        "model": train_cohort_model(spec, X, y),
        "preprocessor": fb,
        "feature_columns": fb.feature_columns,
    }


def _write(root: Path, subdir: str, name: str, bundle: dict[str, Any]) -> str:
    d = root / subdir
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{name}.bundle.pkl"
    p.write_bytes(pickle.dumps(bundle))
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _service(serving_module: Any, models: dict[str, dict[str, Any]]) -> Any:
    service = serving_module.E2IModelService()
    service._model = None
    service._preprocessor = None
    service._feature_columns = None
    service._models = dict(models)
    return service


def _row(i: int = 0) -> dict[str, Any]:
    rec = _raw_frame(n=8, seed=42).iloc[i].to_dict()
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in rec.items()}


# --------------------------------------------------------------------------- identity


def test_fs_discovery_records_the_file_sha(serving_module: Any, tmp_path: Path) -> None:
    sha = _write(tmp_path, "initiation", NAME, _feature_builder_bundle())
    found = serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    assert found[NAME]["bundle_sha256"] == sha


def test_duplicate_names_under_the_root_fail_closed(
    serving_module: Any, tmp_path: Path, caplog: Any
) -> None:
    """F1: os.walk order is not a contract; two files for one name must not be served at random."""
    _write(tmp_path, "initiation", NAME, _feature_builder_bundle())
    _write(tmp_path, "archive", NAME, _feature_builder_bundle(seed=7))
    other = "persistence_kisqali_goldstd_lr_v1"
    _write(tmp_path, "persistence", other, _feature_builder_bundle(seed=3))
    found = serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    assert NAME not in found
    assert other in found  # one duplicated name does not sink the others
    assert "duplicate" in caplog.text.lower() and "archive" in caplog.text


def test_model_info_and_predict_report_the_served_sha(serving_module: Any, tmp_path: Path) -> None:
    sha = _write(tmp_path, "initiation", NAME, _ct_bundle())
    service = _service(
        serving_module, serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    )

    info = asyncio.run(service.model_info(serving_module.ModelInfoInput(model_name=NAME)))
    assert info["bundle_sha256"] == sha
    assert info["model_id"] == NAME  # unchanged contract: the serving name

    out = asyncio.run(
        service.predict(serving_module.PredictionInput(model_name=NAME, raw_features=[_row()]))
    )
    assert out.error is None
    assert out.model_id == NAME
    assert out.bundle_sha256 == sha

    batch = asyncio.run(
        service.predict_batch(
            serving_module.BatchPredictionInput(
                batch_id="b1", model_name=NAME, raw_features=[_row(0), _row(1)]
            )
        )
    )
    assert batch.bundle_sha256 == sha

    shap_out = asyncio.run(
        service.shap(serving_module.ShapInput(model_name=NAME, raw_features=[_row()]))
    )
    assert shap_out.error is None
    assert shap_out.bundle_sha256 == sha


def test_legacy_default_path_reports_no_sha(serving_module: Any) -> None:
    service = _service(serving_module, {})
    info = asyncio.run(service.model_info())
    assert info.get("bundle_sha256") is None


# ------------------------------------------------------------------ sklearn_ct_v1 adapter


def test_ct_bundle_raw_predict_equals_the_bundle_exactly(
    serving_module: Any, tmp_path: Path
) -> None:
    bundle = _ct_bundle()
    _write(tmp_path, "initiation", NAME, bundle)
    service = _service(
        serving_module, serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    )

    rows = [_row(i) for i in range(8)]
    out = asyncio.run(
        service.predict(serving_module.PredictionInput(model_name=NAME, raw_features=rows))
    )
    assert out.error is None, out.error
    expected = bundle["model"].predict_proba(bundle["preprocessor"].transform(pd.DataFrame(rows)))[
        :, 1
    ]
    np.testing.assert_array_equal(np.asarray(out.probabilities), expected)
    assert out.encoded_feature_columns == bundle["feature_columns"]

    # Request key order must not matter: the adapter selects keep_columns in fit order.
    shuffled = [dict(reversed(list(r.items()))) for r in rows]
    out2 = asyncio.run(
        service.predict(serving_module.PredictionInput(model_name=NAME, raw_features=shuffled))
    )
    np.testing.assert_array_equal(np.asarray(out2.probabilities), expected)


def test_ct_bundle_model_info_exposes_the_raw_contract(serving_module: Any, tmp_path: Path) -> None:
    bundle = _ct_bundle()
    _write(tmp_path, "initiation", NAME, bundle)
    service = _service(
        serving_module, serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    )
    info = asyncio.run(service.model_info(serving_module.ModelInfoInput(model_name=NAME)))
    assert info["keep_columns"] == bundle["keep_columns"]
    assert info["feature_columns"] == bundle["feature_columns"]
    assert "geographic_region_south" in info["feature_columns"]


def test_ct_bundle_shap_is_additive_over_the_encoded_vector(
    serving_module: Any, tmp_path: Path
) -> None:
    bundle = _ct_bundle()
    _write(tmp_path, "initiation", NAME, bundle)
    service = _service(
        serving_module, serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    )
    row = _row(3)
    out = asyncio.run(service.shap(serving_module.ShapInput(model_name=NAME, raw_features=[row])))
    assert out.error is None, out.error
    assert out.encoded_feature_columns == bundle["feature_columns"]
    enc = bundle["preprocessor"].transform(pd.DataFrame([row]))
    inner = bundle["model"].calibrated_classifiers_[0].estimator
    margin = float(inner.decision_function(enc)[0])
    assert out.base_value + sum(out.shap_values.values()) == pytest.approx(margin, abs=1e-6)


def test_ct_bundle_string_in_numeric_column_fails_closed(
    serving_module: Any, tmp_path: Path
) -> None:
    _write(tmp_path, "initiation", NAME, _ct_bundle())
    service = _service(
        serving_module, serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    )
    bad = dict(_row(), disease_severity="high")
    with pytest.raises(RuntimeError, match="Numeric covariate 'disease_severity'"):
        asyncio.run(
            service.predict(serving_module.PredictionInput(model_name=NAME, raw_features=[bad]))
        )


def test_ct_bundle_missing_covariate_fails_closed(serving_module: Any, tmp_path: Path) -> None:
    _write(tmp_path, "initiation", NAME, _ct_bundle())
    service = _service(
        serving_module, serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    )
    row = _row()
    del row["insurance_type"]
    with pytest.raises(RuntimeError, match="insurance_type"):
        asyncio.run(
            service.predict(serving_module.PredictionInput(model_name=NAME, raw_features=[row]))
        )


def test_ct_bundle_with_inconsistent_contract_is_not_served(
    serving_module: Any, tmp_path: Path
) -> None:
    """A bundle whose declared feature_columns disagree with its fitted preprocessor would
    label SHAP values with the wrong names — refuse to load it rather than serve it."""
    bad = _ct_bundle()
    bad["feature_columns"] = bad["feature_columns"][:-1]
    _write(tmp_path, "initiation", NAME, bad)
    assert NAME not in serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))


def test_unknown_bundle_format_is_not_served(serving_module: Any, tmp_path: Path) -> None:
    bundle = _ct_bundle()
    bundle["bundle_format"] = "sklearn_ct_v2"
    _write(tmp_path, "initiation", NAME, bundle)
    assert NAME not in serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))


def test_feature_builder_bundles_are_unchanged(serving_module: Any, tmp_path: Path) -> None:
    """The existing 12 served bundles (no bundle_format key) keep their FeatureBuilder path."""
    bundle = _feature_builder_bundle()
    _write(tmp_path, "initiation", NAME, bundle)
    found = serving_module._discover_goldstd_bundles_from_fs(str(tmp_path))
    assert isinstance(found[NAME]["preprocessor"], FeatureBuilder)
