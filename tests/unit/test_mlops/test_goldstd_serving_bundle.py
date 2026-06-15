"""Part 1 (#39) — bundle the FeatureBuilder as the gold-standard model's preprocessor.

The gold-standard cohort models train on the FeatureBuilder-encoded 9-column
frame, so the bare estimator (``CalibratedClassifierCV``) alone cannot serve a
RAW 3-covariate request. The BentoML service unwraps a bundle dict
(``{"model", "preprocessor", "feature_columns"}``) and applies the preprocessor;
this test pins the bundle's *serialization* contract:

  - ``serialize_model_bundle`` writes a pickle whose payload is the dict the
    BentoML ``E2IModelService.__init__`` already unwraps.
  - The bundle round-trips: a RAW 3-covariate row → ``preprocessor.transform``
    → ``model.predict_proba`` yields a finite probability (the verified
    in-process disproof for cohort INITIATION).
  - ``feature_columns`` equals the fitted FeatureBuilder's encoded column order
    (the 9 numeric encoded features SHAP must run over).

No mocks: this fits a REAL FeatureBuilder + CalibratedClassifierCV on a small
synthetic-shaped frame and asserts the real numeric round-trip.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder
from src.mlops.prediction_synthesizer_deploy import serialize_model_bundle


def _fit_initiation_bundle_inputs() -> tuple[object, FeatureBuilder]:
    """Fit a REAL FeatureBuilder + calibrated LR on a synthetic-shaped frame."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "disease_severity": rng.normal(5, 1.5, n).round(2),
            "academic_hcp": rng.integers(0, 2, n),
            "geographic_region": rng.choice(["midwest", "northeast", "south", "west"], n),
            "treatment_initiated": rng.integers(0, 2, n),
        }
    )
    fb = FeatureBuilder(INITIATION)
    X, y = fb.build_from_frame(df)
    model = train_cohort_model(INITIATION, X, y)
    return model, fb


class TestSerializeModelBundle:
    def test_bundle_payload_shape_matches_service_unwrap(self, tmp_path: Path) -> None:
        """The pickle payload is the dict the BentoML service unwraps."""
        model, fb = _fit_initiation_bundle_inputs()
        path = serialize_model_bundle(
            model=model,
            preprocessor=fb,
            feature_columns=fb.feature_columns,
            artifact_dir=tmp_path,
            model_name="initiation_remibrutinib_goldstd_lr_v1",
        )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        assert isinstance(payload, dict)
        assert set(payload.keys()) == {"model", "preprocessor", "feature_columns"}
        assert payload["feature_columns"] == fb.feature_columns
        # The preprocessor is the fitted FeatureBuilder (has transform()).
        assert hasattr(payload["preprocessor"], "transform")

    def test_bundle_feature_columns_are_nine_encoded(self, tmp_path: Path) -> None:
        """feature_columns is the 9 encoded numeric features (the SHAP vector)."""
        model, fb = _fit_initiation_bundle_inputs()
        path = serialize_model_bundle(
            model=model,
            preprocessor=fb,
            feature_columns=list(model.feature_names_in_),
            artifact_dir=tmp_path,
            model_name="m",
        )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        cols = payload["feature_columns"]
        assert len(cols) == 9
        # Encoded names, all numeric-by-construction (one-hot + numeric + __isna).
        assert "geographic_region_northeast" in cols
        assert "disease_severity" in cols
        assert "academic_hcp__isna" in cols

    def test_bundle_round_trips_raw_row_to_finite_probability(self, tmp_path: Path) -> None:
        """RAW 3-covariate row → preprocessor.transform → predict_proba → finite."""
        model, fb = _fit_initiation_bundle_inputs()
        path = serialize_model_bundle(
            model=model,
            preprocessor=fb,
            feature_columns=fb.feature_columns,
            artifact_dir=tmp_path,
            model_name="m",
        )
        with open(path, "rb") as fh:
            payload = pickle.load(fh)

        raw = pd.DataFrame(
            [{"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}]
        )
        encoded = payload["preprocessor"].transform(raw)
        # Encoded columns equal the bundle's feature_columns (the model's contract).
        assert list(encoded.columns) == payload["feature_columns"]
        proba = payload["model"].predict_proba(encoded)[:, 1]
        assert proba.shape == (1,)
        assert np.isfinite(proba[0])
        assert 0.0 <= float(proba[0]) <= 1.0

    def test_returns_absolute_path(self, tmp_path: Path) -> None:
        model, fb = _fit_initiation_bundle_inputs()
        path = serialize_model_bundle(
            model=model,
            preprocessor=fb,
            feature_columns=fb.feature_columns,
            artifact_dir=tmp_path,
            model_name="abs_check",
        )
        assert Path(path).is_absolute()
        assert Path(path).exists()
        assert path.endswith("abs_check.bundle.pkl")
