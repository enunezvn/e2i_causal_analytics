"""Part 2 (#39) — BentoML service serves RAW covariates via a FeatureBuilder.

The gold-standard cohort bundles carry a fitted ``FeatureBuilder`` as their
``preprocessor`` and the 9 ENCODED column names as ``feature_columns``. A caller
supplies the 3 RAW covariates (``disease_severity``, ``academic_hcp``,
``geographic_region``) — including the categorical ``geographic_region`` STRING —
and the service must:

  - apply ``preprocessor.transform(raw_df)`` (raw → 9 encoded numeric) before
    ``model.predict_proba`` (the verified in-process disproof);
  - expose BOTH the raw covariate names (``keep_columns``) and the encoded
    ``feature_columns`` from ``/model_info`` so the explain route knows which
    RAW names to fetch and which ENCODED vector SHAP runs over;
  - keep the legacy bare-estimator / numeric-matrix path unchanged.

These tests fit a REAL FeatureBuilder + calibrated LR (no mocks) and assert the
real numeric round-trip through the service's raw path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import INITIATION
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder


def _fit_real_bundle() -> tuple[Any, FeatureBuilder]:
    rng = np.random.default_rng(7)
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


class TestRawFeaturesSchema:
    def test_raw_features_field_accepts_dict_rows(self, serving_module: Any) -> None:
        inp = serving_module.PredictionInput(
            raw_features=[
                {"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}
            ]
        )
        assert inp.raw_features == [
            {"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}
        ]
        # Legacy numeric matrix still defaults empty.
        assert inp.features == []


@pytest.mark.asyncio
class TestRawCovariatePredict:
    async def test_raw_covariates_predict_real_probability(self, serving_module: Any) -> None:
        """RAW {disease_severity, academic_hcp, geographic_region} → real proba."""
        model, fb = _fit_real_bundle()
        service = serving_module.E2IModelService()
        service._model = model
        service._preprocessor = fb
        service._feature_columns = fb.feature_columns

        out = await service.predict(
            serving_module.PredictionInput(
                raw_features=[
                    {"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}
                ]
            )
        )
        assert out.feature_source == "raw_covariates"
        assert len(out.probabilities) == 1
        p = out.probabilities[0]
        assert np.isfinite(p) and 0.0 <= p <= 1.0
        # Matches the in-process round-trip exactly.
        raw = pd.DataFrame(
            [{"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}]
        )
        expected = float(model.predict_proba(fb.transform(raw))[:, 1][0])
        assert abs(p - expected) < 1e-9

    async def test_raw_covariates_categorical_string_does_not_crash(
        self, serving_module: Any
    ) -> None:
        """The categorical geographic_region STRING must be one-hot-encoded, not
        coerced to float (which would crash / fabricate)."""
        model, fb = _fit_real_bundle()
        service = serving_module.E2IModelService()
        service._model = model
        service._preprocessor = fb
        service._feature_columns = fb.feature_columns

        out = await service.predict(
            serving_module.PredictionInput(
                raw_features=[
                    {"disease_severity": 7.2, "academic_hcp": 1, "geographic_region": "west"},
                    {"disease_severity": 2.1, "academic_hcp": 0, "geographic_region": "south"},
                ]
            )
        )
        assert len(out.probabilities) == 2
        assert all(np.isfinite(p) for p in out.probabilities)


class TestModelInfoServingContract:
    def test_model_info_exposes_keep_columns_and_feature_columns(self, serving_module: Any) -> None:
        """/model_info exposes RAW keep_columns AND encoded feature_columns."""
        model, fb = _fit_real_bundle()
        service = serving_module.E2IModelService()
        service._model = model
        service._preprocessor = fb
        service._feature_columns = fb.feature_columns
        service._model_tag = None

        import asyncio

        info = asyncio.run(service.model_info())
        # Encoded order (SHAP vector) — unchanged contract.
        assert info["feature_columns"] == fb.feature_columns
        # NEW: raw covariate names the caller must supply.
        assert info["keep_columns"] == list(fb.keep_columns)
        assert "geographic_region" in info["keep_columns"]

    def test_model_info_keep_columns_none_for_bare_estimator(self, serving_module: Any) -> None:
        """A bare estimator (no FeatureBuilder preprocessor) exposes keep_columns
        = None — there is no raw covariate contract, only the encoded order."""

        class _Bare:
            feature_names_in_ = np.array(["x1", "x2"])

            def predict(self, arr):
                return np.zeros(len(arr))

        service = serving_module.E2IModelService()
        service._model = _Bare()
        service._preprocessor = None
        service._feature_columns = None
        service._model_tag = None

        import asyncio

        info = asyncio.run(service.model_info())
        assert info["keep_columns"] is None
        assert info["feature_columns"] == ["x1", "x2"]


@pytest.mark.asyncio
class TestBackwardCompatNumericPath:
    async def test_legacy_numeric_matrix_still_predicts(self, serving_module: Any) -> None:
        """The legacy pre-encoded numeric ``features`` path is unchanged."""

        class _LinModel:
            feature_names_in_ = np.array(["a", "b"])

            def predict(self, arr):
                return (np.asarray(arr).sum(axis=1) > 0).astype(int)

            def predict_proba(self, arr):
                s = np.asarray(arr).sum(axis=1)
                p = 1 / (1 + np.exp(-s))
                return np.column_stack([1 - p, p])

        service = serving_module.E2IModelService()
        service._model = _LinModel()
        service._preprocessor = None
        service._feature_columns = ["a", "b"]

        out = await service.predict(serving_module.PredictionInput(features=[[1.0, 2.0]]))
        assert out.feature_source == "user_provided"
        assert len(out.probabilities) == 1
