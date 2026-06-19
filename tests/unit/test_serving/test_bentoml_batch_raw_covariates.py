"""Batch RAW-covariate path: ``predict_batch`` accepts ``raw_features`` (#cohort-scoring).

Mirrors ``test_bentoml_raw_covariates.py`` but for the BATCH endpoint, which the
cohort-scoring API path uses to score a whole holdout cohort in ONE chunked call
(rather than N single-predict round-trips — patient holdout is ~5k rows/brand).

The batch raw path must:
  - accept ``raw_features`` (List[Dict]) + an optional ``model_name`` (#39 routing);
  - encode each row via the bundled FeatureBuilder and return PER-ROW probabilities;
  - match the in-process vectorized round-trip exactly;
  - fail closed on a missing required covariate;
  - keep the legacy pre-encoded numeric ``features`` matrix path unchanged.

Fits a REAL FeatureBuilder + calibrated LR (no mocks).
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


class TestBatchRawFeaturesSchema:
    def test_batch_input_accepts_raw_features_and_model_name(self, serving_module: Any) -> None:
        inp = serving_module.BatchPredictionInput(
            batch_id="b1",
            raw_features=[
                {"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}
            ],
            model_name="initiation_kisqali_goldstd_lr_v1",
        )
        assert inp.raw_features == [
            {"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}
        ]
        assert inp.model_name == "initiation_kisqali_goldstd_lr_v1"
        # Legacy numeric matrix still defaults empty.
        assert inp.features == []


@pytest.mark.asyncio
class TestBatchRawCovariatePredict:
    async def test_batch_raw_covariates_return_per_row_probabilities(
        self, serving_module: Any
    ) -> None:
        """RAW rows -> one vectorized transform -> per-row real probabilities."""
        model, fb = _fit_real_bundle()
        service = serving_module.E2IModelService()
        service._model = model
        service._preprocessor = fb
        service._feature_columns = fb.feature_columns

        rows = [
            {"disease_severity": 7.2, "academic_hcp": 1, "geographic_region": "west"},
            {"disease_severity": 2.1, "academic_hcp": 0, "geographic_region": "south"},
            {"disease_severity": 5.0, "academic_hcp": 1, "geographic_region": "northeast"},
        ]
        out = await service.predict_batch(
            serving_module.BatchPredictionInput(batch_id="b1", raw_features=rows)
        )
        assert out.total_samples == 3
        assert len(out.predictions) == 3
        assert len(out.probabilities) == 3
        assert all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in out.probabilities)
        # Matches the in-process VECTORIZED round-trip exactly (single transform).
        expected = model.predict_proba(fb.transform(pd.DataFrame(rows)))[:, 1].tolist()
        for got, exp in zip(out.probabilities, expected, strict=True):
            assert abs(got - exp) < 1e-9

    async def test_batch_raw_missing_required_covariate_fails_closed(
        self, serving_module: Any
    ) -> None:
        """A row omitting a required keep_column must fail closed (no zero-fill)."""
        model, fb = _fit_real_bundle()
        service = serving_module.E2IModelService()
        service._model = model
        service._preprocessor = fb
        service._feature_columns = fb.feature_columns

        with pytest.raises(RuntimeError):
            await service.predict_batch(
                serving_module.BatchPredictionInput(
                    batch_id="b1",
                    raw_features=[{"disease_severity": 5.61, "geographic_region": "northeast"}],
                )
            )


@pytest.mark.asyncio
class TestBatchNumericBackwardCompat:
    async def test_legacy_numeric_batch_still_predicts(self, serving_module: Any) -> None:
        """The legacy pre-encoded numeric ``features`` matrix path is unchanged."""

        class _LinModel:
            feature_names_in_ = np.array(["a", "b"])

            def predict(self, arr):
                return (np.asarray(arr).sum(axis=1) > 0).astype(int)

        service = serving_module.E2IModelService()
        service._model = _LinModel()
        service._preprocessor = None
        service._feature_columns = ["a", "b"]

        out = await service.predict_batch(
            serving_module.BatchPredictionInput(batch_id="b1", features=[[1.0, 2.0], [-1.0, -2.0]])
        )
        assert out.total_samples == 2
        assert len(out.predictions) == 2
