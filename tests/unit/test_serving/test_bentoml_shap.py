"""Fix #2a (#39) — BentoML ``/shap`` endpoint computes real gold-standard SHAP.

Option B: the gold-standard cohort bundles (model + fitted FeatureBuilder +
encoded ``feature_columns``) live ONLY in the BentoML container's mount, so SHAP
for these models is computed HERE — where the bundle is loaded — not in the API
process. The ``/shap`` endpoint:

  - routes by ``model_name`` (same ``_resolve_active`` pattern as /predict);
  - encodes the RAW covariates via the bundled FeatureBuilder (raw -> encoded);
  - runs ``shap.LinearExplainer`` over the inner LogisticRegression of the
    CalibratedClassifierCV, producing per-ENCODED-feature SHAP values that
    satisfy additivity exactly (base + sum(shap) == inner-LR margin);
  - FAILS CLOSED (``error`` set, empty value maps) for an unknown model_name, a
    non-FeatureBuilder model, or any explainer failure — NO fabricated SHAP.

These tests fit a REAL FeatureBuilder + calibrated LR (no mocks of business
logic) and assert the real SHAP round-trip through the service's /shap path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.mlops.gold_standard_eval.cohort_deployer import train_cohort_model
from src.mlops.gold_standard_eval.cohort_spec import INITIATION, make_patient_spec
from src.mlops.gold_standard_eval.feature_builder import FeatureBuilder


def _fit_bundle(spec, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = 400
    df = pd.DataFrame(
        {
            "disease_severity": rng.normal(5, 1.5, n).round(2),
            "academic_hcp": rng.integers(0, 2, n),
            "geographic_region": rng.choice(["midwest", "northeast", "south", "west"], n),
            spec.label_column: rng.integers(0, 2, n),
        }
    )
    fb = FeatureBuilder(spec)
    X, y = fb.build_from_frame(df)
    model = train_cohort_model(spec, X, y)
    return {"model": model, "preprocessor": fb, "feature_columns": fb.feature_columns}


def _service_with_models(serving_module: Any, models: dict[str, dict]) -> Any:
    service = serving_module.E2IModelService()
    service._model = None
    service._preprocessor = None
    service._feature_columns = None
    service._models = dict(models)
    return service


@pytest.mark.asyncio
class TestShapEndpoint:
    async def test_shap_returns_real_per_encoded_feature_values(self, serving_module: Any) -> None:
        """/shap routed by model_name returns real per-ENCODED-feature SHAP that
        satisfies additivity (base + sum(shap) == inner-LR margin)."""
        bundle = _fit_bundle(make_patient_spec("initiation", "Kisqali"), 7)
        service = _service_with_models(serving_module, {"initiation_kisqali_goldstd_lr_v1": bundle})
        raw = {"disease_severity": 5.26, "academic_hcp": 0, "geographic_region": "northeast"}

        out = await service.shap(
            serving_module.ShapInput(
                model_name="initiation_kisqali_goldstd_lr_v1", raw_features=[raw]
            )
        )

        assert out.error is None
        assert out.explainer_type == "LinearExplainer"
        # One SHAP value per ENCODED feature column.
        assert set(out.shap_values.keys()) == set(bundle["feature_columns"])
        assert all(np.isfinite(v) for v in out.shap_values.values())
        assert out.encoded_feature_columns == list(bundle["feature_columns"])

        # Additivity: base + sum(shap) == the inner LR decision margin over the
        # SAME encoded vector (the verified disproof).
        model = bundle["model"]
        fb = bundle["preprocessor"]
        enc = np.asarray(fb.transform(pd.DataFrame([raw])), dtype=float)
        inner = model.calibrated_classifiers_[0]
        est = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
        margin = float(est.decision_function(enc)[0])
        assert out.base_value + sum(out.shap_values.values()) == pytest.approx(margin, abs=1e-6)

    async def test_shap_top_k_limits_results(self, serving_module: Any) -> None:
        """top_k caps the returned SHAP map by descending |shap|."""
        bundle = _fit_bundle(INITIATION, 3)
        service = _service_with_models(serving_module, {"csu_initiation_goldstd_lr_v1": bundle})
        out = await service.shap(
            serving_module.ShapInput(
                model_name="csu_initiation_goldstd_lr_v1",
                raw_features=[
                    {"disease_severity": 7.2, "academic_hcp": 1, "geographic_region": "west"}
                ],
                top_k=2,
            )
        )
        assert out.error is None
        assert len(out.shap_values) == 2

    async def test_shap_unknown_model_name_fails_closed(self, serving_module: Any) -> None:
        """An unknown model_name returns error (no fabricated SHAP for the
        default/wrong model)."""
        bundle = _fit_bundle(INITIATION, 1)
        service = _service_with_models(serving_module, {"initiation_kisqali_goldstd_lr_v1": bundle})
        out = await service.shap(
            serving_module.ShapInput(
                model_name="does_not_exist_goldstd_lr_v1",
                raw_features=[
                    {"disease_severity": 5.0, "academic_hcp": 0, "geographic_region": "south"}
                ],
            )
        )
        assert out.error is not None
        assert "does_not_exist_goldstd_lr_v1" in out.error
        assert out.shap_values == {}

    async def test_shap_no_feature_builder_fails_closed(self, serving_module: Any) -> None:
        """A bare model with no FeatureBuilder preprocessor fails closed (the raw
        covariates cannot be encoded → no audit-grade SHAP)."""

        class _Bare:
            coef_ = np.array([[0.1, 0.2]])

            def decision_function(self, arr):
                return np.asarray(arr).sum(axis=1)

        service = _service_with_models(
            serving_module,
            {"bare_goldstd_lr_v1": {"model": _Bare(), "preprocessor": None, "feature_columns": []}},
        )
        out = await service.shap(
            serving_module.ShapInput(
                model_name="bare_goldstd_lr_v1", raw_features=[{"disease_severity": 5.0}]
            )
        )
        assert out.error is not None
        assert "FeatureBuilder" in out.error
        assert out.shap_values == {}

    async def test_shap_missing_covariate_fails_closed(self, serving_module: Any) -> None:
        """A missing required RAW covariate fails closed (no fabricated value)."""
        bundle = _fit_bundle(make_patient_spec("initiation", "Kisqali"), 5)
        service = _service_with_models(serving_module, {"initiation_kisqali_goldstd_lr_v1": bundle})
        out = await service.shap(
            serving_module.ShapInput(
                model_name="initiation_kisqali_goldstd_lr_v1",
                raw_features=[
                    {"disease_severity": 5.0, "geographic_region": "south"}
                ],  # no academic_hcp
            )
        )
        assert out.error is not None
        assert "academic_hcp" in out.error
