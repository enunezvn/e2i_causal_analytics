"""Increment 2 (#39) — multi-model BentoML serving + routing by model_name.

The service loads a DICT of gold-standard bundles keyed by serving name (the
registry ``model_name``) and routes ``/predict`` and ``/model_info`` by a
``model_name`` field in the request. Two different model_names return two
DIFFERENT real probabilities (true routing, not a single shared model). The
legacy single-model / auto-discover path remains the default + fallback (tier0,
legacy numeric/Feast contracts) and must NOT regress.

Fits REAL FeatureBuilders + calibrated LRs (no mocks of business logic).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

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
            # T9: the 4 prognostic drivers for the 7-covariate persistence/discontinuation
            # cohorts (initiation's FeatureBuilder ignores them via its 3-col allowlist).
            "insurance_type": rng.choice(["commercial", "medicare", "medicaid"], n),
            "age_at_diagnosis": rng.integers(18, 85, n),
            "comorbidity_burden": rng.integers(0, 6, n),
            "prior_therapy_lines": rng.integers(0, 4, n),
            spec.label_column: rng.integers(0, 2, n),
        }
    )
    fb = FeatureBuilder(spec)
    X, y = fb.build_from_frame(df)
    model = train_cohort_model(spec, X, y)
    return {"model": model, "preprocessor": fb, "feature_columns": fb.feature_columns}


def _service_with_models(serving_module: Any, models: dict[str, dict]) -> Any:
    """Build a service with a multi-model registry, no real discovery."""
    service = serving_module.E2IModelService()
    # Reset any auto-discovered legacy default so the test is deterministic.
    service._model = None
    service._preprocessor = None
    service._feature_columns = None
    service._models = dict(models)
    return service


class TestMultiModelRegistry:
    def test_available_models_listed_in_model_info(self, serving_module: Any) -> None:
        models = {
            "initiation_remibrutinib_goldstd_lr_v1": _fit_bundle(
                make_patient_spec("initiation", "Remibrutinib"), 1
            ),
            "persistence_fabhalta_goldstd_lr_v1": _fit_bundle(
                make_patient_spec("persistence", "Fabhalta"), 2
            ),
        }
        service = _service_with_models(serving_module, models)
        import asyncio

        info = asyncio.run(service.model_info())
        assert set(info["available_models"]) == set(models.keys())


@pytest.mark.asyncio
class TestRoutingByModelName:
    async def test_two_model_names_two_real_probabilities(self, serving_module: Any) -> None:
        """/predict routed by model_name returns the routed model's real proba —
        two different models give two different probabilities for the same raw row."""
        m_init = _fit_bundle(make_patient_spec("initiation", "Remibrutinib"), 1)
        m_pers = _fit_bundle(make_patient_spec("persistence", "Fabhalta"), 2)
        service = _service_with_models(
            serving_module,
            {
                "initiation_remibrutinib_goldstd_lr_v1": m_init,
                "persistence_fabhalta_goldstd_lr_v1": m_pers,
            },
        )
        raw_init = {"disease_severity": 5.61, "academic_hcp": 0, "geographic_region": "northeast"}
        # T9: persistence carries 4 extra prognostic drivers. The service validates EVERY
        # key against the ROUTED model's own covariate types, so each model gets exactly
        # its own set (mirrors production: predictions.py builds raw_features from the
        # cohort's base_covariates — never a superset).
        raw_pers = {
            **raw_init,
            "insurance_type": "commercial",
            "age_at_diagnosis": 52,
            "comorbidity_burden": 1,
            "prior_therapy_lines": 0,
        }

        out_init = await service.predict(
            serving_module.PredictionInput(
                model_name="initiation_remibrutinib_goldstd_lr_v1", raw_features=[raw_init]
            )
        )
        out_pers = await service.predict(
            serving_module.PredictionInput(
                model_name="persistence_fabhalta_goldstd_lr_v1", raw_features=[raw_pers]
            )
        )
        # Both real, finite probabilities.
        assert np.isfinite(out_init.probabilities[0])
        assert np.isfinite(out_pers.probabilities[0])
        # Routed to the correct model: the value matches that model's own transform.
        exp_init = float(
            m_init["model"].predict_proba(
                m_init["preprocessor"].transform(pd.DataFrame([raw_init]))
            )[:, 1][0]
        )
        assert abs(out_init.probabilities[0] - exp_init) < 1e-9
        # Two distinct models → distinct probabilities (independently fit).
        assert out_init.probabilities[0] != out_pers.probabilities[0]
        assert out_init.model_id == "initiation_remibrutinib_goldstd_lr_v1"
        assert out_pers.model_id == "persistence_fabhalta_goldstd_lr_v1"

    async def test_model_name_routes_feature_view_and_numeric_paths(
        self, serving_module: Any
    ) -> None:
        class _ModelA:
            feature_names_in_ = np.array(["a", "b"])

            def predict(self, arr):
                return (np.asarray(arr).sum(axis=1) > 0).astype(int)

            def predict_proba(self, arr):
                s = np.asarray(arr).sum(axis=1)
                p = 1 / (1 + np.exp(-s))
                return np.column_stack([1 - p, p])

        class _ModelB:
            feature_names_in_ = np.array(["a", "b"])

            def predict(self, arr):
                return (np.asarray(arr).sum(axis=1) > 0).astype(int)

            def predict_proba(self, arr):
                s = 2.0 * np.asarray(arr).sum(axis=1)
                p = 1 / (1 + np.exp(-s))
                return np.column_stack([1 - p, p])

        service = _service_with_models(
            serving_module,
            {
                "initiation_remibrutinib_goldstd_lr_v1": {
                    "model": _ModelA(),
                    "preprocessor": None,
                    "feature_columns": ["a", "b"],
                },
                "persistence_fabhalta_goldstd_lr_v1": {
                    "model": _ModelB(),
                    "preprocessor": None,
                    "feature_columns": ["a", "b"],
                },
            },
        )
        service._fetch_features_from_feast = AsyncMock(return_value=[[1.0, 2.0]])

        feast_out = await service.predict(
            serving_module.PredictionInput(
                model_name="persistence_fabhalta_goldstd_lr_v1",
                entity_ids=["E1"],
                feature_view="fv",
            )
        )
        numeric_out = await service.predict(
            serving_module.PredictionInput(
                model_name="initiation_remibrutinib_goldstd_lr_v1",
                features=[[1.0, 2.0]],
            )
        )

        assert feast_out.model_id == "persistence_fabhalta_goldstd_lr_v1"
        assert numeric_out.model_id == "initiation_remibrutinib_goldstd_lr_v1"
        assert feast_out.probabilities[0] == pytest.approx(1 / (1 + np.exp(-6.0)))
        assert numeric_out.probabilities[0] == pytest.approx(1 / (1 + np.exp(-3.0)))

    async def test_model_info_by_model_name_returns_that_models_contract(
        self, serving_module: Any
    ) -> None:
        m_init = _fit_bundle(make_patient_spec("initiation", "Remibrutinib"), 1)
        service = _service_with_models(
            serving_module, {"initiation_remibrutinib_goldstd_lr_v1": m_init}
        )
        info = await service.model_info(
            serving_module.ModelInfoInput(model_name="initiation_remibrutinib_goldstd_lr_v1")
        )
        assert info["model_id"] == "initiation_remibrutinib_goldstd_lr_v1"
        assert info["keep_columns"] == list(m_init["preprocessor"].keep_columns)
        assert info["feature_columns"] == m_init["feature_columns"]

    async def test_unknown_model_name_fails_closed(self, serving_module: Any) -> None:
        """An unknown model_name FAILS CLOSED (no silent wrong-model prediction)."""
        service = _service_with_models(
            serving_module,
            {"initiation_remibrutinib_goldstd_lr_v1": _fit_bundle(INITIATION, 1)},
        )
        out = await service.predict(
            serving_module.PredictionInput(
                model_name="does_not_exist_goldstd_lr_v1",
                raw_features=[
                    {"disease_severity": 5.0, "academic_hcp": 0, "geographic_region": "west"}
                ],
            )
        )
        # Fail-closed signal in the response (no fabricated prediction).
        assert out.predictions == []
        assert out.probabilities == []
        assert out.model_id in ("unknown_model", "error")
        assert out.error is not None
        assert "does_not_exist_goldstd_lr_v1" in out.error


@pytest.mark.asyncio
class TestLegacyBackwardCompat:
    async def test_no_model_name_uses_legacy_default(self, serving_module: Any) -> None:
        """No model_name → the legacy single default model (tier0/numeric path)."""

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
        service._models = {}  # no multi-model registry

        out = await service.predict(serving_module.PredictionInput(features=[[1.0, 2.0]]))
        assert out.feature_source == "user_provided"
        assert len(out.probabilities) == 1
        assert out.error is None

    async def test_model_info_no_name_legacy_default(self, serving_module: Any) -> None:
        class _Bare:
            feature_names_in_ = np.array(["x1", "x2"])

        service = serving_module.E2IModelService()
        service._model = _Bare()
        service._preprocessor = None
        service._feature_columns = None
        service._model_tag = "tier0_legacy:v1"
        service._models = {}

        info = await service.model_info()
        assert info["model_id"] == "tier0_legacy:v1"
        assert info["feature_columns"] == ["x1", "x2"]
        # available_models present but empty when only the legacy default exists.
        assert info["available_models"] == []
