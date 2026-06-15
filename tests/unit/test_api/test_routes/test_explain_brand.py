"""Increment 2 (#39) — brand dimension + per-brand serving-name routing.

The gold-standard models are per-brand. The explain request gains an optional
``brand``; the route resolves the serving ``model_name`` =
``f"{cohort}_{brand}_goldstd_lr_v1"`` (and the HCP-grain
``hcp_adoption_{brand}_goldstd_lr_v1``) and sends THAT to BentoML so the
multi-model service routes correctly. Default brand is explicit when omitted.
The #532/#576 fail-loud contracts are preserved. HCP-adoption is wired as a real
cohort (HCP grain; its covariates come from the HCP FeatureBuilder/spec on main).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from src.api.routes.explain import (
    GOLDSTD_COHORT_MODEL_TYPES,
    ModelType,
    RealTimeSHAPService,
    goldstd_serving_name,
)


class TestModelTypeHcpAddition:
    def test_hcp_adoption_is_a_model_type(self) -> None:
        assert "hcp_adoption" in {m.value for m in ModelType}

    def test_hcp_adoption_is_goldstd(self) -> None:
        assert ModelType.HCP_ADOPTION in GOLDSTD_COHORT_MODEL_TYPES

    def test_legacy_and_patient_cohorts_still_present(self) -> None:
        values = {m.value for m in ModelType}
        assert {"propensity", "initiation", "persistence", "discontinuation"} <= values


class TestServingNameResolution:
    def test_patient_cohort_serving_name(self) -> None:
        assert (
            goldstd_serving_name(ModelType.INITIATION, "Remibrutinib")
            == "initiation_remibrutinib_goldstd_lr_v1"
        )
        assert (
            goldstd_serving_name(ModelType.PERSISTENCE, "Fabhalta")
            == "persistence_fabhalta_goldstd_lr_v1"
        )

    def test_hcp_cohort_serving_name(self) -> None:
        assert (
            goldstd_serving_name(ModelType.HCP_ADOPTION, "Kisqali")
            == "hcp_adoption_kisqali_goldstd_lr_v1"
        )

    def test_brand_is_case_insensitive(self) -> None:
        assert (
            goldstd_serving_name(ModelType.INITIATION, "remibrutinib")
            == "initiation_remibrutinib_goldstd_lr_v1"
        )

    def test_unknown_brand_fails_closed(self) -> None:
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as ei:
            goldstd_serving_name(ModelType.INITIATION, "NotABrand")
        assert ei.value.status_code == 422


class TestExplainRequestBrand:
    def test_brand_optional_default_none(self) -> None:
        from src.api.routes.explain import ExplainRequest

        req = ExplainRequest(patient_id="P1", model_type=ModelType.INITIATION)
        assert req.brand is None

    def test_brand_accepted(self) -> None:
        from src.api.routes.explain import ExplainRequest

        req = ExplainRequest(patient_id="P1", model_type=ModelType.INITIATION, brand="Fabhalta")
        assert req.brand == "Fabhalta"


def _info(keep_columns, feature_columns):
    return {"keep_columns": keep_columns, "feature_columns": feature_columns}


@pytest.mark.asyncio
class TestGetPredictionSendsPerBrandServingName:
    async def test_patient_cohort_sends_serving_name(self) -> None:
        info = _info(
            ["disease_severity", "academic_hcp", "geographic_region"],
            ["disease_severity", "academic_hcp", "geographic_region_west"],
        )
        client = AsyncMock()
        client.get_model_info = AsyncMock(return_value=info)
        client.predict = AsyncMock(
            return_value={
                "probabilities": [0.42],
                "model_id": "initiation_fabhalta_goldstd_lr_v1",
                "encoded_features": [[5.0, 1.0, 1.0]],
                "encoded_feature_columns": info["feature_columns"],
            }
        )
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        service.bentoml_client = client
        service._initialized = True

        await service.get_prediction(
            features={
                "disease_severity": 5.0,
                "academic_hcp": 1,
                "geographic_region": "west",
            },
            model_type=ModelType.INITIATION,
            brand="Fabhalta",
        )
        # Both /model_info and /predict must be keyed by the per-brand serving name.
        assert client.get_model_info.call_args.args[0] == "initiation_fabhalta_goldstd_lr_v1"
        assert client.predict.call_args.kwargs["model_name"] == "initiation_fabhalta_goldstd_lr_v1"

    async def test_hcp_cohort_sends_serving_name_and_shap_over_encoded(self) -> None:
        keep = [
            "peer_influence_score",
            "influence_network_size",
            "years_experience",
            "specialty",
            "geographic_region",
        ]
        # Realistic encoded set: specialty + geographic_region are one-hot
        # (categorical); the other 3 are bare numeric columns. This is what the
        # HCP FeatureBuilder produces and what _infer_categorical_covariates reads.
        fcols = [
            "specialty_dermatology",
            "specialty_oncology",
            "years_experience",
            "peer_influence_score",
            "influence_network_size",
            "geographic_region_northeast",
            "geographic_region_west",
        ]
        client = AsyncMock()
        client.get_model_info = AsyncMock(return_value=_info(keep, fcols))
        client.predict = AsyncMock(
            return_value={
                "probabilities": [0.08],
                "model_id": "hcp_adoption_remibrutinib_goldstd_lr_v1",
                "encoded_features": [[1.0, 0.0, 26.0, 1.31, 3.0, 1.0, 0.0]],
                "encoded_feature_columns": fcols,
            }
        )
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        service.bentoml_client = client
        service._initialized = True

        out = await service.get_prediction(
            features={
                "peer_influence_score": 1.31,
                "influence_network_size": 3,
                "years_experience": 26,
                "specialty": "dermatology",
                "geographic_region": "northeast",
            },
            model_type=ModelType.HCP_ADOPTION,
            brand="Remibrutinib",
        )
        assert (
            client.predict.call_args.kwargs["model_name"]
            == "hcp_adoption_remibrutinib_goldstd_lr_v1"
        )
        # SHAP runs over the ENCODED vector (numeric), not the 5 raw covariates.
        mf = out["model_features"]
        assert set(mf.keys()) == set(fcols)
        assert all(isinstance(v, float) for v in mf.values())

    async def test_goldstd_without_brand_uses_explicit_default(self) -> None:
        """Omitting brand routes to an explicit default brand (not a silent skip)."""
        info = _info(
            ["disease_severity", "academic_hcp", "geographic_region"],
            ["disease_severity", "geographic_region_west"],
        )
        client = AsyncMock()
        client.get_model_info = AsyncMock(return_value=info)
        client.predict = AsyncMock(
            return_value={
                "probabilities": [0.5],
                "model_id": "x",
                "encoded_features": [[5.0, 1.0]],
                "encoded_feature_columns": info["feature_columns"],
            }
        )
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        service.bentoml_client = client
        service._initialized = True
        await service.get_prediction(
            features={
                "disease_severity": 5.0,
                "academic_hcp": 1,
                "geographic_region": "west",
            },
            model_type=ModelType.INITIATION,
        )
        sent = client.predict.call_args.kwargs["model_name"]
        # Default brand resolves to a concrete per-brand serving name.
        assert sent.startswith("initiation_") and sent.endswith("_goldstd_lr_v1")


class TestHcpFeatureRefs:
    def test_hcp_refs_present_and_raw(self) -> None:
        from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

        refs = MODEL_FEATURE_REFS["hcp_adoption"]
        # HCP-grain raw covariates served from a (HCP) view; raw names match the spec.
        assert any("peer_influence_score" in r for r in refs)
        assert any("specialty" in r for r in refs)

    def test_every_model_type_has_refs(self) -> None:
        from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

        for m in ModelType:
            assert m.value in MODEL_FEATURE_REFS
