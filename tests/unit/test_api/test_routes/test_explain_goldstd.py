"""Part 4 (#39) — explain route reconciled to serve gold-standard cohort SHAP.

Covers the additive taxonomy reconcile and the numeric-guard branch:

  - ``ModelType`` gains the gold-standard cohort families
    (initiation / persistence / discontinuation) WITHOUT removing the legacy
    propensity / risk_stratification / churn / next_best_action members.
  - ``MODEL_FEATURE_REFS`` maps each gold-standard cohort to the RAW covariate
    refs on the new ``goldstd_cohort_features`` Feast view (so ``get_features``
    fetches the 3 RAW KEEP_COLUMNS covariates).
  - ``resolve_canonical_model_features`` RELAXES the strict numeric guard when
    ``/model_info`` exposes ``keep_columns`` (a FeatureBuilder model): the RAW
    categorical ``geographic_region`` STRING is allowed (the preprocessor will
    one-hot-encode it), while the legacy guard still FAILS CLOSED for
    non-FeatureBuilder models (the #532/#576 audit contract).

These tests mock only EXTERNAL boundaries (the BentoML client's HTTP responses),
not business logic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api.routes.explain import ModelType, RealTimeSHAPService


class TestModelTypeTaxonomyReconcile:
    def test_legacy_members_preserved(self) -> None:
        """The 4 legacy model types remain (no regression for existing callers)."""
        values = {m.value for m in ModelType}
        assert {
            "propensity",
            "risk_stratification",
            "churn_prediction",
            "next_best_action",
        } <= values

    def test_goldstd_cohort_members_added(self) -> None:
        """The gold-standard cohort families are now explainable model types."""
        values = {m.value for m in ModelType}
        assert {"initiation", "persistence", "discontinuation"} <= values

    def test_every_model_type_has_feature_refs(self) -> None:
        """The lockstep invariant: every ModelType value has a registry entry
        (the existing test_model_feature_refs invariant must still hold after
        adding the cohorts)."""
        from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

        for m in ModelType:
            assert m.value in MODEL_FEATURE_REFS, f"missing refs for {m.value}"

    def test_goldstd_refs_point_at_raw_covariate_view(self) -> None:
        """Gold-standard cohorts fetch the 7 RAW _BASE7 covariates from the
        goldstd_cohort_features view (T9/T11 enrichment): the base 3 plus the 4
        arm-independent prognostic drivers the enriched models consume."""
        from src.feature_store.model_feature_refs import MODEL_FEATURE_REFS

        for cohort in ("initiation", "persistence", "discontinuation"):
            refs = MODEL_FEATURE_REFS[cohort]
            assert refs == [
                "goldstd_cohort_features:disease_severity",
                "goldstd_cohort_features:academic_hcp",
                "goldstd_cohort_features:geographic_region",
                "goldstd_cohort_features:insurance_type",
                "goldstd_cohort_features:age_at_diagnosis",
                "goldstd_cohort_features:comorbidity_burden",
                "goldstd_cohort_features:prior_therapy_lines",
            ]

    def test_get_feature_refs_for_model_returns_goldstd_raw(self) -> None:
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        refs = service._get_feature_refs_for_model(ModelType.INITIATION)
        assert refs == [
            "goldstd_cohort_features:disease_severity",
            "goldstd_cohort_features:academic_hcp",
            "goldstd_cohort_features:geographic_region",
            "goldstd_cohort_features:insurance_type",
            "goldstd_cohort_features:age_at_diagnosis",
            "goldstd_cohort_features:comorbidity_burden",
            "goldstd_cohort_features:prior_therapy_lines",
        ]


def _service_with_model_info(info: dict) -> RealTimeSHAPService:
    service = RealTimeSHAPService.__new__(RealTimeSHAPService)
    client = AsyncMock()
    client.get_model_info = AsyncMock(return_value=info)
    service.bentoml_client = client
    return service


@pytest.mark.asyncio
class TestResolveCanonicalFeaturesGoldstdBranch:
    async def test_raw_categorical_allowed_when_keep_columns_present(self) -> None:
        """A FeatureBuilder model (keep_columns present) validates against the RAW
        covariate names and ALLOWS the categorical geographic_region STRING."""
        info = {
            "feature_columns": [
                "disease_severity__isna",
                "disease_severity",
                "academic_hcp__isna",
                "academic_hcp",
                "geographic_region_northeast",
            ],
            "keep_columns": ["disease_severity", "academic_hcp", "geographic_region"],
        }
        service = _service_with_model_info(info)
        features = {
            "disease_severity": 5.61,
            "academic_hcp": 0,
            "geographic_region": "northeast",  # categorical STRING — must be allowed
        }
        resolved = await service.resolve_canonical_model_features(features, ModelType.INITIATION)
        # Resolved keyed by the RAW keep_columns, geographic_region kept as string.
        assert set(resolved.keys()) == {
            "disease_severity",
            "academic_hcp",
            "geographic_region",
        }
        assert resolved["geographic_region"] == "northeast"
        assert resolved["disease_severity"] == 5.61

    async def test_goldstd_missing_raw_covariate_fails_closed(self) -> None:
        """A required RAW covariate that is missing/null still FAILS CLOSED (422)."""
        info = {
            "feature_columns": ["disease_severity", "geographic_region_northeast"],
            "keep_columns": ["disease_severity", "academic_hcp", "geographic_region"],
        }
        service = _service_with_model_info(info)
        features = {"disease_severity": 5.61, "geographic_region": "northeast"}  # no academic_hcp
        with pytest.raises(HTTPException) as ei:
            await service.resolve_canonical_model_features(features, ModelType.INITIATION)
        assert ei.value.status_code == 422
        assert "academic_hcp" in str(ei.value.detail)

    async def test_goldstd_numeric_covariate_rejects_string(self) -> None:
        """Only the categorical geographic_region field may be a string."""
        info = {
            "feature_columns": ["disease_severity", "geographic_region_northeast"],
            "keep_columns": ["disease_severity", "academic_hcp", "geographic_region"],
        }
        service = _service_with_model_info(info)
        features = {
            "disease_severity": "5.61",
            "academic_hcp": 0,
            "geographic_region": "northeast",
        }
        with pytest.raises(HTTPException) as ei:
            await service.resolve_canonical_model_features(features, ModelType.INITIATION)
        assert ei.value.status_code == 422
        assert "disease_severity" in str(ei.value.detail)

    async def test_legacy_numeric_guard_unchanged_without_keep_columns(self) -> None:
        """A non-FeatureBuilder model (no keep_columns) keeps the strict numeric
        guard: a string feature still FAILS CLOSED (the #532/#576 contract)."""
        info = {
            "feature_columns": ["f1", "f2"],
            "keep_columns": None,  # bare estimator — no raw covariate contract
        }
        service = _service_with_model_info(info)
        features = {"f1": 1.0, "f2": "not_a_number"}
        with pytest.raises(HTTPException) as ei:
            await service.resolve_canonical_model_features(features, ModelType.PROPENSITY)
        assert ei.value.status_code == 422
        assert "numeric" in str(ei.value.detail).lower()

    async def test_legacy_propensity_numeric_still_passes(self) -> None:
        """Backward compat: a legacy numeric request still resolves to floats."""
        info = {"feature_columns": ["f1", "f2"], "keep_columns": None}
        service = _service_with_model_info(info)
        resolved = await service.resolve_canonical_model_features(
            {"f1": 1.0, "f2": 2}, ModelType.PROPENSITY
        )
        assert resolved == {"f1": 1.0, "f2": 2.0}


@pytest.mark.asyncio
class TestComputeShapGoldstdBranch:
    """Fix #2c (#39, Option B): compute_shap delegates gold-standard cohorts to
    the BentoML /shap endpoint (no MLflow load) and maps the response into the
    legacy compute_shap shape. Legacy non-goldstd path is unchanged."""

    @staticmethod
    def _service_with_shap(shap_return: dict) -> RealTimeSHAPService:
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        client = AsyncMock()
        client.get_shap = AsyncMock(return_value=shap_return)
        service.bentoml_client = client
        service._initialized = True
        return service

    async def test_goldstd_compute_shap_calls_bentoml_shap_not_mlflow(self) -> None:
        """For a gold-standard cohort, compute_shap calls /shap with the serving
        name + RAW covariates and returns real per-encoded-feature contributions
        — and NEVER touches the MLflow explainer."""
        service = self._service_with_shap(
            {
                "shap_values": {
                    "disease_severity": 0.8799,
                    "geographic_region_northeast": -0.1817,
                    "academic_hcp": 0.0,
                },
                "base_value": -0.8342,
                "encoded_feature_columns": [
                    "disease_severity",
                    "geographic_region_northeast",
                    "academic_hcp",
                ],
                "explainer_type": "LinearExplainer",
                "model_id": "initiation_kisqali_goldstd_lr_v1",
            }
        )
        # If the legacy in-process explainer were used it would explode (no real
        # MLflow); assert it is NOT consulted.
        service.shap_explainer = AsyncMock()
        service.shap_explainer.compute_shap_values = AsyncMock(
            side_effect=AssertionError("legacy MLflow explainer must not be called for goldstd")
        )

        out = await service.compute_shap(
            features={
                "disease_severity": 5.26,
                "geographic_region_northeast": 1.0,
                "academic_hcp": 0.0,
            },
            model_type=ModelType.INITIATION,
            model_version_id="initiation_kisqali_goldstd_lr_v1",
            top_k=5,
            serving_name="initiation_kisqali_goldstd_lr_v1",
            raw_features={
                "disease_severity": 5.26,
                "academic_hcp": 0,
                "geographic_region": "northeast",
            },
        )

        # /shap called with the routed serving name + RAW covariates.
        called = service.bentoml_client.get_shap.call_args
        assert called.kwargs["model_name"] == "initiation_kisqali_goldstd_lr_v1"
        assert called.kwargs["raw_features"][0]["geographic_region"] == "northeast"
        # Mapped into the legacy compute_shap shape.
        assert out["base_value"] == pytest.approx(-0.8342)
        assert out["explainer_type"] == "LinearExplainer"
        assert out["shap_sum"] == pytest.approx(0.8799 - 0.1817 + 0.0)
        names = {c.feature_name for c in out["contributions"]}
        assert "disease_severity" in names
        top = out["contributions"][0]
        assert top.feature_name == "disease_severity"  # largest |shap|
        assert top.shap_value == pytest.approx(0.8799)

    async def test_goldstd_service_error_fails_closed_502(self) -> None:
        """A fail-closed /shap response (error set) surfaces as 502 — no
        fabricated SHAP."""
        service = self._service_with_shap({"shap_values": {}, "error": "Unknown model_name: bad"})
        with pytest.raises(HTTPException) as ei:
            await service.compute_shap(
                features={"x": 1.0},
                model_type=ModelType.HCP_ADOPTION,
                model_version_id="hcp_adoption_kisqali_goldstd_lr_v1",
                serving_name="hcp_adoption_kisqali_goldstd_lr_v1",
                raw_features={"a": 1},
            )
        assert ei.value.status_code == 502

    async def test_goldstd_missing_context_fails_closed_500(self) -> None:
        """Missing serving_name/raw_features (internal wiring error) → 500."""
        service = self._service_with_shap({"shap_values": {"x": 0.1}})
        with pytest.raises(HTTPException) as ei:
            await service.compute_shap(
                features={"x": 1.0},
                model_type=ModelType.INITIATION,
                model_version_id="initiation_kisqali_goldstd_lr_v1",
                serving_name=None,  # not threaded through
                raw_features=None,
            )
        assert ei.value.status_code == 500

    async def test_goldstd_empty_shap_map_fails_closed_502(self) -> None:
        """An empty shap_values map (no error field) still fails closed (502)."""
        service = self._service_with_shap({"shap_values": {}, "base_value": 0.0})
        with pytest.raises(HTTPException) as ei:
            await service.compute_shap(
                features={"x": 1.0},
                model_type=ModelType.PERSISTENCE,
                model_version_id="persistence_kisqali_goldstd_lr_v1",
                serving_name="persistence_kisqali_goldstd_lr_v1",
                raw_features={"a": 1},
            )
        assert ei.value.status_code == 502


class TestGoldstdCohortFamilyMapping:
    def test_per_brand_and_base_names_map_to_family(self) -> None:
        from src.api.routes.explain import _goldstd_cohort_family

        assert _goldstd_cohort_family("csu_initiation_goldstd_lr_v1") == "initiation"
        assert _goldstd_cohort_family("initiation_remibrutinib_goldstd_lr_v1") == "initiation"
        assert _goldstd_cohort_family("pnh_persistence_goldstd_lr_v1") == "persistence"
        assert _goldstd_cohort_family("persistence_kisqali_goldstd_lr_v1") == "persistence"
        assert _goldstd_cohort_family("pnh_discontinuation_goldstd_lr_v1") == "discontinuation"
        assert _goldstd_cohort_family("discontinuation_fabhalta_goldstd_lr_v1") == "discontinuation"

    def test_hcp_adoption_per_brand_names_map_to_family(self) -> None:
        """The 4th cohort (HCP-grain adoption) must also map to its family, else
        /explain/models reports ``latest_version: null`` for hcp_adoption even
        though the 3 ``hcp_adoption_{brand}_goldstd_lr_v1`` models ARE registered."""
        from src.api.routes.explain import _goldstd_cohort_family

        assert _goldstd_cohort_family("hcp_adoption_remibrutinib_goldstd_lr_v1") == "hcp_adoption"
        assert _goldstd_cohort_family("hcp_adoption_fabhalta_goldstd_lr_v1") == "hcp_adoption"
        assert _goldstd_cohort_family("hcp_adoption_kisqali_goldstd_lr_v1") == "hcp_adoption"

    def test_non_goldstd_name_returns_none(self) -> None:
        from src.api.routes.explain import _goldstd_cohort_family

        assert _goldstd_cohort_family("csu_treatment_initiation_live_v1") is None
        assert _goldstd_cohort_family("propensity") is None


@pytest.mark.asyncio
class TestGetPredictionGoldstdRawPath:
    async def test_sends_raw_features_and_shap_over_encoded(self) -> None:
        """Gold-standard get_prediction sends RAW covariates to BentoML and
        returns the ENCODED vector (from the response) as the SHAP features."""
        info = {
            "feature_columns": [
                "disease_severity__isna",
                "disease_severity",
                "academic_hcp__isna",
                "academic_hcp",
                "geographic_region_northeast",
            ],
            "keep_columns": ["disease_severity", "academic_hcp", "geographic_region"],
        }
        client = AsyncMock()
        client.get_model_info = AsyncMock(return_value=info)
        client.predict = AsyncMock(
            return_value={
                "probabilities": [0.3516],
                "model_id": "csu_initiation_goldstd_lr_v1:bundle",
                "encoded_features": [[0.0, 5.61, 0.0, 0.0, 1.0]],
                "encoded_feature_columns": info["feature_columns"],
            }
        )
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        service.bentoml_client = client
        service._initialized = True

        out = await service.get_prediction(
            features={
                "disease_severity": 5.61,
                "academic_hcp": 0,
                "geographic_region": "northeast",
            },
            model_type=ModelType.INITIATION,
        )
        # BentoML was called with RAW covariates, not a numeric matrix.
        called = client.predict.call_args.kwargs["input_data"]
        assert "raw_features" in called
        assert called["raw_features"][0]["geographic_region"] == "northeast"
        # SHAP features are the ENCODED numeric vector (all numeric, 5 cols).
        mf = out["model_features"]
        assert set(mf.keys()) == set(info["feature_columns"])
        assert all(isinstance(v, float) for v in mf.values())
        assert out["prediction_probability"] == pytest.approx(0.3516)

    async def test_fails_closed_when_no_encoded_vector(self) -> None:
        """If the service omits the encoded vector, FAIL CLOSED (502) — do not
        run SHAP over the raw covariates and mislabel it audit-grade."""
        info = {
            "feature_columns": ["disease_severity", "geographic_region_northeast"],
            "keep_columns": ["disease_severity", "academic_hcp", "geographic_region"],
        }
        client = AsyncMock()
        client.get_model_info = AsyncMock(return_value=info)
        client.predict = AsyncMock(
            return_value={"probabilities": [0.4]}  # no encoded_features
        )
        service = RealTimeSHAPService.__new__(RealTimeSHAPService)
        service.bentoml_client = client
        service._initialized = True
        with pytest.raises(HTTPException) as ei:
            await service.get_prediction(
                features={
                    "disease_severity": 5.61,
                    "academic_hcp": 0,
                    "geographic_region": "northeast",
                },
                model_type=ModelType.INITIATION,
            )
        assert ei.value.status_code == 502
