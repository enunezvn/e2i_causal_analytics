"""Part 3 (#39) — Feast view for the gold-standard cohort RAW covariates.

Asserts the ``goldstd_cohort_features`` FeatureView is defined correctly (so
``feast apply`` would register it) and serves exactly the 3 leakage-safe
KEEP_COLUMNS covariates per ``patient``. These tests require the feast package
(installed in CI / the Feast container) and are skipped where feast is absent,
mirroring ``test_feast_entities.py``.
"""

import sys
from pathlib import Path

import pytest

try:
    import feast  # noqa: F401

    HAS_FEAST = True
except ImportError:
    HAS_FEAST = False

pytestmark = pytest.mark.skipif(
    not HAS_FEAST, reason="feast package not installed - install with: pip install feast"
)

# Add feature_repo to path for imports (mirrors test_feast_entities.py).
feature_repo_path = Path(__file__).parent.parent.parent.parent / "feature_repo"
sys.path.insert(0, str(feature_repo_path))


class TestGoldstdCohortFeatureView:
    def test_view_importable_and_named(self) -> None:
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        assert goldstd_cohort_features_fv is not None
        assert goldstd_cohort_features_fv.name == "goldstd_cohort_features"

    def test_entity_is_patient_only(self) -> None:
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        # Gold-standard serving is keyed on patient alone (not patient_brand).
        entity_names = list(goldstd_cohort_features_fv.entities)
        assert entity_names == ["patient"]

    def test_schema_is_exactly_keep_columns(self) -> None:
        """The view serves EXACTLY the 7 RAW _BASE7 covariates the enriched
        gold-standard patient models consume (T9/T11 DGP enrichment) — no
        post-index/leakage columns leak into the serving contract. The 4 new
        prognostic drivers are arm-independent (ATE/CATE preserved) and are the
        ones the live Feature-Importance page must surface alongside the base 3."""
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        served = {f.name for f in goldstd_cohort_features_fv.schema}
        assert served == {
            "disease_severity",
            "academic_hcp",
            "geographic_region",
            "insurance_type",
            "age_at_diagnosis",
            "comorbidity_burden",
            "prior_therapy_lines",
        }

    def test_geographic_region_served_as_string(self) -> None:
        """The categorical covariates are served as STRINGs (the FeatureBuilder
        one-hot-encodes them), not coerced to a numeric type."""
        from feast.types import String
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        by_name = {f.name: f for f in goldstd_cohort_features_fv.schema}
        assert by_name["geographic_region"].dtype == String
        # insurance_type is also categorical (one-hot -> insurance_type_* columns).
        assert by_name["insurance_type"].dtype == String

    def test_new_prognostic_drivers_typed_int(self) -> None:
        """The 3 numeric prognostic drivers are served as Int64 (the
        FeatureBuilder median-imputes/scales them; they are NOT one-hot)."""
        from feast.types import Int64
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        by_name = {f.name: f for f in goldstd_cohort_features_fv.schema}
        for col in ("age_at_diagnosis", "comorbidity_burden", "prior_therapy_lines"):
            assert by_name[col].dtype == Int64, col

    def test_view_online_enabled(self) -> None:
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        assert goldstd_cohort_features_fv.online is True

    def test_registered_in_feature_view_map(self) -> None:
        from features import FEATURE_VIEW_MAP, get_feature_view

        assert "goldstd_cohort_features" in FEATURE_VIEW_MAP
        resolved = get_feature_view("goldstd_cohort_features")
        assert resolved.name == "goldstd_cohort_features"
