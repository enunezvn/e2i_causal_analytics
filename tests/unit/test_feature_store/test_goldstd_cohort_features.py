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
        """The view serves EXACTLY the 3 RAW KEEP_COLUMNS covariates — no
        post-index/leakage columns leak into the serving contract."""
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        served = {f.name for f in goldstd_cohort_features_fv.schema}
        assert served == {"disease_severity", "academic_hcp", "geographic_region"}

    def test_geographic_region_served_as_string(self) -> None:
        """The categorical covariate is served as a STRING (the FeatureBuilder
        one-hot-encodes it), not coerced to a numeric type."""
        from feast.types import String
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        by_name = {f.name: f for f in goldstd_cohort_features_fv.schema}
        assert by_name["geographic_region"].dtype == String

    def test_view_online_enabled(self) -> None:
        from features.goldstd_cohort_features import goldstd_cohort_features_fv

        assert goldstd_cohort_features_fv.online is True

    def test_registered_in_feature_view_map(self) -> None:
        from features import FEATURE_VIEW_MAP, get_feature_view

        assert "goldstd_cohort_features" in FEATURE_VIEW_MAP
        resolved = get_feature_view("goldstd_cohort_features")
        assert resolved.name == "goldstd_cohort_features"
