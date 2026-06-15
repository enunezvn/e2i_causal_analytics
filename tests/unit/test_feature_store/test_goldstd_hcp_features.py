"""Increment 2 (#39) — Feast view for the gold-standard HCP-adoption covariates.

The HCP-adoption models are HCP-grain (entity ``hcp``); this view serves the 5
RAW leakage-safe covariates from ``hcp_profiles``. Requires feast (CI / Feast
container); skipped where feast is absent, mirroring test_feast_entities.py.
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

feature_repo_path = Path(__file__).parent.parent.parent.parent / "feature_repo"
sys.path.insert(0, str(feature_repo_path))


class TestGoldstdHcpFeatureView:
    def test_view_importable_and_named(self) -> None:
        from features.goldstd_hcp_features import goldstd_hcp_features_fv

        assert goldstd_hcp_features_fv is not None
        assert goldstd_hcp_features_fv.name == "goldstd_hcp_features"

    def test_entity_is_hcp_only(self) -> None:
        from features.goldstd_hcp_features import goldstd_hcp_features_fv

        assert list(goldstd_hcp_features_fv.entities) == ["hcp"]

    def test_schema_is_exactly_hcp_covariates(self) -> None:
        from features.goldstd_hcp_features import goldstd_hcp_features_fv

        served = {f.name for f in goldstd_hcp_features_fv.schema}
        assert served == {
            "peer_influence_score",
            "influence_network_size",
            "years_experience",
            "specialty",
            "geographic_region",
        }

    def test_categoricals_served_as_string(self) -> None:
        from feast.types import String
        from features.goldstd_hcp_features import goldstd_hcp_features_fv

        by_name = {f.name: f for f in goldstd_hcp_features_fv.schema}
        assert by_name["specialty"].dtype == String
        assert by_name["geographic_region"].dtype == String

    def test_online_enabled_and_registered(self) -> None:
        from features import FEATURE_VIEW_MAP, get_feature_view
        from features.goldstd_hcp_features import goldstd_hcp_features_fv

        assert goldstd_hcp_features_fv.online is True
        assert "goldstd_hcp_features" in FEATURE_VIEW_MAP
        assert get_feature_view("goldstd_hcp_features").name == "goldstd_hcp_features"
