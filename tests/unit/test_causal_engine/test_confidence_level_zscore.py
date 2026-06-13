"""RED-FIRST tests for #27: honest confidence-level labeling across causal CIs.

Backlog #27: the causal pipeline hardcodes a 95% CI (+/- 1.96 sigma) but never
exposes the confidence level in its response schemas, so the frontend cannot
honestly label CIs. These tests pin the desired behavior BEFORE the fix:

  (a) a single shared z-score helper derives z from the confidence level
      (scipy-exact), and at the 0.95 default reproduces the legacy 1.96 so
      existing numeric behavior is UNCHANGED;
  (b) at confidence_level=0.90 the CI half-width uses 1.645*sigma, not 1.96*sigma
      (the math test -- the cheapest disproof of the whole approach);
  (c) the causal response schemas carry confidence_level with default 0.95.

No mocking, no fabricated estimator outputs -- this exercises the real z-score
math and the real Pydantic schemas.
"""

import math

import pytest


class TestZScoreHelper:
    """The single shared z-score helper (src/causal/stats.py)."""

    def test_default_095_reproduces_legacy_196(self):
        """confidence_level=0.95 -> z == 1.96 (rounded), preserving legacy CI math."""
        from src.causal.stats import z_score_for_confidence

        z = z_score_for_confidence(0.95)
        # scipy norm.ppf(0.975) == 1.9599639845... -> rounds to 1.96
        assert round(z, 2) == 1.96, f"0.95 must map to ~1.96, got {z}"

    def test_090_maps_to_1645(self):
        """confidence_level=0.90 -> z == 1.645 (rounded), NOT 1.96.

        This is the cheapest disproof of the whole #27 approach: if 0.90 still
        produced 1.96 the CI would be mislabeled. norm.ppf(0.95) == 1.6448536...
        """
        from src.causal.stats import z_score_for_confidence

        z = z_score_for_confidence(0.90)
        assert round(z, 3) == 1.645, f"0.90 must map to ~1.645, got {z}"
        assert not math.isclose(z, 1.96, abs_tol=0.05), "0.90 must NOT be 1.96"

    def test_099_maps_to_2576(self):
        """confidence_level=0.99 -> z == 2.576 (rounded)."""
        from src.causal.stats import z_score_for_confidence

        z = z_score_for_confidence(0.99)
        assert round(z, 3) == 2.576, f"0.99 must map to ~2.576, got {z}"

    def test_ci_half_width_scales_with_level(self):
        """The CI half-width = z * sigma uses 1.645*sigma at 0.90, 1.96*sigma at 0.95."""
        from src.causal.stats import z_score_for_confidence

        sigma = 0.10
        hw_90 = z_score_for_confidence(0.90) * sigma
        hw_95 = z_score_for_confidence(0.95) * sigma
        assert round(hw_90, 4) == round(1.645 * sigma, 4)
        assert round(hw_95, 4) == round(1.96 * sigma, 4)
        # A wider confidence level must produce a wider interval.
        assert hw_95 > hw_90

    def test_alpha_helper_matches_confidence_helper(self):
        """z_score_for_alpha(alpha) == z_score_for_confidence(1 - alpha).

        The CATE estimator threads a *significance_level* (alpha); the consensus
        path threads a *confidence_level*. Both must resolve to the same z so the
        two CIs are consistent.
        """
        from src.causal.stats import z_score_for_alpha, z_score_for_confidence

        for alpha in (0.01, 0.05, 0.10):
            assert math.isclose(
                z_score_for_alpha(alpha),
                z_score_for_confidence(1 - alpha),
                rel_tol=1e-9,
            )

    def test_rejects_out_of_range_levels(self):
        """A confidence level outside (0, 1) is a programming error, not a CI."""
        from src.causal.stats import z_score_for_confidence

        for bad in (0.0, 1.0, -0.1, 1.5):
            with pytest.raises(ValueError):
                z_score_for_confidence(bad)


class TestSchemaConfidenceLevelField:
    """The causal response schemas must carry confidence_level (default 0.95)."""

    def test_parallel_response_has_confidence_level_default_095(self):
        from datetime import datetime, timezone

        from src.api.schemas.causal import AnalysisStatus, ParallelPipelineResponse

        resp = ParallelPipelineResponse(
            pipeline_id="p1",
            status=AnalysisStatus.COMPLETED,
            consensus_method="variance_weighted",
            total_latency_ms=1,
            created_at=datetime.now(timezone.utc),
        )
        assert resp.confidence_level == 0.95

    def test_sequential_response_has_confidence_level_default_095(self):
        from datetime import datetime, timezone

        from src.api.schemas.causal import AnalysisStatus, SequentialPipelineResponse

        resp = SequentialPipelineResponse(
            pipeline_id="s1",
            status=AnalysisStatus.COMPLETED,
            stages_completed=1,
            stages_total=1,
            total_latency_ms=1,
            created_at=datetime.now(timezone.utc),
        )
        assert resp.confidence_level == 0.95

    def test_hierarchical_cate_response_has_confidence_level_default_095(self):
        from datetime import datetime, timezone

        from src.api.schemas.causal import AnalysisStatus, HierarchicalAnalysisResponse

        resp = HierarchicalAnalysisResponse(
            analysis_id="h1",
            status=AnalysisStatus.COMPLETED,
            segmentation_method="quantile",
            estimator_type="causal_forest",
            latency_ms=1,
            created_at=datetime.now(timezone.utc),
        )
        assert resp.confidence_level == 0.95

    def test_confidence_level_is_settable(self):
        """Callers can label a non-default level (e.g. a 90% CI)."""
        from datetime import datetime, timezone

        from src.api.schemas.causal import AnalysisStatus, ParallelPipelineResponse

        resp = ParallelPipelineResponse(
            pipeline_id="p2",
            status=AnalysisStatus.COMPLETED,
            consensus_method="variance_weighted",
            total_latency_ms=1,
            created_at=datetime.now(timezone.utc),
            confidence_level=0.90,
        )
        assert resp.confidence_level == 0.90
