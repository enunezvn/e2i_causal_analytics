"""Unit tests for the SkillMatcher class."""

import pytest

from src.skills.matcher import SkillMatch, SkillMatcher


class TestSkillMatch:
    """Tests for SkillMatch dataclass."""

    def test_repr(self):
        """Test string representation."""
        match = SkillMatch(
            skill_path="test/skill.md",
            score=0.85,
            matched_triggers=["test", "example"],
            skill_name="Test Skill",
        )

        repr_str = repr(match)
        assert "test/skill.md" in repr_str
        assert "0.85" in repr_str


class TestSkillMatcher:
    """Tests for SkillMatcher class."""

    @pytest.fixture
    def matcher(self):
        """Create a SkillMatcher."""
        return SkillMatcher()

    def test_find_matches_kpi_query(self, matcher):
        """Test finding skills for a KPI query."""
        matches = matcher.find_matches("calculate TRx for Kisqali")

        assert len(matches) > 0
        # Should find the KPI calculation skill
        kpi_match = next((m for m in matches if "kpi-calculation" in m.skill_path), None)
        assert kpi_match is not None
        assert kpi_match.score > 0.3

    def test_find_matches_causal_query(self, matcher):
        """Test finding skills for a causal analysis query."""
        matches = matcher.find_matches("identify confounders for causal analysis")

        assert len(matches) > 0
        # Should find the confounder identification skill
        confounder_match = next((m for m in matches if "confounder" in m.skill_path), None)
        assert confounder_match is not None

    def test_find_matches_empty_query(self, matcher):
        """Test with empty query."""
        matches = matcher.find_matches("")
        assert matches == []

    def test_find_matches_whitespace_query(self, matcher):
        """Test with whitespace-only query."""
        matches = matcher.find_matches("   ")
        assert matches == []

    def test_find_matches_top_k(self, matcher):
        """Test limiting results with top_k."""
        matches = matcher.find_matches("pharma commercial analytics", top_k=2)
        assert len(matches) <= 2

    def test_find_matches_min_score(self, matcher):
        """Test filtering by minimum score."""
        matches = matcher.find_matches("TRx calculation", min_score=0.5)

        for match in matches:
            assert match.score >= 0.5

    def test_find_matches_brand_boost(self, matcher):
        """Test that brand terms boost relevant skills."""
        # Query with brand name should boost pharma-commercial skills
        matches_with_brand = matcher.find_matches("Kisqali market share analysis")
        matches_without_brand = matcher.find_matches("market share analysis")

        # Both should return results
        assert len(matches_with_brand) > 0
        assert len(matches_without_brand) > 0

    def test_tokenize(self, matcher):
        """Test query tokenization."""
        tokens = matcher._tokenize("Calculate TRx for Kisqali brand")

        assert "calculate" in tokens
        assert "trx" in tokens
        assert "kisqali" in tokens
        assert "brand" in tokens

    def test_rebuild_index(self, matcher):
        """Test rebuilding the index."""
        # First query builds index
        matcher.find_matches("test query")
        assert matcher._indexed

        # Rebuild should reset and rebuild
        matcher.rebuild_index()
        assert matcher._indexed

    def test_matched_triggers_recorded(self, matcher):
        """Test that matched triggers are recorded."""
        matches = matcher.find_matches("calculate TRx prescription metrics")

        assert len(matches) > 0
        # At least one match should have recorded triggers
        has_triggers = any(len(m.matched_triggers) > 0 for m in matches)
        assert has_triggers

    def test_phrase_matching(self, matcher):
        """Test matching multi-word phrases."""
        matches = matcher.find_matches("market share calculation")

        assert len(matches) > 0
        # Should find KPI skill with "market share" trigger
        kpi_match = next((m for m in matches if "kpi-calculation" in m.skill_path), None)
        assert kpi_match is not None


class TestSkillMatcherCategoryBoost:
    """Tests for category-based scoring boosts."""

    @pytest.fixture
    def matcher(self):
        """Create a SkillMatcher."""
        return SkillMatcher()

    def test_kpi_keywords_boost_pharma_skills(self, matcher):
        """Test that KPI keywords boost pharma-commercial skills."""
        boost = matcher._calculate_category_boost(
            "trx nrx conversion", "pharma-commercial/kpi-calculation.md"
        )
        assert boost > 0

    def test_causal_keywords_boost_causal_skills(self, matcher):
        """Test that causal keywords boost causal-inference skills."""
        boost = matcher._calculate_category_boost(
            "confounder ate cate", "causal-inference/confounder-identification.md"
        )
        assert boost > 0

    def test_unrelated_keywords_no_boost(self, matcher):
        """Test that unrelated keywords don't provide boost."""
        boost = matcher._calculate_category_boost(
            "random unrelated words", "pharma-commercial/kpi-calculation.md"
        )
        assert boost == 0

    def test_boost_capped(self, matcher):
        """Test that boost is capped at 0.5."""
        # Query with many matching keywords
        boost = matcher._calculate_category_boost(
            "trx nrx prescription market share conversion kpi adherence roi",
            "pharma-commercial/kpi-calculation.md",
        )
        assert boost <= 0.5
