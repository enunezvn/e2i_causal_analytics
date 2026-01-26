"""Unit tests for the SkillLoader class."""

import pytest

from src.skills.loader import Skill, SkillLoader, SkillMetadata


class TestSkillMetadata:
    """Tests for SkillMetadata dataclass."""

    def test_from_dict_full(self):
        """Test creating SkillMetadata from a complete dictionary."""
        data = {
            "name": "Test Skill",
            "version": "2.0",
            "description": "A test skill",
            "triggers": ["test", "example"],
            "agents": ["agent1", "agent2"],
            "author": "Test Author",
            "categories": ["cat1", "cat2"],
        }
        metadata = SkillMetadata.from_dict(data)

        assert metadata.name == "Test Skill"
        assert metadata.version == "2.0"
        assert metadata.description == "A test skill"
        assert metadata.triggers == ["test", "example"]
        assert metadata.agents == ["agent1", "agent2"]
        assert metadata.author == "Test Author"
        assert metadata.categories == ["cat1", "cat2"]

    def test_from_dict_minimal(self):
        """Test creating SkillMetadata with minimal data."""
        data = {"name": "Minimal Skill"}
        metadata = SkillMetadata.from_dict(data)

        assert metadata.name == "Minimal Skill"
        assert metadata.version == "1.0"
        assert metadata.description == ""
        assert metadata.triggers == []
        assert metadata.agents == []

    def test_from_dict_missing_name(self):
        """Test creating SkillMetadata without a name."""
        data = {"description": "No name"}
        metadata = SkillMetadata.from_dict(data)

        assert metadata.name == "Unknown"


class TestSkill:
    """Tests for Skill dataclass."""

    def test_get_section_exact_match(self):
        """Test getting a section with exact name match."""
        skill = Skill(
            path="test/skill.md",
            metadata=SkillMetadata(name="Test"),
            content="# Test Content",
            sections={"TRx (Total Prescriptions)": "TRx content here"},
        )

        result = skill.get_section("TRx (Total Prescriptions)")
        assert result == "TRx content here"

    def test_get_section_case_insensitive(self):
        """Test getting a section with case-insensitive match."""
        skill = Skill(
            path="test/skill.md",
            metadata=SkillMetadata(name="Test"),
            content="# Test Content",
            sections={"TRx (Total Prescriptions)": "TRx content here"},
        )

        result = skill.get_section("trx (total prescriptions)")
        assert result == "TRx content here"

    def test_get_section_not_found(self):
        """Test getting a non-existent section."""
        skill = Skill(
            path="test/skill.md",
            metadata=SkillMetadata(name="Test"),
            content="# Test Content",
            sections={"Section A": "Content A"},
        )

        result = skill.get_section("Non-existent")
        assert result is None

    def test_get_sections_matching(self):
        """Test getting sections matching a pattern."""
        skill = Skill(
            path="test/skill.md",
            metadata=SkillMetadata(name="Test"),
            content="# Test Content",
            sections={
                "TRx Metrics": "TRx content",
                "NRx Metrics": "NRx content",
                "Other Section": "Other content",
            },
        )

        result = skill.get_sections_matching(r".*Metrics$")
        assert len(result) == 2
        assert "TRx Metrics" in result
        assert "NRx Metrics" in result


class TestSkillLoader:
    """Tests for SkillLoader class."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader with the project's skills directory."""
        return SkillLoader()

    def test_load_kpi_calculation(self, loader):
        """Test loading the KPI calculation skill."""
        skill = loader.load("pharma-commercial/kpi-calculation.md")

        assert skill.metadata.name == "KPI Calculation Procedures"
        assert "calculate TRx" in skill.metadata.triggers
        assert "gap_analyzer" in skill.metadata.agents

    def test_load_section(self, loader):
        """Test loading a specific section."""
        section = loader.load_section(
            "pharma-commercial/kpi-calculation.md", "TRx (Total Prescriptions)"
        )

        assert section is not None
        assert "Total prescriptions dispensed" in section

    def test_load_skill_not_found(self, loader):
        """Test loading a non-existent skill."""
        with pytest.raises(FileNotFoundError):
            loader.load("non-existent/skill.md")

    def test_parse_frontmatter(self, loader):
        """Test parsing YAML frontmatter."""
        content = """---
name: Test Skill
version: 1.0
triggers:
  - test
  - example
---

# Content here
"""
        skill = loader._parse_skill("test.md", content)

        assert skill.metadata.name == "Test Skill"
        assert skill.metadata.triggers == ["test", "example"]

    def test_parse_sections(self, loader):
        """Test parsing markdown sections."""
        content = """# Main Title

Introduction text.

## Section One

Section one content.

### Subsection

Subsection content.

## Section Two

Section two content.
"""
        sections = loader._parse_sections(content)

        assert "Main Title" in sections
        assert "Section One" in sections
        assert "Subsection" in sections
        assert "Section Two" in sections

    def test_list_skills(self, loader):
        """Test listing available skills."""
        skills = loader.list_skills("pharma-commercial")

        assert len(skills) > 0
        assert any("kpi-calculation.md" in s for s in skills)

    def test_list_skills_all_categories(self, loader):
        """Test listing skills from all categories."""
        skills = loader.list_skills()

        # Should include skills from multiple categories
        assert len(skills) > 0

    def test_caching(self, loader):
        """Test that skills are cached."""
        # Load same skill twice
        skill1 = loader.load("pharma-commercial/kpi-calculation.md")
        skill2 = loader.load("pharma-commercial/kpi-calculation.md")

        # Should be the same cached object
        assert skill1 is skill2

    def test_clear_cache(self, loader):
        """Test clearing the cache."""
        skill1 = loader.load("pharma-commercial/kpi-calculation.md")
        loader.clear_cache()
        skill2 = loader.load("pharma-commercial/kpi-calculation.md")

        # Should be different objects after cache clear
        assert skill1 is not skill2


class TestSkillLoaderConfounderSkill:
    """Tests for loading the confounder identification skill."""

    @pytest.fixture
    def loader(self):
        """Create a SkillLoader."""
        return SkillLoader()

    def test_load_confounder_skill(self, loader):
        """Test loading the confounder identification skill."""
        skill = loader.load("causal-inference/confounder-identification.md")

        assert skill.metadata.name == "Confounder Identification for Pharma Analytics"
        assert "causal_impact" in skill.metadata.agents

    def test_get_hcp_targeting_confounders(self, loader):
        """Test getting HCP targeting confounders section."""
        section = loader.load_section(
            "causal-inference/confounder-identification.md",
            "HCP Targeting → Prescription Impact",
        )

        assert section is not None
        assert "Territory potential" in section
        assert "HCP specialty" in section
