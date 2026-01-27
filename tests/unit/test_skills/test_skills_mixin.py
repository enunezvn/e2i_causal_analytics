"""Unit tests for SkillsMixin functionality.

Tests the mixin that provides skill loading capabilities to agents.
"""

import pytest

from src.agents.base import SkillsMixin
from src.skills import Skill, SkillMatch


class MockAgent(SkillsMixin):
    """Mock agent class for testing SkillsMixin."""

    def __init__(self):
        """Initialize mock agent."""
        self.agent_name = "mock_agent"


class TestSkillsMixinInitialization:
    """Tests for SkillsMixin initialization and lazy loading."""

    def test_mixin_lazy_initialization(self):
        """Test that skill loader is lazily initialized."""
        agent = MockAgent()

        # Should not have loader initially
        assert not hasattr(agent, "_skill_loader") or agent._skill_loader is None

        # Access loader
        loader = agent._get_skill_loader()

        # Now it should be initialized
        assert loader is not None
        assert agent._skill_loader is not None

    def test_mixin_lazy_matcher_initialization(self):
        """Test that skill matcher is lazily initialized."""
        agent = MockAgent()

        # Should not have matcher initially
        assert not hasattr(agent, "_skill_matcher") or agent._skill_matcher is None

        # Access matcher
        matcher = agent._get_skill_matcher()

        # Now it should be initialized
        assert matcher is not None
        assert agent._skill_matcher is not None

    def test_loaded_skills_initialization(self):
        """Test that loaded skills list is properly initialized."""
        agent = MockAgent()

        # Should not have list initially
        assert not hasattr(agent, "_loaded_skills")

        # Ensure list
        skills = agent._ensure_loaded_skills_list()

        # Now it should be initialized
        assert skills == []
        assert agent._loaded_skills == []


class TestSkillsMixinLoadSkill:
    """Tests for load_skill method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    @pytest.mark.asyncio
    async def test_load_existing_skill(self, agent):
        """Test loading an existing skill file."""
        skill = await agent.load_skill("gap-analysis/roi-estimation.md")

        assert skill is not None
        assert isinstance(skill, Skill)
        assert skill.metadata.name == "ROI Estimation Procedures"
        assert "gap_analyzer" in skill.metadata.agents

    @pytest.mark.asyncio
    async def test_load_nonexistent_skill(self, agent):
        """Test loading a non-existent skill returns None."""
        skill = await agent.load_skill("nonexistent/skill.md")

        assert skill is None

    @pytest.mark.asyncio
    async def test_load_skill_tracked(self, agent):
        """Test that loaded skills are tracked."""
        skill = await agent.load_skill("gap-analysis/roi-estimation.md")

        assert skill is not None
        assert skill in agent._loaded_skills
        assert len(agent._loaded_skills) == 1

    @pytest.mark.asyncio
    async def test_load_same_skill_twice(self, agent):
        """Test that loading same skill twice returns cached version."""
        skill1 = await agent.load_skill("gap-analysis/roi-estimation.md")
        skill2 = await agent.load_skill("gap-analysis/roi-estimation.md")

        assert skill1 is skill2
        assert len(agent._loaded_skills) == 1

    @pytest.mark.asyncio
    async def test_load_multiple_skills(self, agent):
        """Test loading multiple different skills."""
        skill1 = await agent.load_skill("gap-analysis/roi-estimation.md")
        skill2 = await agent.load_skill("causal-inference/dowhy-workflow.md")

        assert skill1 is not None
        assert skill2 is not None
        assert len(agent._loaded_skills) == 2


class TestSkillsMixinLoadSkillSection:
    """Tests for load_skill_section method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    @pytest.mark.asyncio
    async def test_load_existing_section(self, agent):
        """Test loading an existing section."""
        # Use "Standard Revenue Multipliers" which has actual content
        # ("Revenue Impact Calculation" is empty since content is in subsections)
        section = await agent.load_skill_section(
            "gap-analysis/roi-estimation.md",
            "Standard Revenue Multipliers"
        )

        assert section is not None
        assert "TRx" in section
        assert "$500" in section

    @pytest.mark.asyncio
    async def test_load_nonexistent_section(self, agent):
        """Test loading a non-existent section returns None."""
        section = await agent.load_skill_section(
            "gap-analysis/roi-estimation.md",
            "Nonexistent Section"
        )

        assert section is None

    @pytest.mark.asyncio
    async def test_load_section_from_nonexistent_skill(self, agent):
        """Test loading section from non-existent skill returns None."""
        section = await agent.load_skill_section(
            "nonexistent/skill.md",
            "Some Section"
        )

        assert section is None


class TestSkillsMixinFindRelevantSkills:
    """Tests for find_relevant_skills method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    @pytest.mark.asyncio
    async def test_find_relevant_skills(self, agent):
        """Test finding relevant skills for a query."""
        matches = await agent.find_relevant_skills("ROI calculation")

        assert len(matches) > 0
        assert all(isinstance(m, SkillMatch) for m in matches)

    @pytest.mark.asyncio
    async def test_find_relevant_skills_top_k(self, agent):
        """Test that top_k limits results."""
        matches = await agent.find_relevant_skills("analysis", top_k=2)

        assert len(matches) <= 2

    @pytest.mark.asyncio
    async def test_find_relevant_skills_sorted_by_score(self, agent):
        """Test that results are sorted by score descending."""
        matches = await agent.find_relevant_skills("causal estimation")

        if len(matches) >= 2:
            scores = [m.score for m in matches]
            assert scores == sorted(scores, reverse=True)


class TestSkillsMixinGetSkillContext:
    """Tests for get_skill_context method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    def test_get_skill_context_empty(self, agent):
        """Test getting context when no skills loaded."""
        context = agent.get_skill_context()

        assert context == ""

    @pytest.mark.asyncio
    async def test_get_skill_context_single_skill(self, agent):
        """Test getting context with one loaded skill."""
        await agent.load_skill("gap-analysis/roi-estimation.md")
        context = agent.get_skill_context()

        assert "ROI Estimation Procedures" in context
        assert "TRx" in context

    @pytest.mark.asyncio
    async def test_get_skill_context_multiple_skills(self, agent):
        """Test getting context with multiple loaded skills."""
        await agent.load_skill("gap-analysis/roi-estimation.md")
        await agent.load_skill("causal-inference/dowhy-workflow.md")
        context = agent.get_skill_context()

        assert "ROI Estimation Procedures" in context
        assert "DoWhy" in context
        assert "---" in context  # Separator between skills


class TestSkillsMixinClearLoadedSkills:
    """Tests for clear_loaded_skills method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    @pytest.mark.asyncio
    async def test_clear_loaded_skills(self, agent):
        """Test clearing loaded skills."""
        await agent.load_skill("gap-analysis/roi-estimation.md")
        assert len(agent._loaded_skills) == 1

        agent.clear_loaded_skills()

        assert len(agent._loaded_skills) == 0

    def test_clear_loaded_skills_empty(self, agent):
        """Test clearing when no skills loaded."""
        agent.clear_loaded_skills()

        assert agent._loaded_skills == []


class TestSkillsMixinGetLoadedSkillNames:
    """Tests for get_loaded_skill_names method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    def test_get_loaded_skill_names_empty(self, agent):
        """Test getting names when no skills loaded."""
        names = agent.get_loaded_skill_names()

        assert names == []

    @pytest.mark.asyncio
    async def test_get_loaded_skill_names(self, agent):
        """Test getting names of loaded skills."""
        await agent.load_skill("gap-analysis/roi-estimation.md")
        await agent.load_skill("causal-inference/dowhy-workflow.md")
        names = agent.get_loaded_skill_names()

        assert len(names) == 2
        assert "ROI Estimation Procedures" in names
        assert "DoWhy Causal Estimation Workflow" in names


class TestSkillsMixinLoadSkillsForAgent:
    """Tests for load_skills_for_agent method."""

    @pytest.fixture
    def agent(self):
        """Create a mock agent."""
        return MockAgent()

    @pytest.mark.asyncio
    async def test_load_skills_for_causal_impact(self, agent):
        """Test loading skills for causal_impact agent."""
        skills = await agent.load_skills_for_agent("causal_impact")

        assert len(skills) > 0
        for skill in skills:
            assert "causal_impact" in skill.metadata.agents

    @pytest.mark.asyncio
    async def test_load_skills_for_gap_analyzer(self, agent):
        """Test loading skills for gap_analyzer agent."""
        skills = await agent.load_skills_for_agent("gap_analyzer")

        assert len(skills) > 0
        for skill in skills:
            assert "gap_analyzer" in skill.metadata.agents

    @pytest.mark.asyncio
    async def test_load_skills_for_nonexistent_agent(self, agent):
        """Test loading skills for non-existent agent returns empty list."""
        skills = await agent.load_skills_for_agent("nonexistent_agent")

        assert skills == []

    @pytest.mark.asyncio
    async def test_load_skills_for_agent_tracked(self, agent):
        """Test that skills loaded for agent are tracked."""
        skills = await agent.load_skills_for_agent("gap_analyzer")

        assert len(skills) > 0
        assert len(agent._loaded_skills) == len(skills)
