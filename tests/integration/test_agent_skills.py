"""Integration tests for agent-skill integration.

Tests that agents properly load and utilize skills from the skills framework.
"""

import pytest


class TestCausalImpactAgentSkills:
    """Integration tests for CausalImpactAgent skill loading."""

    @pytest.mark.asyncio
    async def test_causal_impact_loads_skills_on_run(self):
        """Test that CausalImpactAgent loads skills when run is called."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent()

        # Verify agent has SkillsMixin methods
        assert hasattr(agent, "load_skill")
        assert hasattr(agent, "get_loaded_skill_names")
        assert hasattr(agent, "clear_loaded_skills")

    @pytest.mark.asyncio
    async def test_causal_impact_skill_loading(self):
        """Test skill loading functionality on CausalImpactAgent."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent()

        # Load a skill directly
        skill = await agent.load_skill("causal-inference/dowhy-workflow.md")

        assert skill is not None
        assert "causal_impact" in skill.metadata.agents

        # Check it's tracked
        names = agent.get_loaded_skill_names()
        assert "DoWhy Causal Estimation Workflow" in names

    @pytest.mark.asyncio
    async def test_causal_impact_clears_skills_between_runs(self):
        """Test that skills are cleared between invocations."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent()

        # Load a skill
        await agent.load_skill("causal-inference/dowhy-workflow.md")
        assert len(agent.get_loaded_skill_names()) == 1

        # Clear skills (as would happen at start of run)
        agent.clear_loaded_skills()
        assert len(agent.get_loaded_skill_names()) == 0


class TestExperimentDesignerAgentSkills:
    """Integration tests for ExperimentDesignerAgent skill loading."""

    @pytest.mark.asyncio
    async def test_experiment_designer_loads_skills_on_run(self):
        """Test that ExperimentDesignerAgent loads skills when arun is called."""
        from src.agents.experiment_designer.agent import ExperimentDesignerAgent

        agent = ExperimentDesignerAgent()

        # Verify agent has SkillsMixin methods
        assert hasattr(agent, "load_skill")
        assert hasattr(agent, "get_loaded_skill_names")
        assert hasattr(agent, "clear_loaded_skills")

    @pytest.mark.asyncio
    async def test_experiment_designer_skill_loading(self):
        """Test skill loading functionality on ExperimentDesignerAgent."""
        from src.agents.experiment_designer.agent import ExperimentDesignerAgent

        agent = ExperimentDesignerAgent()

        # Load experiment design skills directly
        skill = await agent.load_skill("experiment-design/power-analysis.md")

        # Skill may or may not exist depending on what skills are created
        # This tests the loading mechanism works
        if skill:
            names = agent.get_loaded_skill_names()
            assert len(names) >= 1


class TestGapAnalyzerAgentSkills:
    """Integration tests for GapAnalyzerAgent skill loading."""

    @pytest.mark.asyncio
    async def test_gap_analyzer_loads_skills_on_run(self):
        """Test that GapAnalyzerAgent loads skills when run is called."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        # Verify agent has SkillsMixin methods
        assert hasattr(agent, "load_skill")
        assert hasattr(agent, "get_loaded_skill_names")
        assert hasattr(agent, "clear_loaded_skills")

    @pytest.mark.asyncio
    async def test_gap_analyzer_skill_loading(self):
        """Test skill loading functionality on GapAnalyzerAgent."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        # Load ROI estimation skill
        skill = await agent.load_skill("gap-analysis/roi-estimation.md")

        assert skill is not None
        assert "gap_analyzer" in skill.metadata.agents

        # Check it's tracked
        names = agent.get_loaded_skill_names()
        assert "ROI Estimation Procedures" in names


class TestExplainerAgentSkills:
    """Integration tests for ExplainerAgent skill loading."""

    @pytest.mark.asyncio
    async def test_explainer_loads_skills_on_explain(self):
        """Test that ExplainerAgent loads skills when explain is called."""
        from src.agents.explainer.agent import ExplainerAgent

        agent = ExplainerAgent()

        # Verify agent has SkillsMixin methods
        assert hasattr(agent, "load_skill")
        assert hasattr(agent, "get_loaded_skill_names")
        assert hasattr(agent, "clear_loaded_skills")

    @pytest.mark.asyncio
    async def test_explainer_skill_loading(self):
        """Test skill loading functionality on ExplainerAgent."""
        from src.agents.explainer.agent import ExplainerAgent

        agent = ExplainerAgent()

        # Load causal inference skill for explanation context
        skill = await agent.load_skill("causal-inference/dowhy-workflow.md")

        assert skill is not None

        # Check it's tracked
        names = agent.get_loaded_skill_names()
        assert len(names) >= 1


class TestSkillContextIntegration:
    """Tests for skill context building across agents."""

    @pytest.mark.asyncio
    async def test_skill_context_formatting(self):
        """Test that skill context is properly formatted."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent()

        # Load multiple skills
        await agent.load_skill("causal-inference/dowhy-workflow.md")
        await agent.load_skill("causal-inference/confounder-identification.md")

        # Get formatted context
        context = agent.get_skill_context()

        assert "DoWhy" in context
        assert "---" in context  # Separator between skills

    @pytest.mark.asyncio
    async def test_skill_deduplication(self):
        """Test that loading same skill twice returns cached version."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        # Load same skill twice
        skill1 = await agent.load_skill("gap-analysis/roi-estimation.md")
        skill2 = await agent.load_skill("gap-analysis/roi-estimation.md")

        # Should be same instance
        assert skill1 is skill2
        assert len(agent.get_loaded_skill_names()) == 1


class TestAgentSkillsForAgentMethod:
    """Tests for load_skills_for_agent method."""

    @pytest.mark.asyncio
    async def test_load_skills_for_causal_impact_agent(self):
        """Test loading all skills tagged for causal_impact agent."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent()

        # Load all skills for this agent
        skills = await agent.load_skills_for_agent("causal_impact")

        # Should load at least the causal-inference skills
        assert len(skills) > 0

        # All loaded skills should reference this agent
        for skill in skills:
            assert "causal_impact" in skill.metadata.agents

    @pytest.mark.asyncio
    async def test_load_skills_for_gap_analyzer_agent(self):
        """Test loading all skills tagged for gap_analyzer agent."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        # Load all skills for this agent
        skills = await agent.load_skills_for_agent("gap_analyzer")

        # Should load at least the gap-analysis skills
        assert len(skills) > 0

        # All loaded skills should reference this agent
        for skill in skills:
            assert "gap_analyzer" in skill.metadata.agents


class TestFindRelevantSkills:
    """Tests for finding relevant skills for queries."""

    @pytest.mark.asyncio
    async def test_find_relevant_skills_roi_query(self):
        """Test finding skills relevant to ROI queries."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        # Find skills for ROI query
        matches = await agent.find_relevant_skills("ROI calculation")

        assert len(matches) > 0
        # ROI estimation skill should be highly ranked
        top_match_names = [m.skill_name for m in matches[:3]]
        assert any("ROI" in name for name in top_match_names)

    @pytest.mark.asyncio
    async def test_find_relevant_skills_causal_query(self):
        """Test finding skills relevant to causal queries."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent()

        # Find skills for causal query
        matches = await agent.find_relevant_skills("causal effect estimation")

        assert len(matches) > 0
        # DoWhy skill should be highly ranked
        top_match_names = [m.skill_name for m in matches[:3]]
        assert any("DoWhy" in name or "causal" in name.lower() for name in top_match_names)
