"""Workflow integration tests for agent-skill integration.

Tests that agents properly load skills during their actual workflow execution,
not just manual skill loading. These tests mock the graph execution to avoid
API calls while verifying skill loading behavior.

Test Coverage:
- CausalImpactAgent.run() loads causal-inference skills
- ExplainerAgent.explain() loads context-appropriate skills
- GapAnalyzerAgent.run() loads gap-analysis skills
- Skills are cleared between workflow invocations
- Brand-specific skills load conditionally
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Any, Dict


class TestCausalImpactWorkflowSkills:
    """Test skill loading during CausalImpactAgent workflow execution."""

    @pytest.mark.asyncio
    async def test_run_loads_core_causal_skills(self):
        """Test that run() loads core causal inference skills."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        # Mock the graph.ainvoke to avoid actual execution
        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            # Execute the workflow
            await agent.run({
                "query": "test query",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
            })

            # Verify core skills were loaded
            skill_names = agent.get_loaded_skill_names()
            assert "Confounder Identification for Pharma Analytics" in skill_names
            assert "DoWhy Causal Estimation Workflow" in skill_names

    @pytest.mark.asyncio
    async def test_run_loads_brand_skill_when_brand_specified(self):
        """Test that run() loads brand-specific skill when brand is in input."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            # Execute with brand specified
            await agent.run({
                "query": "test query",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
                "brand": "Remibrutinib",
            })

            # Verify brand skill was loaded
            skill_names = agent.get_loaded_skill_names()
            assert "Brand-Specific Analytics" in skill_names

    @pytest.mark.asyncio
    async def test_run_does_not_load_brand_skill_without_brand(self):
        """Test that run() does not load brand skill when brand is not specified."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            # Execute without brand
            await agent.run({
                "query": "test query",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
            })

            # Verify brand skill was NOT loaded
            skill_names = agent.get_loaded_skill_names()
            assert "Brand-Specific Analytics" not in skill_names

    @pytest.mark.asyncio
    async def test_run_clears_skills_from_previous_invocation(self):
        """Test that skills are cleared at the start of each run()."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        # Pre-load some skills (simulating previous invocation)
        await agent.load_skill("gap-analysis/roi-estimation.md")
        assert len(agent.get_loaded_skill_names()) == 1

        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            # Execute workflow
            await agent.run({
                "query": "test query",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
            })

            # Verify old skills were cleared and only causal skills remain
            skill_names = agent.get_loaded_skill_names()
            assert "ROI Estimation Procedures" not in skill_names
            assert "DoWhy Causal Estimation Workflow" in skill_names

    @pytest.mark.asyncio
    async def test_skill_context_available_during_workflow(self):
        """Test that skill context is available for use during workflow."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        # Track when skill context is retrieved
        skill_context_captured = None

        async def capture_skill_context(state):
            nonlocal skill_context_captured
            skill_context_captured = agent.get_skill_context()
            return state

        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            # Side effect to capture skill context during execution
            async def mock_execution(state):
                nonlocal skill_context_captured
                skill_context_captured = agent.get_skill_context()
                return mock_final_state

            mock_invoke.side_effect = mock_execution

            await agent.run({
                "query": "test query",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
            })

            # Verify skill context was available during workflow
            assert skill_context_captured is not None
            assert "DoWhy" in skill_context_captured


class TestGapAnalyzerWorkflowSkills:
    """Test skill loading during GapAnalyzerAgent workflow execution."""

    @pytest.mark.asyncio
    async def test_run_loads_core_gap_analysis_skills(self):
        """Test that run() loads core gap analysis skills."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        mock_final_state = {
            "gaps_detected": [{"gap_id": "1"}],
            "prioritized_opportunities": [],
            "quick_wins": [],
            "strategic_bets": [],
            "total_addressable_value": 0.0,
            "total_gap_value": 0.0,
            "segments_analyzed": 2,
            "executive_summary": "Test",
            "key_insights": [],
            "segments": ["seg1", "seg2"],
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            await agent.run({
                "query": "find ROI opportunities",
                "metrics": ["TRx", "NRx"],
                "segments": ["Northeast", "Southwest"],
                "brand": "Kisqali",
            })

            skill_names = agent.get_loaded_skill_names()
            assert "ROI Estimation Procedures" in skill_names
            # Note: gap-analysis/opportunity-sizing.md skill doesn't exist yet
            # The agent attempts to load it but gracefully continues without it

    @pytest.mark.asyncio
    async def test_run_loads_brand_skill_when_brand_specified(self):
        """Test that run() loads brand-specific skill when brand is in input."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        mock_final_state = {
            "gaps_detected": [{"gap_id": "1"}],
            "prioritized_opportunities": [],
            "quick_wins": [],
            "strategic_bets": [],
            "total_addressable_value": 0.0,
            "total_gap_value": 0.0,
            "segments_analyzed": 2,
            "executive_summary": "Test",
            "key_insights": [],
            "segments": ["seg1", "seg2"],
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            await agent.run({
                "query": "find ROI opportunities",
                "metrics": ["TRx", "NRx"],
                "segments": ["Northeast", "Southwest"],
                "brand": "Kisqali",
            })

            skill_names = agent.get_loaded_skill_names()
            assert "Brand-Specific Analytics" in skill_names

    @pytest.mark.asyncio
    async def test_run_clears_skills_between_invocations(self):
        """Test that skills are cleared between run() invocations."""
        from src.agents.gap_analyzer.agent import GapAnalyzerAgent

        agent = GapAnalyzerAgent(enable_mlflow=False, enable_opik=False)

        # Pre-load causal skills (simulating previous different analysis)
        await agent.load_skill("causal-inference/dowhy-workflow.md")
        assert "DoWhy Causal Estimation Workflow" in agent.get_loaded_skill_names()

        mock_final_state = {
            "gaps_detected": [{"gap_id": "1"}],
            "prioritized_opportunities": [],
            "quick_wins": [],
            "strategic_bets": [],
            "total_addressable_value": 0.0,
            "total_gap_value": 0.0,
            "segments_analyzed": 2,
            "executive_summary": "Test",
            "key_insights": [],
            "segments": ["seg1", "seg2"],
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            await agent.run({
                "query": "find ROI opportunities",
                "metrics": ["TRx", "NRx"],
                "segments": ["Northeast", "Southwest"],
                "brand": "Kisqali",
            })

            skill_names = agent.get_loaded_skill_names()
            # Old causal skill should be cleared
            assert "DoWhy Causal Estimation Workflow" not in skill_names
            # New gap analysis skills should be present
            assert "ROI Estimation Procedures" in skill_names


class TestExplainerWorkflowSkills:
    """Test skill loading during ExplainerAgent workflow execution."""

    @pytest.mark.asyncio
    async def test_explain_loads_causal_skill_for_causal_results(self):
        """Test that explain() loads causal skill when results contain causal data."""
        from src.agents.explainer.agent import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        mock_final_state = {
            "executive_summary": "Test summary",
            "detailed_explanation": "Test explanation",
            "narrative_sections": [],
            "extracted_insights": [],
            "key_themes": [],
            "visual_suggestions": [],
            "follow_up_questions": [],
            "total_latency_ms": 100,
            "model_used": "deterministic",
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        # Create mock graph
        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)

        with patch.object(agent, "_get_graph", return_value=mock_graph):
            # Results with causal effect data
            causal_results = [
                {"treatment_effect": 0.5, "confidence": 0.9}
            ]

            await agent.explain(
                analysis_results=causal_results,
                query="explain the causal effect",
            )

            skill_names = agent.get_loaded_skill_names()
            assert "DoWhy Causal Estimation Workflow" in skill_names

    @pytest.mark.asyncio
    async def test_explain_does_not_load_causal_skill_for_non_causal_results(self):
        """Test that explain() doesn't load causal skill for non-causal results."""
        from src.agents.explainer.agent import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        mock_final_state = {
            "executive_summary": "Test summary",
            "detailed_explanation": "Test explanation",
            "narrative_sections": [],
            "extracted_insights": [],
            "key_themes": [],
            "visual_suggestions": [],
            "follow_up_questions": [],
            "total_latency_ms": 100,
            "model_used": "deterministic",
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)

        with patch.object(agent, "_get_graph", return_value=mock_graph):
            # Results without causal data
            non_causal_results = [
                {"metric": "TRx", "value": 1000}
            ]

            await agent.explain(
                analysis_results=non_causal_results,
                query="explain the metrics",
            )

            skill_names = agent.get_loaded_skill_names()
            assert "DoWhy Causal Estimation Workflow" not in skill_names

    @pytest.mark.asyncio
    async def test_explain_loads_brand_skill_from_memory_config(self):
        """Test that explain() loads brand skill when brand is in memory_config."""
        from src.agents.explainer.agent import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        mock_final_state = {
            "executive_summary": "Test summary",
            "detailed_explanation": "Test explanation",
            "narrative_sections": [],
            "extracted_insights": [],
            "key_themes": [],
            "visual_suggestions": [],
            "follow_up_questions": [],
            "total_latency_ms": 100,
            "model_used": "deterministic",
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)

        with patch.object(agent, "_get_graph", return_value=mock_graph):
            await agent.explain(
                analysis_results=[{"metric": "TRx", "value": 1000}],
                query="explain metrics",
                memory_config={"brand": "Fabhalta"},
            )

            skill_names = agent.get_loaded_skill_names()
            assert "Brand-Specific Analytics" in skill_names

    @pytest.mark.asyncio
    async def test_explain_clears_skills_between_calls(self):
        """Test that skills are cleared between explain() calls."""
        from src.agents.explainer.agent import ExplainerAgent

        agent = ExplainerAgent(use_llm=False)

        # Pre-load some skills
        await agent.load_skill("gap-analysis/roi-estimation.md")
        assert len(agent.get_loaded_skill_names()) == 1

        mock_final_state = {
            "executive_summary": "Test summary",
            "detailed_explanation": "Test explanation",
            "narrative_sections": [],
            "extracted_insights": [],
            "key_themes": [],
            "visual_suggestions": [],
            "follow_up_questions": [],
            "total_latency_ms": 100,
            "model_used": "deterministic",
            "errors": [],
            "warnings": [],
            "status": "completed",
        }

        mock_graph = MagicMock()
        mock_graph.ainvoke = AsyncMock(return_value=mock_final_state)

        with patch.object(agent, "_get_graph", return_value=mock_graph):
            await agent.explain(
                analysis_results=[{"metric": "TRx", "value": 1000}],
                query="explain metrics",
            )

            skill_names = agent.get_loaded_skill_names()
            # Old skill should be cleared
            assert "ROI Estimation Procedures" not in skill_names


class TestSkillWorkflowRobustness:
    """Test skill loading robustness during workflow execution."""

    @pytest.mark.asyncio
    async def test_workflow_continues_if_skill_loading_fails(self):
        """Test that workflow proceeds even if skill loading fails."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            # Mock load_skill to raise an exception
            with patch.object(agent, "load_skill", side_effect=Exception("Skill loading failed")):
                # Workflow should still complete
                result = await agent.run({
                    "query": "test query",
                    "treatment_var": "treatment",
                    "outcome_var": "outcome",
                    "confounders": ["conf1"],
                    "data_source": "test_table",
                })

                # Workflow should have completed
                assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_multiple_sequential_runs_isolate_skills(self):
        """Test that multiple sequential runs have isolated skill contexts."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        mock_final_state = {
            "query_id": "test-123",
            "status": "completed",
            "interpretation": {"narrative": "Test narrative"},
            "estimation_result": {"ate": 0.5, "method": "propensity_score"},
            "refutation_results": {},
            "sensitivity_analysis": {},
            "causal_graph": {},
        }

        with patch.object(agent.graph, "ainvoke", new_callable=AsyncMock) as mock_invoke:
            mock_invoke.return_value = mock_final_state

            # First run with brand
            await agent.run({
                "query": "test query 1",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
                "brand": "Remibrutinib",
            })

            skills_after_first = set(agent.get_loaded_skill_names())
            assert "Brand-Specific Analytics" in skills_after_first

            # Second run without brand
            await agent.run({
                "query": "test query 2",
                "treatment_var": "treatment",
                "outcome_var": "outcome",
                "confounders": ["conf1"],
                "data_source": "test_table",
            })

            skills_after_second = set(agent.get_loaded_skill_names())
            # Brand skill should NOT be present in second run
            assert "Brand-Specific Analytics" not in skills_after_second


class TestSkillGuidanceMethod:
    """Test the get_skill_guidance method during workflow."""

    @pytest.mark.asyncio
    async def test_get_skill_guidance_returns_context_after_loading(self):
        """Test get_skill_guidance returns skill content after loading."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        # Before loading, guidance should be empty
        guidance = agent.get_skill_guidance("estimation")
        assert guidance == ""

        # Load skills
        await agent.load_skill("causal-inference/dowhy-workflow.md")

        # After loading, guidance should contain skill content
        guidance = agent.get_skill_guidance("estimation")
        assert guidance != ""
        assert "DoWhy" in guidance

    @pytest.mark.asyncio
    async def test_skill_guidance_available_for_different_phases(self):
        """Test that skill guidance is accessible for different analysis phases."""
        from src.agents.causal_impact.agent import CausalImpactAgent

        agent = CausalImpactAgent(enable_mlflow=False)

        # Load all skills
        await agent.load_skill("causal-inference/confounder-identification.md")
        await agent.load_skill("causal-inference/dowhy-workflow.md")

        # Guidance should be available for any phase
        graph_guidance = agent.get_skill_guidance("graph_building")
        estimation_guidance = agent.get_skill_guidance("estimation")
        refutation_guidance = agent.get_skill_guidance("refutation")

        # All should return the same full context (phase filtering is done by nodes)
        assert graph_guidance == estimation_guidance == refutation_guidance
        assert "confounder" in graph_guidance.lower()
