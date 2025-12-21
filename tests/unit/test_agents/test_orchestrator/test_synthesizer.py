"""Tests for synthesizer node."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agents.orchestrator.nodes.synthesizer import (
    SynthesizerNode,
    synthesize_response,
)


class TestSynthesizerNode:
    """Test SynthesizerNode."""

    @pytest.mark.asyncio
    async def test_synthesize_single_agent(self):
        """Test synthesis with single successful agent."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": True,
                    "result": {
                        "narrative": "HCP engagement has a significant positive effect on conversions (ATE=0.12).",
                        "recommendations": [
                            "Increase HCP engagement",
                            "Focus on high-potential HCPs",
                        ],
                        "confidence": 0.87,
                        "follow_up_suggestions": ["Design A/B test"],
                    },
                    "error": None,
                    "latency_ms": 1500,
                }
            ],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 1600,
        }

        result = await synthesizer.execute(state)

        assert "synthesized_response" in result
        assert "HCP engagement" in result["synthesized_response"]
        assert result["response_confidence"] == 0.87
        assert len(result["recommendations"]) == 2
        assert result["current_phase"] == "complete"
        assert result["status"] == "completed"
        assert result["synthesis_latency_ms"] >= 0
        assert result["total_latency_ms"] > 0

    @pytest.mark.asyncio
    async def test_synthesize_multiple_agents(self):
        """Test synthesis with multiple successful agents."""
        # Mock LLM response
        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Synthesized response combining all agent insights."
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        synthesizer = SynthesizerNode()
        synthesizer.llm = mock_llm

        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": True,
                    "result": {
                        "narrative": "HCP engagement drives conversions.",
                        "recommendations": ["Increase engagement"],
                        "confidence": 0.87,
                    },
                    "error": None,
                    "latency_ms": 1500,
                },
                {
                    "agent_name": "gap_analyzer",
                    "success": True,
                    "result": {
                        "narrative": "Identified gaps in Northeast region.",
                        "recommendations": ["Expand coverage"],
                        "confidence": 0.82,
                    },
                    "error": None,
                    "latency_ms": 1200,
                },
            ],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 2800,
        }

        result = await synthesizer.execute(state)

        # Verify LLM was called
        assert mock_llm.ainvoke.called
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert "causal_impact" in call_args
        assert "gap_analyzer" in call_args

        # Verify synthesis
        assert (
            result["synthesized_response"] == "Synthesized response combining all agent insights."
        )
        assert result["response_confidence"] == pytest.approx(
            0.845, 0.01
        )  # Average of 0.87 and 0.82
        assert "Increase engagement" in result["recommendations"]
        assert "Expand coverage" in result["recommendations"]

    @pytest.mark.asyncio
    async def test_synthesize_all_failed(self):
        """Test synthesis when all agents fail."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": False,
                    "result": None,
                    "error": "Agent timed out after 30000ms",
                    "latency_ms": 30000,
                },
                {
                    "agent_name": "gap_analyzer",
                    "success": False,
                    "result": None,
                    "error": "Connection error",
                    "latency_ms": 500,
                },
            ],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 30500,
        }

        result = await synthesizer.execute(state)

        assert "unable to complete" in result["synthesized_response"].lower()
        assert "causal_impact" in result["synthesized_response"]
        assert "gap_analyzer" in result["synthesized_response"]
        assert result["response_confidence"] == 0.0
        assert result["status"] == "failed"
        assert "Simplify your question" in result["follow_up_suggestions"]

    @pytest.mark.asyncio
    async def test_synthesize_mixed_success_failure(self):
        """Test synthesis with some agents succeeding and some failing."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": True,
                    "result": {
                        "narrative": "HCP engagement drives conversions.",
                        "recommendations": ["Increase engagement"],
                        "confidence": 0.87,
                    },
                    "error": None,
                    "latency_ms": 1500,
                },
                {
                    "agent_name": "gap_analyzer",
                    "success": False,
                    "result": None,
                    "error": "Timeout",
                    "latency_ms": 20000,
                },
            ],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 21500,
        }

        result = await synthesizer.execute(state)

        # Should extract from successful agent only
        assert "HCP engagement" in result["synthesized_response"]
        assert result["status"] == "completed"  # At least one succeeded
        assert result["response_confidence"] == 0.87

    @pytest.mark.asyncio
    async def test_synthesize_empty_results(self):
        """Test synthesis with empty results list."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 0,
        }

        result = await synthesizer.execute(state)

        # Should generate error response
        assert result["status"] == "failed"
        assert result["response_confidence"] == 0.0

    @pytest.mark.asyncio
    async def test_total_latency_calculation(self):
        """Test total latency calculation."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": True,
                    "result": {
                        "narrative": "Test narrative",
                        "confidence": 0.8,
                    },
                    "error": None,
                    "latency_ms": 1000,
                }
            ],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 1200,
        }

        result = await synthesizer.execute(state)

        # Total = classification + routing + dispatch + synthesis
        assert result["total_latency_ms"] >= 300 + 20 + 1200  # >= 1520ms
        assert result["synthesis_latency_ms"] >= 0

    @pytest.mark.asyncio
    async def test_synthesize_response_function(self):
        """Test standalone synthesize_response function."""
        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": True,
                    "result": {
                        "narrative": "Test narrative",
                        "recommendations": ["Test rec"],
                        "confidence": 0.85,
                    },
                    "error": None,
                    "latency_ms": 1000,
                }
            ],
            "classification_latency_ms": 300,
            "routing_latency_ms": 20,
            "dispatch_latency_ms": 1200,
        }

        result = await synthesize_response(state)

        assert "synthesized_response" in result
        assert result["status"] == "completed"


class TestExtractResponse:
    """Test _extract_response method."""

    def test_extract_with_narrative(self):
        """Test extraction when result has narrative field."""
        synthesizer = SynthesizerNode()

        result = {
            "agent_name": "causal_impact",
            "success": True,
            "result": {
                "narrative": "Analysis shows significant effect.",
                "recommendations": ["Rec 1", "Rec 2"],
                "confidence": 0.87,
                "follow_up_suggestions": ["Follow up 1"],
            },
        }

        extracted = synthesizer._extract_response(result)

        assert extracted["response"] == "Analysis shows significant effect."
        assert extracted["confidence"] == 0.87
        assert extracted["recommendations"] == ["Rec 1", "Rec 2"]
        assert extracted["follow_ups"] == ["Follow up 1"]

    def test_extract_with_response_field(self):
        """Test extraction when result has response instead of narrative."""
        synthesizer = SynthesizerNode()

        result = {
            "agent_name": "explainer",
            "success": True,
            "result": {
                "response": "Explanation of the analysis.",
                "recommendations": ["Rec 1"],
                "confidence": 0.75,
            },
        }

        extracted = synthesizer._extract_response(result)

        assert extracted["response"] == "Explanation of the analysis."
        assert extracted["confidence"] == 0.75

    def test_extract_with_minimal_result(self):
        """Test extraction with minimal result fields."""
        synthesizer = SynthesizerNode()

        result = {"agent_name": "test_agent", "success": True, "result": {}}

        extracted = synthesizer._extract_response(result)

        assert extracted["response"] == "{}"  # Fallback to str(agent_output)
        assert extracted["confidence"] == 0.5  # Default
        assert extracted["recommendations"] == []
        assert extracted["follow_ups"] == []


class TestSynthesizeMultiple:
    """Test _synthesize_multiple method."""

    @pytest.mark.asyncio
    async def test_synthesize_multiple_success(self):
        """Test successful multi-agent synthesis."""
        # Mock LLM
        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Unified synthesis of all insights."
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        synthesizer = SynthesizerNode()
        synthesizer.llm = mock_llm

        results = [
            {
                "agent_name": "causal_impact",
                "result": {
                    "narrative": "First agent narrative.",
                    "recommendations": ["Rec 1"],
                    "confidence": 0.9,
                },
            },
            {
                "agent_name": "gap_analyzer",
                "result": {
                    "narrative": "Second agent narrative.",
                    "recommendations": ["Rec 2", "Rec 3"],
                    "confidence": 0.8,
                },
            },
        ]

        synthesized = await synthesizer._synthesize_multiple(results)

        assert synthesized["response"] == "Unified synthesis of all insights."
        assert synthesized["confidence"] == 0.85  # Average of 0.9 and 0.8
        assert len(synthesized["recommendations"]) == 3
        assert "Rec 1" in synthesized["recommendations"]
        assert "Rec 2" in synthesized["recommendations"]

    @pytest.mark.asyncio
    async def test_synthesize_multiple_llm_failure(self):
        """Test multi-agent synthesis with LLM failure (fallback to concatenation)."""
        # Mock LLM that raises exception
        mock_llm = MagicMock()
        mock_llm.ainvoke = AsyncMock(side_effect=Exception("LLM error"))

        synthesizer = SynthesizerNode()
        synthesizer.llm = mock_llm

        results = [
            {
                "agent_name": "causal_impact",
                "result": {
                    "narrative": "First narrative.",
                    "recommendations": ["Rec 1"],
                    "confidence": 0.9,
                },
            },
            {
                "agent_name": "gap_analyzer",
                "result": {
                    "narrative": "Second narrative.",
                    "recommendations": ["Rec 2"],
                    "confidence": 0.8,
                },
            },
        ]

        synthesized = await synthesizer._synthesize_multiple(results)

        # Should fallback to concatenation
        assert "First narrative" in synthesized["response"]
        assert "Second narrative" in synthesized["response"]
        assert synthesized["confidence"] == 0.85

    @pytest.mark.asyncio
    async def test_synthesize_multiple_truncates_narratives(self):
        """Test that long narratives are truncated in synthesis prompt."""
        # Create long narratives
        long_narrative = "A" * 1000  # 1000 chars

        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Synthesized."
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        synthesizer = SynthesizerNode()
        synthesizer.llm = mock_llm

        results = [
            {
                "agent_name": "agent1",
                "result": {"narrative": long_narrative, "confidence": 0.8},
            }
        ]

        await synthesizer._synthesize_multiple(results)

        # Check that prompt was created (and narrative was truncated to 500 chars)
        call_args = mock_llm.ainvoke.call_args[0][0]
        assert len(call_args) < 1500  # Should be much shorter due to truncation

    @pytest.mark.asyncio
    async def test_synthesize_multiple_limits_recommendations(self):
        """Test that recommendations are limited to top 5."""
        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Synthesized."
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        synthesizer = SynthesizerNode()
        synthesizer.llm = mock_llm

        results = [
            {
                "agent_name": "agent1",
                "result": {
                    "narrative": "Narrative 1",
                    "recommendations": ["R1", "R2", "R3"],
                    "confidence": 0.8,
                },
            },
            {
                "agent_name": "agent2",
                "result": {
                    "narrative": "Narrative 2",
                    "recommendations": ["R4", "R5", "R6"],
                    "confidence": 0.7,
                },
            },
        ]

        synthesized = await synthesizer._synthesize_multiple(results)

        # Should limit to 5 recommendations
        assert len(synthesized["recommendations"]) == 5


class TestGenerateErrorResponse:
    """Test _generate_error_response method."""

    def test_generate_error_single_failure(self):
        """Test error response with single failed agent."""
        synthesizer = SynthesizerNode()

        failed_results = [{"agent_name": "causal_impact", "error": "Agent timed out after 30000ms"}]

        error_response = synthesizer._generate_error_response(failed_results)

        assert "unable to complete" in error_response["response"].lower()
        assert "causal_impact" in error_response["response"]
        assert "timed out" in error_response["response"].lower()
        assert error_response["confidence"] == 0.0
        assert len(error_response["recommendations"]) == 0
        assert "Simplify your question" in error_response["follow_ups"]

    def test_generate_error_multiple_failures(self):
        """Test error response with multiple failed agents."""
        synthesizer = SynthesizerNode()

        failed_results = [
            {"agent_name": "causal_impact", "error": "Timeout"},
            {"agent_name": "gap_analyzer", "error": "Connection error"},
            {"agent_name": "explainer", "error": "Unknown error"},
        ]

        error_response = synthesizer._generate_error_response(failed_results)

        assert "causal_impact" in error_response["response"]
        assert "gap_analyzer" in error_response["response"]
        assert "explainer" in error_response["response"]
        assert "Timeout" in error_response["response"]
        assert "Connection error" in error_response["response"]


class TestConfidenceCalculation:
    """Test confidence calculation."""

    @pytest.mark.asyncio
    async def test_confidence_single_agent(self):
        """Test confidence with single agent."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [
                {
                    "agent_name": "causal_impact",
                    "success": True,
                    "result": {"narrative": "Test", "confidence": 0.92},
                }
            ],
            "classification_latency_ms": 0,
            "routing_latency_ms": 0,
            "dispatch_latency_ms": 0,
        }

        result = await synthesizer.execute(state)

        assert result["response_confidence"] == 0.92

    @pytest.mark.asyncio
    async def test_confidence_multiple_agents(self):
        """Test confidence averaging with multiple agents."""
        mock_llm = MagicMock()
        mock_llm_response = MagicMock()
        mock_llm_response.content = "Synthesized."
        mock_llm.ainvoke = AsyncMock(return_value=mock_llm_response)

        synthesizer = SynthesizerNode()
        synthesizer.llm = mock_llm

        state = {
            "agent_results": [
                {
                    "agent_name": "agent1",
                    "success": True,
                    "result": {"narrative": "Test", "confidence": 0.9},
                },
                {
                    "agent_name": "agent2",
                    "success": True,
                    "result": {"narrative": "Test", "confidence": 0.7},
                },
                {
                    "agent_name": "agent3",
                    "success": True,
                    "result": {"narrative": "Test", "confidence": 0.8},
                },
            ],
            "classification_latency_ms": 0,
            "routing_latency_ms": 0,
            "dispatch_latency_ms": 0,
        }

        result = await synthesizer.execute(state)

        # Average: (0.9 + 0.7 + 0.8) / 3 = 0.8
        assert result["response_confidence"] == pytest.approx(0.8, 0.01)

    @pytest.mark.asyncio
    async def test_confidence_failed_agents(self):
        """Test confidence when all agents fail."""
        synthesizer = SynthesizerNode()

        state = {
            "agent_results": [{"agent_name": "agent1", "success": False, "error": "Failed"}],
            "classification_latency_ms": 0,
            "routing_latency_ms": 0,
            "dispatch_latency_ms": 0,
        }

        result = await synthesizer.execute(state)

        assert result["response_confidence"] == 0.0
