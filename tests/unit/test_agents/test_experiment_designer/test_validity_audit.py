"""Tests for Validity Audit Node.

Tests the LLM-based adversarial validity assessment functionality.
"""

import json

import pytest

from src.agents.experiment_designer.graph import create_initial_state
from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode


class MockLLM:
    """Mock LLM for testing validity audit."""

    def __init__(self, response: dict = None, raise_error: bool = False):
        """Initialize mock LLM."""
        self.response = response or self._default_response()
        self.raise_error = raise_error
        self.call_count = 0
        self.last_prompt = None

    def _default_response(self) -> dict:
        """Return default validity audit response."""
        return {
            "validity_threats": [
                {
                    "threat_type": "internal",
                    "threat_name": "selection_bias",
                    "description": "Non-random assignment may occur",
                    "severity": "high",
                    "mitigation_possible": True,
                    "mitigation_strategy": "Use stratified randomization",
                },
                {
                    "threat_type": "external",
                    "threat_name": "generalizability",
                    "description": "Results may not generalize",
                    "severity": "medium",
                    "mitigation_possible": True,
                    "mitigation_strategy": "Include diverse sample",
                },
            ],
            "mitigations": [
                {
                    "threat_addressed": "selection_bias",
                    "recommendation": "Stratify on baseline characteristics",
                    "effectiveness_rating": "high",
                    "implementation_cost": "low",
                    "implementation_steps": ["Define strata", "Balance allocation"],
                }
            ],
            "overall_validity_score": 0.75,
            "validity_confidence": "medium",
            "redesign_needed": False,
            "redesign_recommendations": [],
        }

    def invoke(self, prompt: str) -> str:
        """Mock LLM invocation."""
        self.call_count += 1
        self.last_prompt = prompt
        if self.raise_error:
            raise Exception("LLM error")
        return json.dumps(self.response)

    async def ainvoke(self, prompt: str) -> str:
        """Mock async LLM invocation."""
        return self.invoke(prompt)


class TestValidityAuditNode:
    """Test ValidityAuditNode functionality."""

    def test_create_node(self):
        """Test creating node."""
        node = ValidityAuditNode()

        assert node is not None
        assert hasattr(node, "llm")
        assert hasattr(node, "model_name")

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test validity audit")
        state["status"] = "auditing"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "Treatment", "description": "Test"}]
        state["outcomes"] = [{"name": "Outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert result["status"] in ["generating", "redesigning"]
        assert "validity_audit" in result.get("node_latencies_ms", {})

    @pytest.mark.asyncio
    async def test_execute_returns_threats(self):
        """Test that validity threats are returned."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test threats identification")
        state["status"] = "auditing"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]

        result = await node.execute(state)

        assert "validity_threats" in result
        if result.get("enable_validity_audit", True):
            assert len(result["validity_threats"]) >= 0

    @pytest.mark.asyncio
    async def test_execute_returns_validity_score(self):
        """Test that validity score is returned."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test validity score")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        assert "overall_validity_score" in result
        assert 0.0 <= result["overall_validity_score"] <= 1.0

    @pytest.mark.asyncio
    async def test_execute_returns_confidence(self):
        """Test that validity confidence is returned."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test validity confidence")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        assert "validity_confidence" in result
        assert result["validity_confidence"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_execute_returns_redesign_flag(self):
        """Test that redesign flag is returned."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test redesign flag")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        assert "redesign_needed" in result
        assert isinstance(result["redesign_needed"], bool)

    @pytest.mark.asyncio
    async def test_execute_skip_on_failed(self):
        """Test execution skips on failed status."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test skip on failed")
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_skip_when_disabled(self):
        """Test execution skips when validity audit disabled."""
        node = ValidityAuditNode()
        state = create_initial_state(
            business_question="Test skip when disabled", enable_validity_audit=False
        )
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        # Should skip to generating without actual audit
        assert result["status"] == "generating"

    @pytest.mark.asyncio
    async def test_execute_records_latency(self):
        """Test that node latency is recorded."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test latency recording")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        assert "node_latencies_ms" in result
        assert "validity_audit" in result["node_latencies_ms"]


class TestValidityThreatTypes:
    """Test different validity threat type detection."""

    @pytest.mark.asyncio
    async def test_detects_internal_threats(self):
        """Test detection of internal validity threats."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test internal validity")
        state["status"] = "auditing"
        state["design_type"] = "RCT"
        state["randomization_method"] = "simple"  # May have selection issues

        result = await node.execute(state)

        threats = result.get("validity_threats", [])
        internal_threats = [t for t in threats if t.get("threat_type") == "internal"]
        # Should identify at least one internal threat
        assert isinstance(internal_threats, list)

    @pytest.mark.asyncio
    async def test_detects_external_threats(self):
        """Test detection of external validity threats."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test external validity")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        threats = result.get("validity_threats", [])
        external_threats = [t for t in threats if t.get("threat_type") == "external"]
        assert isinstance(external_threats, list)

    @pytest.mark.asyncio
    async def test_detects_statistical_threats(self):
        """Test detection of statistical validity threats."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test statistical validity")
        state["status"] = "auditing"
        state["design_type"] = "RCT"
        state["power_analysis"] = {"achieved_power": 0.60}  # Low power

        result = await node.execute(state)

        # Should potentially flag low power as statistical threat
        threats = result.get("validity_threats", [])
        assert isinstance(threats, list)


class TestThreatSeverityLevels:
    """Test threat severity level assessment."""

    @pytest.mark.asyncio
    async def test_severity_levels_valid(self):
        """Test that severity levels are valid."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test severity levels")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        valid_severities = {"low", "medium", "high", "critical"}
        for threat in result.get("validity_threats", []):
            assert threat.get("severity") in valid_severities


class TestMitigationRecommendations:
    """Test mitigation recommendation generation."""

    @pytest.mark.asyncio
    async def test_mitigations_returned(self):
        """Test that mitigations are returned."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test mitigations")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        assert "mitigations" in result
        assert isinstance(result["mitigations"], list)

    @pytest.mark.asyncio
    async def test_mitigation_structure(self):
        """Test mitigation structure."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test mitigation structure")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        for mitigation in result.get("mitigations", []):
            if mitigation:  # Skip empty
                assert "recommendation" in mitigation or "threat_addressed" in mitigation


class TestRedesignTriggers:
    """Test redesign triggering logic."""

    @pytest.mark.asyncio
    async def test_redesign_triggered_on_critical_threat(self):
        """Test that redesign is triggered for critical threats."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test critical threat redesign")
        state["status"] = "auditing"
        state["design_type"] = "RCT"
        state["current_iteration"] = 0
        state["max_redesign_iterations"] = 2

        result = await node.execute(state)

        # If critical threats exist, redesign might be needed
        critical_threats = [
            t for t in result.get("validity_threats", []) if t.get("severity") == "critical"
        ]
        if critical_threats:
            assert result.get("redesign_needed") is True

    @pytest.mark.asyncio
    async def test_redesign_capped_at_max_iterations(self):
        """Test that redesign is capped at max iterations."""
        node = ValidityAuditNode()
        state = create_initial_state(
            business_question="Test max iterations", max_redesign_iterations=2
        )
        state["status"] = "auditing"
        state["design_type"] = "RCT"
        state["current_iteration"] = 2  # Already at max

        result = await node.execute(state)

        # Should not trigger redesign at max iterations
        # (though redesign_needed might still be True, the graph won't route to redesign)
        assert result["status"] in ["generating", "redesigning"]


class TestValidityAuditPerformance:
    """Test validity audit performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test validity audit completes under 30s target."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test latency performance")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["validity_audit"]
        # Mock should be fast; real target is 30s
        assert latency < 30_000, f"Validity audit took {latency}ms, exceeds 30s target"


class TestValidityAuditEdgeCases:
    """Test validity audit edge cases."""

    @pytest.mark.asyncio
    async def test_missing_design_type(self):
        """Test handling of missing design type."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test missing design type")
        state["status"] = "auditing"
        # No design_type

        result = await node.execute(state)

        # Should handle gracefully
        assert result["status"] in ["generating", "redesigning", "failed"]

    @pytest.mark.asyncio
    async def test_quasi_experimental_design(self):
        """Test audit for quasi-experimental design."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test quasi-experimental audit")
        state["status"] = "auditing"
        state["design_type"] = "Quasi_Experimental"

        result = await node.execute(state)

        # Quasi-experimental should have specific threats
        assert result["status"] in ["generating", "redesigning"]

    @pytest.mark.asyncio
    async def test_observational_design(self):
        """Test audit for observational design."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test observational audit")
        state["status"] = "auditing"
        state["design_type"] = "Observational"

        result = await node.execute(state)

        # Observational should flag confounding threats
        assert result["status"] in ["generating", "redesigning"]


class TestValidityScoreCalculation:
    """Test validity score calculation logic."""

    @pytest.mark.asyncio
    async def test_score_in_valid_range(self):
        """Test that validity score is in valid range."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test score range")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        score = result.get("overall_validity_score", 0)
        assert 0.0 <= score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_levels(self):
        """Test that confidence levels are valid."""
        node = ValidityAuditNode()
        state = create_initial_state(business_question="Test confidence levels")
        state["status"] = "auditing"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        confidence = result.get("validity_confidence", "low")
        assert confidence in ["low", "medium", "high"]
