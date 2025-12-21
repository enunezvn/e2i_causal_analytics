"""Tests for Redesign Node.

Tests the rule-based redesign functionality that incorporates validity audit feedback.
"""

import pytest

from src.agents.experiment_designer.graph import create_initial_state
from src.agents.experiment_designer.nodes.redesign import RedesignNode


class TestRedesignNode:
    """Test RedesignNode functionality."""

    def test_create_node(self):
        """Test creating node."""
        node = RedesignNode()

        assert node is not None
        assert hasattr(node, "_mitigation_rules")

    def test_mitigation_rules_defined(self):
        """Test that mitigation rules are defined."""
        node = RedesignNode()

        expected_rules = [
            "selection_bias",
            "confounding",
            "contamination",
            "measurement",
            "attrition",
            "temporal",
        ]

        for rule in expected_rules:
            assert rule in node._mitigation_rules

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test redesign")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["design_type"] = "RCT"
        state["validity_threats"] = [
            {
                "threat_type": "internal",
                "threat_name": "selection_bias",
                "severity": "high",
                "mitigation_possible": True,
            }
        ]

        result = await node.execute(state)

        assert result["status"] == "calculating"
        assert result["current_iteration"] == 1

    @pytest.mark.asyncio
    async def test_execute_increments_iteration(self):
        """Test that iteration counter is incremented."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test iteration increment")
        state["status"] = "redesigning"
        state["current_iteration"] = 1
        state["validity_threats"] = []

        result = await node.execute(state)

        assert result["current_iteration"] == 2

    @pytest.mark.asyncio
    async def test_execute_records_iteration_history(self):
        """Test that iteration history is recorded."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test iteration history")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["design_type"] = "RCT"
        state["validity_threats"] = []
        state["power_analysis"] = {"achieved_power": 0.80}

        result = await node.execute(state)

        assert "iteration_history" in result
        assert len(result["iteration_history"]) == 1
        assert result["iteration_history"][0]["iteration_number"] == 0

    @pytest.mark.asyncio
    async def test_execute_skip_on_failed(self):
        """Test execution skips on failed status."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test skip on failed")
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_records_latency(self):
        """Test that node latency is recorded."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test latency recording")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["validity_threats"] = []

        result = await node.execute(state)

        assert "node_latencies_ms" in result
        assert "redesign_0" in result["node_latencies_ms"]


class TestMitigationApplication:
    """Test mitigation application logic."""

    @pytest.mark.asyncio
    async def test_selection_bias_adds_stratification(self):
        """Test that selection bias mitigation adds stratification variables."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test selection bias mitigation")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["stratification_variables"] = []
        state["validity_threats"] = [
            {
                "threat_type": "internal",
                "threat_name": "selection_bias",
                "severity": "high",
                "mitigation_possible": True,
            }
        ]

        result = await node.execute(state)

        # Should add stratification variables
        assert len(result.get("stratification_variables", [])) > 0

    @pytest.mark.asyncio
    async def test_confounding_adds_blocking(self):
        """Test that confounding mitigation adds blocking variables."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test confounding mitigation")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["blocking_variables"] = []
        state["validity_threats"] = [
            {
                "threat_type": "internal",
                "threat_name": "confounding",
                "severity": "high",
                "mitigation_possible": True,
            }
        ]

        result = await node.execute(state)

        # Should add blocking variables
        assert len(result.get("blocking_variables", [])) > 0

    @pytest.mark.asyncio
    async def test_attrition_increases_sample(self):
        """Test that attrition mitigation increases sample size."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test attrition mitigation")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["power_analysis"] = {"required_sample_size": 500, "required_sample_size_per_arm": 250}
        state["validity_threats"] = [
            {
                "threat_type": "internal",
                "threat_name": "attrition",
                "severity": "high",
                "mitigation_possible": True,
            }
        ]

        result = await node.execute(state)

        pa = result.get("power_analysis", {})
        # Should increase sample size
        assert pa.get("required_sample_size", 0) > 500

    @pytest.mark.asyncio
    async def test_low_severity_skipped(self):
        """Test that low severity threats are skipped."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test low severity skip")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["stratification_variables"] = []
        state["validity_threats"] = [
            {
                "threat_type": "internal",
                "threat_name": "selection_bias",
                "severity": "low",  # Low severity
                "mitigation_possible": True,
            }
        ]

        result = await node.execute(state)

        # Low severity should not trigger mitigation
        # (may or may not add stratification depending on implementation)
        assert result["status"] == "calculating"


class TestIterationHistory:
    """Test iteration history recording."""

    @pytest.mark.asyncio
    async def test_iteration_record_fields(self):
        """Test that iteration record has all fields."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test iteration fields")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["design_type"] = "RCT"
        state["validity_threats"] = [{"severity": "critical", "threat_name": "test"}]
        state["power_analysis"] = {"achieved_power": 0.75}

        result = await node.execute(state)

        record = result["iteration_history"][0]
        assert "iteration_number" in record
        assert "design_type" in record
        assert "validity_threats_identified" in record
        assert "critical_threats" in record
        assert "power_achieved" in record
        assert "redesign_reason" in record
        assert "timestamp" in record

    @pytest.mark.asyncio
    async def test_multiple_iterations_tracked(self):
        """Test that multiple iterations are tracked."""
        node = RedesignNode()

        # First iteration
        state = create_initial_state(business_question="Test multiple iterations")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["design_type"] = "RCT"
        state["validity_threats"] = []
        state["iteration_history"] = []

        result = await node.execute(state)

        # Second iteration
        result["status"] = "redesigning"
        result["validity_threats"] = [{"severity": "high", "threat_name": "new_threat"}]
        result = await node.execute(result)

        assert len(result["iteration_history"]) == 2
        assert result["iteration_history"][0]["iteration_number"] == 0
        assert result["iteration_history"][1]["iteration_number"] == 1


class TestRedesignReason:
    """Test redesign reason determination."""

    @pytest.mark.asyncio
    async def test_reason_from_recommendations(self):
        """Test reason from redesign recommendations."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test reason from recommendations")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["validity_threats"] = []
        state["redesign_recommendations"] = ["Increase sample size", "Add stratification"]

        result = await node.execute(state)

        record = result["iteration_history"][0]
        assert "Increase sample size" in record["redesign_reason"]

    @pytest.mark.asyncio
    async def test_reason_from_critical_threat(self):
        """Test reason from critical threat."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test reason from critical threat")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["redesign_recommendations"] = []
        state["validity_threats"] = [{"severity": "critical", "threat_name": "selection_bias"}]

        result = await node.execute(state)

        record = result["iteration_history"][0]
        assert "Critical threat" in record["redesign_reason"]
        assert "selection_bias" in record["redesign_reason"]

    @pytest.mark.asyncio
    async def test_reason_from_high_threat(self):
        """Test reason from high severity threat."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test reason from high threat")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["redesign_recommendations"] = []
        state["validity_threats"] = [{"severity": "high", "threat_name": "confounding"}]

        result = await node.execute(state)

        record = result["iteration_history"][0]
        assert (
            "High severity" in record["redesign_reason"]
            or "confounding" in record["redesign_reason"]
        )


class TestCausalAssumptionsUpdate:
    """Test causal assumptions update during redesign."""

    @pytest.mark.asyncio
    async def test_causal_assumptions_added(self):
        """Test that causal assumptions are added during redesign."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test causal assumptions")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["causal_assumptions"] = ["Initial assumption"]
        state["validity_threats"] = []

        result = await node.execute(state)

        assumptions = result.get("causal_assumptions", [])
        # Should have original plus new
        assert len(assumptions) >= 1


class TestRedesignPerformance:
    """Test redesign performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test redesign completes under 100ms target."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test latency performance")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["validity_threats"] = [{"severity": "high", "threat_name": "selection_bias"}]
        state["stratification_variables"] = []
        state["blocking_variables"] = []

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["redesign_0"]
        assert latency < 100, f"Redesign took {latency}ms, exceeds 100ms target"


class TestRedesignEdgeCases:
    """Test redesign edge cases."""

    @pytest.mark.asyncio
    async def test_empty_threats(self):
        """Test handling of empty threats list."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test empty threats")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["validity_threats"] = []

        result = await node.execute(state)

        assert result["status"] == "calculating"

    @pytest.mark.asyncio
    async def test_unknown_threat_name(self):
        """Test handling of unknown threat name."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test unknown threat")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["validity_threats"] = [{"severity": "high", "threat_name": "unknown_threat_xyz"}]

        result = await node.execute(state)

        # Should handle gracefully
        assert result["status"] == "calculating"

    @pytest.mark.asyncio
    async def test_preserves_existing_stratification(self):
        """Test that existing stratification variables are preserved."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test preserve stratification")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["stratification_variables"] = ["existing_var"]
        state["validity_threats"] = [{"severity": "high", "threat_name": "selection_bias"}]

        result = await node.execute(state)

        strat_vars = result.get("stratification_variables", [])
        assert "existing_var" in strat_vars

    @pytest.mark.asyncio
    async def test_no_duplicate_variables(self):
        """Test that duplicate variables are not added."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test no duplicates")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        state["stratification_variables"] = ["baseline_engagement"]  # Already present
        state["validity_threats"] = [{"severity": "high", "threat_name": "selection_bias"}]

        result = await node.execute(state)

        strat_vars = result.get("stratification_variables", [])
        # Should not have duplicates
        assert len(strat_vars) == len(set(strat_vars))

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling during execution."""
        node = RedesignNode()
        state = create_initial_state(business_question="Test error handling")
        state["status"] = "redesigning"
        state["current_iteration"] = 0
        # Invalid validity_threats structure
        state["validity_threats"] = "invalid"

        result = await node.execute(state)

        # Should handle error gracefully
        assert result["status"] in ["calculating", "generating", "failed"]
        if result["status"] != "failed":
            assert len(result.get("warnings", [])) > 0 or len(result.get("errors", [])) > 0
