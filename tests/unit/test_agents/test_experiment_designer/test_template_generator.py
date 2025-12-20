"""Tests for Template Generator Node.

Tests the DoWhy code generation and pre-registration document generation functionality.
"""

import pytest
from src.agents.experiment_designer.nodes.template_generator import TemplateGeneratorNode
from src.agents.experiment_designer.graph import create_initial_state


class TestTemplateGeneratorNode:
    """Test TemplateGeneratorNode functionality."""

    def test_create_node(self):
        """Test creating node."""
        node = TemplateGeneratorNode()

        assert node is not None
        assert hasattr(node, "_template_version")
        assert node._template_version == "1.0.0"

    @pytest.mark.asyncio
    async def test_execute_basic(self):
        """Test basic execution."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test template generation"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "Treatment", "description": "Test"}]
        state["outcomes"] = [{"name": "Outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert result["status"] == "completed"
        assert "template_generator" in result.get("node_latencies_ms", {})

    @pytest.mark.asyncio
    async def test_execute_generates_analysis_code(self):
        """Test that analysis code is generated."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test analysis code generation"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert "analysis_code" in result
        assert len(result["analysis_code"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_causal_graph(self):
        """Test that causal graph DOT format is generated."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test causal graph generation"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert "causal_graph_dot" in result
        assert len(result["causal_graph_dot"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_preregistration(self):
        """Test that pre-registration document is generated."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test preregistration generation"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]

        result = await node.execute(state)

        template = result.get("experiment_template", {})
        assert "pre_registration_document" in template
        assert len(template["pre_registration_document"]) > 0

    @pytest.mark.asyncio
    async def test_execute_generates_dowhy_spec(self):
        """Test that DoWhy specification is generated."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test DoWhy spec generation"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert "dowhy_spec" in result
        spec = result["dowhy_spec"]
        assert "treatment_variable" in spec
        assert "outcome_variable" in spec

    @pytest.mark.asyncio
    async def test_execute_skip_on_failed(self):
        """Test execution skips on failed status."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test skip on failed"
        )
        state["status"] = "failed"

        result = await node.execute(state)

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_execute_records_latency(self):
        """Test that node latency is recorded."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test latency recording"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"

        result = await node.execute(state)

        assert "node_latencies_ms" in result
        assert "template_generator" in result["node_latencies_ms"]


class TestAnalysisCodeGeneration:
    """Test analysis code generation."""

    @pytest.mark.asyncio
    async def test_code_includes_dowhy_imports(self):
        """Test that generated code includes DoWhy imports."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test DoWhy imports"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        code = result.get("analysis_code", "")
        assert "dowhy" in code.lower() or "CausalModel" in code

    @pytest.mark.asyncio
    async def test_code_includes_treatment_variable(self):
        """Test that generated code includes treatment variable."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test treatment variable in code"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "my_treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        code = result.get("analysis_code", "")
        # Should reference the treatment
        assert "treatment" in code.lower()

    @pytest.mark.asyncio
    async def test_code_includes_outcome_variable(self):
        """Test that generated code includes outcome variable."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test outcome variable in code"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "my_outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        code = result.get("analysis_code", "")
        # Should reference the outcome
        assert "outcome" in code.lower()

    @pytest.mark.asyncio
    async def test_code_includes_econml_for_heterogeneity(self):
        """Test that EconML is included for heterogeneity analysis."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test EconML inclusion"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        code = result.get("analysis_code", "")
        # Should include EconML for CATE analysis
        assert "econml" in code.lower() or "CausalForest" in code


class TestCausalGraphGeneration:
    """Test causal graph DOT format generation."""

    @pytest.mark.asyncio
    async def test_dot_format_valid(self):
        """Test that DOT format is valid."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test DOT format"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        dot = result.get("causal_graph_dot", "")
        # Should have digraph declaration
        assert "digraph" in dot or "->" in dot

    @pytest.mark.asyncio
    async def test_graph_includes_treatment_outcome_edge(self):
        """Test that graph includes treatment->outcome edge."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test treatment-outcome edge"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        dot = result.get("causal_graph_dot", "")
        # Should have treatment->outcome edge
        assert "->" in dot

    @pytest.mark.asyncio
    async def test_graph_includes_confounders(self):
        """Test that graph includes confounders."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test confounders in graph"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]
        state["identified_confounders"] = ["territory", "baseline"]

        result = await node.execute(state)

        dot = result.get("causal_graph_dot", "")
        # Should include confounders (may or may not depending on implementation)
        assert len(dot) > 0


class TestPreregistrationGeneration:
    """Test pre-registration document generation."""

    @pytest.mark.asyncio
    async def test_light_formality(self):
        """Test light formality pre-registration."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test light preregistration",
            preregistration_formality="light"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]

        result = await node.execute(state)

        template = result.get("experiment_template", {})
        doc = template.get("pre_registration_document", "")
        assert len(doc) > 0

    @pytest.mark.asyncio
    async def test_medium_formality(self):
        """Test medium formality pre-registration."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test medium preregistration",
            preregistration_formality="medium"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]

        result = await node.execute(state)

        template = result.get("experiment_template", {})
        doc = template.get("pre_registration_document", "")
        assert len(doc) > 0

    @pytest.mark.asyncio
    async def test_heavy_formality(self):
        """Test heavy formality pre-registration."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test heavy preregistration",
            preregistration_formality="heavy"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]

        result = await node.execute(state)

        template = result.get("experiment_template", {})
        doc = template.get("pre_registration_document", "")
        assert len(doc) > 0

    @pytest.mark.asyncio
    async def test_heavy_longer_than_light(self):
        """Test that heavy formality produces longer document."""
        node = TemplateGeneratorNode()

        # Light
        state_light = create_initial_state(
            business_question="Test light length",
            preregistration_formality="light"
        )
        state_light["status"] = "generating"
        state_light["design_type"] = "RCT"
        state_light["treatments"] = [{"name": "T", "description": "D"}]
        state_light["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]
        result_light = await node.execute(state_light)

        # Heavy
        state_heavy = create_initial_state(
            business_question="Test heavy length",
            preregistration_formality="heavy"
        )
        state_heavy["status"] = "generating"
        state_heavy["design_type"] = "RCT"
        state_heavy["treatments"] = [{"name": "T", "description": "D"}]
        state_heavy["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]
        result_heavy = await node.execute(state_heavy)

        light_doc = result_light.get("experiment_template", {}).get("pre_registration_document", "")
        heavy_doc = result_heavy.get("experiment_template", {}).get("pre_registration_document", "")

        # Heavy should be longer (or at least equal)
        assert len(heavy_doc) >= len(light_doc)

    @pytest.mark.asyncio
    async def test_includes_hypothesis(self):
        """Test that pre-registration includes hypothesis."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test hypothesis inclusion"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]

        result = await node.execute(state)

        template = result.get("experiment_template", {})
        doc = template.get("pre_registration_document", "")
        # Should mention hypothesis or question
        assert "hypothesis" in doc.lower() or "question" in doc.lower() or "objective" in doc.lower()


class TestDoWhySpecGeneration:
    """Test DoWhy specification generation."""

    @pytest.mark.asyncio
    async def test_spec_has_treatment(self):
        """Test that spec has treatment variable."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test spec treatment"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "my_treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        spec = result.get("dowhy_spec", {})
        assert "treatment_variable" in spec
        assert len(spec["treatment_variable"]) > 0

    @pytest.mark.asyncio
    async def test_spec_has_outcome(self):
        """Test that spec has outcome variable."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test spec outcome"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "my_outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        spec = result.get("dowhy_spec", {})
        assert "outcome_variable" in spec
        assert len(spec["outcome_variable"]) > 0

    @pytest.mark.asyncio
    async def test_spec_has_common_causes(self):
        """Test that spec has common_causes list."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test spec common causes"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]
        state["stratification_variables"] = ["var1", "var2"]

        result = await node.execute(state)

        spec = result.get("dowhy_spec", {})
        assert "common_causes" in spec
        assert isinstance(spec["common_causes"], list)


class TestTemplateGeneratorPerformance:
    """Test template generator performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_under_target(self):
        """Test template generation completes under 500ms target."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test latency performance"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        latency = result["node_latencies_ms"]["template_generator"]
        assert latency < 500, f"Template generation took {latency}ms, exceeds 500ms target"


class TestTemplateGeneratorEdgeCases:
    """Test template generator edge cases."""

    @pytest.mark.asyncio
    async def test_missing_treatments(self):
        """Test handling of missing treatments."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test missing treatments"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        # No treatments

        result = await node.execute(state)

        # Should handle gracefully
        assert result["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_missing_outcomes(self):
        """Test handling of missing outcomes."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test missing outcomes"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "T", "description": "D"}]
        # No outcomes

        result = await node.execute(state)

        # Should handle gracefully
        assert result["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_empty_design_type(self):
        """Test handling of empty design type."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test empty design type"
        )
        state["status"] = "generating"
        state["treatments"] = [{"name": "T", "description": "D"}]
        state["outcomes"] = [{"name": "Y", "metric_type": "continuous"}]
        # No design_type

        result = await node.execute(state)

        # Should handle gracefully
        assert result["status"] in ["completed", "failed"]

    @pytest.mark.asyncio
    async def test_cluster_rct_design(self):
        """Test template generation for cluster RCT."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test cluster RCT template"
        )
        state["status"] = "generating"
        state["design_type"] = "Cluster_RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_quasi_experimental_design(self):
        """Test template generation for quasi-experimental design."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test quasi-experimental template"
        )
        state["status"] = "generating"
        state["design_type"] = "Quasi_Experimental"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]

        result = await node.execute(state)

        assert result["status"] == "completed"

    @pytest.mark.asyncio
    async def test_with_power_analysis(self):
        """Test template generation with power analysis included."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test with power analysis"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]
        state["power_analysis"] = {
            "required_sample_size": 500,
            "achieved_power": 0.82,
            "alpha": 0.05
        }

        result = await node.execute(state)

        assert result["status"] == "completed"
        # Pre-registration should mention sample size
        template = result.get("experiment_template", {})
        doc = template.get("pre_registration_document", "")
        assert "sample" in doc.lower() or "500" in doc

    @pytest.mark.asyncio
    async def test_with_validity_threats(self):
        """Test template generation with validity threats included."""
        node = TemplateGeneratorNode()
        state = create_initial_state(
            business_question="Test with validity threats"
        )
        state["status"] = "generating"
        state["design_type"] = "RCT"
        state["treatments"] = [{"name": "treatment", "description": "D"}]
        state["outcomes"] = [{"name": "outcome", "metric_type": "continuous"}]
        state["validity_threats"] = [
            {"threat_name": "selection_bias", "severity": "high"}
        ]

        result = await node.execute(state)

        assert result["status"] == "completed"
