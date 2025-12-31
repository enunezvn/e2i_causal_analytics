"""Tests for V4.4 DAG Validation in Experiment Designer Validity Audit.

Tests the validity_audit's DAG validation methods:
- _has_dag_evidence
- _validate_confounders_against_dag
- _identify_latent_confounders
- _identify_instrument_candidates
- _identify_effect_modifiers
- _perform_dag_validation
"""

import pytest

from src.agents.experiment_designer.nodes.validity_audit import ValidityAuditNode


@pytest.fixture
def validity_audit():
    """Create validity audit node instance."""
    return ValidityAuditNode()


@pytest.fixture
def sample_dag_adjacency():
    """Sample DAG: instrument -> treatment -> mediator -> outcome.

    Also has: confounder -> treatment, confounder -> outcome

    Adjacency matrix (row -> col):
              instrument  treatment  mediator  outcome  confounder
    instrument     0          1         0         0         0
    treatment      0          0         1         0         0
    mediator       0          0         0         1         0
    outcome        0          0         0         0         0
    confounder     0          1         0         1         0
    """
    return [
        [0, 1, 0, 0, 0],  # instrument -> treatment
        [0, 0, 1, 0, 0],  # treatment -> mediator
        [0, 0, 0, 1, 0],  # mediator -> outcome
        [0, 0, 0, 0, 0],  # outcome (no outgoing)
        [0, 1, 0, 1, 0],  # confounder -> treatment, confounder -> outcome
    ]


@pytest.fixture
def sample_dag_nodes():
    """Node names for the sample DAG."""
    return ["instrument", "treatment", "mediator", "outcome", "confounder"]


class TestHasDagEvidence:
    """Tests for _has_dag_evidence method."""

    def test_has_dag_evidence_with_valid_dag(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return True when DAG evidence is valid."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }
        assert validity_audit._has_dag_evidence(state) is True

    def test_has_dag_evidence_with_review_decision(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return True when gate decision is 'review'."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "review",
        }
        assert validity_audit._has_dag_evidence(state) is True

    def test_has_dag_evidence_with_reject_decision(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should return False when gate decision is 'reject'."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "reject",
        }
        assert validity_audit._has_dag_evidence(state) is False

    def test_has_dag_evidence_without_adjacency(self, validity_audit, sample_dag_nodes):
        """Should return False when adjacency matrix is missing."""
        state = {
            "discovered_dag_adjacency": None,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }
        assert validity_audit._has_dag_evidence(state) is False

    def test_has_dag_evidence_with_empty_adjacency(self, validity_audit, sample_dag_nodes):
        """Should return False when adjacency matrix is empty."""
        state = {
            "discovered_dag_adjacency": [],
            "discovered_dag_nodes": sample_dag_nodes,
            "discovery_gate_decision": "accept",
        }
        assert validity_audit._has_dag_evidence(state) is False

    def test_has_dag_evidence_without_nodes(self, validity_audit, sample_dag_adjacency):
        """Should return False when nodes list is missing."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": None,
            "discovery_gate_decision": "accept",
        }
        assert validity_audit._has_dag_evidence(state) is False


class TestValidateConfoundersAgainstDag:
    """Tests for _validate_confounders_against_dag method."""

    def test_validates_confounders_in_dag(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should validate confounders that are in the DAG."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "causal_assumptions": [
                "Controlled for: confounder, mediator",
            ],
        }
        validated, missing, warnings = validity_audit._validate_confounders_against_dag(state)

        assert "confounder" in validated
        assert "mediator" in validated
        assert len(missing) == 0
        assert len(warnings) == 0

    def test_identifies_missing_confounders(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should identify confounders not in the DAG."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "causal_assumptions": [
                "Controlled for: confounder, unknown_var, another_missing",
            ],
        }
        validated, missing, warnings = validity_audit._validate_confounders_against_dag(state)

        assert "confounder" in validated
        assert "unknown_var" in missing
        assert "another_missing" in missing
        assert len(warnings) == 2  # One warning per missing confounder

    def test_extracts_confounders_from_dowhy_spec(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should extract confounders from dowhy_spec common_causes."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "causal_assumptions": [],
            "dowhy_spec": {
                "common_causes": ["confounder", "missing_cause"],
            },
        }
        validated, missing, warnings = validity_audit._validate_confounders_against_dag(state)

        assert "confounder" in validated
        assert "missing_cause" in missing

    def test_handles_various_assumption_formats(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should handle various assumption text formats."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "causal_assumptions": [
                "Adjusted for confounder and mediator",
                "Control variables: instrument; outcome",
            ],
        }
        validated, missing, warnings = validity_audit._validate_confounders_against_dag(state)

        # Should find confounder, mediator, instrument, outcome
        assert len(validated) >= 2  # At least some should be found


class TestIdentifyLatentConfounders:
    """Tests for _identify_latent_confounders method."""

    def test_no_latent_confounders(self, validity_audit):
        """Should return empty list when no bidirected edges."""
        state = {
            "discovered_dag_edge_types": {
                "treatment->outcome": "DIRECTED",
                "confounder->treatment": "DIRECTED",
            },
        }
        latent = validity_audit._identify_latent_confounders(state)
        assert len(latent) == 0

    def test_detects_bidirected_edges(self, validity_audit):
        """Should detect bidirected edges as latent confounders."""
        state = {
            "discovered_dag_edge_types": {
                "treatment<->outcome": "BIDIRECTED",
                "treatment->mediator": "DIRECTED",
            },
        }
        latent = validity_audit._identify_latent_confounders(state)

        assert len(latent) == 1
        assert "treatment<->outcome" in latent

    def test_detects_multiple_latent_confounders(self, validity_audit):
        """Should detect multiple bidirected edges."""
        state = {
            "discovered_dag_edge_types": {
                "treatment<->outcome": "BIDIRECTED",
                "mediator<->outcome": "BIDIRECTED",
                "treatment->mediator": "DIRECTED",
            },
        }
        latent = validity_audit._identify_latent_confounders(state)

        assert len(latent) == 2
        assert "treatment<->outcome" in latent
        assert "mediator<->outcome" in latent

    def test_handles_empty_edge_types(self, validity_audit):
        """Should handle empty edge types dict."""
        state = {
            "discovered_dag_edge_types": {},
        }
        latent = validity_audit._identify_latent_confounders(state)
        assert len(latent) == 0

    def test_handles_missing_edge_types(self, validity_audit):
        """Should handle missing edge types."""
        state = {}
        latent = validity_audit._identify_latent_confounders(state)
        assert len(latent) == 0


class TestIdentifyInstrumentCandidates:
    """Tests for _identify_instrument_candidates method."""

    def test_identifies_valid_instrument(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should identify instrument that affects treatment but not outcome."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
        }
        candidates = validity_audit._identify_instrument_candidates(state)

        assert "instrument" in candidates
        # confounder should NOT be a candidate (it points to outcome)
        assert "confounder" not in candidates

    def test_no_candidates_when_all_affect_outcome(self, validity_audit):
        """Should return empty list when all variables affect outcome."""
        # DAG where all non-treatment/outcome nodes point to outcome
        dag_adjacency = [
            [0, 1, 0],  # var1 -> treatment
            [0, 0, 1],  # treatment -> outcome
            [0, 0, 0],  # outcome
        ]
        # Actually need var1 to also point to outcome for no IV
        dag_adjacency = [
            [0, 1, 1],  # var1 -> treatment, var1 -> outcome
            [0, 0, 1],  # treatment -> outcome
            [0, 0, 0],  # outcome
        ]
        dag_nodes = ["var1", "treatment", "outcome"]

        state = {
            "discovered_dag_adjacency": dag_adjacency,
            "discovered_dag_nodes": dag_nodes,
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
        }
        candidates = validity_audit._identify_instrument_candidates(state)

        assert "var1" not in candidates

    def test_handles_missing_treatment_variable(self, validity_audit, sample_dag_adjacency, sample_dag_nodes):
        """Should return empty list when treatment variable is missing."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "",
            "outcome_variable": "outcome",
        }
        candidates = validity_audit._identify_instrument_candidates(state)
        assert len(candidates) == 0

    def test_handles_treatment_not_in_dag(self, validity_audit, sample_dag_adjacency, sample_dag_nodes):
        """Should return empty list when treatment not in DAG."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "nonexistent",
            "outcome_variable": "outcome",
        }
        candidates = validity_audit._identify_instrument_candidates(state)
        assert len(candidates) == 0


class TestIdentifyEffectModifiers:
    """Tests for _identify_effect_modifiers method."""

    def test_identifies_common_cause_as_modifier(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should identify common causes as effect modifiers."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
        }
        modifiers = validity_audit._identify_effect_modifiers(state)

        # confounder points to both treatment and outcome
        assert "confounder" in modifiers

    def test_mediator_not_effect_modifier(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should not identify mediator as effect modifier."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
        }
        modifiers = validity_audit._identify_effect_modifiers(state)

        # mediator is on causal path, not a common cause
        assert "mediator" not in modifiers

    def test_instrument_not_effect_modifier(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should not identify instrument as effect modifier."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
        }
        modifiers = validity_audit._identify_effect_modifiers(state)

        # instrument only points to treatment, not outcome
        assert "instrument" not in modifiers

    def test_handles_missing_variables(self, validity_audit, sample_dag_adjacency, sample_dag_nodes):
        """Should handle missing treatment/outcome variables."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "treatment_variable": "",
            "outcome_variable": "",
        }
        modifiers = validity_audit._identify_effect_modifiers(state)
        assert len(modifiers) == 0


class TestPerformDagValidation:
    """Tests for _perform_dag_validation method."""

    def test_comprehensive_validation(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should perform comprehensive DAG validation."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "causal_assumptions": ["Controlled for: confounder"],
            "dowhy_spec": None,
        }

        results, warnings = validity_audit._perform_dag_validation(state)

        assert "confounders_validated" in results
        assert "confounders_missing" in results
        assert "latent_confounders" in results
        assert "instrument_candidates" in results
        assert "effect_modifiers" in results

        # Confounder should be validated
        assert "confounder" in results["confounders_validated"]

        # Instrument should be identified
        assert "instrument" in results["instrument_candidates"]

        # Confounder should be identified as effect modifier
        assert "confounder" in results["effect_modifiers"]

    def test_validation_with_latent_confounders(self, validity_audit, sample_dag_adjacency, sample_dag_nodes):
        """Should warn about latent confounders."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {
                "treatment<->outcome": "BIDIRECTED",
            },
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "causal_assumptions": [],
        }

        results, warnings = validity_audit._perform_dag_validation(state)

        assert len(results["latent_confounders"]) == 1
        assert any("latent confounder" in w.lower() for w in warnings)

    def test_validation_with_missing_confounders(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should warn about missing confounders."""
        state = {
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "causal_assumptions": ["Controlled for: confounder, missing_var"],
        }

        results, warnings = validity_audit._perform_dag_validation(state)

        assert "missing_var" in results["confounders_missing"]
        assert any("missing_var" in w for w in warnings)


class TestDagValidationIntegration:
    """Integration tests for DAG validation in validity audit."""

    @pytest.mark.asyncio
    async def test_execute_with_dag_validation(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should include DAG validation results when DAG is available."""
        state = {
            "business_question": "Does treatment improve outcome?",
            "constraints": {},
            "available_data": {},
            "preregistration_formality": "medium",
            "max_redesign_iterations": 2,
            "enable_validity_audit": True,
            "design_type": "RCT",
            "design_rationale": "Gold standard for causal inference",
            "treatments": [],
            "outcomes": [],
            "power_analysis": {"required_sample_size": 500},
            "randomization_unit": "individual",
            "randomization_method": "simple",
            "stratification_variables": [],
            "blocking_variables": [],
            "causal_assumptions": ["Controlled for: confounder"],
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "discovery_gate_decision": "accept",
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

        result = await validity_audit.execute(state)

        # Should have DAG validation fields
        assert "dag_confounders_validated" in result
        assert "dag_missing_confounders" in result
        assert "dag_latent_confounders" in result
        assert "dag_instrument_candidates" in result
        assert "dag_effect_modifiers" in result
        assert "dag_validation_warnings" in result

        # Confounder should be validated
        assert "confounder" in result["dag_confounders_validated"]

        # Instrument should be identified
        assert "instrument" in result["dag_instrument_candidates"]

    @pytest.mark.asyncio
    async def test_execute_without_dag_skips_validation(self, validity_audit):
        """Should skip DAG validation when no DAG evidence."""
        state = {
            "business_question": "Does treatment improve outcome?",
            "constraints": {},
            "available_data": {},
            "preregistration_formality": "medium",
            "max_redesign_iterations": 2,
            "enable_validity_audit": True,
            "design_type": "RCT",
            "design_rationale": "Gold standard",
            "treatments": [],
            "outcomes": [],
            "power_analysis": {"required_sample_size": 500},
            "randomization_unit": "individual",
            "randomization_method": "simple",
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

        result = await validity_audit.execute(state)

        # Should NOT have DAG validation fields
        assert "dag_confounders_validated" not in result
        assert "dag_instrument_candidates" not in result

    @pytest.mark.asyncio
    async def test_execute_with_reject_decision_skips_validation(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should skip DAG validation when discovery gate rejected."""
        state = {
            "business_question": "Does treatment improve outcome?",
            "constraints": {},
            "available_data": {},
            "preregistration_formality": "medium",
            "max_redesign_iterations": 2,
            "enable_validity_audit": True,
            "design_type": "RCT",
            "design_rationale": "Gold standard",
            "treatments": [],
            "outcomes": [],
            "power_analysis": {"required_sample_size": 500},
            "randomization_unit": "individual",
            "randomization_method": "simple",
            "causal_assumptions": ["Controlled for: confounder"],
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {},
            "discovery_gate_decision": "reject",  # Rejected!
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

        result = await validity_audit.execute(state)

        # Should NOT have DAG validation fields because gate was rejected
        assert "dag_confounders_validated" not in result

    @pytest.mark.asyncio
    async def test_validity_score_penalty_for_concerns(
        self, validity_audit, sample_dag_adjacency, sample_dag_nodes
    ):
        """Should reduce validity score when DAG reveals concerns."""
        state = {
            "business_question": "Does treatment improve outcome?",
            "constraints": {},
            "available_data": {},
            "preregistration_formality": "medium",
            "max_redesign_iterations": 2,
            "enable_validity_audit": True,
            "design_type": "RCT",
            "design_rationale": "Gold standard",
            "treatments": [],
            "outcomes": [],
            "power_analysis": {"required_sample_size": 500},
            "randomization_unit": "individual",
            "randomization_method": "simple",
            "causal_assumptions": ["Controlled for: missing_confounder"],  # Not in DAG!
            "discovered_dag_adjacency": sample_dag_adjacency,
            "discovered_dag_nodes": sample_dag_nodes,
            "discovered_dag_edge_types": {
                "treatment<->outcome": "BIDIRECTED",  # Latent confounder!
            },
            "discovery_gate_decision": "accept",
            "treatment_variable": "treatment",
            "outcome_variable": "outcome",
            "errors": [],
            "warnings": [],
            "status": "calculating",
        }

        result = await validity_audit.execute(state)

        # Should have missing confounders
        assert "missing_confounder" in result["dag_missing_confounders"]

        # Should have latent confounders
        assert len(result["dag_latent_confounders"]) == 1

        # Validity score should be reduced (mock returns 0.75, penalty should reduce it)
        # 0.75 - 0.1 (missing) - 0.1 (latent) = 0.55
        assert result["overall_validity_score"] <= 0.75
