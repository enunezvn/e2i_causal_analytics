"""Tests for Simulate Intervention Tool.

Tests the digital twin simulation tool functionality for pre-screening
A/B test interventions.

NOTE: Uses direct module imports to avoid triggering LLM initialization
in the experiment_designer package __init__.py.
"""

import logging
from unittest.mock import patch
from uuid import uuid4

import pytest


# Direct module import to avoid package __init__.py side effects
def _import_tool_module():
    """Import tool module directly, bypassing package __init__."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "simulate_intervention_tool",
        "src/agents/experiment_designer/tools/simulate_intervention_tool.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Planted per-region effect of a high (above-median) email_campaign_count on conversion.
PLANTED_EFFECT = {"northeast": 0.45, "west": 0.30, "south": 0.30, "midwest": 0.30}


def _planted_cohort(n_per_region: int = 300, seed: int = 3):
    """A per-HCP cohort frame with the columns ``load_cohort_frame`` returns."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    frames = []
    for region, tau in PLANTED_EFFECT.items():
        market = rng.uniform(0.0, 1.0, n_per_region)
        frames.append(
            pd.DataFrame(
                {
                    "region": region,
                    "email_campaign_count": rng.poisson(3 + 4 * market).astype(float),
                    "market_share": market,
                    "triggers_total_count": rng.poisson(60, n_per_region).astype(float),
                    "_tau": tau,
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    treated = (df["email_campaign_count"] > df["email_campaign_count"].median()).astype(float)
    df["cohort_conversion_outcome"] = (
        0.2 + 0.3 * df["market_share"] + df["_tau"] * treated + rng.normal(0, 0.05, len(df))
    )
    return df.drop(columns="_tau")


def _scoped_client(client):
    """Stand-in for ``loop_scoped_async_supabase_client`` yielding ``client``."""
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _scoped():
        yield client

    return _scoped


def _region_population(n_per_region: int = 130):
    import numpy as np

    from src.digital_twin.models.twin_models import Brand, DigitalTwin, TwinPopulation, TwinType

    rng = np.random.default_rng(5)
    twins = [
        DigitalTwin(
            twin_type=TwinType.HCP,
            brand=Brand.KISQALI,
            features={"region": region, "decile": int(rng.integers(1, 11))},
            baseline_outcome=float(rng.uniform(0.1, 0.3)),
            baseline_propensity=float(rng.uniform(0.2, 0.5)),
        )
        for region in PLANTED_EFFECT
        for _ in range(n_per_region)
    ]
    return TwinPopulation(twin_type=TwinType.HCP, brand=Brand.KISQALI, twins=twins, size=len(twins))


# Import schemas and models that don't trigger LLM init


# Lazy import the actual tool module
@pytest.fixture(scope="module")
def tool_module():
    """Fixture to lazily import the tool module."""
    return _import_tool_module()


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionInput:
    """Test input schema validation."""

    def test_valid_input_minimal(self, tool_module):
        """Test creating input with minimal required fields."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="email_campaign",
            brand="Kisqali",
        )

        assert input_schema.intervention_type == "email_campaign"
        assert input_schema.brand == "Kisqali"
        assert input_schema.target_population == "hcp"  # default
        assert input_schema.twin_count == 10000  # default
        assert input_schema.confidence_level == 0.95  # default

    def test_valid_input_full(self, tool_module):
        """Test creating input with all fields populated."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="speaker_program_invitation",
            brand="Fabhalta",
            target_population="patient",
            channel="in_person",
            frequency="monthly",
            duration_weeks=12,
            target_deciles=[1, 2],
            target_specialties=["oncology", "hematology"],
            target_regions=["northeast"],
            twin_count=5000,
            confidence_level=0.90,
        )

        assert input_schema.intervention_type == "speaker_program_invitation"
        assert input_schema.brand == "Fabhalta"
        assert input_schema.target_population == "patient"
        assert input_schema.channel == "in_person"
        assert input_schema.frequency == "monthly"
        assert input_schema.duration_weeks == 12
        assert input_schema.target_deciles == [1, 2]
        assert input_schema.target_specialties == ["oncology", "hematology"]
        assert input_schema.target_regions == ["northeast"]
        assert input_schema.twin_count == 5000
        assert input_schema.confidence_level == 0.90

    def test_default_target_population(self, tool_module):
        """Test default target population is HCP."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="call_frequency_increase",
            brand="Remibrutinib",
        )

        assert input_schema.target_population == "hcp"

    def test_default_duration_weeks(self, tool_module):
        """Test default duration is 8 weeks."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="email_campaign",
            brand="Kisqali",
        )

        assert input_schema.duration_weeks == 8

    def test_optional_fields_are_none(self, tool_module):
        """Test that optional fields default to None."""
        SimulateInterventionInput = tool_module.SimulateInterventionInput
        input_schema = SimulateInterventionInput(
            intervention_type="digital_engagement",
            brand="Fabhalta",
        )

        assert input_schema.channel is None
        assert input_schema.frequency is None
        assert input_schema.target_deciles is None
        assert input_schema.target_specialties is None
        assert input_schema.target_regions is None


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateIntervention:
    """Test simulate_intervention tool function."""

    def test_returns_deploy_recommendation(self, tool_module):
        """Test tool returns deploy recommendation for high-effect intervention."""
        simulate_intervention = tool_module.simulate_intervention
        # Use mock result generator with known intervention type
        result = simulate_intervention.invoke(
            {
                "intervention_type": "speaker_program_invitation",  # High base effect
                "brand": "Kisqali",
            }
        )

        assert "recommendation" in result
        assert result["recommendation"] in ["deploy", "skip", "refine"]
        assert "simulation_id" in result
        assert "simulated_ate" in result

    def test_returns_required_output_fields(self, tool_module):
        """Test all required output fields are present."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "email_campaign",
                "brand": "Fabhalta",
            }
        )

        # Check all required fields
        required_fields = [
            "simulation_id",
            "recommendation",
            "recommendation_rationale",
            "simulated_ate",
            "confidence_interval",
            "recommended_sample_size",
            "recommended_duration_weeks",
            "simulation_confidence",
            "fidelity_warning",
            "fidelity_warning_reason",
            "top_segments",
        ]

        for field in required_fields:
            assert field in result, f"Missing required field: {field}"

    def test_confidence_interval_calculated(self, tool_module):
        """Test confidence interval is properly calculated."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "call_frequency_increase",
                "brand": "Remibrutinib",
                "confidence_level": 0.95,
            }
        )

        ci = result["confidence_interval"]
        assert isinstance(ci, tuple) or isinstance(ci, list)
        assert len(ci) == 2
        assert ci[0] <= ci[1]  # Lower bound <= upper bound

    def test_top_segments_returned(self, tool_module):
        """Test top performing segments are returned."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "peer_influence_activation",
                "brand": "Kisqali",
            }
        )

        top_segments = result["top_segments"]
        assert isinstance(top_segments, list)
        # Mock result returns 3 segments
        if top_segments:
            segment = top_segments[0]
            assert "dimension" in segment
            assert "segment" in segment
            assert "ate" in segment

    def test_handles_invalid_brand(self, tool_module):
        """Test handling of invalid brand."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "email_campaign",
                "brand": "InvalidBrand",
            }
        )

        # Should return refine recommendation with error
        assert result["recommendation"] == "refine"
        assert (
            "error" in result.get("recommendation_rationale", "").lower()
            or "failed" in result.get("recommendation_rationale", "").lower()
        )

    def test_handles_invalid_population(self, tool_module):
        """Test handling of invalid target population."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "email_campaign",
                "brand": "Kisqali",
                "target_population": "invalid_population",
            }
        )

        # Should return error response
        assert result["recommendation"] == "refine"

    def test_fidelity_warning_included(self, tool_module):
        """Test fidelity warning field is included in response."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "sample_distribution",
                "brand": "Fabhalta",
            }
        )

        assert "fidelity_warning" in result
        assert isinstance(result["fidelity_warning"], bool)

    def test_fidelity_status_is_explicit(self, tool_module):
        """#2206: the chat tool states the model's fidelity status, so a NULL model
        fidelity reads as 'unvalidated' in the designer's output, not as passed."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "sample_distribution",
                "brand": "Fabhalta",
            }
        )

        assert "fidelity_status" in result
        if result["simulation_id"] != "error":
            assert result["fidelity_status"] in {"unvalidated", "validated", "below_threshold"}
            if result["fidelity_status"] == "unvalidated":
                assert result["fidelity_warning"] is True


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionFailsClosed:
    """R2/H3: with no loadable twin model the tool must FAIL CLOSED (honest error
    result), never a fabricated 'deploy' with fidelity_warning=False off a random
    base effect."""

    def test_does_not_fabricate_when_no_model(self, monkeypatch, tool_module):
        from unittest.mock import AsyncMock, MagicMock

        # Force the fail-closed path: a reachable client, a usable cohort, but NO active
        # trained model.
        monkeypatch.setattr(
            "src.memory.services.factories.loop_scoped_async_supabase_client",
            _scoped_client(MagicMock()),
        )
        monkeypatch.setattr(
            "src.digital_twin.effect.cohort_loader.load_cohort_frame",
            AsyncMock(return_value=_planted_cohort()),
        )
        fake_repo = MagicMock()
        fake_repo.list_active_models = AsyncMock(return_value=[])
        # tool_module is a fresh importlib instance; patch its module-global binding.
        monkeypatch.setattr(tool_module, "TwinRepository", MagicMock(return_value=fake_repo))
        tool_module._twin_cache.clear()

        out = tool_module.simulate_intervention.invoke(
            {
                # The planted cohort identifies email_campaign, so the gate passes and the
                # missing model is what the tool meets.
                "intervention_type": "email_campaign",
                "brand": "Kisqali",
                "target_population": "hcp",
                "twin_count": 10000,
            }
        )
        fake_repo.list_active_models.assert_awaited_once()
        # No fabricated success: a missing model yields refine/error + a fidelity warning.
        assert out["recommendation"] != "deploy"
        assert out["fidelity_warning"] is True
        assert out["simulated_ate"] == 0.0
        # The fabricator is deleted.
        assert not hasattr(tool_module, "_create_mock_result")

    def test_a_failure_returns_no_exception_text_and_logs_it(
        self, monkeypatch, tool_module, caplog
    ):
        """#2020 E1: the node copies both fields into the experiment_designer warnings, and the
        orchestrator stringifies that agent's whole output into the answer."""
        raw = (
            "Input X contains NaN. For further information visit "
            "https://errors.pydantic.dev/2.12/v/value_error"
        )

        def _raise(*args, **kwargs):
            raise ValueError(raw)

        from unittest.mock import AsyncMock, MagicMock

        # A usable cohort, so the failure is raised past the identification gate (#2025).
        monkeypatch.setattr(
            "src.memory.services.factories.loop_scoped_async_supabase_client",
            _scoped_client(MagicMock()),
        )
        monkeypatch.setattr(
            "src.digital_twin.effect.cohort_loader.load_cohort_frame",
            AsyncMock(return_value=_planted_cohort()),
        )
        monkeypatch.setattr(tool_module, "_get_or_create_twins", _raise)
        with caplog.at_level(logging.ERROR):
            out = tool_module.simulate_intervention.invoke(
                {"intervention_type": "email_campaign", "brand": "Kisqali"}
            )

        assert (
            out["recommendation_rationale"]
            == "Simulation failed. Please check inputs and try again."
        )
        assert out["fidelity_warning_reason"] == "the twin simulation could not be completed"
        for leak in ("NaN", "pydantic.dev"):
            assert leak not in str(out), f"{leak!r} reached the tool output"
        # Every other field of the error result is unchanged.
        assert out["recommendation"] == "refine"
        assert out["simulated_ate"] == 0.0
        assert out["fidelity_warning"] is True
        assert raw in caplog.text


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionEstimatesOnTheCohort:
    """#2025: the pre-screen built its engine with no effect provider, so it simulated the
    engine's synthetic default (a planted ATE of 0.15) and returned it as the estimate. It
    now estimates on the brand's cohort the way ``/digital-twin/simulate`` does, and
    refuses when the cohort cannot identify the intervention.

    The engine, the cohort provider and ``CohortCausalEstimator`` run for real; only the
    database read and the model registry in front of them are replaced."""

    @pytest.mark.timeout(180)
    def test_returns_the_planted_cohort_effect_for_the_target_region(
        self, monkeypatch, tool_module
    ):
        from unittest.mock import ANY, AsyncMock, MagicMock

        load = AsyncMock(return_value=_planted_cohort())
        monkeypatch.setattr("src.digital_twin.effect.cohort_loader.load_cohort_frame", load)
        monkeypatch.setattr(
            "src.memory.services.factories.loop_scoped_async_supabase_client",
            _scoped_client(MagicMock()),
        )
        monkeypatch.setattr(
            tool_module, "_get_or_create_twins", lambda *a, **k: _region_population()
        )

        out = tool_module.simulate_intervention.invoke(
            {
                "intervention_type": "email_campaign",
                "brand": "Kisqali",
                "target_regions": ["northeast"],
            }
        )

        load.assert_awaited_once_with(ANY, "Kisqali")
        assert out["simulation_id"] != "error", out["recommendation_rationale"]
        # The northeast effect, not the synthetic default (0.15) and not the whole-cohort
        # average (~0.34): the target regions reach the estimator.
        assert out["simulated_ate"] == pytest.approx(PLANTED_EFFECT["northeast"], abs=0.04)

    def test_refuses_without_generating_twins_when_the_cohort_is_unusable(
        self, monkeypatch, tool_module
    ):
        from unittest.mock import AsyncMock, MagicMock

        import pandas as pd

        monkeypatch.setattr(
            "src.digital_twin.effect.cohort_loader.load_cohort_frame",
            AsyncMock(return_value=pd.DataFrame()),
        )
        monkeypatch.setattr(
            "src.memory.services.factories.loop_scoped_async_supabase_client",
            _scoped_client(MagicMock()),
        )
        generated = []
        monkeypatch.setattr(
            tool_module,
            "_get_or_create_twins",
            lambda *a, **k: generated.append(a) or _region_population(),
        )

        out = tool_module.simulate_intervention.invoke(
            {"intervention_type": "email_campaign", "brand": "Kisqali"}
        )

        assert generated == []
        assert out["recommendation"] == "refine"
        assert out["simulated_ate"] == 0.0
        assert out["confidence_interval"] == (0.0, 0.0)
        assert out["fidelity_warning"] is True
        assert out["recommendation_rationale"] == (
            "No effect data available for intervention 'email_campaign' and brand 'Kisqali': "
            "the connected cohort cannot identify this intervention, so a causal effect "
            "cannot be estimated (no fabricated effect is returned)."
        )


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestDigitalTwinWorkflow:
    """Test DigitalTwinWorkflow class."""

    def test_propose_experiment_skip(self, tool_module):
        """Test workflow returns skip action for low-effect intervention."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        # Mock simulate_intervention to return skip
        with patch.object(tool_module, "simulate_intervention") as mock_sim:
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "skip",
                "recommendation_rationale": "Effect below threshold",
                "simulated_ate": 0.02,
                "confidence_interval": (-0.01, 0.05),
                "recommended_sample_size": None,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.75,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": [],
            }

            result = workflow.propose_experiment(
                intervention_type="email_campaign",
                brand="Kisqali",
            )

        assert result["action"] == "SKIP"
        assert "reason" in result

    def test_propose_experiment_design(self, tool_module):
        """Test workflow returns design action for promising intervention."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        # Mock simulate_intervention to return deploy
        with patch.object(tool_module, "simulate_intervention") as mock_sim:
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "deploy",
                "recommendation_rationale": "Strong predicted effect",
                "simulated_ate": 0.12,
                "confidence_interval": (0.08, 0.16),
                "recommended_sample_size": 2000,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.85,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": [{"dimension": "decile", "segment": "1-2", "ate": 0.15, "n": 1000}],
            }

            result = workflow.propose_experiment(
                intervention_type="speaker_program_invitation",
                brand="Kisqali",
            )

        assert result["action"] == "DESIGN"
        assert "prior_estimate" in result
        assert "recommended_sample_size" in result

    def test_passes_prior_estimate(self, tool_module):
        """Test workflow passes prior estimate to experiment designer."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        with patch.object(tool_module, "simulate_intervention") as mock_sim:
            expected_ate = 0.10
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "deploy",
                "recommendation_rationale": "Proceed with test",
                "simulated_ate": expected_ate,
                "confidence_interval": (0.06, 0.14),
                "recommended_sample_size": 2500,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.80,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": [],
            }

            result = workflow.propose_experiment(
                intervention_type="call_frequency_increase",
                brand="Fabhalta",
            )

        assert result["prior_estimate"]["ate"] == expected_ate
        assert "ci" in result["prior_estimate"]

    def test_includes_top_segments(self, tool_module):
        """Test workflow includes top segments in design action."""
        DigitalTwinWorkflow = tool_module.DigitalTwinWorkflow
        workflow = DigitalTwinWorkflow()

        segments = [{"dimension": "specialty", "segment": "oncology", "ate": 0.14, "n": 500}]

        with patch.object(tool_module, "simulate_intervention") as mock_sim:
            # Note: simulate_intervention is called directly (not .invoke())
            mock_sim.return_value = {
                "simulation_id": str(uuid4()),
                "recommendation": "deploy",
                "recommendation_rationale": "Proceed",
                "simulated_ate": 0.11,
                "confidence_interval": (0.07, 0.15),
                "recommended_sample_size": 2000,
                "recommended_duration_weeks": 8,
                "simulation_confidence": 0.82,
                "fidelity_warning": False,
                "fidelity_warning_reason": None,
                "top_segments": segments,
            }

            result = workflow.propose_experiment(
                intervention_type="digital_engagement",
                brand="Remibrutinib",
            )

        assert result["top_segments"] == segments


# NOTE: the old TestMockResultGeneration class was removed with the fabricator it
# tested (_create_mock_result) — it asserted the fabricated-effect contract that
# H3 eliminates. Fail-closed behaviour is covered by
# TestSimulateInterventionFailsClosed above.


@pytest.mark.xdist_group(name="experiment_designer_tools")
class TestSimulateInterventionEdgeCases:
    """Test edge cases for simulate_intervention tool."""

    def test_minimal_twin_count(self, tool_module):
        """Test simulation with minimal twin count."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "email_campaign",
                "brand": "Kisqali",
                "twin_count": 100,
            }
        )

        assert "simulation_id" in result
        assert result["simulation_confidence"] >= 0

    def test_maximum_twin_count(self, tool_module):
        """Test simulation with high twin count."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "email_campaign",
                "brand": "Fabhalta",
                "twin_count": 50000,
            }
        )

        assert "simulation_id" in result

    def test_all_brands_supported(self, tool_module):
        """Test all supported brands work."""
        simulate_intervention = tool_module.simulate_intervention
        brands = ["Remibrutinib", "Fabhalta", "Kisqali"]

        for brand in brands:
            result = simulate_intervention.invoke(
                {
                    "intervention_type": "email_campaign",
                    "brand": brand,
                }
            )

            assert result["recommendation"] in ["deploy", "skip", "refine"], (
                f"Failed for brand: {brand}"
            )

    def test_all_intervention_types(self, tool_module):
        """Test all known intervention types."""
        simulate_intervention = tool_module.simulate_intervention
        intervention_types = [
            "email_campaign",
            "call_frequency_increase",
            "speaker_program_invitation",
            "sample_distribution",
            "peer_influence_activation",
            "digital_engagement",
        ]

        for itype in intervention_types:
            result = simulate_intervention.invoke(
                {
                    "intervention_type": itype,
                    "brand": "Kisqali",
                }
            )

            assert result["recommendation"] in ["deploy", "skip", "refine"], (
                f"Failed for intervention type: {itype}"
            )

    def test_unknown_intervention_type(self, tool_module):
        """Test handling of unknown intervention type."""
        simulate_intervention = tool_module.simulate_intervention
        result = simulate_intervention.invoke(
            {
                "intervention_type": "unknown_intervention",
                "brand": "Kisqali",
            }
        )

        # Should still return valid result with default effect
        assert "simulation_id" in result
        assert "simulated_ate" in result
