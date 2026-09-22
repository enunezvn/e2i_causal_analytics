"""``digital_twin_simulate_tool``: the chat surface for digital-twin simulation (#2211).

Why a chat tool and not a routing change to the composer: the two failing asks are SINGLE
questions. ``tool_composer_tool`` is the multi-faceted brain — its decomposer refuses fewer
than two sub-questions (``config/agent_config.yaml`` ``min_sub_questions: 2``) and, run for
real on the two asks (2026-09-22), inflated each into FIVE sub-questions (baseline
descriptives, a registry causal step, a comparison) around one simulation step whose intent
was EXPERIMENTAL — the fallback for which is ``power_calculator``. #2015 (PR #2042) built
``counterfactual_simulator`` as "the chat tool" that states the same numbers as the Digital
Twin page; this tool is the direct chat face of that same function, the shape #2115 used for
``forecast_kpi_tool`` / ``kpi_forecaster``.

The numbers are never this tool's own: every success payload is the engine's
``SimulationResults`` — proven here by running the REAL engine (real twins, the real cohort
provider over a planted-effect cohort, the real ``CohortCausalEstimator``) behind the real
``counterfactual_simulator``, with only the database and MLflow lookups in front of it
replaced (the same seams ``test_counterfactual_simulator_2015`` documents).
"""

from __future__ import annotations

import asyncio
import inspect
import uuid

import pytest

from src.agents.tool_composer import tool_registrations as tr
from src.api.routes import chat_twin_simulation_tool as cts
from src.api.routes.chatbot_tools import E2I_CHATBOT_TOOLS, E2I_TOOL_MAP

pytestmark = pytest.mark.timeout(240)


def _run(**kwargs):
    result = cts.digital_twin_simulate_tool.ainvoke(kwargs)
    return asyncio.run(result) if inspect.isawaitable(result) else result


# --------------------------------------------------------------------------- registration
def test_the_tool_is_bound_to_both_chat_brains():
    """``E2I_CHATBOT_TOOLS`` is what copilotkit's chat_node and chatbot_graph bind."""
    names = {t.name for t in E2I_CHATBOT_TOOLS}
    assert "digital_twin_simulate_tool" in names
    assert E2I_TOOL_MAP["digital_twin_simulate_tool"] is cts.digital_twin_simulate_tool


def test_the_description_carries_the_cues_of_the_failing_asks():
    """The model selects tools by description. TW.1 said 'use the digital twin … simulate',
    TW.2 'run a counterfactual … what would happen … if we increased call frequency'."""
    desc = cts.digital_twin_simulate_tool.description.lower()
    for cue in ("digital twin", "simulat", "counterfactual", "what would happen"):
        assert cue in desc, f"{cue!r} missing from the tool description"


def test_the_description_names_every_engine_intervention_and_nothing_else():
    import re

    from src.digital_twin.effect.provider import SUPPORTED_INTERVENTIONS

    schema = cts.digital_twin_simulate_tool.args_schema.model_json_schema()
    text = schema["properties"]["intervention"]["description"]
    named = {tok for tok in re.findall(r"[a-z_]+", text) if "_" in tok}
    assert named == set(SUPPORTED_INTERVENTIONS)
    in_description = {tok for tok in re.findall(r"[a-z_]+", cts.DESCRIPTION) if "_" in tok}
    assert in_description - {"causal_analysis_tool"} == set(SUPPORTED_INTERVENTIONS)


def test_the_tool_input_is_the_engine_input():
    """intervention, brand, optional regions — what ``counterfactual_simulator`` takes.
    Anything else (an expected effect, a duration) would be a number the engine ignores."""
    fields = set(cts.DigitalTwinSimulateInput.model_fields)
    assert fields == {"intervention", "brand", "target_regions"}


# --------------------------------------------------------------------------- refusals
def test_a_non_catalog_intervention_is_refused_by_name_with_no_numbers():
    payload = _run(intervention="free lunch for oncologists", brand="Kisqali")
    assert payload["success"] is False
    assert "free lunch for oncologists" in payload["error"]
    assert payload["reason_code"] == "invalid_input_value"
    assert set(payload["interventions_available"]) == set(cts.intervention_names())
    for key in ("effect", "ci_lower", "ci_upper", "recommendation"):
        assert key not in payload


def test_a_brand_without_a_twin_model_is_refused_by_name():
    payload = _run(intervention="email_campaign", brand="Cosentyx")
    assert payload["success"] is False
    assert "Cosentyx" in payload["error"]
    assert payload["reason_code"] == "invalid_input_value"


def test_a_non_region_target_is_refused_not_ignored():
    payload = _run(intervention="email_campaign", brand="Kisqali", target_regions=["oncologists"])
    assert payload["success"] is False
    assert "oncologists" in payload["error"]


@pytest.mark.parametrize(
    ("phrase", "expected"),
    [
        ("email_campaign", "email_campaign"),
        ("Email Campaign", "email_campaign"),
        ("email campaign", "email_campaign"),
        ("Increased Call Frequency", "call_frequency_increase"),
        ("increase call frequency", "call_frequency_increase"),
        ("call frequency", "call_frequency_increase"),
        ("speaker program", "speaker_program_invitation"),
        ("samples", None),
        ("", None),
        ("lunch", None),
    ],
)
def test_user_phrasing_resolves_to_one_catalog_intervention_or_none(phrase, expected):
    """The model passes what the user said; a catalog value, its label, or a phrase whose
    every word belongs to exactly one catalog entry resolves. Nothing is guessed: 'samples'
    is not 'sample_distribution' and an empty or unrelated phrase is None, so the engine's
    own refusal names the catalog."""
    assert cts.resolve_intervention(phrase) == expected


def test_an_unreachable_database_is_reported_as_retryable_not_as_absence(caplog):
    """The unit tree pins Supabase to a dead port (#1420). The answer must say the
    simulation could not run right now — never that the platform has no simulation."""
    payload = _run(intervention="email_campaign", brand="Kisqali")
    assert payload["success"] is False
    assert payload["retryable"] is True
    assert "could be read" in payload["error"]
    assert "Digital Twin" in payload["capability"]
    assert "not include" not in payload["error"]


# --------------------------------------------------------------------------- delegation
def _engine_seams(monkeypatch):
    """Replace the two network lookups in front of the engine with the 2015 fixtures.

    ``TwinRepository.list_active_models`` is a class attribute and ``tr._load_cohort_provider``
    / ``tr._run_twin_simulation`` are module globals: ``counterfactual_simulator`` resolves all
    three at call time, so these patches are what it uses. The simulation itself is the real
    ``_simulate_population`` on the real engine.
    """
    from src.digital_twin.effect.provider import CohortEffectDataProvider
    from src.digital_twin.twin_repository import TwinRepository
    from tests.unit.test_agents.test_tool_composer.test_counterfactual_simulator_2015 import (
        _cohort,
        _population,
    )

    provider = CohortEffectDataProvider(_cohort())
    seen = {}
    model_id = str(uuid.uuid4())

    async def list_active_models(self, twin_type=None, brand=None):
        return [{"model_id": model_id, "brand": brand, "mlflow_model_uri": None}]

    async def load_provider(client, intervention_type, brand_value):
        return provider

    def run_twin_simulation(model_row, prov, frame, intervention_type, brand_value, regions):
        out = tr._simulate_population(
            _population(),
            provider=prov,
            frame=frame,
            intervention_type=intervention_type,
            regions=regions,
            model_id=model_row["model_id"],
        )
        seen["engine"] = out
        return out

    monkeypatch.setattr(TwinRepository, "list_active_models", list_active_models)
    monkeypatch.setattr(tr, "_load_cohort_provider", load_provider)
    monkeypatch.setattr(tr, "_run_twin_simulation", run_twin_simulation)
    return seen


def test_the_success_payload_is_the_engines_result_verbatim(monkeypatch):
    seen = _engine_seams(monkeypatch)
    payload = _run(intervention="email campaign", brand="Kisqali")
    assert payload["success"] is True, payload
    engine_result, targeted = seen["engine"]
    assert targeted is None
    assert payload["intervention_type"] == "email_campaign"
    assert payload["brand"] == "Kisqali"
    assert payload["effect"] == pytest.approx(engine_result.simulated_ate)
    assert payload["ci_lower"] == pytest.approx(engine_result.simulated_ci_lower)
    assert payload["ci_upper"] == pytest.approx(engine_result.simulated_ci_upper)
    assert payload["effect_scope"] == "cohort"
    assert payload["cohort_effect"] == payload["effect"]
    assert payload["recommendation"] in {"deploy", "refine", "skip"}
    assert payload["twin_count"] == engine_result.twin_count
    # What the answer must carry: the synthetic-gold provenance and where the same
    # simulation lives outside chat.
    assert payload["evidence_is_synthetic"] is True
    assert "synthetic-gold" in " ".join(payload["assumptions"])
    assert payload["surfaces"]["endpoint"] == "POST /api/digital-twin/simulate"
    assert "digital-twin" in payload["surfaces"]["page"]
    assert payload["query_type"] == "digital_twin_simulation"


def test_a_region_target_is_answered_on_that_region(monkeypatch):
    seen = _engine_seams(monkeypatch)
    payload = _run(intervention="email_campaign", brand="Kisqali", target_regions=["Northeast"])
    assert payload["success"] is True, payload
    _engine_result, targeted = seen["engine"]
    assert targeted is not None and targeted.regions == ["northeast"]
    assert payload["effect_scope"] == "targeted regions ['northeast']"
    assert payload["target_regions"] == ["northeast"]
    assert payload["effect"] == pytest.approx(targeted.effect)
