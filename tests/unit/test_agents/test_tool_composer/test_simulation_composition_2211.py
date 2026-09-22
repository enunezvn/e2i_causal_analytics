"""A simulation step inside a composition reaches ``counterfactual_simulator`` (#2211).

The composer is the multi-faceted brain, not the front door for a single simulation ask
(that is ``digital_twin_simulate_tool``). But a genuinely multi-step ask — "simulate an
email campaign for Kisqali AND compare it with the current gap" — is decomposed by an LLM
whose EXPERIMENTAL intent is documented as "test design OR simulation", and the planner's
fallback for EXPERIMENTAL is ``power_calculator``. Run for real on 2026-09-22, both #2211
asks decomposed with their simulation step labelled EXPERIMENTAL ("Simulate the email
campaign intervention on Kisqali HCPs using the digital twin model"), so a planner LLM
miss would have sized an experiment instead of simulating one. Same shape as #2115's
forecast route: a simulation-shaped EXPERIMENTAL ask goes to the simulator when it is
registered, and degrades to the old mapping when it is not.
"""

from __future__ import annotations

import pytest


class _Registry:
    def __init__(self, known):
        self.known = set(known)

    def validate_tool_exists(self, name):
        return name in self.known

    def get_schema(self, name):
        return None


class _SubQuestion:
    def __init__(self, question, intent, sq_id="sq1"):
        self.question = question
        self.intent = intent
        self.id = sq_id


def _map(
    question,
    intent,
    known=("counterfactual_simulator", "power_calculator", "causal_effect_estimator"),
):
    from src.agents.tool_composer.planner import ToolPlanner

    planner = ToolPlanner.__new__(ToolPlanner)
    planner.registry = _Registry(known)
    return planner._get_fallback_mapping(_SubQuestion(question, intent))


@pytest.mark.parametrize(
    "question",
    [
        # The two real decompositions of the #2211 asks (2026-09-22).
        "Simulate the email campaign intervention on Kisqali HCPs using the digital twin model",
        "What would the simulated Fabhalta conversion rate be if call frequency for its HCPs were increased?",
        "Run a counterfactual: what happens to conversion if we increase call frequency?",
        "Use the digital twin to predict the effect of a speaker program invitation",
    ],
)
def test_a_simulation_shaped_experimental_ask_maps_to_the_simulator(question):
    mapping = _map(question, "EXPERIMENTAL")
    assert mapping is not None
    assert mapping.tool_name == "counterfactual_simulator", f"{question!r} -> {mapping.tool_name}"
    assert mapping.source_agent == "experiment_designer"


@pytest.mark.parametrize(
    "question",
    [
        "How many patients do we need per arm to detect d=0.2?",
        "What sample size gives 80% power for the pilot?",
        "Design an A/B test for the new detailing cadence",
    ],
)
def test_a_design_shaped_experimental_ask_still_maps_to_the_power_calculator(question):
    """The asks the old mapping was RIGHT about must keep their tool."""
    mapping = _map(question, "EXPERIMENTAL")
    assert mapping is not None
    assert mapping.tool_name == "power_calculator", f"{question!r} -> {mapping.tool_name}"


@pytest.mark.parametrize(
    ("question", "intent"),
    [
        ("What drives Kisqali conversion?", "CAUSAL"),
        ("What is the causal effect of rep visits on Fabhalta TRx?", "CAUSAL"),
    ],
)
def test_an_ordinary_causal_ask_does_not_reach_the_simulator(question, intent):
    """The negative case the issue asks for: a driver question stays on the estimator."""
    mapping = _map(question, intent)
    assert mapping is not None
    assert mapping.tool_name == "causal_effect_estimator", f"{question!r} -> {mapping.tool_name}"


def test_the_simulation_route_degrades_when_the_simulator_is_not_registered():
    mapping = _map(
        "Simulate an email campaign for Kisqali with the digital twin",
        "EXPERIMENTAL",
        known=("power_calculator",),
    )
    assert mapping is not None
    assert mapping.tool_name == "power_calculator"


def test_a_simulation_cue_without_the_experimental_intent_is_not_hijacked():
    """Only the EXPERIMENTAL branch changes; the intent map for the others is untouched."""
    mapping = _map(
        "Simulate the descriptive baseline for Kisqali",
        "DESCRIPTIVE",
        known=("cohort_statistics", "counterfactual_simulator"),
    )
    assert mapping is not None
    assert mapping.tool_name == "cohort_statistics"
