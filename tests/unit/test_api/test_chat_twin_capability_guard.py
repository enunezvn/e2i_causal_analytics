"""The answer may not deny a capability the platform has (#2211): the twin capability guard.

The 2026-09-22 answer opened with *"The E2I platform doesn't include a 'digital twin'
simulation capability — there's no tool that runs a counterfactual/simulated intervention
forward and predicts its effect."* while ``/digital-twin/health`` reported three brands
simulable. The routing fix (``digital_twin_simulate_tool`` + prompt guidance) is the primary
defence; this guard is the residual one, and it is a CAPABILITY check, not a phrase list:

* the phrase family only DECIDES WHETHER TO ASK — it never asserts anything;
* the correction is derived from the live probe that ``/digital-twin/health`` runs
  (active twin models, brands whose cohort can identify an intervention) and claims exactly
  what that probe establishes — an active model row and usable cohort effect data — never
  that a run would succeed (the AFTER run of 2026-09-22 had an active row and an
  unreachable model registry). A dark twin gets a note that says so; an unmeasurable probe
  still corrects the existence claim (the tool IS bound) but promises nothing.

What would make the check pass while the condition is false? A denial phrased outside the
family goes uncorrected — that residual is documented, and the primary defence is the
routing. A probe that lies is the same lie the health endpoint tells; the guard adds none.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage

from src.api.routes import chat_twin_capability as guard

TW1_DENIAL = (
    "The E2I platform doesn't include a \"digital twin\" simulation capability — there's no "
    "tool that runs a counterfactual/simulated intervention forward and predicts its effect. "
    "I don't want to fabricate that kind of output.\n\nWhat I *can* do is check the causal-path "
    "registry for whether an email/engagement-type driver has a modeled causal effect."
)

TW2_ANSWER = (
    "**Fabhalta — call-frequency counterfactual on conversion: no direct causal path exists in "
    'the registry**\n\nI queried the causal-path registry for both "conversion" and "call '
    'frequency" for Fabhalta. There is **no causal chain linking call frequency to '
    "conversion** in the results returned."
)

#: The two AFTER answers of 2026-09-22 (whole AG-UI graph in-process, real tools; the model
#: registry was unreachable from that host, so the tool reported an outage). Both AFFIRM the
#: capability with a negated absence — "not a missing capability", "not a platform
#: limitation" — and a first draft of the detector fired on exactly that.
AFTER_TW1_OUTAGE = (
    "The digital twin simulation for Kisqali (email_campaign intervention) could not run right "
    'now:\n\n- **Error**: "counterfactual_simulator: trained twin model 428cea1e for Kisqali/hcp '
    'could not be loaded from the model registry."\n- The tool explicitly flags this as **a '
    "service outage, not a missing capability** — the platform does support this simulation, "
    "but the trained Kisqali HCP twin model isn't loading from the registry at the moment.\n"
    "- It's marked **retryable**.\n\nSince the simulation didn't return a result, I don't have "
    "any predicted effect size, confidence interval, or per-region breakdown to report — and no "
    "DEPLOY/REFINE/SKIP recommendation was generated. I can't answer the \"what effect / should "
    'we deploy" question with real numbers right now.\n\n**Options:**\n- I can retry the '
    "simulation now in case the registry issue was transient.\n- You could try the Digital Twin "
    "page (`/digital-twin`) directly, which hits the same simulation engine."
)
AFTER_TW2_OUTAGE = (
    "The counterfactual simulation didn't run this time.\n\n**What happened:** I called the "
    "Digital Twin engine (`digital_twin_simulate_tool`) for `call_frequency_increase` on "
    "**Fabhalta**'s HCP cohort, and it failed with:\n\n> \"The simulation could not run right "
    "now: counterfactual_simulator: trained twin model 796a07b1 for Fabhalta/hcp could not be "
    "loaded from the model registry. This is a service outage, not a missing capability — retry, "
    'or use the Digital Twin page."\n\n**Key point:** this is a service outage (the trained twin '
    "model couldn't be loaded from the registry), not a platform limitation — the digital twin "
    "does support call-frequency-increase simulations for Fabhalta's HCP cohort, and it's "
    "marked retryable."
)


# --------------------------------------------------------------------------- detection
@pytest.mark.parametrize(
    "text",
    [
        TW1_DENIAL,
        "There is no digital twin simulation tool on this platform.",
        "There is not any digital twin simulation tool here.",
        "The platform does not have a counterfactual simulation capability.",
        "The platform lacks a digital twin simulation capability.",
        "I can't run a simulation here — no tool simulates an intervention forward.",
        "Digital-twin simulation isn't available in this assistant.",
        "We don't offer what-if simulation of interventions.",
    ],
)
def test_a_denial_of_the_simulation_capability_is_detected(text):
    assert guard.denies_simulation_capability(text)


@pytest.mark.parametrize(
    "text",
    [
        TW2_ANSWER,
        AFTER_TW1_OUTAGE,
        AFTER_TW2_OUTAGE,
        "No causal chain links call frequency to conversion in the registry.",
        "There is no evidence from the digital twin simulation tool that calls improve conversion.",
        "The digital twin simulation predicts an effect of 0.135 [0.092, 0.178].",
        "The simulation could not run for Fabhalta: the cohort has no usable call_frequency "
        "channel (no_treatment_contrast). Re-run the backfill and try again.",
        "I cannot run a simulation: the model registry is unavailable.",
        "I cannot run the simulation because the twin model for Kisqali could not be loaded.",
        "Kisqali TRx was 12,400 last month.",
        "",
    ],
)
def test_a_registry_negative_or_an_honest_engine_refusal_is_not_a_denial(text):
    """A query-scoped negative (rule 10), a finding phrased with 'no evidence', and the
    engine's own refusal or outage (a stated cause) are legitimate."""
    assert not guard.denies_simulation_capability(text)


# --------------------------------------------------------------------------- correction
def _probe(brands, simulable, measured=True):
    async def probe():
        return guard.TwinCapability(
            model_brands=list(brands), simulable_brands=list(simulable), measured=measured
        )

    return probe


async def test_no_denial_means_no_probe_and_no_note():
    calls = []

    async def probe():
        calls.append(1)
        return guard.TwinCapability(["Kisqali"], ["Kisqali"], True)

    for honest in (TW2_ANSWER, AFTER_TW1_OUTAGE, AFTER_TW2_OUTAGE):
        assert await guard.simulation_denial_correction(honest, probe=probe) is None
    assert calls == [], "the probe is a database round-trip; it runs only on a denial"


async def test_a_denial_while_the_twin_is_simulable_is_corrected_from_the_probe():
    note = await guard.simulation_denial_correction(
        TW1_DENIAL, probe=_probe(["Fabhalta", "Kisqali", "Remibrutinib"], ["Kisqali", "Fabhalta"])
    )
    assert note is not None
    assert "digital_twin_simulate_tool" in note
    assert "POST /api/digital-twin/simulate" in note
    # The brands come from the probe, not from a canned sentence — and the claim is the
    # probe's: an active model with usable cohort effect data, not a promised run.
    tail = note.split("health check")[1]
    assert "Fabhalta" in tail and "Kisqali" in tail and "Remibrutinib" not in tail
    assert "active twin model" in note and "usable cohort effect data" in note
    assert "can simulate" not in note and "it can run" not in note
    assert "email_campaign" in note and "call_frequency_increase" in note


async def test_a_denial_while_the_twin_is_dark_is_corrected_without_promising_a_run():
    """Models exist, no brand's cohort can identify an intervention (the 2026-09-21 state):
    the capability exists, and the note says why it cannot run — never that it can."""
    note = await guard.simulation_denial_correction(
        TW1_DENIAL, probe=_probe(["Fabhalta", "Kisqali"], [])
    )
    assert note is not None
    assert "cannot run" in note
    assert "Fabhalta" in note and "Kisqali" in note
    assert "active twin model with usable" not in note


async def test_an_unmeasurable_probe_corrects_existence_but_claims_nothing_about_running():
    note = await guard.simulation_denial_correction(
        TW1_DENIAL, probe=_probe([], [], measured=False)
    )
    assert note is not None
    assert "could not verify" in note
    assert "active twin model with usable" not in note and "cannot run" not in note


async def test_a_probe_that_raises_never_breaks_the_answer(caplog):
    async def probe():
        raise ConnectionError("db down")

    note = await guard.simulation_denial_correction(TW1_DENIAL, probe=probe)
    assert note is not None and "could not verify" in note


async def test_the_default_probe_is_the_health_endpoints_own_check():
    """``twin_capability`` runs the same two reads ``/digital-twin/health`` runs; against the
    unit tree's dead Supabase port (#1420) it must report UNMEASURED, not dark."""
    cap = await guard.twin_capability()
    assert cap.measured is False
    assert cap.model_brands == [] and cap.simulable_brands == []


# --------------------------------------------------------------------------- graph wiring
class _DenyingLLM:
    """Boundary fake for get_chat_llm: streams the TW.1 denial as its answer."""

    def bind_tools(self, tools, tool_choice=None):
        return self

    async def astream(self, messages):
        yield AIMessageChunk(content=TW1_DENIAL)


def _graph_boundaries(copilotkit_mod, emitted):
    async def emit(config, text):
        emitted.append(text)

    return (
        patch.object(copilotkit_mod, "get_chat_llm", lambda **kw: _DenyingLLM()),
        patch.object(copilotkit_mod, "_ensure_conversation_exists", AsyncMock(return_value=False)),
        patch.object(copilotkit_mod, "_persist_message_sync", MagicMock(return_value=None)),
        patch.object(copilotkit_mod, "_record_analytics_sync", MagicMock(return_value=None)),
        patch.object(copilotkit_mod, "_collect_copilot_learning_signal", AsyncMock()),
        patch.object(copilotkit_mod, "copilotkit_emit_state", AsyncMock()),
        patch.object(copilotkit_mod, "copilotkit_emit_message", emit),
    )


async def test_a_direct_denial_through_the_real_agui_graph_carries_the_correction():
    """The chat_node seam: through ``create_e2i_chat_agent`` (the AG-UI brain's real entry
    point), a no-tool answer that denies the capability leaves the node with the correction
    appended and emitted. The probe is the module attribute the guard resolves AT CALL TIME,
    so patching it is what the node uses."""
    from src.api.routes import copilotkit as copilotkit_mod

    emitted: list = []
    with patch.multiple(guard, twin_capability=_probe(["Kisqali"], ["Kisqali"])):
        patches = _graph_boundaries(copilotkit_mod, emitted)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            graph = copilotkit_mod.create_e2i_chat_agent()
            result = await graph.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content="Use the digital twin to simulate an email campaign for Kisqali"
                        )
                    ]
                },
                config={"configurable": {"thread_id": "test-2211-direct"}},
            )

    final = result["messages"][-1].content
    assert TW1_DENIAL in final
    assert "digital_twin_simulate_tool" in final
    assert any("digital_twin_simulate_tool" in t for t in emitted), emitted


async def test_a_synthesized_denial_after_a_tool_turn_carries_the_correction_too():
    """The synthesize seam, through the same compiled graph: the state is placed as if the
    tools node had just run (a causal_analysis_tool turn — the 2026-09-22 shape), the graph
    is resumed, and synthesize_node's streamed denial leaves with the correction appended
    to the user-visible answer and emitted. No source counting: the seam runs."""
    from src.api.routes import copilotkit as copilotkit_mod

    emitted: list = []
    with patch.multiple(guard, twin_capability=_probe(["Kisqali"], ["Kisqali"])):
        patches = _graph_boundaries(copilotkit_mod, emitted)
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5], patches[6]:
            graph = copilotkit_mod.create_e2i_chat_agent()
            config = {"configurable": {"thread_id": "test-2211-synth"}}
            graph.update_state(
                config,
                {
                    "messages": [
                        HumanMessage(content="Use the digital twin to simulate an email campaign"),
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": "causal_analysis_tool",
                                    "args": {"kpi_name": "email campaign", "brand": "Kisqali"},
                                    "id": "call-1",
                                }
                            ],
                        ),
                        ToolMessage(
                            content='{"success": true, "causal_chains_found": 0}',
                            name="causal_analysis_tool",
                            tool_call_id="call-1",
                        ),
                    ]
                },
                as_node="tools",
            )
            result = await graph.ainvoke(None, config=config)

    final = result["messages"][-1].content
    assert TW1_DENIAL in final
    assert "digital_twin_simulate_tool" in final
    assert any("digital_twin_simulate_tool" in t for t in emitted), emitted
