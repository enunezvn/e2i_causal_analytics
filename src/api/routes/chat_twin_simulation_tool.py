"""``digital_twin_simulate_tool``: the chat surface for digital-twin simulation (#2211).

On 2026-09-22, with the cohort restored and ``/digital-twin/health`` reporting three brands
simulable, the AG-UI brain answered "Use the digital twin to simulate an email campaign
intervention for Kisqali HCPs" with *"The E2I platform doesn't include a 'digital twin'
simulation capability"* and called ``causal_analysis_tool`` (tools it could not see a twin
in). #2015 (PR #2042) built ``counterfactual_simulator`` as the chat's twin engine, but it is a
COMPOSABLE tool: reachable only through ``tool_composer_tool``, whose decomposer refuses fewer
than two sub-questions and, run for real on the two asks, inflated each into five steps with
the simulation step labelled EXPERIMENTAL. A single simulation ask is not a composition.

This tool is the direct chat face of that same function — the shape #2115 gave forecasting
(``forecast_kpi_tool`` for chat, ``kpi_forecaster`` for the composer). It delegates to
``counterfactual_simulator`` so the effect, interval, recommendation and experiment size a
chat answer states are the ones the Digital Twin page states for the same request. A second
implementation here would eventually disagree with the page inside the same answer.

IT LIVES IN ITS OWN MODULE BY CONSTRAINT: ``chatbot_tools.py`` and ``copilotkit.py`` are
size-ratchet pinned. Everything under ``src.digital_twin`` is imported lazily: importing any
module of that package costs 17 s and +548 MB (measured 2026-09-22), far too much for a
route module that is imported at API start.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from src.data.per_hcp_cohort_columns import INTERVENTION_TREATMENT_MAP

logger = logging.getLogger(__name__)

QUERY_TYPE = "digital_twin_simulation"

#: Where the same simulation lives outside chat. Named in every payload so the answer can
#: point the user at the page instead of denying the capability.
SURFACES: Dict[str, str] = {
    "page": "Digital Twin page (/digital-twin)",
    "endpoint": "POST /api/digital-twin/simulate",
    "chat_tool": "digital_twin_simulate_tool",
}

NOTE = (
    "A digital-twin SIMULATION on the brand's synthetic-gold per-HCP cohort, not an observed "
    "result: the effect is a causal-forest estimate of the intervention's effect on HCP "
    "conversion, with its 95% interval, the per-region effects, the DEPLOY / REFINE / SKIP "
    "recommendation and the per-arm experiment size the Digital Twin page states for the same "
    "request."
)

_WORD_RE = re.compile(r"[a-z0-9]+")
#: Words that carry no intervention meaning in a user's phrasing of one.
_STOPWORDS = frozenset({"a", "an", "the", "of", "for", "to", "and", "with", "more", "our", "its"})


def intervention_names() -> List[str]:
    """The engine's catalog values, from the side-effect-free contract module."""
    return list(INTERVENTION_TREATMENT_MAP)


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def resolve_intervention(phrase: Optional[str]) -> Optional[str]:
    """Map the user's phrasing onto ONE catalog intervention, or ``None``.

    A catalog value or its label (spaces or hyphens for underscores, any case) resolves
    directly; otherwise a phrase resolves only when it carries AT LEAST TWO meaningful words
    and every one of them belongs to exactly one catalog entry's words ("call frequency" ->
    call_frequency_increase; "increase call frequency" too, "increase" being a stem of that
    entry's "increased"). One word is never enough (codex r1 #1: "increase", "quality" and
    "distribution" each belong to one entry and would have driven a real simulation the user
    never named), "samples" is not "sample_distribution", and an ambiguous or unrelated
    phrase is None — so the engine's refusal names the catalog and the answer can say what
    it does serve.
    """
    if not isinstance(phrase, str) or not phrase.strip():
        return None
    from src.digital_twin.effect.provider import INTERVENTION_CATALOG  # lazy: heavy package

    wanted = re.sub(r"[\s\-]+", "_", phrase.strip().lower())
    for value, label in INTERVENTION_CATALOG:
        if wanted in (value, re.sub(r"[\s\-]+", "_", label.lower())):
            return value
    asked = _tokens(phrase)
    if len(asked) < 2:
        return None
    matches = []
    for value, label in INTERVENTION_CATALOG:
        vocabulary = _tokens(value.replace("_", " ")) | _tokens(label)
        if all(any(v.startswith(word) for v in vocabulary) for word in asked):
            matches.append(value)
    return matches[0] if len(matches) == 1 else None


class DigitalTwinSimulateInput(BaseModel):
    """Arguments for digital_twin_simulate_tool — what the twin engine can use."""

    intervention: str = Field(
        ...,
        description=(
            "The intervention to simulate, one of: "
            + ", ".join(INTERVENTION_TREATMENT_MAP)
            + ". Pass the user's phrasing if it names one of these (e.g. 'email campaign', "
            "'increase call frequency'); anything outside this catalog is refused by name."
        ),
    )
    brand: str = Field(
        ...,
        description="Kisqali, Fabhalta or Remibrutinib — the brand whose HCP twin to simulate.",
    )
    target_regions: Optional[List[str]] = Field(
        None,
        description=(
            "Optional US census regions to simulate on (northeast, south, midwest, west). "
            "The twin model's effects vary only by region, so no other targeting is served."
        ),
    )


async def run_twin_simulation(
    intervention: str, brand: str, target_regions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Run the digital-twin engine through ``counterfactual_simulator`` and report honestly.

    Success is the engine's ``SimulationResults`` verbatim plus provenance; a refusal is the
    engine's own reason with its code and the catalog; an unreachable model or cohort is
    reported as a retryable outage, never as the absence of the capability.
    """
    from src.agents.tool_composer import tool_registrations as tr
    from src.agents.tool_composer.errors import ToolInputError, ToolRefusalError
    from src.agents.tool_composer.executor import SyncToolTimeout

    resolved = resolve_intervention(intervention) or intervention
    base: Dict[str, Any] = {
        "query_type": QUERY_TYPE,
        "intervention_requested": intervention,
        "brand_requested": brand,
        "surfaces": dict(SURFACES),
        "capability": (
            "The platform runs digital-twin intervention simulations (Digital Twin page, "
            f"{SURFACES['endpoint']}, this tool)."
        ),
    }
    try:
        result = await tr.counterfactual_simulator(
            intervention=resolved, brand=brand, target_entities=target_regions
        )
    except (ToolInputError, ToolRefusalError) as exc:
        logger.info("digital_twin_simulate_tool refused: %s", exc)
        return {
            **base,
            "success": False,
            "error": str(exc),
            "reason_code": exc.reason_code.value,
            "details": dict(exc.details),
            "retryable": False,
            # The engine's CATALOG — the names it recognises — not what this brand can run
            # right now; per-brand availability is what a refusal's reason code states
            # (codex r3 #1).
            "intervention_catalog": intervention_names(),
        }
    except SyncToolTimeout as exc:
        logger.warning("digital_twin_simulate_tool timed out: %s", exc)
        return {**base, "success": False, "error": str(exc), "retryable": True}
    except Exception as exc:  # noqa: BLE001 - the outage is the honest answer
        logger.warning("digital_twin_simulate_tool could not run: %s", exc)
        return {
            **base,
            "success": False,
            "error": (
                f"The simulation could not run right now (service outage): {exc} The "
                "capability exists — retry, or use the Digital Twin page."
            ),
            "retryable": True,
        }
    return {
        **base,
        "success": True,
        **result.model_dump(),
        "evidence_is_synthetic": True,
        "note": NOTE,
    }


#: The description the model selects on. Written for the model, derived from the contract
#: module so it can never name an intervention the engine refuses.
DESCRIPTION = (
    "Simulate a commercial intervention for a brand with the Digital Twin engine — the tool "
    'for "use the digital twin", "simulate <intervention> for <brand>", "run a counterfactual" '
    'and "what would happen to <brand> conversion if we <intervention>" asks. Runs the engine '
    "behind the Digital Twin page's POST /api/digital-twin/simulate on the brand's per-HCP twin "
    "cohort and returns the simulated effect on HCP conversion with its 95% interval, "
    "per-region effects, a DEPLOY / REFINE / SKIP recommendation and the per-arm experiment "
    f"size. Interventions: {', '.join(INTERVENTION_TREATMENT_MAP)}. Brands: Kisqali, Fabhalta, "
    "Remibrutinib. Takes about a minute. This is a simulation on a synthetic-gold cohort — say "
    "so — and it is NOT causal_analysis_tool, which reports observed drivers from the "
    "causal-path registry and never simulates an intervention."
)


@tool(args_schema=DigitalTwinSimulateInput, description=DESCRIPTION)
async def digital_twin_simulate_tool(
    intervention: str, brand: str, target_regions: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Simulate an intervention with the Digital Twin engine (see ``DESCRIPTION``)."""
    return await run_twin_simulation(intervention, brand, target_regions)
