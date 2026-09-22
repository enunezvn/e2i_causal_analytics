"""The answer may not deny a capability the platform has: the twin capability guard (#2211).

The 2026-09-22 answer opened with *"The E2I platform doesn't include a 'digital twin'
simulation capability — there's no tool that runs a counterfactual/simulated intervention
forward"* while ``/digital-twin/health`` reported three brands simulable. The routing fix
(``digital_twin_simulate_tool`` bound to the chat brains, named in both prompts) is the
primary defence; this is the residual one, for the turn where the model still narrates a
platform-level negative (prompt rule 10) about simulation.

IT IS A CAPABILITY CHECK, NOT A PHRASE LIST. The phrase family below only decides whether to
ASK; it asserts nothing. The correction is derived from :func:`twin_capability`, the same two
reads ``/digital-twin/health`` runs (active twin models; brands whose cohort can identify an
intervention, via ``digital_twin_capability.simulable_brands`` and its 300-s cache). A twin
that is dark gets a note that says it cannot run and why; a probe that cannot measure still
corrects the EXISTENCE claim — the tool is bound, the page and the endpoint exist — and
promises nothing about running. What the check cannot catch is a denial phrased outside the
family; that residual is accepted because the routing is the defence, not this note.

Kept out of ``copilotkit.py`` (size-ratchet pinned), which calls
:func:`simulation_denial_correction` at its two answer seams, the way it appends the #1691
superlative note: the answer has already streamed, so the correction is appended, never
rewritten in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Optional, Sequence

from src.data.per_hcp_cohort_columns import INTERVENTION_TREATMENT_MAP

logger = logging.getLogger(__name__)

_SUBJECT = r"(?:digital[\s-]?twin|simulat\w*|counterfactual\w*|what[\s-]if)"
#: "no …" / "isn't a …" / "not a …", except when what follows is itself an absence word:
#: "not a missing capability" and "not a platform limitation" affirm the capability.
_NEG = (
    r"(?:\bno\b|\b(?:isn'?t|is not|not)\s+an?\b)"
    r"(?!\s+(?:missing|lack\w*|absen\w*|unavail\w*|(?:platform\s+)?limitation))"
)
#: Shapes of a platform-level negative about simulation, each measured on 2026-09-22's
#: answer or its obvious paraphrases. Every pattern needs a NEGATION and a SUBJECT within a
#: short span, so "no causal chain links call frequency to conversion" (a registry negative)
#: and "the simulation could not run for Fabhalta" (the engine's own refusal) do not match.
_DENIAL_PATTERNS = tuple(
    re.compile(p, re.IGNORECASE | re.DOTALL)
    for p in (
        # "doesn't include a digital twin simulation capability" / "does not have ... simulation"
        r"\b(?:doesn'?t|does not|don'?t|do not|didn'?t|did not)\s+"
        r"(?:include|have|offer|provide|support|expose)\b[^.\n]{0,60}?" + _SUBJECT,
        # "there's no tool that runs a counterfactual" / "isn't a tool that simulates" — the
        # capability noun first, the subject after it. A negated ABSENCE ("not a missing
        # capability", "not a platform limitation") is an affirmation — measured on the
        # 2026-09-22 AFTER answers, where the tool's own outage text tripped a first draft.
        _NEG + r"[^.\n]{0,40}?"
        r"\b(?:tool|capability|feature|engine|way|function|module)s?\b[^.\n]{0,80}?" + _SUBJECT,
        # "no digital twin simulation tool on this platform" — the subject first.
        _NEG + r"[^.\n]{0,40}?" + _SUBJECT + r"[^.\n]{0,40}?"
        r"\b(?:tool|capability|feature|engine|way|function|module)s?\b",
        # "digital-twin simulation isn't available / supported / part of / included"
        _SUBJECT + r"[^.\n]{0,60}?\b(?:isn'?t|is not|aren'?t|are not|not)\s+"
        r"(?:available|supported|part of|included|exposed|offered|something)\b",
        # "I can't run a simulation here" — an assistant-level inability, not an engine refusal
        # (the engine's refusals say WHY: "could not run for <brand>: <reason>").
        r"\b(?:can'?t|cannot|unable to|not able to)\s+(?:run|perform|execute|do)\s+"
        r"(?:an?\s+)?(?:[\w-]+\s+){0,2}?" + _SUBJECT + r"\b(?![^.\n]*\b(?:for|because|:)\b)",
    )
)


def denies_simulation_capability(text: Optional[str]) -> bool:
    """Does ``text`` assert that the platform lacks simulation / twin / counterfactual?"""
    if not text:
        return False
    return any(p.search(text) for p in _DENIAL_PATTERNS)


@dataclass(frozen=True)
class TwinCapability:
    """What the live probe found: the brands with an active twin model, those whose cohort
    can identify at least one intervention right now, and whether that was measured."""

    model_brands: List[str] = field(default_factory=list)
    simulable_brands: List[str] = field(default_factory=list)
    measured: bool = True


async def twin_capability() -> TwinCapability:
    """The reads ``/digital-twin/health`` performs, as a capability statement.

    ``measured=False`` when the repository cannot be read (the health route's ``degraded``
    with zeroed counts) or when every simulability probe errored: unknown, not dark.
    """
    from src.api.routes.digital_twin_capability import brand_is_simulable, simulable_brands
    from src.digital_twin.twin_repository import TwinRepository  # lazy: heavy package
    from src.memory.services.factories import get_async_supabase_client

    try:
        client = await get_async_supabase_client()
        repo = TwinRepository(supabase_client=client)
        models = await repo.list_active_models()
        if not models:
            # list_active_models logs and returns [] on a database error — "no model" and
            # "unreadable" look the same here, and neither licenses a claim about running.
            return TwinCapability([], [], measured=False)
        # The health route's call (it logs the dark state); the per-brand answers below come
        # from its 300-s cache, and None means that brand's probes errored.
        model_brands, _n = await simulable_brands(repo.client, models)
        answers = {b: await brand_is_simulable(repo.client, b) for b in model_brands}
    except Exception as exc:  # noqa: BLE001 - the guard must never break the answer
        logger.warning("twin capability probe failed: %s", exc)
        return TwinCapability([], [], measured=False)
    simulable = [b for b, a in answers.items() if a]
    if not simulable and any(a is None for a in answers.values()):
        return TwinCapability(model_brands, [], measured=False)
    return TwinCapability(model_brands, simulable, measured=True)


def _note(capability: TwinCapability) -> str:
    where = (
        "the Digital Twin page, `POST /api/digital-twin/simulate`, or the chat tool "
        "`digital_twin_simulate_tool`"
    )
    catalog = ", ".join(INTERVENTION_TREATMENT_MAP)
    head = (
        "\n\n**Correction — the platform does have a digital-twin simulation capability.** "
        f"It simulates a commercial intervention forward on a brand's HCP twin cohort via {where} "
        f"(interventions: {catalog})."
    )
    if not capability.measured:
        return head + (
            " I could not verify right now whether a simulation can run; the Digital Twin "
            "page's health readout will say."
        )
    if capability.simulable_brands:
        brands = ", ".join(capability.simulable_brands)
        return head + (
            f" Right now it can simulate for {brands}. Ask me to run one — e.g. 'simulate an "
            "email campaign for Kisqali'."
        )
    brands = ", ".join(capability.model_brands) or "any brand"
    return head + (
        f" Right now it cannot run: {brands} have a trained twin model but no usable cohort "
        "effect data (the per-HCP cohort's planted treatment channels are missing), so every "
        "simulation would refuse until the cohort is restored."
    )


Probe = Callable[[], Awaitable[TwinCapability]]


#: The chat tool whose presence in a turn's tool results settles the question: an answer
#: that just ran the twin cannot be denying that the twin exists.
TWIN_TOOL_NAME = "digital_twin_simulate_tool"


async def simulation_denial_correction(
    answer: Optional[str],
    probe: Optional[Probe] = None,
    tools_ran: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """The correction to append to ``answer``, or ``None`` when it denies nothing.

    ``tools_ran`` are the turn's tool results: when the twin tool itself ran, the answer is
    narrating its payload (a result, a refusal or an outage), so nothing is appended however
    the narration is worded. ``probe`` defaults to :func:`twin_capability` resolved from this
    module AT CALL TIME, so a test that patches ``chat_twin_capability.twin_capability``
    patches what the graph uses. The probe runs only on a denial: it is a database round-trip.
    """
    if tools_ran and TWIN_TOOL_NAME in tools_ran:
        return None
    if not denies_simulation_capability(answer):
        return None
    run = probe or twin_capability
    try:
        capability = await run()
    except Exception as exc:  # noqa: BLE001 - never break the answer
        logger.warning("twin capability probe raised: %s", exc)
        capability = TwinCapability([], [], measured=False)
    logger.warning(
        "[CopilotKit] #2211 twin capability guard: answer denied simulation; probe=%s", capability
    )
    return _note(capability)
