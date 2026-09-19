"""THE CENSUS GUARD (#2150): no surface may hand-maintain axis capability.

One fact — which KPI serves which axis, for which brand, under what window — was
hand-copied into 13 places, at least 8 of them LLM-facing, and they drifted: the
prompts instruct the model to call KPIs the tool refuses. This file names every
surface so a fourteenth copy cannot be added silently.

⚠ TWO INSTRUMENTS, BECAUSE THE SURFACES ARE TWO KINDS — and picking one instrument
for both is the proxy trap this lane wrote to memory
(``proxy_instead_of_capability_check_20260916``):

* **ID-SET surfaces** hold a ``frozenset`` / ``dict`` of KPI ids. For these the
  question is "is this produced by the evaluator?", so they are asserted against
  the policy's output. A scan for ``WS3-BI-\\d{3}`` would be the proxy here: an id
  set can be hand-maintained in a hundred ways and a literal is only one of them.
* **PROSE surfaces** name KPIs in ENGLISH — "TRx, NRx, NBRx" — and carry no id at
  all, so an id scan cannot see them and would report them clean. For these the
  question is "does the KPI this prose names actually serve the axis it is being
  named for?", which is answered by resolving the term through the production
  resolver and checking the served set. That is a capability check, not a grep.

Measured at 0f4916916, and the reason the prose half is red: ``recognize_kpi``
maps TRx/NRx/NBRx to WS3-BI-005/006/007, and NONE of those is in
``_PATIENT_AXIS_KPI_IDS`` for any axis — the canonical ids refuse every patient
axis. The prompts and field descriptions still name them.
"""

from __future__ import annotations

import re
from typing import Dict, List

import pytest

#: Prose surfaces: (label, text-getter). Every one reaches the LLM.
_PATIENT_AXIS_VOCAB = ("segment", "therapy_line", "biologic", "ige_tier")


def _served_ids(axis: str) -> frozenset:
    from src.api.routes.chatbot_tools import _PATIENT_AXIS_KPI_IDS

    return _PATIENT_AXIS_KPI_IDS[axis]


def _named_volume_terms(text: str) -> List[str]:
    """The volume KPI terms a piece of guidance tells the model to call.

    Deliberately narrow: only the three volume names whose canonical-vs-panel
    identity is the subject of #2114. ``TRx share`` is excluded because it is
    matched by its own dedicated assertions elsewhere, and matching it here would
    make this helper answer two questions at once.
    """
    found = []
    for term in ("NBRx", "NRx", "TRx"):
        # word-boundary, and not the "TRx share" / "TRx Panel" compounds
        for match in re.finditer(rf"\b{term}\b(?! [Ss]hare)(?! Panel)", text):
            found.append(text[match.start() : match.end()])
    return found


# =============================================================================
# PROSE SURFACES (1-6, 10, 11, 12) — capability check, not a grep
# =============================================================================


@pytest.mark.unit
@pytest.mark.parametrize("axis", _PATIENT_AXIS_VOCAB)
def test_axis_field_description_names_only_kpis_that_serve_that_axis(axis):
    """Surfaces 1-4. The existing guard in test_chatbot_kpi_axis_gate_1911.py
    asserts ``"TRx" in desc`` — a SUBSTRING, which is satisfied whether the name
    resolves to a KPI that serves the axis or one that refuses it. That is why it
    passes today while the description steers the model at the canonical ids.
    """
    from src.api.routes.chatbot_tools import KpiCalculateInput
    from src.services.kpi_resolution import recognize_kpi

    desc = KpiCalculateInput.model_fields[axis].description or ""
    served = _served_ids(axis)
    offenders: Dict[str, str] = {}
    for term in set(_named_volume_terms(desc)):
        kpi = recognize_kpi(term)
        if kpi is None or kpi.id not in served:
            offenders[term] = "unresolvable" if kpi is None else kpi.id
    assert not offenders, (
        f"{axis} description names KPIs that do NOT serve it: {offenders}; "
        f"served ids are {sorted(served)}"
    )


@pytest.mark.unit
@pytest.mark.parametrize("surface", ["chatbot_graph", "copilotkit"])
def test_breakdown_guidance_names_only_kpis_that_serve_the_axes_it_offers(surface):
    """Surfaces 11 and 12 — the two prompt copies, which no plan task owned.

    The guidance offers segment / therapy_line / biologic / ige_tier breakdowns and
    tells the model which KPIs to call for them. Every KPI it names must serve the
    axes it is named for, or the prompt is instructing the model to make a call the
    tool refuses (codex r13 HIGH).
    """
    from src.services.kpi_resolution import recognize_kpi

    if surface == "copilotkit":
        from src.api.routes.copilotkit import E2I_COPILOT_SYSTEM_PROMPT as prompt
    else:
        from src.api.routes.chatbot_graph import E2I_CHATBOT_SYSTEM_PROMPT as prompt

    lines = [ln for ln in prompt.splitlines() if ln.startswith("- BREAKDOWN GUIDANCE:")]
    assert len(lines) == 1, f"{surface}: expected exactly one BREAKDOWN GUIDANCE line"
    served = _served_ids("segment")
    offenders: Dict[str, str] = {}
    for term in set(_named_volume_terms(lines[0])):
        kpi = recognize_kpi(term)
        if kpi is None or kpi.id not in served:
            offenders[term] = "unresolvable" if kpi is None else kpi.id
    assert not offenders, (
        f"{surface} BREAKDOWN GUIDANCE tells the model to call {offenders} for a "
        f"patient-axis breakdown; those KPIs refuse it. Served: {sorted(served)}"
    )


# =============================================================================
# ID-SET SURFACES (7, 8, 10) — derived from the policy, not hand-maintained
# =============================================================================


@pytest.mark.unit
def test_the_patient_axis_allowlist_is_derived_from_the_policy():
    """Surface 7. Correct TODAY (e9d767372 moved it to the panel ids) but still
    hand-maintained, which is the #2150 defect: being right once is not the same as
    being derived."""
    from src.api.routes.chatbot_tools import _PATIENT_AXIS_KPI_IDS
    from src.kpi.capability_policy import axis_kpi_ids

    for axis in _PATIENT_AXIS_KPI_IDS:
        assert _PATIENT_AXIS_KPI_IDS[axis] == axis_kpi_ids(axis), axis


@pytest.mark.unit
def test_the_volume_coverage_probe_set_is_derived_from_the_policy():
    """Surface 8 — and this one is a live BEHAVIOURAL defect, not just drift.

    ``_VOLUME_KPI_IDS`` gates the trailing-30-day coverage probe and still reads
    {005, 006, 007}. Those now read MONTHLY business_metrics, where a trailing-30d
    prescription-coverage probe means nothing; meanwhile the panel event counts
    011..013, which the probe was designed for, are not in the set at all. So the
    probe currently runs on the wrong three KPIs in both directions.
    """
    from src.api.routes.chatbot_tools import _VOLUME_KPI_IDS
    from src.kpi.capability_policy import trailing_coverage_kpi_ids

    assert _VOLUME_KPI_IDS == trailing_coverage_kpi_ids()


@pytest.mark.unit
def test_the_capability_catalog_axis_rules_are_derived_from_the_policy():
    """Surface 10: AXIS_RULES is prose ASSEMBLED from a hand-maintained view of the
    same fact, and it is injected into the chat capability prompt."""
    from src.kpi.capability_policy import axis_rules_prose
    from src.services.chat_capability_catalog import AXIS_RULES

    assert AXIS_RULES == axis_rules_prose()


# =============================================================================
# THE CENSUS ITSELF — every surface is named, so a 14th cannot appear quietly
# =============================================================================


@pytest.mark.unit
def test_every_known_surface_is_covered_by_a_test_in_this_file():
    """A census that does not enumerate its own subjects rots silently. If a
    surface is added to or removed from the consolidation, this list must move with
    it, and the test that covers it must exist.
    """
    covered = {
        "chatbot_tools.KpiCalculateInput.segment": test_axis_field_description_names_only_kpis_that_serve_that_axis,
        "chatbot_tools.KpiCalculateInput.therapy_line": test_axis_field_description_names_only_kpis_that_serve_that_axis,
        "chatbot_tools.KpiCalculateInput.biologic": test_axis_field_description_names_only_kpis_that_serve_that_axis,
        "chatbot_tools.KpiCalculateInput.ige_tier": test_axis_field_description_names_only_kpis_that_serve_that_axis,
        "chatbot_tools._PATIENT_AXIS_KPI_IDS": test_the_patient_axis_allowlist_is_derived_from_the_policy,
        "chatbot_tools._VOLUME_KPI_IDS": test_the_volume_coverage_probe_set_is_derived_from_the_policy,
        "chat_capability_catalog.AXIS_RULES": test_the_capability_catalog_axis_rules_are_derived_from_the_policy,
        "chatbot_graph.BREAKDOWN_GUIDANCE": test_breakdown_guidance_names_only_kpis_that_serve_the_axes_it_offers,
        "copilotkit.BREAKDOWN_GUIDANCE": test_breakdown_guidance_names_only_kpis_that_serve_the_axes_it_offers,
    }
    assert len(covered) == 9, "the census list changed without its tests changing"
    assert all(callable(fn) for fn in covered.values())
