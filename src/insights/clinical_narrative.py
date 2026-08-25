"""Clinical-context narrative distillation for ONE causal analysis.

One flowing narrative that reads the specific causal result (signed ATE, CI,
robustness gate) through the brand's clinical and competitive context — the
drill-down panel's primary read (spec 2026-08-24). Mirrors causal_discovery.py:
DSPy signature guarded by import, build_grounding, honest fallback, and a
post-generation fabrication guard (identifiers not present in the grounding
reject the sample). Facts come from ClinicalContextService.get_context, fetched
SERVER-side by the route; this module never calls the network itself.

Unlike the digit-free executive-brief/HTE surfaces, this surface REPORTS effect
figures (the causal-discovery insight precedent) — digits are allowed.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.insights.clinical_context import format_clinical_positioning
from src.insights.common import run_signature

logger = logging.getLogger(__name__)

try:
    import dspy

    class ClinicalNarrativeSignature(dspy.Signature):
        """Write ONE flowing narrative (2-4 short paragraphs, no headings, no
        bullet lists) that reads a single causal analysis through the brand's
        clinical and competitive reality, for a pharma brand analyst deciding
        what to do next.

        STRICTLY grounded: use ONLY the facts provided in the inputs. NEVER
        invent trial results, citations, PMIDs, NCT ids, numbers, competitors,
        or label claims. The ONLY numbers allowed are ones present in the
        inputs (the effect estimate, its confidence interval, and figures
        quoted verbatim inside the provided facts).

        Do NOT merely restate the facts — draw the strategic implications they
        license. Translate the label's own boundary into which patient or
        prescriber segments may and may not be recommended for targeting, and be
        precise about WHICH axis the label restricts: an indication limited to
        patients refractory to one prior therapy class excludes only segments
        naive to or responsive on THAT class — segments defined on other axes
        (e.g. exposure to a different therapy class) remain in-label. Read the
        competitive set for what it implies about where this estimate can be
        acted on, and close on what the estimate does and does not license as a
        next commercial action. Every implication must be a stated logical
        consequence of a provided fact — never a new fact.

        When the analysis input says the treatment is a commercial lever
        (access/promotion), the mechanism, endpoints and label describe the
        THERAPY, never the lever — do not read them as evidence about the
        lever.

        Weave absences in honestly (no real-world evidence yet, outcome not
        mapped to any registered endpoint, evidence unavailable) instead of
        omitting them — an absence woven into the story is part of the story.

        The estimate comes from a SYNTHETIC patient cohort; the clinical and
        competitive context is REAL. Keep that boundary explicit and never
        present the estimate as clinical evidence."""

        analysis: str = dspy.InputField(
            desc="What this causal analysis asks: framing, treatment kind, grain"
        )
        result: str = dspy.InputField(
            desc="The estimate: signed ATE, CI, robustness gate verdict, synthetic-cohort boundary"
        )
        clinical_position: str = dspy.InputField(
            desc=(
                "Mechanism of action, approved indication verbatim, limitations of "
                "use, labeled target population / line of therapy"
            )
        )
        competitive_position: str = dspy.InputField(
            desc="Competitive framing for this analysis + the curated rival list"
        )
        trial_endpoints: str = dspy.InputField(
            desc="Registered pivotal trial endpoint measures + whether OUR outcome maps to one"
        )
        evidence: str = dspy.InputField(
            desc=(
                "Label considerations bearing on this outcome (or their honest absence), "
                "public-KG indication edge, real-world evidence or its honest absence"
            )
        )

        narrative: str = dspy.OutputField(
            desc=(
                "2-4 flowing paragraphs; every clause traceable to an input fact or a "
                "stated implication of one; absences woven in honestly; closes on the "
                "actionable strategic read (segment eligibility under the label, "
                "targeting, what this estimate licenses next)"
            )
        )

    DSPY_AVAILABLE = True
except ImportError:
    DSPY_AVAILABLE = False
    ClinicalNarrativeSignature = None  # type: ignore[assignment,misc]


_GATE_PHRASES = {
    "proceed": "survived all robustness checks",
    "review": "needs review (mixed robustness)",
    "block": "failed robustness checks",
}
# Mirrors the panel's MAX_ENDPOINTS_SHOWN: the endpoint list grounds outcome
# definitions, it is not a data table.
_MAX_ENDPOINTS = 5
_LIVE_SOURCES = {"chembl", "clinicaltrials.gov", "pubmed", "openfda"}
_SYNTHETIC_NOTE = (
    "The estimate comes from a synthetic patient cohort (gold-standard demo data); "
    "the clinical and competitive context is real."
)


def _result_sentence(
    treatment: str,
    outcome: str,
    ate: Optional[float],
    lo: Optional[float],
    hi: Optional[float],
    gate: Optional[str],
) -> str:
    if ate is None:
        est = f"No effect estimate was provided for {treatment} -> {outcome}."
    else:
        ci = f" [95% CI {lo:+.4f}, {hi:+.4f}]" if lo is not None and hi is not None else ""
        est = f"Estimated effect of {treatment} on {outcome}: ATE {ate:+.4f}{ci}."
    phrase = _GATE_PHRASES.get((gate or "").lower())
    if phrase:
        est += f" Robustness gate: {gate} — the estimate {phrase}."
    elif gate:
        # An unmapped verdict is reported raw, never claimed absent — the
        # Gate chip echoes the same raw value (omit-or-report discipline).
        est += f" Robustness gate: {gate}."
    else:
        est += " Robustness gate: not reported."
    return est + " " + _SYNTHETIC_NOTE


def build_grounding(
    payload: dict[str, Any],
    *,
    grain: str,
    ate: Optional[float],
    ate_ci_lower: Optional[float],
    ate_ci_upper: Optional[float],
    gate_decision: Optional[str],
) -> dict[str, Any]:
    """Compose the six grounding strings from a ClinicalContextService payload
    + the caller-supplied result. Every string is honest about absences — the
    LM is instructed to weave them in, never to fill them."""
    brand = str(payload.get("brand") or "")
    treatment = str(payload.get("our_treatment") or "")
    outcome = str(payload.get("our_outcome") or "")
    tc = payload.get("treatment_context") or {}

    # -- analysis --------------------------------------------------------
    parts: list[str] = []
    if payload.get("analysis_framing"):
        parts.append(str(payload["analysis_framing"]))
    kind = tc.get("kind")
    label = tc.get("label") or treatment
    if kind == "commercial":
        parts.append(
            f"The treatment '{label}' is a commercial (access/promotion) lever, not a "
            "therapy: the clinical sources below describe the therapy, never this lever."
        )
    elif kind == "clinical_covariate":
        parts.append(
            f"The treatment '{label}' is a patient-state variable used as an observational treatment."
        )
    elif kind == "drug_therapy":
        parts.append(
            f"The treatment '{label}' is a therapy contrast — the clinical sources describe it directly."
        )
    parts.append(f"Analysis grain: {grain}.")
    analysis = " ".join(parts)

    # -- result ----------------------------------------------------------
    result = _result_sentence(treatment, outcome, ate, ate_ci_lower, ate_ci_upper, gate_decision)

    # -- clinical_position ----------------------------------------------
    ind = payload.get("approved_indications") or {}
    lines = [f"{payload.get('drug_name')} — {payload.get('disease')}."]
    mech = (payload.get("mechanism") or {}).get("mechanism_of_action")
    if mech:
        lines.append(f"Mechanism of action: {mech}.")
    if ind.get("indications"):
        lines.append("Approved indication (label, verbatim): " + " | ".join(ind["indications"]))
    if ind.get("limitations_of_use"):
        lines.append(f"Limitations of use: {ind['limitations_of_use']}")
    positioning = format_clinical_positioning(brand)
    if positioning:
        lines.append(positioning)
    clinical_position = " ".join(lines)

    # -- competitive_position -------------------------------------------
    ag = payload.get("analysis_grounding") or {}
    comp_lines: list[str] = []
    if ag.get("competitive_context"):
        comp_lines.append(str(ag["competitive_context"]))
    rivals = list((payload.get("competitor_landscape") or {}).get("competitors") or [])
    if rivals:
        comp_lines.append("Curated rivals: " + "; ".join(rivals) + ".")
    competitive_position = (
        " ".join(comp_lines) or "No competitive context is established for this analysis."
    )

    # -- trial_endpoints -------------------------------------------------
    eps = payload.get("pivotal_endpoints") or {}
    measures = [str(e.get("measure")) for e in (eps.get("endpoints") or []) if e.get("measure")]
    ep_lines: list[str] = []
    if measures:
        extra = f" (+{len(measures) - _MAX_ENDPOINTS} more)" if len(measures) > _MAX_ENDPOINTS else ""
        ep_lines.append(
            "Registered pivotal trial endpoint measures: "
            + "; ".join(measures[:_MAX_ENDPOINTS])
            + extra
            + "."
        )
    else:
        ep_lines.append("No registered trial endpoints are available for this brand.")
    mapped = payload.get("mapped_endpoint")
    if mapped:
        ep_lines.append(f"Our outcome '{outcome}' maps to the real endpoint: {mapped}.")
    else:
        ep_lines.append(f"Our outcome '{outcome}' is not mapped to any registered endpoint.")
    trial_endpoints = " ".join(ep_lines)

    # -- evidence --------------------------------------------------------
    ev_lines: list[str] = []
    considerations = list(ag.get("label_considerations") or [])
    if considerations:
        for c in considerations:
            ev_lines.append(
                f"Label consideration ({c.get('section')}): {c.get('title')} — {c.get('detail')}"
            )
    elif ind.get("source") == "openfda":
        # Provenance discrimination (#1767): an empty list under openfda means
        # "read, nothing bears"; under the fallback it means "could not read".
        ev_lines.append("The FDA label was read and carries nothing bearing on this outcome.")
    else:
        ev_lines.append("The FDA label could not be read for this analysis (curated fallback in use).")
    ce = payload.get("causal_evidence") or {}
    edge = ce.get("indication_edge")
    if edge:
        verb = "an approved therapy for" if edge.get("predicate") == "treats" else "in development for"
        ev_lines.append(
            f"Open Targets records {edge.get('drug_name')} as {verb} {edge.get('disease_name')} "
            f"(max clinical stage: {edge.get('max_clinical_stage')})."
        )
    if ce.get("note"):
        ev_lines.append(str(ce["note"]))
    rwe_titles: list[str] = []
    for key in ("seminal_real_world_evidence", "real_world_evidence"):
        r = payload.get(key)
        if r and r.get("title"):
            pm = f" (PMID {r['pmid']})" if r.get("pmid") else ""
            rwe_titles.append(f"{r['title']}{pm}")
    if rwe_titles:
        ev_lines.append("Real-world evidence: " + " | ".join(rwe_titles))
    else:
        ev_lines.append(
            "No real-world evidence names this brand yet — expected for a recent approval; "
            "real-world evidence typically lags approval by years."
        )
    evidence = " ".join(ev_lines)

    # -- chips -----------------------------------------------------------
    live = sum(
        1
        for s in (
            (payload.get("mechanism") or {}).get("source"),
            eps.get("source"),
            ind.get("source"),
            (payload.get("real_world_evidence") or {}).get("source"),
        )
        if s in _LIVE_SOURCES
    )
    chips = [
        {"label": "Brand", "value": brand},
        {"label": "Analysis", "value": f"{treatment} -> {outcome}"},
        {"label": "Gate", "value": str(gate_decision or "n/a")},
        {"label": "Live sources", "value": f"{live}/4"},
    ]

    return {
        "analysis": analysis,
        "result": result,
        "clinical_position": clinical_position,
        "competitive_position": competitive_position,
        "trial_endpoints": trial_endpoints,
        "evidence": evidence,
        "grounding": chips,
        "context_unavailable": False,
    }


def build_result_only_grounding(
    *,
    brand: str,
    grain: str,
    treatment: str,
    outcome: str,
    ate: Optional[float],
    ate_ci_lower: Optional[float],
    ate_ci_upper: Optional[float],
    gate_decision: Optional[str],
) -> dict[str, Any]:
    """Grounding for the fetch-failed path: the result is all we can honestly
    say. The route renders it through fallback() — never through the LM."""
    unavailable = "The clinical-context sources could not be fetched for this analysis."
    return {
        "analysis": f"Causal analysis of {treatment} -> {outcome} for {brand} at the {grain} grain.",
        "result": _result_sentence(treatment, outcome, ate, ate_ci_lower, ate_ci_upper, gate_decision),
        "clinical_position": unavailable,
        "competitive_position": unavailable,
        "trial_endpoints": unavailable,
        "evidence": unavailable,
        "grounding": [
            {"label": "Brand", "value": brand},
            {"label": "Analysis", "value": f"{treatment} -> {outcome}"},
            {"label": "Gate", "value": str(gate_decision or "n/a")},
            {"label": "Clinical context", "value": "unavailable"},
        ],
        "context_unavailable": True,
    }


# The cheapest fabrication tell for this content type: a citation-shaped
# identifier (PMID / NCT id / DOI / URL) the grounding never contained.
# Plain numbers are NOT scanned — the ATE/CI digits are legitimate here.
# Comparison is on NORMALIZED (scheme, value) tokens, never raw substrings:
# substring membership passed truncated ids (a prefix of a grounded id) and
# rejected legitimate reformattings ("PMID: n" vs the grounded "(PMID n)").
_IDENTIFIER_PATTERNS = (
    ("NCT", re.compile(r"\bNCT(\d{7,8})\b", re.IGNORECASE)),
    ("PMID", re.compile(r"\bPMID[:\s]*(\d{6,9})\b", re.IGNORECASE)),
    ("DOI", re.compile(r"\b(10\.\d{4,9}/[^\s\)\]]+)")),
    # build_grounding composes real-world evidence as title + PMID only, so no
    # URL ever reaches the grounding strings — today any URL in a narrative is
    # fabricated by construction. If a future change composes URLs into the
    # grounding, revisit the trailing-punctuation greediness of \S+.
    ("URL", re.compile(r"(https?://\S+)", re.IGNORECASE)),
)

_GROUNDING_STRING_KEYS = (
    "analysis",
    "result",
    "clinical_position",
    "competitive_position",
    "trial_endpoints",
    "evidence",
)


def _identifier_tokens(text: str) -> set[tuple[str, str]]:
    return {
        (scheme, match.casefold())
        for scheme, pat in _IDENTIFIER_PATTERNS
        for match in pat.findall(text)
    }


def _fabricated_identifiers(narrative: str, g: dict[str, Any]) -> list[str]:
    grounding_text = " ".join(str(g.get(k, "")) for k in _GROUNDING_STRING_KEYS)
    fabricated = _identifier_tokens(narrative) - _identifier_tokens(grounding_text)
    return sorted(f"{scheme} {value}" for scheme, value in fabricated)


def fallback(g: dict[str, Any]) -> dict[str, Any]:
    """Deterministic factual summary of the grounding strings. Public because
    the route calls it directly on the fetch-failed (result-only) path."""
    parts = [g["analysis"], g["result"]]
    if g.get("context_unavailable"):
        parts.append(
            "The clinical-context sources could not be fetched for this analysis, so no "
            "clinical or competitive read can be offered right now."
        )
    else:
        parts.extend(
            [g["clinical_position"], g["competitive_position"], g["trial_endpoints"], g["evidence"]]
        )
    parts.append("(Factual summary — LLM narrative unavailable.)")
    return {
        "insight": "\n\n".join(parts),
        "key_takeaways": [],
        "grounding": g["grounding"],
        "is_fallback": True,
    }


def generate_insight(g: dict[str, Any]) -> dict[str, Any]:
    # Attempt 2 forces a fresh sample (lm_cache=False) — the long-lived API
    # process's in-memory DSPy cache would otherwise replay the identical
    # rejected completion on the retry (home_kpi/exec-brief precedent).
    for attempt in (1, 2):
        pred = run_signature(
            ClinicalNarrativeSignature,
            lm_cache=attempt == 1,
            analysis=g["analysis"],
            result=g["result"],
            clinical_position=g["clinical_position"],
            competitive_position=g["competitive_position"],
            trial_endpoints=g["trial_endpoints"],
            evidence=g["evidence"],
        )
        if pred is None:
            return fallback(g)
        narrative = str(getattr(pred, "narrative", "")).strip()
        if not narrative:
            return fallback(g)
        fabricated = _fabricated_identifiers(narrative, g)
        if not fabricated:
            return {
                "insight": narrative,
                "key_takeaways": [],
                "grounding": g["grounding"],
                "is_fallback": False,
            }
        logger.warning(
            "clinical narrative rejected — fabricated identifiers: %s (attempt %d; %s)",
            fabricated,
            attempt,
            "retrying with a fresh sample" if attempt == 1 else "serving factual fallback",
        )
    return fallback(g)
