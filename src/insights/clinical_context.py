"""Digit-free clinical grounding shared by strategic-insight surfaces.

Commercial outputs do not take place in a clinical vacuum (user directive,
2026-07-12): the HTE strategic insight and the Executive Brief cite the
brand's clinical setting — mechanism of action, indicated disease, label
constraints (boxed-warning presence), and the competitive landscape — as
QUALITATIVE color. The facts are REAL, from the public biomedical/regulatory
fan-out in ``src/services/clinical_context`` (ChEMBL, OpenFDA label,
ClinicalTrials.gov, PubMed; competitor list curated by design).

Two hard rules, mirroring ``causal_context.format_qualitative_context``:

* **Digit-free by construction.** The executive brief's placeholder guard
  fails closed on ANY numeric character in LM output, and the HTE guard
  rejects any numeric claim its grounding cannot vouch for. A digit-bearing
  fragment (pivotal endpoints like "UAS7 at week 12", label populations like
  "12 years and older", doses) is DROPPED, never paraphrased or rounded.
* **Fail-open fetch.** A clinical-context hiccup (unknown brand, provider
  outage, slow fan-out) must never block an insight: the fetch degrades to
  ``None`` and the grounding says clinical context is unavailable.

This package NEVER touches the causal math — context strings feed prompts
only, and their digits (there are none) never enter the guard vocabularies.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# The service fans out to four public REST APIs on a cold cache; each client
# has its own short timeout, but bound the whole fan-out so a pathological
# stall can never hold an insight request hostage.
_FETCH_TIMEOUT_S = 12.0


async def fetch_clinical_payload(brand: Optional[str], outcome: str = "TRx") -> Optional[dict]:
    """The assembled clinical-context payload for ``brand``, or ``None``.

    Wraps the synchronous ``ClinicalContextService`` fan-out in a worker
    thread with an overall timeout. Every failure mode — no brand, unknown
    brand (``KeyError``), provider/client errors, timeout — returns ``None``
    so callers degrade to an honest "no clinical context" grounding.
    """
    if not brand:
        return None
    try:
        from src.services.clinical_context.service import ClinicalContextService

        def _get() -> dict:
            return ClinicalContextService().get_context(brand, outcome)

        return await asyncio.wait_for(asyncio.to_thread(_get), timeout=_FETCH_TIMEOUT_S)
    except Exception as e:  # noqa: BLE001 — fail-open by contract
        logger.warning(
            "clinical context unavailable for %s (%s); insight proceeds without", brand, e
        )
        return None


# Curated, label-derived clinical POSITIONING per brand: the labeled target
# population and line of therapy. Unlike ``format_clinical_context`` (which
# grounds the digit-guarded executive-brief / HTE surfaces and MUST be
# digit-free), this grounds the causal-discovery strategic interpretation — a
# surface that already reports effect figures — so accurate receptor / line-of-
# therapy names (HR+/HER2-, antihistamine-refractory) are kept, not stripped.
# These are REAL prescribing-label facts, NOT invented placeholders, and they
# GATE commercial recommendations by clinical appropriateness: a modeled effect
# favouring a population OUTSIDE the labeled target is clinically off-target even
# when the number looks good (e.g. treatment-naive patients are not the target
# for an antihistamine-refractory indication). Limitations-of-use and
# contraindications are deliberately EXCLUDED (product decision 2026-07-23) —
# this is positioning, not a safety summary.
_CLINICAL_POSITIONING: dict[str, str] = {
    "Remibrutinib": (
        "Labeled target population: chronic spontaneous urticaria that remains "
        "symptomatic despite H1-antihistamine therapy — an antihistamine-refractory, "
        "later-line population, not treatment-naive patients. Treatment-naive or "
        "antihistamine-responsive segments fall outside the label's target even when "
        "their modeled response is favourable, so do not recommend prioritising them."
    ),
    "Fabhalta": (
        "Labeled target population: adults with paroxysmal nocturnal hemoglobinuria, "
        "positioned as an oral monotherapy — including patients switching from anti-C5 "
        "therapy. The commercial target is the diagnosed PNH population; broader anemia "
        "or undiagnosed cohorts are outside the label even when a modeled effect looks strong."
    ),
    "Kisqali": (
        "Labeled target population: HR-positive, HER2-negative breast cancer — advanced or "
        "metastatic disease combined with endocrine therapy, and node-positive early breast "
        "cancer in the adjuvant setting. Endocrine-eligible HR+/HER2- patients are the target; "
        "HER2-positive or hormone-receptor-negative segments are off-label even if modeled "
        "response is high."
    ),
}


def format_clinical_positioning(brand: Optional[str]) -> str:
    """The curated, label-derived target-population + line-of-therapy positioning
    for ``brand`` (empty string if unknown/unbranded).

    Grounds the causal-discovery strategic interpretation so its commercial
    recommendations are GATED by clinical appropriateness — a strong modeled
    effect in a clinically off-target population is not an actionable
    recommendation. Fail-open by contract: an unknown brand yields ``""`` and the
    interpretation proceeds without a clinical gate.
    """
    if not brand:
        return ""
    # Case-insensitive match: a brand-casing drift (e.g. "kisqali") must NOT
    # silently DISABLE the clinical gate — suppressing clinically off-target
    # recommendations is the point, so a silent miss is the failure mode that
    # matters most. A genuinely unknown brand still fails open ("") by contract.
    key = brand.strip().casefold()
    for name, positioning in _CLINICAL_POSITIONING.items():
        if name.casefold() == key:
            return positioning
    return ""


def _digit_free(text: Any) -> Optional[str]:
    """``text`` stripped, if it is a non-empty digit-free string, else None."""
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s or any(ch.isnumeric() for ch in s):
        return None
    return s


def format_clinical_context(payload: Optional[dict]) -> str:
    """One digit-free paragraph of clinical setting, or ``""``.

    Composes only fragments that are digit-free on their own; anything
    carrying a digit (endpoint names, label populations, doses, counts) is
    dropped. The boxed warning is stated by PRESENCE only — its body carries
    case obligations and figures that must not reach the LM.
    """
    if not payload:
        return ""

    brand = _digit_free(payload.get("brand"))
    drug = _digit_free(payload.get("drug_name"))
    disease = _digit_free(payload.get("disease"))
    mechanism = _digit_free((payload.get("mechanism") or {}).get("mechanism_of_action"))

    subject = brand or drug
    if subject is None:
        return ""
    if drug and drug.lower() != subject.lower():
        subject = f"{subject} ({drug})"

    sentences: list[str] = []

    opening = f"Clinical setting for {subject}"
    descriptors: list[str] = []
    if mechanism:
        descriptors.append(mechanism)
    indications = payload.get("approved_indications") or {}
    # Label indication text routinely carries digit-bearing populations; the
    # digit-free disease name stands in for the indicated population.
    indication = next(
        (t for t in (_digit_free(i) for i in indications.get("indications") or []) if t),
        None,
    )
    indicated = indication or disease
    if indicated:
        descriptors.append(f"indicated for {indicated}")
    if not descriptors:
        return ""
    sentences.append(opening + ": " + ", ".join(descriptors) + ".")

    if indications.get("boxed_warning"):
        sentences.append("The FDA label carries a boxed warning.")

    competitors = [
        c
        for c in (
            _digit_free(name)
            for name in (payload.get("competitor_landscape") or {}).get("competitors") or []
        )
        if c
    ][:4]
    if competitors:
        sentences.append("Key competitors (curated reference): " + "; ".join(competitors) + ".")

    sentences.append(
        "Context from public biomedical and regulatory sources; reference "
        "qualitatively — no figures are provided for it."
    )

    text = " ".join(sentences)
    # Belt and braces: composition of digit-free fragments is digit-free, but
    # the guards downstream fail closed, so verify rather than trust.
    if any(ch.isnumeric() for ch in text):
        logger.warning("clinical context formatter produced digits; dropping the context")
        return ""
    return text
