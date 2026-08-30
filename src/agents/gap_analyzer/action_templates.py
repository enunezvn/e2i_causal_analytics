"""Brand-aware ``recommended_action`` templates for the gap_analyzer prioritizer (#1835).

Before #1835 the templates were a metric x difficulty table with only the
segment interpolated, so Kisqali (oncology) and Remibrutinib (CSU) produced the
VERBATIM identical sentence for the same gap, and the Strategic Brief
(``src/insights/executive_brief.py``) narrated the same recommendation for both.

Brand identity and the brand's HCP audience are DERIVED from what the repo
already declares — no brand fact is invented here:

- ``SUPPORTED_BRANDS`` (``src/agents/cohort_constructor/constants.py``): which
  brands exist (the lower-case key is the identity).
- ``Brand`` (``src/ml/synthetic/config.py``): the display name — it MUST match
  the Supabase ``brand_type`` enum, i.e. the value the API stores in
  ``gap_analyses.brand`` and the UI shows.
- ``HCPGenerator.BRAND_SPECIALTY_DIST`` (``src/ml/synthetic/generators/
  hcp_generator.py``): the targeted specialties per brand. MIRRORED below as
  ``BRAND_TARGET_SPECIALTIES`` rather than imported: importing it would pull the
  whole synthetic-generator package (29 modules) into the agent's runtime import
  path; the mirror is pinned to the SSOT by
  ``tests/unit/test_agents/test_gap_analyzer/test_prioritizer_brand_actions_1835.py``
  so any drift fails CI loudly.
- ``INTERVENTION_CATALOG`` (``src/digital_twin/effect/provider.py``): the
  channel vocabulary wherever a template names a channel.

Unknown / missing brand (``competitor``, ``other``, None, a typo) falls OPEN to
the pre-#1835 neutral templates — never a KeyError — so the node keeps
completing for any brand string the API accepts.

Every rendering must stay within ``MAX_ACTION_CHARS`` (the brief truncates the
action at 160 chars); the test enumerates metric x difficulty x gap_type x
brand x longest segment value exhaustively.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from src.agents.cohort_constructor.constants import SUPPORTED_BRANDS
from src.digital_twin.effect.provider import INTERVENTION_CATALOG
from src.ml.synthetic.config import Brand, SpecialtyEnum

# executive_brief._opportunity_line / _lm_opportunity_line: _truncate(action, 160)
MAX_ACTION_CHARS = 160

# Gap-type context appended to every action (unchanged from pre-#1835).
GAP_TYPE_SUFFIXES: Dict[str, str] = {
    "vs_benchmark": " (benchmark-driven)",
    "vs_potential": " (top-decile target)",
    "temporal": " (restore prior performance)",
}

# Mirror of HCPGenerator.BRAND_SPECIALTY_DIST, ordered by targeting share
# (descending; ties keep the SSOT's declaration order). Pinned by
# TestSsotPins.test_target_specialties_mirror_hcp_generator_ordered_by_share.
BRAND_TARGET_SPECIALTIES: Dict[Brand, Tuple[SpecialtyEnum, ...]] = {
    Brand.REMIBRUTINIB: (
        SpecialtyEnum.DERMATOLOGY,  # 0.50
        SpecialtyEnum.ALLERGY_IMMUNOLOGY,  # 0.35 (CSU indication)
        SpecialtyEnum.RHEUMATOLOGY,  # 0.15
    ),
    Brand.FABHALTA: (
        SpecialtyEnum.HEMATOLOGY,  # 0.60 (PNH indication)
        SpecialtyEnum.INTERNAL_MEDICINE,  # 0.30
        SpecialtyEnum.NEUROLOGY,  # 0.10
    ),
    Brand.KISQALI: (SpecialtyEnum.ONCOLOGY,),  # 1.00 (HR+/HER2- breast cancer)
}

# Practitioner noun per specialty (singular; pluralised with a trailing "s").
# Pinned to cover every SpecialtyEnum member.
SPECIALTY_PRACTITIONER: Dict[SpecialtyEnum, str] = {
    SpecialtyEnum.DERMATOLOGY: "dermatologist",
    SpecialtyEnum.HEMATOLOGY: "hematologist",
    SpecialtyEnum.ONCOLOGY: "oncologist",
    SpecialtyEnum.NEUROLOGY: "neurologist",
    SpecialtyEnum.RHEUMATOLOGY: "rheumatologist",
    SpecialtyEnum.INTERNAL_MEDICINE: "internist",
    SpecialtyEnum.GENERAL_PRACTICE: "general practitioner",
    SpecialtyEnum.ALLERGY_IMMUNOLOGY: "allergist",
}

# How many of the brand's top target specialties name the audience. Two keeps
# the audience honest for the split-targeting brands (Remibrutinib 50/35,
# Fabhalta 60/30) while staying inside the 160-char budget.
AUDIENCE_SPECIALTY_COUNT = 2

# Catalog channels the templates name; the label is the catalog's human label
# lower-cased for prose. Pinned: every key must be a SUPPORTED_INTERVENTIONS.
ACTION_CHANNELS: Tuple[str, ...] = (
    "sample_distribution",
    "peer_influence_activation",
    "call_frequency_increase",
    "patient_support_program",
    "digital_engagement",
)
_CATALOG_LABELS: Dict[str, str] = dict(INTERVENTION_CATALOG)
_CHANNEL_WORDS: Dict[str, str] = {ch: _CATALOG_LABELS[ch].lower() for ch in ACTION_CHANNELS}

_BRAND_BY_KEY: Dict[str, Brand] = {b.value.lower(): b for b in Brand}

_DEFAULT = "_default"

# Placeholders: {brand} display name, {audience} singular attributive noun(s)
# ("oncologist", "dermatologist/allergist"), {audiences} plural, {segment_value},
# {segment}, plus one per ACTION_CHANNELS entry (already lower-cased prose).
BRAND_TEMPLATES: Mapping[str, Mapping[str, str]] = {
    "trx": {
        "low": (
            "Launch a {sample_distribution} campaign for {brand} with {audiences} in "
            "{segment_value} ({segment}) to drive TRx growth"
        ),
        "medium": (
            "Implement a multichannel {audience} engagement strategy for {brand} in "
            "{segment_value} to increase TRx"
        ),
        "high": (
            "Execute a market-access and {audience} engagement program for {brand} in "
            "{segment_value} to close the TRx gap"
        ),
    },
    "nrx": {
        "low": (
            "Deploy {brand} educational webinars for {audiences} in {segment_value} to "
            "boost new prescriptions"
        ),
        "medium": (
            "Launch a new-prescriber acquisition campaign for {brand} targeting "
            "{audiences} in {segment_value}"
        ),
        "high": (
            "Develop a {peer_influence_activation} program with {audience} KOLs for "
            "{brand} in {segment_value} for NRx growth"
        ),
    },
    "market_share": {
        "low": (
            "Drive {call_frequency_increase} with {audiences} for {brand} in "
            "{segment_value} to capture share"
        ),
        "medium": (
            "Launch a competitive positioning campaign for {brand} among {audiences} in "
            "{segment_value}"
        ),
        "high": (
            "Execute a full-scale market penetration strategy for {brand} across "
            "{audiences} in {segment_value}"
        ),
    },
    "conversion_rate": {
        "low": (
            "Optimize {brand} {patient_support_program} messaging for {audiences} in "
            "{segment_value}"
        ),
        "medium": (
            "Redesign {brand} patient journey touchpoints with {audiences} for the "
            "{segment_value} segment"
        ),
        "high": (
            "Implement a {patient_support_program} and {audience} enablement program for "
            "{brand} in {segment_value}"
        ),
    },
    "hcp_engagement_score": {
        "low": (
            "Increase {digital_engagement} touchpoints with {audiences} for {brand} in "
            "{segment_value}"
        ),
        "medium": (
            "Launch an omnichannel engagement initiative for {brand} with {audiences} in "
            "{segment_value}"
        ),
        "high": (
            "Build a strategic {audience} partnership program with personalized {brand} "
            "engagement in {segment_value}"
        ),
    },
    _DEFAULT: {
        "low": "Address the {brand} performance gap among {audiences} in {segment_value}",
        "medium": "Implement a targeted {brand} intervention for {audiences} in {segment_value}",
        "high": "Execute a strategic {brand} initiative with {audiences} in {segment_value}",
    },
}

# Pre-#1835 wording, verbatim — the fail-open path for an unknown brand.
NEUTRAL_TEMPLATES: Mapping[str, Mapping[str, str]] = {
    "trx": {
        "low": "Launch targeted sampling campaign in {segment_value} ({segment}) to drive TRx growth",
        "medium": (
            "Implement multichannel engagement strategy for HCPs in {segment_value} to increase TRx"
        ),
        "high": (
            "Execute comprehensive market access and HCP engagement program in {segment_value} "
            "to close TRx gap"
        ),
    },
    "nrx": {
        "low": "Deploy HCP educational webinars in {segment_value} to boost new prescriptions",
        "medium": "Launch new prescriber acquisition campaign targeting {segment_value} specialists",
        "high": "Develop strategic partnership program with KOLs in {segment_value} for NRx growth",
    },
    "market_share": {
        "low": "Increase rep frequency in {segment_value} to capture share",
        "medium": "Launch competitive positioning campaign in {segment_value}",
        "high": (
            "Execute full-scale market penetration strategy in {segment_value} with expanded "
            "resources"
        ),
    },
    "conversion_rate": {
        "low": "Optimize patient starter program messaging for {segment_value}",
        "medium": "Redesign patient journey touchpoints for {segment_value} segment",
        "high": (
            "Implement comprehensive patient support and HCP enablement program in {segment_value}"
        ),
    },
    "hcp_engagement_score": {
        "low": "Increase digital touchpoints with HCPs in {segment_value}",
        "medium": "Launch omnichannel engagement initiative for {segment_value} providers",
        "high": (
            "Build strategic HCP partnership program with personalized engagement for "
            "{segment_value}"
        ),
    },
    _DEFAULT: {
        "low": "Address performance gap in {segment_value}",
        "medium": "Implement targeted intervention in {segment_value}",
        "high": "Execute strategic initiative in {segment_value}",
    },
}

# Metrics with a dedicated template (everything else renders the default row).
ACTION_METRICS: Tuple[str, ...] = tuple(m for m in BRAND_TEMPLATES if m != _DEFAULT)


@dataclass(frozen=True)
class BrandActionContext:
    """What a brand-aware template interpolates."""

    key: str  # SUPPORTED_BRANDS key (lower-case identity)
    name: str  # display name = Supabase brand_type enum value
    audience: str  # singular attributive, e.g. "dermatologist/allergist"

    @property
    def audiences(self) -> str:
        return "/".join(f"{noun}s" for noun in self.audience.split("/"))


def brand_action_context(brand: Optional[str]) -> Optional[BrandActionContext]:
    """Resolve a request brand string (any casing / whitespace) to its context.

    Returns None — the caller then uses the neutral templates — when the brand is
    absent, not a SUPPORTED_BRANDS key, or has no target-specialty entry.
    """
    if not brand:
        return None
    key = str(brand).strip().lower()
    if key not in SUPPORTED_BRANDS:
        return None
    enum = _BRAND_BY_KEY.get(key)
    if enum is None:
        return None
    specialties = BRAND_TARGET_SPECIALTIES.get(enum)
    if not specialties:
        return None
    nouns = [SPECIALTY_PRACTITIONER[s] for s in specialties[:AUDIENCE_SPECIALTY_COUNT]]
    return BrandActionContext(key=key, name=enum.value, audience="/".join(nouns))


def render_action(
    *,
    metric: str,
    difficulty: str,
    segment: str,
    segment_value: object,
    gap_type: str,
    brand: Optional[str],
) -> str:
    """Render the recommended action for one gap.

    Brand-aware when ``brand`` resolves via :func:`brand_action_context`;
    otherwise the pre-#1835 neutral wording. The gap-type suffix is appended in
    both cases.
    """
    context = brand_action_context(brand)
    table = BRAND_TEMPLATES if context is not None else NEUTRAL_TEMPLATES
    templates = table.get(metric, table[_DEFAULT])
    template = templates.get(difficulty, templates["medium"])
    fields: Dict[str, object] = {
        "segment_value": segment_value,
        "segment": segment,
        **_CHANNEL_WORDS,
    }
    if context is not None:
        fields.update(brand=context.name, audience=context.audience, audiences=context.audiences)
    return template.format(**fields) + GAP_TYPE_SUFFIXES.get(gap_type, "")
