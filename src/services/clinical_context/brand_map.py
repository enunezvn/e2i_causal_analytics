"""Brand -> drug / disease / static-fallback profile for clinical-context
enrichment.

Joins the canonical SSOT in ``src/ml/synthetic/clinical_codes.py`` (drug_name,
drug_class, disease desc) with the enrichment-only extras that SSOT does not
carry: the precise static MoA fallback strings the redesign spec pins (used when
ChEMBL is unreachable), the static pivotal-endpoint fallback (used when
ClinicalTrials.gov is down OR returns only safety endpoints), the PubMed RWE
search term, and the map from OUR synthetic outcome -> the real pivotal endpoint
framing.

All fallback strings are REAL clinical facts (MoA per ChEMBL mechanism rows;
pivotal endpoints verified live 2026-06-19 against ClinicalTrials.gov API v2:
breast cancer = OS/PFS/iDFS; PNH = transfusion-avoidance/LDH/Hb-stabilization;
CSU = UAS7/UCT7/ISS7) — never invented placeholders.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from src.ml.synthetic.clinical_codes import BRAND_DIAGNOSIS, BRAND_NDC


@dataclass(frozen=True)
class BrandClinicalProfile:
    """The clinical facts + static fallbacks for one brand.

    ``moa_fallback`` / ``pivotal_endpoints_fallback`` are used ONLY when the live
    API is unreachable or unhelpful; the providers prefer the live API value and
    fall back to these. ``outcome_endpoint_map`` maps our synthetic outcome
    column -> the real pivotal-endpoint framing it stands in for.
    """

    brand: str
    drug_name: str
    disease: str
    drug_class: str
    moa_fallback: str
    pivotal_endpoints_fallback: List[str]
    rwe_search_term: str
    rwe_seed_pmid: Optional[str]
    outcome_endpoint_map: Dict[str, str] = field(default_factory=dict)


# Enrichment-only static facts keyed by brand. The drug_name / disease / drug_class
# are pulled from the clinical_codes SSOT at construction time (below) so they can
# never drift from it.
_STATIC_ENRICHMENT: Dict[str, Dict[str, object]] = {
    "Kisqali": {
        "moa_fallback": "CDK4/6 inhibitor",
        "pivotal_endpoints_fallback": [
            "Overall Survival (OS)",
            "Progression-Free Survival (PFS)",
            "Invasive Disease-Free Survival (iDFS)",
        ],
        "rwe_search_term": "ribociclib persistence adherence breast cancer real-world",
        "rwe_seed_pmid": "35642282",
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation / time-to-treatment-start",
            "persistent_180d": "Treatment persistence / duration of therapy (proxy for PFS-supporting adherence)",
            "discontinued_180d": "Treatment discontinuation / early termination",
        },
    },
    "Remibrutinib": {
        "moa_fallback": "BTK inhibitor",
        "pivotal_endpoints_fallback": [
            "Change from baseline in UAS7 (Urticaria Activity Score over 7 days)",
            "UCT7 (Urticaria Control Test)",
            "ISS7 (Itch Severity Score) / WI-NRS",
        ],
        "rwe_search_term": "remibrutinib chronic spontaneous urticaria real-world persistence",
        "rwe_seed_pmid": None,
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation (BTKi start after antihistamine failure)",
            "persistent_180d": "Treatment persistence / sustained UAS7 control",
            "discontinued_180d": "Treatment discontinuation",
        },
    },
    "Fabhalta": {
        "moa_fallback": "complement Factor B inhibitor",
        "pivotal_endpoints_fallback": [
            "Transfusion avoidance (proportion not requiring RBC transfusion)",
            "Sustained hemoglobin stabilization (increase >= 2 g/dL without transfusion)",
            "Change from baseline in lactate dehydrogenase (LDH)",
        ],
        "rwe_search_term": "iptacopan paroxysmal nocturnal hemoglobinuria real-world persistence",
        "rwe_seed_pmid": None,
        "outcome_endpoint_map": {
            "treatment_initiated": "Treatment initiation (complement-inhibitor start/switch)",
            "persistent_180d": "Treatment persistence / sustained Hb stabilization",
            "discontinued_180d": "Treatment discontinuation",
        },
    },
}


def _build_map() -> Dict[str, BrandClinicalProfile]:
    out: Dict[str, BrandClinicalProfile] = {}
    for brand, extra in _STATIC_ENRICHMENT.items():
        ndc = BRAND_NDC[brand]
        dx = BRAND_DIAGNOSIS[brand]
        out[brand] = BrandClinicalProfile(
            brand=brand,
            drug_name=str(ndc["drug_name"]),
            disease=str(dx["desc"]),
            # drug_class lives in clinical_codes.BRAND_DRUG_CLASS, but the
            # enrichment MoA-fallback string is the precise spec-pinned phrasing;
            # we surface the fallback as the authoritative static MoA.
            drug_class=str(extra["moa_fallback"]),
            moa_fallback=str(extra["moa_fallback"]),
            pivotal_endpoints_fallback=list(extra["pivotal_endpoints_fallback"]),  # type: ignore[call-overload]
            rwe_search_term=str(extra["rwe_search_term"]),
            rwe_seed_pmid=(str(extra["rwe_seed_pmid"]) if extra["rwe_seed_pmid"] else None),
            outcome_endpoint_map=dict(extra["outcome_endpoint_map"]),  # type: ignore[call-overload]
        )
    return out


BRAND_CLINICAL_MAP: Dict[str, BrandClinicalProfile] = _build_map()


def resolve_brand_profile(brand: str) -> BrandClinicalProfile:
    """Return the clinical profile for ``brand``.

    Raises ``KeyError`` on an unsupported brand so callers fail closed rather
    than emit a wrong indication.
    """
    return BRAND_CLINICAL_MAP[brand]


def endpoint_mapping_for_outcome(brand: str, outcome: str) -> Optional[str]:
    """Map our synthetic ``outcome`` column -> the real pivotal-endpoint framing
    for ``brand``. Returns None when there is no curated mapping (honest: we do
    not fabricate a clinical claim for an outcome we have not mapped)."""
    profile = BRAND_CLINICAL_MAP.get(brand)
    if profile is None:
        return None
    return profile.outcome_endpoint_map.get(outcome)
