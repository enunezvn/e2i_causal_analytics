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

    Therapy-label + competitor fields (added 2026-06-20):
    ``indications_fallback`` — curated approved indication strings (used when
    OpenFDA label is unreachable); ``limitations_fallback`` / ``boxed_warning_fallback``
    — curated LoU / boxed-warning text or None when absent for this drug;
    ``competitor_map`` — dict keyed by disease string (lowercased) -> list of
    competitor ``"Brand (generic)"`` strings within the same therapeutic class.
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
    # Therapy-label fallback fields — default-empty so existing construction is unaffected.
    indications_fallback: List[str] = field(default_factory=list)
    limitations_fallback: Optional[str] = None
    boxed_warning_fallback: Optional[str] = None
    # key = disease string lowercased → list of competitor "Brand (generic)" strings
    competitor_map: Dict[str, List[str]] = field(default_factory=dict)
    # Curated brand-SPECIFIC seminal real-world-evidence citation (verified against
    # PubMed). Surfaced as its OWN "seminal RWE" link so the brand of interest always
    # has a brand-faithful reference, INDEPENDENT of what the live PubMed relevance
    # search returns — that search can rank a competitor / CDK4-6-class-comparison
    # paper first (the reported abemaciclib-for-Kisqali confusion). Keys:
    # pmid, title, journal, year, doi. None when no seminal RWE is curated yet.
    seminal_rwe: Optional[Dict[str, str]] = None


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
        # Therapy-label fallback (OpenFDA SPL / prescribing information, verified 2026-06-20)
        "indications_fallback": [
            "HR+/HER2- advanced or metastatic breast cancer (with an aromatase inhibitor or fulvestrant)",
            "HR+/HER2- node-positive early breast cancer, adjuvant (with an aromatase inhibitor)",
        ],
        "limitations_fallback": None,
        "boxed_warning_fallback": None,
        # disease key = "malignant neoplasm of breast" (BRAND_DIAGNOSIS["Kisqali"]["desc"].lower())
        "competitor_map": {
            "malignant neoplasm of breast": [
                "Ibrance (palbociclib)",
                "Verzenio (abemaciclib)",
            ],  # ATC L01EF CDK4/6 inhibitors (OpenFDA/RxClass probe-confirmed)
        },
        # Curated brand-SPECIFIC seminal RWE (ribociclib-only, no competitor in the
        # title — verified against PubMed 2026-07-10). A real-world managed-access
        # study mirroring the MONALEESA-7 pivotal population. Chosen over a live
        # relevance hit precisely because relevance ranks CDK4/6-class comparisons
        # (palbociclib/abemaciclib) first, which read as competitor papers.
        "seminal_rwe": {
            "pmid": "36135090",
            "title": (
                "Real-World Clinical Outcomes of Ribociclib in Combination with a "
                "Non-Steroidal Aromatase Inhibitor and a Luteinizing Hormone-Releasing "
                "Hormone Agonist in Premenopausal HR+/HER2- Advanced Breast Cancer "
                "Patients: An Italian Managed Access Program"
            ),
            "journal": "Current Oncology",
            "year": "2022",
            "doi": "10.3390/curroncol29090521",
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
        # Therapy-label fallback (FDA prescribing information, verified 2026-06-20)
        "indications_fallback": [
            "Chronic spontaneous urticaria (CSU) in adults who remain symptomatic despite H1-antihistamine treatment",
        ],
        "limitations_fallback": "Not indicated for other forms of urticaria.",
        "boxed_warning_fallback": None,
        # disease key = "chronic spontaneous urticaria" (BRAND_DIAGNOSIS["Remibrutinib"]["desc"].lower())
        "competitor_map": {
            "chronic spontaneous urticaria": [
                "Xolair (omalizumab)",
                "Dupixent (dupilumab)",
            ],  # CSU biologics approved for CSU (omalizumab FDA 2014; dupilumab FDA 2025)
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
        # Therapy-label fallback (FDA prescribing information, verified 2026-06-20)
        "indications_fallback": [
            "Paroxysmal nocturnal hemoglobinuria (PNH)",
            "Primary IgA nephropathy (IgAN), to reduce proteinuria",
        ],
        "limitations_fallback": None,
        "boxed_warning_fallback": (
            "Serious infections caused by encapsulated bacteria (e.g., S. pneumoniae, "
            "N. meningitidis, H. influenzae) can occur; complete or update vaccinations "
            "before initiation."
        ),
        # disease key = "paroxysmal nocturnal hemoglobinuria" (BRAND_DIAGNOSIS["Fabhalta"]["desc"].lower())
        # A second key is included for IgAN so providers can resolve either indication.
        "competitor_map": {
            "paroxysmal nocturnal hemoglobinuria": [
                "Soliris (eculizumab)",
                "Ultomiris (ravulizumab)",
                "Empaveli (pegcetacoplan)",
                "Voydeya (danicopan)",
            ],  # Complement inhibitors approved for PNH (FDA/EMA probe-confirmed)
            "primary iga nephropathy": [
                "Tarpeyo (budesonide)",
                "Filspari (sparsentan)",
            ],  # IgAN-approved therapies (FDA 2023-2024)
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
            # Therapy-label + competitor fields (Task 2, 2026-06-20)
            indications_fallback=list(extra.get("indications_fallback", [])),  # type: ignore[call-overload]
            limitations_fallback=(
                str(extra["limitations_fallback"]) if extra.get("limitations_fallback") else None
            ),
            boxed_warning_fallback=(
                str(extra["boxed_warning_fallback"])
                if extra.get("boxed_warning_fallback")
                else None
            ),
            competitor_map={k: list(v) for k, v in extra.get("competitor_map", {}).items()},  # type: ignore[attr-defined]
            seminal_rwe=(
                dict(extra["seminal_rwe"]) if extra.get("seminal_rwe") else None  # type: ignore[call-overload]
            ),
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
